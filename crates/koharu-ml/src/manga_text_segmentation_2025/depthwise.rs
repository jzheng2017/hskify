//! Channel-independent convolution in one CUDA launch. Candle's grouped
//! convolution launches a separate convolution and temporary tensor per channel.
//! Keep the same FP32 accumulation and round only when storing the model dtype.

use std::sync::OnceLock;

use candle_core::{
    CpuStorage, CustomOp2, Layout, Result, Shape,
    backend::BackendStorage,
    bail,
    cuda_backend::{CudaDType, CudaStorage, CudaStorageSlice},
};
use cudarc::{
    driver::{DeviceRepr, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use candle_nn::{Conv2d, Module};

const CUDA_SOURCE: &str = include_str!("depthwise.cu");

pub(super) fn forward(conv: &Conv2d, input: &candle_core::Tensor) -> Result<candle_core::Tensor> {
    if conv.config().groups == 1 {
        return conv.forward(input);
    }
    let op = Depthwise {
        padding: conv.config().padding,
        stride: conv.config().stride,
        dilation: conv.config().dilation,
    };
    let mut output = input
        .contiguous()?
        .apply_op2_no_bwd(&conv.weight().contiguous()?, &op)?;
    if let Some(bias) = conv.bias() {
        output = output.broadcast_add(&bias.reshape((1, bias.elem_count(), 1, 1))?)?;
    }
    Ok(output)
}

struct Depthwise {
    padding: usize,
    stride: usize,
    dilation: usize,
}

impl CustomOp2 for Depthwise {
    fn name(&self) -> &'static str {
        "manga-depthwise-convolution"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        bail!("depthwise CUDA operation requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weights: &CudaStorage,
        weight_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, channels, height, width) = input_layout.shape().dims4()?;
        let (filters, per_filter, kernel_height, kernel_width) = weight_layout.shape().dims4()?;
        if channels != filters || per_filter != 1 || batch == 0 || channels == 0 {
            bail!("depthwise convolution requires one filter per input channel")
        }
        if input.device().id() != weights.device().id() {
            bail!("depthwise input and weights must share a CUDA device")
        }
        let output_height = output_dimension(height, kernel_height, self)?;
        let output_width = output_dimension(width, kernel_width, self)?;
        let shape = Shape::from((batch, channels, output_height, output_width));
        let elements = [batch, channels, output_height, output_width]
            .into_iter()
            .try_fold(1usize, |size, dim| size.checked_mul(dim))
            .ok_or_else(|| candle_core::Error::Msg("depthwise output overflow".into()))?;
        let count = u32::try_from(elements)
            .map_err(|_| candle_core::Error::Msg("depthwise output exceeds CUDA bounds".into()))?;
        let geometry = [
            channels,
            height,
            width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            self.padding,
            self.stride,
            self.dilation,
        ];
        let mut args = [0u32; 10];
        for (arg, value) in args.iter_mut().zip(geometry) {
            *arg = u32::try_from(value).map_err(|_| {
                candle_core::Error::Msg("depthwise geometry exceeds CUDA bounds".into())
            })?;
        }
        let input_range = input_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("depthwise input must be contiguous".into()))?;
        let weight_range = weight_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("depthwise weights must be contiguous".into())
        })?;
        if input_range.1 - input_range.0 > u32::MAX as usize
            || weight_range.1 - weight_range.0 > u32::MAX as usize
        {
            bail!("depthwise input storage exceeds CUDA indexing bounds")
        }
        match (&input.slice, &weights.slice) {
            (CudaStorageSlice::BF16(input_data), CudaStorageSlice::BF16(weight_data)) => launch(
                input,
                input_data,
                weight_data,
                input_range,
                weight_range,
                &args,
                count,
                shape,
                "depthwise_bf16",
            ),
            (CudaStorageSlice::F32(input_data), CudaStorageSlice::F32(weight_data)) => launch(
                input,
                input_data,
                weight_data,
                input_range,
                weight_range,
                &args,
                count,
                shape,
                "depthwise_f32",
            ),
            _ => bail!("depthwise model input and weights require matching BF16 or F32 dtype"),
        }
    }
}

fn output_dimension(input: usize, kernel: usize, op: &Depthwise) -> Result<usize> {
    if input == 0 || kernel == 0 || op.stride == 0 || op.dilation == 0 {
        bail!("depthwise convolution requires nonzero geometry")
    }
    let padded = input
        .checked_add(
            op.padding
                .checked_mul(2)
                .ok_or_else(|| candle_core::Error::Msg("depthwise padding overflow".into()))?,
        )
        .ok_or_else(|| candle_core::Error::Msg("depthwise dimension overflow".into()))?;
    let effective = op
        .dilation
        .checked_mul(kernel - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| candle_core::Error::Msg("depthwise kernel overflow".into()))?;
    if padded > i32::MAX as usize || effective > i32::MAX as usize {
        bail!("depthwise padded geometry exceeds signed CUDA coordinates")
    }
    let remaining = padded
        .checked_sub(effective)
        .ok_or_else(|| candle_core::Error::Msg("depthwise kernel exceeds padded input".into()))?;
    Ok(remaining / op.stride + 1)
}

#[allow(clippy::too_many_arguments)]
fn launch<T: CudaDType + DeviceRepr>(
    input: &CudaStorage,
    input_data: &cudarc::driver::CudaSlice<T>,
    weight_data: &cudarc::driver::CudaSlice<T>,
    input_range: (usize, usize),
    weight_range: (usize, usize),
    args: &[u32],
    count: u32,
    shape: Shape,
    name: &str,
) -> Result<(CudaStorage, Shape)> {
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let ptx = PTX
        .get_or_init(|| {
            compile_ptx_with_opts(
                CUDA_SOURCE,
                CompileOptions {
                    use_fast_math: Some(false),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| candle_core::Error::Msg(error.clone()))?;
    let device = input.device();
    let function = device.get_or_load_custom_func(name, "manga-depthwise-v1", ptx)?;
    let source = input_data.slice(input_range.0..input_range.1);
    let weights = weight_data.slice(weight_range.0..weight_range.1);
    // Every output element is written by exactly one thread before use on the
    // same tracked stream. The validated contiguous views retain their owners.
    let mut output = unsafe { device.alloc::<T>(count as usize)? };
    let mut builder = function.builder();
    builder
        .arg(&source)
        .arg(&weights)
        .arg(&mut output)
        .arg(&count);
    for arg in args {
        builder.arg(arg);
    }
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (count.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|error| candle_core::Error::Cuda(Box::new(error)))?;
    Ok((CudaStorage::wrap_cuda_slice(output, device.clone()), shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Conv2dConfig;

    #[test]
    fn unsafe_depthwise_geometry_is_rejected_before_launch() {
        let op = Depthwise {
            padding: 1,
            stride: 1,
            dilation: 1,
        };
        assert!(output_dimension(0, 3, &op).is_err());
        assert!(output_dimension(1, 5, &op).is_err());
        assert!(output_dimension(usize::MAX, 3, &op).is_err());
        assert!(output_dimension(i32::MAX as usize, 3, &op).is_err());
        assert!(output_dimension(10, 3, &Depthwise { stride: 0, ..op }).is_err());
    }

    #[tokio::test]
    #[ignore = "requires the matching packaged CUDA runtime in HSKIFY_RESOURCES_DIR"]
    async fn cuda_depthwise_matches_cpu_reference_for_precision_strides_and_views()
    -> anyhow::Result<()> {
        let resources = std::env::var_os("HSKIFY_RESOURCES_DIR")
            .ok_or_else(|| anyhow::anyhow!("HSKIFY_RESOURCES_DIR is required"))?;
        let runtime = koharu_runtime::RuntimeManager::new(
            resources,
            koharu_runtime::ComputePolicy::CudaRequired,
        )?;
        runtime.prepare().await?;
        let cuda = Device::new_cuda(0)?;
        for dtype in [DType::F32, DType::BF16] {
            for (kernel, stride, padding, dilation) in [(3, 1, 1, 1), (3, 2, 1, 1), (5, 2, 4, 2)] {
                let channels = 7;
                let data = (0..3 * channels * 13 * 11)
                    .map(|i| (i as f32 * 0.11).sin())
                    .collect::<Vec<_>>();
                let weights = (0..(channels + 1) * kernel * kernel)
                    .map(|i| (i % 11) as f32 * 0.03 - 0.15)
                    .collect::<Vec<_>>();
                let input = Tensor::from_vec(data, (3, channels, 13, 11), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F32)?;
                let weights =
                    Tensor::from_vec(weights, (channels + 1, 1, kernel, kernel), &Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_dtype(DType::F32)?;
                let config = Conv2dConfig {
                    padding,
                    stride,
                    dilation,
                    groups: channels,
                    ..Default::default()
                };
                let input_view = input.narrow(0, 1, 2)?.narrow(2, 2, 9)?.transpose(2, 3)?;
                let weight_view = weights.narrow(0, 1, channels)?;
                let expected = Conv2d::new(weight_view.clone(), None, config)
                    .forward(&input_view)?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let cuda_input = input
                    .to_device(&cuda)?
                    .to_dtype(dtype)?
                    .narrow(0, 1, 2)?
                    .narrow(2, 2, 9)?
                    .transpose(2, 3)?;
                let cuda_weights = weights
                    .to_device(&cuda)?
                    .to_dtype(dtype)?
                    .narrow(0, 1, channels)?;
                let conv = Conv2d::new(cuda_weights, None, config);
                let actual = forward(&conv, &cuda_input)?
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(actual.len(), expected.len());
                let tolerance = if dtype == DType::BF16 { 0.008 } else { 0.00001 };
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    error <= tolerance,
                    "dtype={dtype:?} kernel={kernel} stride={stride} dilation={dilation}: error={error}"
                );
            }
        }
        Ok(())
    }
}
