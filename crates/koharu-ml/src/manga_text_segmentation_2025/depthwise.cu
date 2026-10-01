// No headers or extra runtime files: NVRTC and its device intrinsics are already
// shipped with the pinned CUDA runtime. Storage is NCHW, weights are C x 1 x K x K.
template<typename T> __device__ float read_value(T value);
template<> __device__ float read_value<float>(float value) { return value; }
template<> __device__ float read_value<unsigned short>(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}
template<typename T> __device__ T store_value(float value);
template<> __device__ float store_value<float>(float value) { return value; }
template<> __device__ unsigned short store_value<unsigned short>(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return (bits >> 16) | 0x0040U;
    return (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
}

template<typename T> __device__ void convolve(
    const T* input, const T* weights, T* output, unsigned int count,
    unsigned int channels, unsigned int height, unsigned int width,
    unsigned int out_height, unsigned int out_width,
    unsigned int kernel_height, unsigned int kernel_width,
    unsigned int padding, unsigned int stride, unsigned int dilation
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int x = index % out_width;
    unsigned int y = (index / out_width) % out_height;
    unsigned int plane = index / (out_width * out_height);
    unsigned int channel = plane % channels;
    float sum = 0.0f;
    for (unsigned int ky = 0; ky < kernel_height; ++ky) {
        int iy = (int)(y * stride + ky * dilation) - (int)padding;
        if (iy < 0 || iy >= (int)height) continue;
        for (unsigned int kx = 0; kx < kernel_width; ++kx) {
            int ix = (int)(x * stride + kx * dilation) - (int)padding;
            if (ix < 0 || ix >= (int)width) continue;
            float source = read_value(input[(plane * height + iy) * width + ix]);
            float weight = read_value(weights[(channel * kernel_height + ky) * kernel_width + kx]);
            sum = __fmaf_rn(source, weight, sum);
        }
    }
    output[index] = store_value<T>(sum);
}

#define DEFINE_DEPTHWISE(NAME, TYPE) \
extern "C" __global__ void NAME( \
    const TYPE* input, const TYPE* weights, TYPE* output, unsigned int count, \
    unsigned int channels, unsigned int height, unsigned int width, \
    unsigned int out_height, unsigned int out_width, \
    unsigned int kernel_height, unsigned int kernel_width, \
    unsigned int padding, unsigned int stride, unsigned int dilation \
) { convolve(input, weights, output, count, channels, height, width, \
    out_height, out_width, kernel_height, kernel_width, padding, stride, dilation); }

DEFINE_DEPTHWISE(depthwise_f32, float)
DEFINE_DEPTHWISE(depthwise_bf16, unsigned short)
