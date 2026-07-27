use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};

use koharu_runtime::RuntimeManager;

use crate::prompt::PromptRenderer;
use crate::safe::context::LlamaContext;
use crate::safe::context::params::LlamaContextParams;
use crate::safe::llama_backend::LlamaBackend;
use crate::safe::llama_batch::LlamaBatch;
use crate::safe::model::params::LlamaModelParams;
use crate::safe::model::{AddBos, LlamaModel};
use crate::safe::sampling::LlamaSampler;
use crate::safe::token::LlamaToken;
use crate::{Language, ModelId};

const DEFAULT_GPU_LAYERS: u32 = 1000;
const RESIDENT_CONTEXT_TOKENS: u32 = 1280;
const RESIDENT_PROMPT_BATCH_TOKENS: usize = RESIDENT_CONTEXT_TOKENS as usize;
const MAX_UBATCH: u32 = 256;
const SAKURA_QWEN_CORRECT_EOS_ID: i32 = 151645;
static NEVER_CANCELLED: AtomicBool = AtomicBool::new(false);

pub struct Llm {
    model_id: ModelId,
    prompt_renderer: PromptRenderer,
    eos_token: LlamaToken,
    // SAFETY INVARIANT: `session_context` borrows the pointee in `model`.
    // The model is boxed, so moving `Llm` never moves that pointee. Fields are
    // dropped in declaration order, which releases the context before the
    // model and the backend. `model` is never replaced while the context lives.
    session_context: LlamaContext<'static>,
    prompt_batch: LlamaBatch<'static>,
    token_batch: LlamaBatch<'static>,
    model: Box<LlamaModel>,
    _backend: Arc<LlamaBackend>,
}

// The resident context and batches are private and can only be mutated through
// generation methods that require `&mut Llm`. Shared methods access immutable
// model metadata/tokenization only. Application code additionally serializes
// generation behind its model-state write lock.
unsafe impl Sync for Llm {}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub seed: u64,
    pub split_prompt: bool,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub presence_penalty: f32,
    /// Optional llama.cpp GBNF constraint applied to every sampled token.
    pub grammar: Option<Grammar>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    pub source: String,
    pub root: String,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 1000,
            temperature: 0.1,
            top_k: None,
            top_p: None,
            min_p: None,
            seed: 299792458,
            split_prompt: false,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            grammar: None,
        }
    }
}

impl GenerateOptions {
    /// Build deterministic, unpenalized greedy generation options.
    ///
    /// This is intentionally independent of model-family defaults: callers
    /// that need reproducible structured output should not inherit sampling
    /// filters or presence/repetition penalties from a model preset.
    #[must_use]
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            grammar: None,
            ..Self::default()
        }
    }
}

impl Llm {
    pub async fn load(
        runtime: &RuntimeManager,
        id: ModelId,
        cpu: bool,
        backend: Arc<LlamaBackend>,
    ) -> Result<Self> {
        crate::sys::initialize(runtime)
            .context("failed to initialize llama.cpp runtime bindings")?;
        let model_path = id.get(runtime).await?;

        Self::load_owned_path(id, cpu, model_path, backend, crate::inference_threads()).await
    }

    /// Load a known model family from an already-downloaded GGUF file.
    ///
    /// Normal product code should use [`Self::load`]. This entry point is
    /// useful for opt-in local smoke tests and managed model-pack installers
    /// that have already verified the file.
    pub async fn load_file(
        runtime: &RuntimeManager,
        id: ModelId,
        cpu: bool,
        model_path: PathBuf,
        backend: Arc<LlamaBackend>,
    ) -> Result<Self> {
        crate::sys::initialize(runtime)
            .context("failed to initialize llama.cpp runtime bindings")?;
        Self::load_owned_path(id, cpu, model_path, backend, crate::inference_threads()).await
    }

    pub async fn load_file_with_threads(
        runtime: &RuntimeManager,
        id: ModelId,
        cpu: bool,
        model_path: PathBuf,
        backend: Arc<LlamaBackend>,
        inference_threads: i32,
    ) -> Result<Self> {
        if inference_threads <= 0 {
            bail!("llama.cpp inference thread count must be positive");
        }
        crate::sys::initialize(runtime)
            .context("failed to initialize llama.cpp runtime bindings")?;
        Self::load_owned_path(id, cpu, model_path, backend, inference_threads).await
    }

    async fn load_owned_path(
        id: ModelId,
        cpu: bool,
        model_path: PathBuf,
        backend: Arc<LlamaBackend>,
        inference_threads: i32,
    ) -> Result<Self> {
        tokio::task::spawn_blocking(move || {
            Self::load_from_path(id, cpu, model_path, backend, inference_threads)
        })
        .await
        .context("failed to join llama.cpp model loading task")?
    }

    fn load_from_path(
        id: ModelId,
        cpu: bool,
        model_path: PathBuf,
        backend: Arc<LlamaBackend>,
        inference_threads: i32,
    ) -> Result<Self> {
        let model_params = model_params(cpu, backend.as_ref())?;
        let model = Box::new(
            LlamaModel::load_from_file(backend.as_ref(), &model_path, &model_params)
                .with_context(|| format!("unable to load model from `{}`", model_path.display()))?,
        );

        let chat_template = model
            .meta_val_str("tokenizer.ggml.chat_template")
            .or_else(|_| model.meta_val_str("tokenizer.chat_template"))
            .context("missing chat template in GGUF metadata")?;

        let bos_token = token_text(&model, model.token_bos());
        let (eos_token, eos_text) = eos_token_for(id, &model);
        let prompt_renderer = PromptRenderer::new(id, chat_template, bos_token, eos_text);
        let session_context = model
            .new_context(backend.as_ref(), resident_context_params(inference_threads))
            .context("unable to create permanent llama.cpp context")?;
        // `LlamaContext` stores only a reference to the model pointee. The
        // pointee is kept at a stable address by `Box`, and the field ordering
        // above guarantees that the context is dropped first.
        let session_context = unsafe {
            std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(session_context)
        };

        Ok(Self {
            model_id: id,
            prompt_renderer,
            eos_token,
            session_context,
            prompt_batch: LlamaBatch::new(RESIDENT_PROMPT_BATCH_TOKENS, 1),
            token_batch: LlamaBatch::new(1, 1),
            model,
            _backend: backend,
        })
    }

    pub fn id(&self) -> ModelId {
        self.model_id
    }

    /// Count model tokens without adding a beginning-of-sequence token.
    ///
    /// The direct browser translator uses this to enforce its context budget
    /// against the resident model's real tokenizer rather than a character or
    /// whitespace estimate.
    pub fn token_count(&self, text: &str) -> Result<usize> {
        self.model
            .str_to_token(text, AddBos::Never)
            .map(|tokens| tokens.len())
            .context("failed to count model tokens")
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        target_language: Language,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        self.generate_with_cancel(
            prompt,
            opts,
            target_language,
            system_prompt,
            &NEVER_CANCELLED,
        )
    }

    /// Generate with the normal translation prompt behavior while observing
    /// an external cancellation flag between llama.cpp decode calls.
    pub fn generate_with_cancel(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        target_language: Language,
        system_prompt: Option<&str>,
        cancel: &AtomicBool,
    ) -> Result<String> {
        self.generate_inner(
            prompt,
            opts,
            target_language,
            system_prompt,
            false,
            cancel,
            |_| Ok(()),
        )
    }

    /// Generate with an exact system prompt and optional GBNF constraint.
    ///
    /// Unlike [`Self::generate`], this does not append Koharu's legacy
    /// numbered-block instructions to the supplied system prompt.
    pub fn generate_constrained(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        target_language: Language,
        system_prompt: &str,
        cancel: &AtomicBool,
    ) -> Result<String> {
        self.generate_inner(
            prompt,
            opts,
            target_language,
            Some(system_prompt),
            true,
            cancel,
            |_| Ok(()),
        )
    }

    /// Generate with an exact system prompt and report decoded UTF-8 pieces as
    /// soon as llama.cpp produces them.
    ///
    /// The callback runs synchronously on the sole inference worker. It must
    /// therefore remain non-blocking; browser publication only appends to an
    /// in-memory update log.
    pub fn generate_constrained_streaming(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        target_language: Language,
        system_prompt: &str,
        cancel: &AtomicBool,
        on_piece: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        self.generate_inner(
            prompt,
            opts,
            target_language,
            Some(system_prompt),
            true,
            cancel,
            on_piece,
        )
    }

    fn generate_inner(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        target_language: Language,
        system_prompt: Option<&str>,
        exact_system_prompt: bool,
        cancel: &AtomicBool,
        mut on_piece: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        check_cancelled(cancel)?;
        if opts.max_tokens == 0 {
            return Ok(String::new());
        }

        let prompt = if exact_system_prompt {
            self.prompt_renderer.format_chat_prompt_exact_system(
                prompt.to_string(),
                target_language,
                system_prompt,
            )?
        } else {
            self.prompt_renderer.format_chat_prompt(
                prompt.to_string(),
                target_language,
                system_prompt,
            )?
        };
        tracing::debug!("Generating with prompt:\n{}", prompt);

        let prompt_tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never)
            .context("failed to tokenize prompt")?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt produced no tokens");
        }
        check_cancelled(cancel)?;

        ensure_generation_fits(
            prompt_tokens.len(),
            opts.max_tokens,
            self.session_context.n_ctx(),
        )?;
        let mut sampler = build_sampler(&self.model, opts)?;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        // `llama_decode` may return while CUDA work is still queued. In
        // particular, cancellation can return before sampling performs
        // llama.cpp's implicit synchronization. Wait before clearing memory
        // left by the previous generation.
        self.session_context.synchronize();
        self.session_context.clear_kv_cache();
        self.prompt_batch.clear();
        self.token_batch.clear();

        let start_prompt_processing = Instant::now();
        let mut next_token = if opts.split_prompt {
            process_prompt_split(
                &mut self.session_context,
                &prompt_tokens,
                &mut sampler,
                cancel,
                &mut self.token_batch,
            )?
        } else {
            process_prompt_batch(
                &mut self.session_context,
                &prompt_tokens,
                &mut sampler,
                cancel,
                &mut self.prompt_batch,
            )?
        };
        check_cancelled(cancel)?;
        let prompt_dt = start_prompt_processing.elapsed();

        tracing::info!(
            "{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            rate(prompt_tokens.len(), prompt_dt)
        );

        if should_stop(&self.model, self.eos_token, next_token) {
            tracing::warn!("Early stopping: EOS/EOG token generated at end of prompt");
            return Ok(String::new());
        }

        let start_post_prompt = Instant::now();
        let mut generated = String::new();
        let mut sampled = 0usize;
        let mut position = i32::try_from(prompt_tokens.len()).context("prompt is too long")?;

        while sampled < opts.max_tokens {
            check_cancelled(cancel)?;
            let piece = decode_token(&self.model, next_token, &mut decoder)?;
            generated.push_str(&piece);
            if !piece.is_empty() {
                on_piece(&piece)?;
            }
            sampled += 1;

            if sampled >= opts.max_tokens {
                break;
            }

            self.token_batch.clear();
            self.token_batch
                .add(next_token, position, &[0], true)
                .context("failed to add generated token to llama batch")?;
            self.session_context
                .decode(&mut self.token_batch)
                .context("failed to decode generated token")?;
            check_cancelled_after_decode(&mut self.session_context, cancel)?;
            position += 1;

            next_token = sampler.sample(&self.session_context, self.token_batch.n_tokens() - 1);
            if should_stop(&self.model, self.eos_token, next_token) {
                break;
            }
        }

        let gen_dt = start_post_prompt.elapsed();
        tracing::info!(
            "{:<4} tokens generated: {:.2} token/s",
            sampled,
            rate(sampled, gen_dt)
        );

        Ok(generated)
    }
}

fn process_prompt_batch(
    ctx: &mut LlamaContext<'_>,
    prompt_tokens: &[LlamaToken],
    sampler: &mut LlamaSampler,
    cancel: &AtomicBool,
    batch: &mut LlamaBatch<'_>,
) -> Result<LlamaToken> {
    check_cancelled(cancel)?;
    batch.clear();
    batch
        .add_sequence(prompt_tokens, 0, false)
        .context("failed to build prompt batch")?;
    ctx.decode(batch)
        .context("failed to process prompt batch")?;
    check_cancelled_after_decode(ctx, cancel)?;
    Ok(sampler.sample(ctx, batch.n_tokens() - 1))
}

fn process_prompt_split(
    ctx: &mut LlamaContext<'_>,
    prompt_tokens: &[LlamaToken],
    sampler: &mut LlamaSampler,
    cancel: &AtomicBool,
    batch: &mut LlamaBatch<'_>,
) -> Result<LlamaToken> {
    let last_index = prompt_tokens.len() - 1;

    for (index, token) in prompt_tokens.iter().copied().enumerate() {
        check_cancelled(cancel)?;
        batch.clear();
        batch
            .add(
                token,
                i32::try_from(index).context("prompt is too long")?,
                &[0],
                index == last_index,
            )
            .context("failed to build split prompt batch")?;
        ctx.decode(batch)
            .with_context(|| format!("failed to process prompt token {index}"))?;
        check_cancelled_after_decode(ctx, cancel)?;

        if index == last_index {
            return Ok(sampler.sample(ctx, batch.n_tokens() - 1));
        }
    }

    anyhow::bail!("split prompt processing did not produce a final token")
}

fn should_stop(model: &LlamaModel, eos_token: LlamaToken, token: LlamaToken) -> bool {
    token == eos_token || model.is_eog_token(token)
}

fn model_params(cpu: bool, backend: &LlamaBackend) -> Result<LlamaModelParams> {
    if cpu {
        // Issue #309: default n_gpu_layers is -1 (auto), which may still offload to GPU.
        return Ok(LlamaModelParams::default().with_n_gpu_layers(0));
    }

    if !backend.supports_gpu_offload() {
        bail!(
            "Hskify's browser translation model requires llama.cpp CUDA offload; \
             CPU fallback is disabled"
        );
    }

    Ok(LlamaModelParams::default().with_n_gpu_layers(DEFAULT_GPU_LAYERS))
}

fn resident_context_params(inference_threads: i32) -> LlamaContextParams {
    let n_ctx =
        NonZeroU32::new(RESIDENT_CONTEXT_TOKENS).expect("resident context token count is non-zero");
    let (n_threads, n_threads_batch) = resident_context_thread_counts(inference_threads);
    LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(RESIDENT_CONTEXT_TOKENS)
        .with_n_ubatch(MAX_UBATCH)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads_batch)
}

fn resident_context_thread_counts(inference_threads: i32) -> (i32, i32) {
    (inference_threads, inference_threads)
}

fn ensure_generation_fits(
    prompt_tokens: usize,
    max_tokens: usize,
    context_tokens: u32,
) -> Result<()> {
    let required_ctx = prompt_tokens
        .checked_add(max_tokens)
        .and_then(|tokens| tokens.checked_add(1))
        .context("generation token budget overflowed usize")?;
    let context_tokens = usize::try_from(context_tokens).context("context size exceeds usize")?;
    if required_ctx > context_tokens {
        bail!(
            "prompt plus output budget requires {required_ctx} tokens, exceeding Hskify's \
             permanent {context_tokens}-token llama context"
        );
    }
    if prompt_tokens > RESIDENT_PROMPT_BATCH_TOKENS {
        bail!(
            "prompt contains {prompt_tokens} tokens, exceeding Hskify's preallocated \
             {RESIDENT_PROMPT_BATCH_TOKENS}-token prompt batch"
        );
    }
    Ok(())
}

fn build_sampler(model: &LlamaModel, opts: &GenerateOptions) -> Result<LlamaSampler> {
    let mut samplers = Vec::new();

    if let Some(grammar) = &opts.grammar {
        samplers.push(
            LlamaSampler::grammar(model, &grammar.source, &grammar.root)
                .context("failed to initialize generation grammar")?,
        );
    }

    let has_repeat = (opts.repeat_penalty - 1.0).abs() >= f32::EPSILON && opts.repeat_last_n > 0;
    let has_presence = opts.presence_penalty.abs() >= f32::EPSILON;
    if has_repeat || has_presence {
        samplers.push(LlamaSampler::penalties(
            i32::try_from(opts.repeat_last_n).unwrap_or(i32::MAX),
            if has_repeat { opts.repeat_penalty } else { 1.0 },
            0.0,
            opts.presence_penalty,
        ));
    }

    if opts.temperature <= 0.0 {
        samplers.push(LlamaSampler::greedy());
        return Ok(LlamaSampler::chain_simple(samplers));
    }

    if let Some(top_k) = opts.top_k.filter(|value| *value > 0) {
        samplers.push(LlamaSampler::top_k(
            i32::try_from(top_k).unwrap_or(i32::MAX),
        ));
    }
    if let Some(top_p) = opts.top_p {
        samplers.push(LlamaSampler::top_p(top_p as f32, 1));
    }
    if let Some(min_p) = opts.min_p.filter(|&v| v > 0.0) {
        samplers.push(LlamaSampler::min_p(min_p as f32, 1));
    }

    samplers.push(LlamaSampler::temp(opts.temperature as f32));
    samplers.push(LlamaSampler::dist(opts.seed as u32));
    Ok(LlamaSampler::chain_simple(samplers))
}

fn check_cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("cancelled");
    }
    Ok(())
}

fn check_cancelled_after_decode(ctx: &mut LlamaContext<'_>, cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        // A decode may return while its CUDA graph is still queued. Do not
        // release Hskify's sole CUDA permit until that work has stopped.
        ctx.synchronize();
        anyhow::bail!("cancelled");
    }
    Ok(())
}

fn eos_token_for(id: ModelId, model: &LlamaModel) -> (LlamaToken, String) {
    let token = match id {
        ModelId::Sakura1_5bQwen2_5v1_0 => LlamaToken::new(SAKURA_QWEN_CORRECT_EOS_ID),
        _ => model.token_eos(),
    };
    (token, token_text(model, token))
}

fn token_text(model: &LlamaModel, token: LlamaToken) -> String {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    match model.token_to_piece(token, &mut decoder, true, None) {
        Ok(piece) if !piece.is_empty() => piece,
        _ => token.to_string(),
    }
}

fn decode_token(
    model: &LlamaModel,
    token: LlamaToken,
    decoder: &mut encoding_rs::Decoder,
) -> Result<String> {
    model
        .token_to_piece(token, decoder, true, None)
        .context("failed to decode generated token")
}

fn rate(tokens: usize, duration: std::time::Duration) -> f64 {
    if duration.as_secs_f64() > 0.0 {
        tokens as f64 / duration.as_secs_f64()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GenerateOptions, RESIDENT_CONTEXT_TOKENS, ensure_generation_fits,
        resident_context_thread_counts,
    };

    #[test]
    fn greedy_options_do_not_inherit_sampling_or_penalties() {
        let options = GenerateOptions::greedy(73);

        assert_eq!(options.max_tokens, 73);
        assert_eq!(options.temperature, 0.0);
        assert_eq!(options.top_k, None);
        assert_eq!(options.top_p, None);
        assert_eq!(options.min_p, None);
        assert_eq!(options.repeat_penalty, 1.0);
        assert_eq!(options.repeat_last_n, 0);
        assert_eq!(options.presence_penalty, 0.0);
        assert_eq!(options.grammar, None);
    }

    #[test]
    fn resident_context_budget_accepts_its_exact_boundary() {
        let context_tokens = RESIDENT_CONTEXT_TOKENS;
        let prompt_tokens = context_tokens as usize - 2;

        ensure_generation_fits(prompt_tokens, 1, context_tokens).unwrap();
    }

    #[test]
    fn resident_context_budget_rejects_one_token_over_its_boundary() {
        let context_tokens = RESIDENT_CONTEXT_TOKENS;
        let prompt_tokens = context_tokens as usize - 1;

        let error = ensure_generation_fits(prompt_tokens, 1, context_tokens).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("exceeding Hskify's permanent 1280-token llama context")
        );
    }

    #[test]
    fn resident_context_uses_the_explicit_thread_count() {
        assert_eq!(resident_context_thread_counts(6), (6, 6));
    }
}
