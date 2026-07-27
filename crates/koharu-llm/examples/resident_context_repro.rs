//! Deterministic A/B probe for llama.cpp context and batch reuse.
//!
//! This deliberately bypasses `Llm` so context reuse and batch reuse can be
//! varied independently while keeping the model, prompt, context dimensions,
//! sampling, and decode loop identical.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::{fmt::Write as _, fs};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use koharu_llm::prompt::PromptRenderer;
use koharu_llm::safe::context::LlamaContext;
use koharu_llm::safe::context::params::LlamaContextParams;
use koharu_llm::safe::llama_backend::LlamaBackend;
use koharu_llm::safe::llama_batch::LlamaBatch;
use koharu_llm::safe::model::params::LlamaModelParams;
use koharu_llm::safe::model::{AddBos, LlamaModel};
use koharu_llm::safe::sampling::LlamaSampler;
use koharu_llm::safe::token::LlamaToken;
use koharu_llm::{GenerateOptions, Language, Llm, ModelId};
use koharu_runtime::{ComputePolicy, RuntimeManager};
use serde::{Deserialize, Serialize};

const N_BATCH: u32 = 4096;
const N_UBATCH: u32 = 512;
const PROMPT_BATCH_CAPACITY: usize = N_BATCH as usize;

const BENCHMARK_PROMPT_REVISION: &str = "direct-hsk-en-zh-2026-07-26";
const BENCHMARK_LEVEL: u8 = 5;
const BENCHMARK_SYSTEM_PROMPT: &str = "\
Translate only the numbered English manga lines after INPUT into concise, natural \
Simplified Chinese for a reader targeting cumulative HSK 2.0 level 5. Prefer \
vocabulary at or below that level and short grammar. Preserve meaning, tone, relationships, \
protected names, ASCII numbers, every negation, and question intent. C and N records are \
reference-only; never output them. The numbers are temporary positions, not application IDs. \
Output each requested position once in the same order, exactly \
`<position><TAB><Simplified Chinese>`. Never copy the English input or add a third field. Output no \
other text.";
const BENCHMARK_REPAIR_SYSTEM_PROMPT: &str = "\
Repair this one English-to-Simplified-Chinese manga translation for a reader targeting \
cumulative HSK 2.0 level 5. Fix every listed problem and prefer short natural \
phrasing with vocabulary at or below that level. Preserve meaning, tone, relationships, protected \
names, ASCII numbers, every negation, and question intent. N records are reference-only. \
Output exactly one line containing only the corrected Simplified Chinese text: no position, label, \
tab, English, explanation, Markdown, JSON, or application ID.";

#[derive(Debug, Parser)]
#[command(about = "Probe resident llama.cpp context determinism")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    runtime_root: PathBuf,
    /// Evidence from the failing corrected benchmark, used only to reconstruct
    /// the exact first four primary prompts and intervening repair.
    #[arg(long)]
    benchmark_evidence: PathBuf,
    #[arg(long, default_value_t = 25)]
    runs: usize,
    #[arg(long, default_value_t = 96)]
    max_tokens: usize,
    #[arg(long, default_value_t = 4096)]
    context_tokens: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RunEvidence {
    phase: &'static str,
    run: usize,
    prompt_index: usize,
    prompt_tokens: usize,
    output_hash: String,
    output_bytes: usize,
    matches_fresh: bool,
    output: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PriorEvidence {
    batches: Vec<PriorBatch>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PriorBatch {
    user_prompt_tokens: usize,
    max_output_tokens: usize,
    repairs: Vec<PriorRepair>,
    items: Vec<PriorItem>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PriorRepair {
    item_id: String,
    user_prompt_tokens: usize,
    max_output_tokens: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PriorItem {
    id: String,
    source_english: String,
    candidate_chinese: Option<String>,
    primary_candidate_chinese: Option<String>,
    expected_names: Vec<String>,
    critical_proxy_failures: Vec<String>,
}

struct BenchmarkRequest {
    label: String,
    user: String,
    system: &'static str,
    tokens: Vec<LlamaToken>,
    max_tokens: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.runs >= 25, "--runs must be at least 25");
    ensure!(args.max_tokens > 0, "--max-tokens must be positive");
    ensure!(args.context_tokens > 0, "--context-tokens must be positive");

    let runtime = RuntimeManager::new(&args.runtime_root, ComputePolicy::PreferGpu)?;
    runtime
        .prepare()
        .await
        .context("failed to prepare CUDA runtime")?;
    koharu_llm::sys::initialize(&runtime)?;
    let backend = Arc::new(LlamaBackend::init()?);
    ensure!(
        backend.supports_gpu_offload(),
        "CUDA/GPU offload is unavailable"
    );

    let model = LlamaModel::load_from_file(
        backend.as_ref(),
        &args.model,
        &LlamaModelParams::default().with_n_gpu_layers(1000),
    )
    .with_context(|| format!("failed to load `{}`", args.model.display()))?;
    let renderer = prompt_renderer(&model)?;
    let benchmark_requests = benchmark_requests(&args.benchmark_evidence, &renderer, &model)?;
    let prompts = benchmark_requests
        .iter()
        .take(5)
        .map(|request| request.tokens.clone())
        .collect::<Vec<_>>();
    ensure!(
        prompts.len() >= 2,
        "chapter 5 benchmark evidence has fewer than two replayable requests"
    );
    for tokens in &prompts {
        ensure!(
            tokens.len() + args.max_tokens + 1 <= args.context_tokens as usize,
            "prompt and output do not fit in the context"
        );
    }

    let mut evidence = Vec::new();
    let mut fresh_outputs = Vec::with_capacity(prompts.len());
    for (prompt_index, tokens) in prompts.iter().enumerate() {
        let mut context = new_context(&model, &backend, args.context_tokens)?;
        let mut prompt_batch = LlamaBatch::new(PROMPT_BATCH_CAPACITY, 1);
        let mut token_batch = LlamaBatch::new(1, 1);
        let output = generate_once(
            &model,
            &mut context,
            tokens,
            args.max_tokens,
            &mut prompt_batch,
            &mut token_batch,
            false,
            None,
        )?;
        evidence.push(record(
            "fresh-context-reference",
            prompt_index,
            prompt_index,
            tokens.len(),
            &output,
            true,
        ));
        fresh_outputs.push(output);
    }

    {
        let mut context = new_context(&model, &backend, args.context_tokens)?;
        for run in 0..args.runs {
            let prompt_index = run % prompts.len();
            let mut prompt_batch = LlamaBatch::new(PROMPT_BATCH_CAPACITY, 1);
            let mut token_batch = LlamaBatch::new(1, 1);
            let output = generate_once(
                &model,
                &mut context,
                &prompts[prompt_index],
                args.max_tokens,
                &mut prompt_batch,
                &mut token_batch,
                true,
                None,
            )?;
            evidence.push(record(
                "resident-context-fresh-batches",
                run,
                prompt_index,
                prompts[prompt_index].len(),
                &output,
                output == fresh_outputs[prompt_index],
            ));
        }
    }

    {
        let mut context = new_context(&model, &backend, args.context_tokens)?;
        let mut prompt_batch = LlamaBatch::new(PROMPT_BATCH_CAPACITY, 1);
        let mut token_batch = LlamaBatch::new(1, 1);
        for run in 0..args.runs {
            let prompt_index = run % prompts.len();
            let output = generate_once(
                &model,
                &mut context,
                &prompts[prompt_index],
                args.max_tokens,
                &mut prompt_batch,
                &mut token_batch,
                true,
                None,
            )?;
            evidence.push(record(
                "resident-context-reused-batches",
                run,
                prompt_index,
                prompts[prompt_index].len(),
                &output,
                output == fresh_outputs[prompt_index],
            ));
        }

        let cancelled = generate_once(
            &model,
            &mut context,
            &prompts[1],
            args.max_tokens,
            &mut prompt_batch,
            &mut token_batch,
            true,
            Some(3),
        );
        ensure!(
            cancelled
                .as_ref()
                .is_err_and(|error| error.to_string() == "cancelled after decode 3"),
            "cancellation probe unexpectedly returned {cancelled:?}"
        );
        let output = generate_once(
            &model,
            &mut context,
            &prompts[0],
            args.max_tokens,
            &mut prompt_batch,
            &mut token_batch,
            true,
            None,
        )?;
        evidence.push(record(
            "resident-after-cancellation",
            args.runs,
            0,
            prompts[0].len(),
            &output,
            output == fresh_outputs[0],
        ));
    }

    for (request_index, request) in benchmark_requests.iter().enumerate() {
        let required_context = request
            .tokens
            .len()
            .checked_add(request.max_tokens)
            .and_then(|tokens| tokens.checked_add(1))
            .context("dynamic reference context size overflowed")?;
        let n_ctx = u32::try_from(required_context).context("dynamic context exceeds u32")?;
        let n_batch =
            u32::try_from(request.tokens.len()).context("dynamic prompt batch exceeds u32")?;
        let mut context =
            new_context_config(&model, &backend, n_ctx, n_batch, n_batch.min(N_UBATCH))?;
        let mut prompt_batch = LlamaBatch::new(request.tokens.len(), 1);
        let mut token_batch = LlamaBatch::new(1, 1);
        let output = generate_once(
            &model,
            &mut context,
            &request.tokens,
            request.max_tokens,
            &mut prompt_batch,
            &mut token_batch,
            false,
            None,
        )?;
        eprintln!(
            "BENCHMARK_DYNAMIC request={} label={} n_ctx={} n_batch={} n_ubatch={} hash={:016x}",
            request_index,
            request.label,
            n_ctx,
            n_batch,
            n_batch.min(N_UBATCH),
            fnv1a64(output.as_bytes())
        );
        evidence.push(record(
            "benchmark-dynamic-context-reference",
            request_index,
            request_index,
            request.tokens.len(),
            &output,
            true,
        ));
    }

    let mut benchmark_fresh_outputs = Vec::with_capacity(benchmark_requests.len());
    for (request_index, request) in benchmark_requests.iter().enumerate() {
        let mut context = new_context(&model, &backend, args.context_tokens)?;
        let mut prompt_batch = LlamaBatch::new(PROMPT_BATCH_CAPACITY, 1);
        let mut token_batch = LlamaBatch::new(1, 1);
        let output = generate_once(
            &model,
            &mut context,
            &request.tokens,
            request.max_tokens,
            &mut prompt_batch,
            &mut token_batch,
            false,
            None,
        )?;
        evidence.push(record(
            "benchmark-fresh-context-reference",
            request_index,
            request_index,
            request.tokens.len(),
            &output,
            true,
        ));
        eprintln!(
            "BENCHMARK_FRESH request={} label={} hash={:016x}",
            request_index,
            request.label,
            fnv1a64(output.as_bytes())
        );
        benchmark_fresh_outputs.push(output);
    }

    {
        let mut context = new_context(&model, &backend, args.context_tokens)?;
        for (request_index, request) in benchmark_requests.iter().enumerate() {
            let mut prompt_batch = LlamaBatch::new(PROMPT_BATCH_CAPACITY, 1);
            let mut token_batch = LlamaBatch::new(1, 1);
            let output = generate_once(
                &model,
                &mut context,
                &request.tokens,
                request.max_tokens,
                &mut prompt_batch,
                &mut token_batch,
                true,
                None,
            )?;
            evidence.push(record(
                "benchmark-resident-context-fresh-batches",
                request_index,
                request_index,
                request.tokens.len(),
                &output,
                output == benchmark_fresh_outputs[request_index],
            ));
        }
    }

    {
        let mut context = new_context(&model, &backend, args.context_tokens)?;
        let mut prompt_batch = LlamaBatch::new(PROMPT_BATCH_CAPACITY, 1);
        let mut token_batch = LlamaBatch::new(1, 1);
        for (request_index, request) in benchmark_requests.iter().enumerate() {
            let output = generate_once(
                &model,
                &mut context,
                &request.tokens,
                request.max_tokens,
                &mut prompt_batch,
                &mut token_batch,
                true,
                None,
            )?;
            evidence.push(record(
                "benchmark-resident-context-reused-batches",
                request_index,
                request_index,
                request.tokens.len(),
                &output,
                output == benchmark_fresh_outputs[request_index],
            ));
        }
    }

    drop(renderer);
    drop(model);
    if args.context_tokens == 4096 {
        let mut llm = Llm::load_file(
            &runtime,
            ModelId::Qwen3_5_4b,
            false,
            args.model.clone(),
            Arc::clone(&backend),
        )
        .await
        .context("failed to load product Llm for benchmark replay")?;
        for (request_index, request) in benchmark_requests.iter().enumerate() {
            ensure!(
                llm.token_count(&request.user)? > 0,
                "product tokenizer returned an empty benchmark user prompt"
            );
            let options = GenerateOptions::greedy(request.max_tokens);
            let mut streamed = String::new();
            let output = if request_index == 0 {
                llm.generate_constrained(
                    &request.user,
                    &options,
                    Language::ChineseSimplified,
                    request.system,
                    &AtomicBool::new(false),
                )?
            } else {
                llm.generate_constrained_streaming(
                    &request.user,
                    &options,
                    Language::ChineseSimplified,
                    request.system,
                    &AtomicBool::new(false),
                    |piece| {
                        streamed.push_str(piece);
                        Ok(())
                    },
                )?
            };
            if request_index > 0 {
                ensure!(
                    streamed == output,
                    "streamed output differs from returned output"
                );
            }
            ensure!(
                llm.token_count(&output)? > 0,
                "product tokenizer returned an empty output"
            );
            evidence.push(record(
                "benchmark-product-resident-context-reused-batches",
                request_index,
                request_index,
                request.tokens.len(),
                &output,
                output == benchmark_fresh_outputs[request_index],
            ));
        }
    }

    let mismatches = evidence.iter().filter(|entry| !entry.matches_fresh).count();
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    ensure!(
        mismatches == 0,
        "{mismatches} outputs differed from fresh context"
    );
    Ok(())
}

fn benchmark_requests(
    evidence_path: &PathBuf,
    renderer: &PromptRenderer,
    model: &LlamaModel,
) -> Result<Vec<BenchmarkRequest>> {
    let prior: PriorEvidence = serde_json::from_slice(
        &fs::read(evidence_path)
            .with_context(|| format!("failed to read `{}`", evidence_path.display()))?,
    )
    .with_context(|| format!("failed to parse `{}`", evidence_path.display()))?;
    ensure!(
        prior.batches.len() >= 4,
        "benchmark evidence has fewer than four batches"
    );
    for batch in &prior.batches {
        for item in &batch.items {
            ensure!(
                is_chapter5_region_id(&item.id),
                "benchmark evidence contains non-chapter-5 region ID `{}`",
                item.id
            );
        }
    }

    let mut requests = Vec::new();
    let warmup = &prior.batches[0];
    let warmup_user = benchmark_primary_prompt(warmup)?;
    let warmup_tokens = benchmark_tokens(
        renderer,
        model,
        &warmup_user,
        BENCHMARK_SYSTEM_PROMPT,
        warmup.user_prompt_tokens,
    )?;
    requests.push(BenchmarkRequest {
        label: "warmup-batch-1".to_owned(),
        user: warmup_user,
        system: BENCHMARK_SYSTEM_PROMPT,
        tokens: warmup_tokens,
        max_tokens: warmup.max_output_tokens,
    });

    for (batch_index, batch) in prior.batches.iter().take(4).enumerate() {
        let user = benchmark_primary_prompt(batch)?;
        let tokens = benchmark_tokens(
            renderer,
            model,
            &user,
            BENCHMARK_SYSTEM_PROMPT,
            batch.user_prompt_tokens,
        )?;
        requests.push(BenchmarkRequest {
            label: format!("primary-batch-{}", batch_index + 1),
            user,
            system: BENCHMARK_SYSTEM_PROMPT,
            tokens,
            max_tokens: batch.max_output_tokens,
        });

        for (repair_index, repair_evidence) in batch.repairs.iter().enumerate() {
            let repair_region = batch
                .items
                .iter()
                .find(|item| item.id == repair_evidence.item_id)
                .with_context(|| {
                    format!(
                        "repair item {} is missing from batch {}",
                        repair_evidence.item_id,
                        batch_index + 1
                    )
                })?;
            ensure!(
                !repair_region.critical_proxy_failures.is_empty(),
                "repair item {} has no recorded failure reasons",
                repair_region.id
            );
            let mut repair = String::new();
            write_common_prompt(
                &mut repair,
                prior.batches[..batch_index]
                    .iter()
                    .flat_map(|previous| previous.items.iter())
                    .filter_map(|item| {
                        item.candidate_chinese
                            .as_deref()
                            .map(|chinese| (item.source_english.as_str(), chinese))
                    }),
                std::iter::empty(),
            );
            writeln!(
                &mut repair,
                "SOURCE\t{}\nREJECTED\t{}\nFIX\t{}\nANSWER",
                compact(&repair_region.source_english),
                compact(
                    repair_region
                        .primary_candidate_chinese
                        .as_deref()
                        .unwrap_or("<missing>")
                ),
                repair_region.critical_proxy_failures.join("; "),
            )
            .expect("String writes cannot fail");
            let repair_tokens = benchmark_tokens(
                renderer,
                model,
                &repair,
                BENCHMARK_REPAIR_SYSTEM_PROMPT,
                repair_evidence.user_prompt_tokens,
            )?;
            requests.push(BenchmarkRequest {
                label: format!(
                    "repair-after-batch-{}-{}",
                    batch_index + 1,
                    repair_index + 1
                ),
                user: repair,
                system: BENCHMARK_REPAIR_SYSTEM_PROMPT,
                tokens: repair_tokens,
                max_tokens: repair_evidence.max_output_tokens,
            });
        }
    }

    Ok(requests)
}

fn benchmark_primary_prompt(batch: &PriorBatch) -> Result<String> {
    let names = batch
        .items
        .iter()
        .flat_map(|item| item.expected_names.iter())
        .map(String::as_str)
        .collect::<Vec<_>>();
    ensure!(
        names.is_empty(),
        "chapter 5 benchmark evidence contains protected names but no committed source-to-Chinese glossary"
    );
    let mut prompt = String::new();
    write_common_prompt(&mut prompt, std::iter::empty(), std::iter::empty());
    writeln!(&mut prompt, "INPUT\t{}", batch.items.len()).expect("String writes cannot fail");
    for (index, item) in batch.items.iter().enumerate() {
        writeln!(
            &mut prompt,
            "{}\t{}",
            index + 1,
            compact(&item.source_english)
        )
        .expect("String writes cannot fail");
    }
    Ok(prompt)
}

fn is_chapter5_region_id(value: &str) -> bool {
    value
        .strip_prefix("30ysp-ch5-p")
        .and_then(|suffix| suffix.split_once("-r"))
        .is_some_and(|(page, region)| {
            page.len() == 3
                && region.len() == 2
                && page.bytes().all(|byte| byte.is_ascii_digit())
                && region.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn write_common_prompt<'a>(
    prompt: &mut String,
    context: impl IntoIterator<Item = (&'a str, &'a str)>,
    names: impl IntoIterator<Item = (&'a str, &'a str)>,
) {
    let context = context.into_iter().collect::<Vec<_>>();
    let names = names.into_iter().collect::<Vec<_>>();
    writeln!(
        prompt,
        "REV\t{BENCHMARK_PROMPT_REVISION}\nLEVEL\t{BENCHMARK_LEVEL}"
    )
    .expect("String writes cannot fail");
    writeln!(prompt, "CONTEXT\t{}", context.len()).expect("String writes cannot fail");
    for (source, chinese) in context {
        writeln!(prompt, "C\t{}\t{}", compact(source), compact(chinese))
            .expect("String writes cannot fail");
    }
    writeln!(prompt, "NAMES\t{}", names.len()).expect("String writes cannot fail");
    for (source, chinese) in names {
        writeln!(prompt, "N\t{}\t{}", compact(source), compact(chinese))
            .expect("String writes cannot fail");
    }
}

fn benchmark_tokens(
    renderer: &PromptRenderer,
    model: &LlamaModel,
    user: &str,
    system: &str,
    expected_user_tokens: usize,
) -> Result<Vec<LlamaToken>> {
    let actual_user_tokens = model
        .str_to_token(user, AddBos::Never)
        .context("failed to tokenize benchmark user prompt")?
        .len();
    ensure!(
        actual_user_tokens == expected_user_tokens,
        "benchmark user-prompt token mismatch: expected {expected_user_tokens}, got {actual_user_tokens}"
    );
    let wire = renderer.format_chat_prompt_exact_system(
        user.to_owned(),
        Language::ChineseSimplified,
        Some(system),
    )?;
    model
        .str_to_token(&wire, AddBos::Never)
        .context("failed to tokenize benchmark wire prompt")
}

fn compact(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn new_context<'model>(
    model: &'model LlamaModel,
    backend: &LlamaBackend,
    context_tokens: u32,
) -> Result<LlamaContext<'model>> {
    new_context_config(model, backend, context_tokens, N_BATCH, N_UBATCH)
}

fn new_context_config<'model>(
    model: &'model LlamaModel,
    backend: &LlamaBackend,
    context_tokens: u32,
    batch_tokens: u32,
    ubatch_tokens: u32,
) -> Result<LlamaContext<'model>> {
    let n_ctx = NonZeroU32::new(context_tokens).context("context size must be non-zero")?;
    model
        .new_context(
            backend,
            LlamaContextParams::default()
                .with_n_ctx(Some(n_ctx))
                .with_n_batch(batch_tokens)
                .with_n_ubatch(ubatch_tokens)
                .with_n_threads(koharu_llm::inference_threads())
                .with_n_threads_batch(koharu_llm::inference_threads()),
        )
        .context("failed to create llama.cpp context")
}

#[allow(clippy::too_many_arguments)]
fn generate_once(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    prompt_tokens: &[LlamaToken],
    max_tokens: usize,
    prompt_batch: &mut LlamaBatch<'_>,
    token_batch: &mut LlamaBatch<'_>,
    reset_context: bool,
    cancel_after_decode: Option<usize>,
) -> Result<String> {
    if reset_context {
        context.synchronize();
        context.clear_kv_cache();
    }
    prompt_batch.clear();
    token_batch.clear();

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    prompt_batch
        .add_sequence(prompt_tokens, 0, false)
        .context("failed to build prompt batch")?;
    context
        .decode(prompt_batch)
        .context("failed to decode prompt")?;
    let mut next_token = sampler.sample(context, prompt_batch.n_tokens() - 1);
    let mut output = String::new();
    let mut position = i32::try_from(prompt_tokens.len()).context("prompt is too long")?;

    for sampled in 0..max_tokens {
        if next_token == model.token_eos() || model.is_eog_token(next_token) {
            break;
        }
        output.push_str(
            &model
                .token_to_piece(next_token, &mut decoder, true, None)
                .context("failed to decode token piece")?,
        );
        if sampled + 1 == max_tokens {
            break;
        }

        token_batch.clear();
        token_batch
            .add(next_token, position, &[0], true)
            .context("failed to build token batch")?;
        context
            .decode(token_batch)
            .context("failed to decode generated token")?;
        if cancel_after_decode == Some(sampled + 1) {
            context.synchronize();
            bail!("cancelled after decode {}", sampled + 1);
        }
        position += 1;
        next_token = sampler.sample(context, token_batch.n_tokens() - 1);
    }

    Ok(output)
}

fn prompt_renderer(model: &LlamaModel) -> Result<PromptRenderer> {
    let template = model
        .meta_val_str("tokenizer.ggml.chat_template")
        .or_else(|_| model.meta_val_str("tokenizer.chat_template"))
        .context("model is missing its chat template")?;
    Ok(PromptRenderer::new(
        ModelId::Qwen3_5_4b,
        template,
        token_text(model, model.token_bos()),
        token_text(model, model.token_eos()),
    ))
}

fn token_text(model: &LlamaModel, token: LlamaToken) -> String {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    match model.token_to_piece(token, &mut decoder, true, None) {
        Ok(piece) if !piece.is_empty() => piece,
        _ => token.to_string(),
    }
}

fn record(
    phase: &'static str,
    run: usize,
    prompt_index: usize,
    prompt_tokens: usize,
    output: &str,
    matches_fresh: bool,
) -> RunEvidence {
    RunEvidence {
        phase,
        run,
        prompt_index,
        prompt_tokens,
        output_hash: format!("{:016x}", fnv1a64(output.as_bytes())),
        output_bytes: output.len(),
        matches_fresh,
        output: output.to_owned(),
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
