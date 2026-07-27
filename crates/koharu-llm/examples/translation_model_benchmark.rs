//! Reproducible CUDA benchmark for the direct English-to-HSK-Chinese prompt.
//!
//! This is intentionally an example binary rather than product code. It reads
//! the completed 30 Years Since the Prologue chapter 5 annotations, retains all detector
//! gold for fixture validation, runs only the English translation targets
//! through the three supplied GGUFs sequentially, and writes raw evidence only
//! below an ignored output directory.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use koharu_llm::direct_hsk_protocol::{
    DIRECT_HSK_PROMPT_HASH as PROMPT_HASH, DIRECT_HSK_PROMPT_REVISION as PROMPT_REVISION,
    DIRECT_HSK_VALIDATOR_HASH as VALIDATOR_HASH, DirectHskContext, DirectHskName,
    context_budget_text, primary_system_prompt, primary_user_prompt,
    repair_system_prompt as shared_repair_system_prompt, repair_user_prompt,
};
use koharu_llm::safe::llama_backend::LlamaBackend;
use koharu_llm::{GenerateOptions, Language, Llm, ModelId};
use koharu_runtime::{ComputePolicy, RuntimeManager};
use serde::{Deserialize, Serialize};

const REQUESTED_HSK_LEVEL: u8 = 5;
const BATCH_MAX: usize = 6;
const CONTEXT_MAX_UTTERANCES: usize = 6;
const CONTEXT_MAX_TOKENS: usize = 256;
const MIN_OUTPUT_TOKENS: usize = 24;
const MAX_OUTPUT_TOKENS: usize = 256;
const OUTPUT_TOKENS_PER_UTTERANCE: usize = 4;

#[derive(Parser, Debug)]
#[command(about = "Benchmark direct chapter 5 English-to-HSK-Chinese translation")]
struct Args {
    #[arg(long)]
    annotations: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    runtime_root: PathBuf,
    #[arg(long)]
    qwen4b: PathBuf,
    #[arg(long)]
    qwen2b: PathBuf,
    #[arg(long)]
    hy_mt2: PathBuf,
    /// Run one candidate ID and retain its evidence for later assembly.
    #[arg(long)]
    candidate: Option<String>,
    /// Assemble final JSON and the blinded packet from three completed
    /// candidate evidence files without loading a model.
    #[arg(long, default_value_t = false)]
    assemble_only: bool,
    /// Run only the listed one-based corrected batches through the three
    /// minimal prompt variants and write protocol-probe.json.
    #[arg(long, value_delimiter = ',')]
    protocol_probe_batches: Vec<usize>,
}

#[derive(Clone, Debug)]
struct Candidate {
    id: &'static str,
    display_name: &'static str,
    model_id: ModelId,
    path: PathBuf,
    expected_bytes: u64,
    expected_sha256: &'static str,
    repository_revision: &'static str,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PageAnnotations {
    page: PageIdentity,
    regions: Vec<GoldRegion>,
}

#[derive(Debug, Deserialize)]
struct PageIdentity {
    order: usize,
    file: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkManifest {
    id: String,
    page_count: usize,
    annotation_status: AnnotationStatus,
    total_expected_region_count: Option<usize>,
    total_expected_dialogue_bubble_count: Option<usize>,
    total_expected_english_translation_target_count: Option<usize>,
    total_expected_untouched_exclusion_count: Option<usize>,
    images: Vec<ManifestImage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnnotationStatus {
    status: String,
    completed_page_count: usize,
    required_page_count: usize,
    reason_code: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestImage {
    order: usize,
    file: String,
    annotation: Option<String>,
}

struct LoadedGold {
    regions: Vec<GoldRegion>,
    page_count: usize,
    detected_bubble_gold: usize,
    english_translation_targets: usize,
    untouched_exclusions: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoldRegion {
    id: String,
    kind: String,
    normalized_english: String,
    #[serde(default)]
    simplified_chinese: String,
    #[serde(default)]
    hsk_tokens: Vec<GoldToken>,
    translation_target: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoldToken {
    text: String,
    classification: String,
}

#[derive(Clone, Debug)]
struct ContextItem {
    source_english: String,
    chinese: String,
}

#[derive(Clone, Copy, Debug)]
struct NameMapping {
    source: &'static str,
    chinese: &'static str,
}

// Chapter 5 currently has no committed English-to-Chinese name glossary.
// Do not infer or carry names over from another benchmark.
const NAME_MAPPINGS: &[NameMapping] = &[];

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkEvidence {
    schema_version: u8,
    workload: WorkloadEvidence,
    protocol: ProtocolEvidence,
    candidates: Vec<CandidateEvidence>,
    selection: SelectionEvidence,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkloadEvidence {
    id: &'static str,
    annotation_files: usize,
    detected_bubble_gold: usize,
    english_translation_targets: usize,
    untouched_exclusions: usize,
    accepted_regions: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProtocolEvidence {
    prompt_revision: &'static str,
    prompt_hash: &'static str,
    validator_hash: &'static str,
    requested_hsk_level: u8,
    batch_min: usize,
    batch_max: usize,
    batch_sizes: Vec<usize>,
    preceding_utterance_limit: usize,
    preceding_context_token_limit: usize,
    decoding: &'static str,
    warmup_batches_per_candidate: usize,
    cpu_thread_cap: usize,
    name_policy: &'static str,
    number_policy: &'static str,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CandidateEvidence {
    id: String,
    display_name: String,
    runtime_model_id: String,
    model_path: String,
    repository_revision: String,
    expected_bytes: u64,
    expected_sha256: String,
    load_ms: f64,
    warmup_ms: f64,
    warmup_output_tokens: usize,
    batches: Vec<BatchEvidence>,
    summary: CandidateSummary,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchEvidence {
    batch_index: usize,
    item_ids: Vec<String>,
    item_count: usize,
    context_count: usize,
    context_tokens: usize,
    user_prompt_tokens: usize,
    max_output_tokens: usize,
    output_tokens: usize,
    first_piece_ms: Option<f64>,
    first_complete_line_ms: Option<f64>,
    wall_ms: f64,
    total_wall_ms: f64,
    output_tokens_per_second: f64,
    raw_output: String,
    ignored_output_lines: Vec<String>,
    repairs: Vec<RepairEvidence>,
    items: Vec<ItemEvidence>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RepairEvidence {
    item_id: String,
    user_prompt_tokens: usize,
    max_output_tokens: usize,
    output_tokens: usize,
    first_piece_ms: Option<f64>,
    first_complete_line_ms: Option<f64>,
    wall_ms: f64,
    raw_output: String,
    ignored_output_lines: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ItemEvidence {
    id: String,
    source_english: String,
    approved_chinese: String,
    candidate_chinese: Option<String>,
    structured_success: bool,
    parse_issue: Option<String>,
    primary_candidate_chinese: Option<String>,
    primary_structured_success: bool,
    primary_parse_issue: Option<String>,
    repair_attempted: bool,
    repair_candidate_chinese: Option<String>,
    repair_structured_success: Option<bool>,
    repair_parse_issue: Option<String>,
    expected_names: Vec<String>,
    missing_names: Vec<String>,
    expected_ascii_numbers: Vec<String>,
    actual_ascii_numbers: Vec<String>,
    question_required: bool,
    question_preserved: bool,
    exact_reference_match: bool,
    character_unigram_f1: f64,
    character_bigram_f1: f64,
    critical_proxy_failures: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CandidateSummary {
    regions: usize,
    primary_structured_successes: usize,
    primary_structured_success_rate: f64,
    structured_successes: usize,
    structured_success_rate: f64,
    repair_batches: usize,
    repaired_items: usize,
    repaired_items_passing_final_validation: usize,
    protected_names: usize,
    protected_names_preserved: usize,
    protected_name_preservation_rate: f64,
    ascii_numbers: usize,
    ascii_numbers_preserved: usize,
    ascii_number_preservation_rate: f64,
    names_and_numbers: usize,
    names_and_numbers_preserved: usize,
    names_and_numbers_preservation_rate: f64,
    questions: usize,
    questions_preserved: usize,
    items_with_critical_proxy_failures: usize,
    critical_proxy_failure_count: usize,
    exact_reference_matches: usize,
    mean_character_unigram_f1: f64,
    mean_character_bigram_f1: f64,
    warm_batch_latency_p50_ms: f64,
    warm_batch_latency_p95_ms: f64,
    warm_first_line_p50_ms: Option<f64>,
    warm_total_ms: f64,
    primary_output_tokens: usize,
    repair_output_tokens: usize,
    output_tokens: usize,
    aggregate_output_tokens_per_second: f64,
    automated_zero_critical_proxy_failures: bool,
    names_and_numbers_at_least_99_percent: bool,
    human_naturalness_matches_qwen4b: Option<bool>,
    qualifies_as_smaller_replacement: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SelectionEvidence {
    selected_candidate_id: &'static str,
    reason: &'static str,
    human_naturalness_review: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProtocolProbeEvidence {
    model_id: String,
    model_path: String,
    expected_sha256: String,
    prompt_revision: &'static str,
    prompt_hash: &'static str,
    load_ms: f64,
    warmup_ms: f64,
    corrected_batch_count: usize,
    requested_batches: Vec<usize>,
    results: Vec<ProtocolProbeResult>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProtocolProbeResult {
    variant: &'static str,
    batch_index: usize,
    item_ids: Vec<String>,
    system_prompt: String,
    user_prompt: String,
    max_output_tokens: usize,
    wall_ms: f64,
    raw_output: String,
    exact_ordered_shape: bool,
    parsed_positions: usize,
}

#[derive(Clone, Debug)]
enum ParsedLine {
    Candidate(String),
    Malformed(Option<String>),
}

#[derive(Clone, Debug)]
struct ParsedItem {
    text: Option<String>,
    issue: Option<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        std::env::var("KOHARU_INFERENCE_THREADS").ok().as_deref() == Some("6"),
        "set KOHARU_INFERENCE_THREADS=6"
    );
    fs::create_dir_all(&args.output)
        .with_context(|| format!("failed to create `{}`", args.output.display()))?;

    let gold = load_gold(&args.annotations)?;
    let untouched_exclusions = gold
        .regions
        .iter()
        .filter(|region| region.translation_target == Some(false))
        .count();
    ensure!(
        untouched_exclusions == gold.untouched_exclusions,
        "manifest expects {} untouched exclusions, found {untouched_exclusions}",
        gold.untouched_exclusions
    );
    let regions = gold
        .regions
        .iter()
        .filter(|region| region.translation_target != Some(false))
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        regions.len() == gold.english_translation_targets,
        "manifest expects {} English translation targets, found {}",
        gold.english_translation_targets,
        regions.len(),
    );
    validate_name_glossary(&regions)?;
    let batch_sizes = regions
        .chunks(BATCH_MAX)
        .map(<[_]>::len)
        .collect::<Vec<_>>();
    ensure!(
        batch_sizes
            .iter()
            .all(|size| (3..=BATCH_MAX).contains(size)),
        "all benchmark batches must contain 3 through {BATCH_MAX} regions"
    );

    let candidates = vec![
        Candidate {
            id: "qwen3.5-4b-q4-k-m",
            display_name: "Qwen3.5 4B Q4_K_M",
            model_id: ModelId::Qwen3_5_4b,
            path: args.qwen4b,
            expected_bytes: 2_740_937_888,
            expected_sha256: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
            repository_revision: "e87f176479d0855a907a41277aca2f8ee7a09523",
        },
        Candidate {
            id: "qwen3.5-2b-q4-k-m",
            display_name: "Qwen3.5 2B Q4_K_M",
            model_id: ModelId::Qwen3_5_2b,
            path: args.qwen2b,
            expected_bytes: 1_280_835_840,
            expected_sha256: "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223",
            repository_revision: "f6d5376be1edb4d416d56da11e5397a961aca8ae",
        },
        Candidate {
            id: "hy-mt2-1.8b-q4-k-m",
            display_name: "Hy-MT2 1.8B Q4_K_M",
            model_id: ModelId::HyMt2_1_8b,
            path: args.hy_mt2,
            expected_bytes: 1_133_080_448,
            expected_sha256: "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699",
            repository_revision: "1cd5208700acedef4ef93019b6cfc148b8522d45",
        },
    ];
    if args.assemble_only {
        ensure!(
            args.protocol_probe_batches.is_empty(),
            "--assemble-only and --protocol-probe-batches are mutually exclusive"
        );
        let completed = load_completed_candidates(&args.output, &candidates, &regions)?;
        write_final_evidence(&args.output, &gold, &regions, batch_sizes, completed)?;
        return Ok(());
    }

    let candidates_to_run = if !args.protocol_probe_batches.is_empty() {
        ensure!(
            args.candidate
                .as_deref()
                .is_none_or(|id| id == candidates[0].id),
            "the protocol probe is intentionally Qwen3.5 4B-only"
        );
        vec![&candidates[0]]
    } else {
        match args.candidate.as_deref() {
            Some(id) => {
                let candidate = candidates
                    .iter()
                    .find(|candidate| candidate.id == id)
                    .with_context(|| format!("unknown candidate ID `{id}`"))?;
                vec![candidate]
            }
            None => candidates.iter().collect(),
        }
    };
    for candidate in &candidates_to_run {
        let bytes = fs::metadata(&candidate.path)
            .with_context(|| format!("missing `{}`", candidate.path.display()))?
            .len();
        ensure!(
            bytes == candidate.expected_bytes,
            "{} byte mismatch: expected {}, got {}",
            candidate.display_name,
            candidate.expected_bytes,
            bytes
        );
    }

    let runtime = RuntimeManager::new(&args.runtime_root, ComputePolicy::PreferGpu)?;
    runtime
        .prepare()
        .await
        .context("failed to prepare llama.cpp CUDA runtime")?;
    koharu_llm::sys::initialize(&runtime)?;
    let backend = Arc::new(LlamaBackend::init()?);
    ensure!(
        backend.supports_gpu_offload(),
        "CUDA/GPU offload is unavailable; refusing to record a CPU benchmark"
    );

    if !args.protocol_probe_batches.is_empty() {
        let result = run_protocol_probe(
            &runtime,
            Arc::clone(&backend),
            candidates_to_run[0],
            &regions,
            &args.protocol_probe_batches,
        )
        .await?;
        fs::write(
            args.output.join("protocol-probe.json"),
            serde_json::to_vec_pretty(&result)?,
        )?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "output": args.output,
                "probeResults": result.results.len(),
            }))?
        );
        return Ok(());
    }

    let mut evidence = Vec::with_capacity(candidates_to_run.len());
    for candidate in candidates_to_run {
        eprintln!("BENCHMARK_CANDIDATE_START {}", candidate.id);
        let result = run_candidate(&runtime, Arc::clone(&backend), candidate, &regions).await?;
        let candidate_dir = args.output.join("candidates").join(candidate.id);
        fs::create_dir_all(&candidate_dir)?;
        fs::write(
            candidate_dir.join("evidence.json"),
            serde_json::to_vec_pretty(&result)?,
        )?;
        eprintln!(
            "BENCHMARK_CANDIDATE_DONE {} p50_ms={:.3} structured={}/{} critical_proxy_items={} names_numbers={}/{}",
            candidate.id,
            result.summary.warm_batch_latency_p50_ms,
            result.summary.structured_successes,
            result.summary.regions,
            result.summary.items_with_critical_proxy_failures,
            result.summary.names_and_numbers_preserved,
            result.summary.names_and_numbers,
        );
        evidence.push(result);
        // `result` owns no model state. run_candidate drops its Llm before
        // returning, so candidates cannot overlap on the GPU.
    }

    if args.candidate.is_some() {
        let candidate = evidence
            .first()
            .expect("one explicitly selected candidate completed");
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "output": args.output,
                "candidateId": candidate.id,
                "summary": candidate.summary,
                "assemblyPending": true,
            }))?
        );
        return Ok(());
    }

    write_final_evidence(&args.output, &gold, &regions, batch_sizes, evidence)
}

fn load_completed_candidates(
    output: &Path,
    candidates: &[Candidate],
    regions: &[GoldRegion],
) -> Result<Vec<CandidateEvidence>> {
    candidates
        .iter()
        .map(|candidate| {
            let path = output
                .join("candidates")
                .join(candidate.id)
                .join("evidence.json");
            let mut evidence: CandidateEvidence = serde_json::from_slice(
                &fs::read(&path)
                    .with_context(|| format!("missing completed evidence `{}`", path.display()))?,
            )
            .with_context(|| format!("failed to parse `{}`", path.display()))?;
            let completed_ids = evidence
                .batches
                .iter()
                .flat_map(|batch| batch.item_ids.iter())
                .collect::<Vec<_>>();
            let expected_ids = regions.iter().map(|region| &region.id).collect::<Vec<_>>();
            ensure!(
                completed_ids == expected_ids,
                "completed evidence `{}` does not match the {}-region English translation target workload",
                path.display(),
                regions.len()
            );
            rescore_candidate(&mut evidence, regions);
            fs::write(&path, serde_json::to_vec_pretty(&evidence)?)
                .with_context(|| format!("failed to update `{}`", path.display()))?;
            Ok(evidence)
        })
        .collect()
}

fn rescore_candidate(candidate: &mut CandidateEvidence, regions: &[GoldRegion]) {
    for (batch, gold) in candidate.batches.iter_mut().zip(regions.chunks(BATCH_MAX)) {
        let (primary, ignored) = parse_numbered_output(&batch.raw_output, gold);
        batch.ignored_output_lines = ignored;
        let mut final_parsed = primary.clone();
        let mut repair_by_index = HashMap::new();
        let names = protected_names(gold);
        for repair in &mut batch.repairs {
            if let Some(index) = gold.iter().position(|region| region.id == repair.item_id) {
                let (repair_parsed, repair_ignored) =
                    parse_repair_output(&repair.raw_output, &gold[index].normalized_english);
                repair.ignored_output_lines = repair_ignored;
                if deterministic_problems(&gold[index], &repair_parsed, &names).is_empty() {
                    final_parsed[index] = repair_parsed.clone();
                }
                repair_by_index.insert(index, repair_parsed);
            }
        }
        batch.total_wall_ms = batch.wall_ms
            + batch
                .repairs
                .iter()
                .map(|repair| repair.wall_ms)
                .sum::<f64>();
        batch.items = build_item_evidence(gold, &final_parsed, &names, &primary, &repair_by_index);
    }
    candidate.summary = summarize(&candidate.batches);
}

fn write_final_evidence(
    output: &Path,
    gold: &LoadedGold,
    regions: &[GoldRegion],
    batch_sizes: Vec<usize>,
    evidence: Vec<CandidateEvidence>,
) -> Result<()> {
    ensure!(
        evidence.len() == 3,
        "final assembly requires exactly three candidates"
    );
    let benchmark = BenchmarkEvidence {
        schema_version: 1,
        workload: WorkloadEvidence {
            id: "30-years-since-the-prologue-chapter-5",
            annotation_files: gold.page_count,
            detected_bubble_gold: gold.detected_bubble_gold,
            english_translation_targets: regions.len(),
            untouched_exclusions: gold.untouched_exclusions,
            accepted_regions: regions.len(),
        },
        protocol: ProtocolEvidence {
            prompt_revision: PROMPT_REVISION,
            prompt_hash: PROMPT_HASH,
            validator_hash: VALIDATOR_HASH,
            requested_hsk_level: REQUESTED_HSK_LEVEL,
            batch_min: 3,
            batch_max: BATCH_MAX,
            batch_sizes,
            preceding_utterance_limit: CONTEXT_MAX_UTTERANCES,
            preceding_context_token_limit: CONTEXT_MAX_TOKENS,
            decoding: "greedy; temperature=0; no top-k/top-p/min-p; no repeat or presence penalty",
            warmup_batches_per_candidate: 1,
            cpu_thread_cap: 6,
            name_policy: "longest matching protected English name is supplied with its approved Chinese form; output must contain the exact Chinese form",
            number_policy: "numeric values from ASCII digit runs must be preserved exactly and in order; equivalent Chinese numeral output is normalized only for validation",
        },
        candidates: evidence,
        selection: SelectionEvidence {
            selected_candidate_id: "qwen3.5-4b-q4-k-m",
            reason: "Operational fallback per plan: Qwen3.5 4B is selected provisionally because no smaller candidate can qualify without a completed row-level critical-error and naturalness audit, at least 99% exact name/number preservation, and naturalness matching Qwen3.5 4B. This is not a qualified smaller-model winner.",
            human_naturalness_review: "pending; no human score was inferred or invented",
        },
    };
    fs::write(
        output.join("benchmark.json"),
        serde_json::to_vec_pretty(&benchmark)?,
    )?;
    write_review_packet(output, regions, &benchmark.candidates)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "output": output,
            "selectedCandidateId": benchmark.selection.selected_candidate_id,
            "candidateSummaries": benchmark.candidates.iter().map(|candidate| serde_json::json!({
                "id": candidate.id,
                "summary": candidate.summary,
            })).collect::<Vec<_>>(),
        }))?
    );
    Ok(())
}

fn load_gold(directory: &Path) -> Result<LoadedGold> {
    let fixture_root = directory
        .parent()
        .context("annotations directory must be inside the chapter 5 fixture root")?;
    let manifest_path = fixture_root.join("manifest.json");
    let manifest: BenchmarkManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("failed to read `{}`", manifest_path.display()))?,
    )
    .with_context(|| format!("failed to parse `{}`", manifest_path.display()))?;
    ensure!(
        manifest.id == "30-years-since-the-prologue-chapter-5",
        "unexpected benchmark ID `{}`",
        manifest.id
    );
    ensure!(
        manifest.images.len() == manifest.page_count,
        "manifest pageCount {} does not match {} images",
        manifest.page_count,
        manifest.images.len()
    );
    ensure!(
        manifest.annotation_status.status == "complete"
            && manifest.annotation_status.completed_page_count == manifest.page_count
            && manifest.annotation_status.required_page_count == manifest.page_count,
        "chapter 5 gold fixture is incomplete: status={}, completedPageCount={}, requiredPageCount={}, reasonCode={:?}",
        manifest.annotation_status.status,
        manifest.annotation_status.completed_page_count,
        manifest.annotation_status.required_page_count,
        manifest.annotation_status.reason_code
    );
    let detected_bubble_gold = manifest
        .total_expected_dialogue_bubble_count
        .context("complete manifest is missing totalExpectedDialogueBubbleCount")?;
    let reviewed_region_count = manifest
        .total_expected_region_count
        .context("complete manifest is missing totalExpectedRegionCount")?;
    let english_translation_targets = manifest
        .total_expected_english_translation_target_count
        .context("complete manifest is missing totalExpectedEnglishTranslationTargetCount")?;
    let untouched_exclusions = manifest
        .total_expected_untouched_exclusion_count
        .context("complete manifest is missing totalExpectedUntouchedExclusionCount")?;
    ensure!(
        english_translation_targets + untouched_exclusions == reviewed_region_count,
        "manifest translation targets and untouched exclusions do not equal total reviewed regions"
    );

    let mut pages = Vec::new();
    for (index, image) in manifest.images.iter().enumerate() {
        let order = index + 1;
        ensure!(
            image.order == order,
            "manifest page order {} is not contiguous at {order}",
            image.order
        );
        ensure!(
            image.file == format!("{order:03}.webp"),
            "manifest page filename mismatch at {order}"
        );
        let annotation = image
            .annotation
            .as_deref()
            .with_context(|| format!("manifest page {order} has no annotation"))?;
        ensure!(
            annotation == format!("annotations/{order:03}.json"),
            "manifest page {order} annotation path is not canonical"
        );
        let path = fixture_root.join(annotation);
        let page: PageAnnotations = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("failed to read `{}`", path.display()))?,
        )
        .with_context(|| format!("failed to parse `{}`", path.display()))?;
        ensure!(
            page.page.order == order,
            "{} page order mismatch",
            path.display()
        );
        ensure!(
            page.page.file == image.file,
            "{} page filename mismatch",
            path.display()
        );
        for (region_index, region) in page.regions.iter().enumerate() {
            ensure!(
                region.id == format!("30ysp-ch5-p{order:03}-r{region_index:02}"),
                "{} region ID is not canonical",
                region.id
            );
            ensure!(
                matches!(region.kind.as_str(), "dialogue" | "thought" | "narration"),
                "unsupported region kind `{}`",
                region.kind
            );
            ensure!(
                !region.normalized_english.trim().is_empty(),
                "empty normalized English gold text for {}",
                region.id
            );
            let confident_english = has_confident_english_text(&region.normalized_english);
            ensure!(
                matches!(
                    (confident_english, region.translation_target),
                    (true, None) | (false, Some(false))
                ),
                "{} must omit translationTarget only for confident Latin English, or set translationTarget=false for an untouched ambiguous exclusion",
                region.id
            );
            ensure!(
                !confident_english || !region.simplified_chinese.trim().is_empty(),
                "{} is a translation target with empty Chinese gold",
                region.id
            );
        }
        pages.extend(page.regions);
    }
    ensure!(
        pages.len() == reviewed_region_count,
        "manifest expects {reviewed_region_count} reviewed regions, found {}",
        pages.len()
    );
    let found_detector_gold = pages
        .iter()
        .filter(|region| matches!(region.kind.as_str(), "dialogue" | "thought"))
        .count();
    ensure!(
        found_detector_gold == detected_bubble_gold,
        "manifest expects {detected_bubble_gold} detector-gold regions, found {found_detector_gold}"
    );
    Ok(LoadedGold {
        regions: pages,
        page_count: manifest.page_count,
        detected_bubble_gold,
        english_translation_targets,
        untouched_exclusions,
    })
}

fn has_confident_english_text(text: &str) -> bool {
    let alphabetic = text
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect::<Vec<_>>();
    !alphabetic.is_empty()
        && alphabetic
            .iter()
            .all(|character| is_latin_letter(*character))
}

fn is_latin_letter(character: char) -> bool {
    character.is_ascii_alphabetic()
        || matches!(
            character as u32,
            0x00c0..=0x00ff | 0x0100..=0x017f | 0x0180..=0x024f | 0x1e00..=0x1eff
        )
}

async fn run_protocol_probe(
    runtime: &RuntimeManager,
    backend: Arc<LlamaBackend>,
    candidate: &Candidate,
    regions: &[GoldRegion],
    requested_batches: &[usize],
) -> Result<ProtocolProbeEvidence> {
    let batch_count = regions.len() / BATCH_MAX;
    ensure!(
        regions.len().is_multiple_of(BATCH_MAX),
        "protocol probe requires the corrected 18x6 workload"
    );
    ensure!(
        requested_batches
            .iter()
            .all(|batch| (1..=batch_count).contains(batch)),
        "protocol probe batch indices must be from 1 through {batch_count}"
    );

    let load_start = Instant::now();
    let mut llm = Llm::load_file(
        runtime,
        candidate.model_id,
        false,
        candidate.path.clone(),
        backend,
    )
    .await
    .with_context(|| format!("failed to load {}", candidate.display_name))?;
    let load_ms = duration_ms(load_start);

    let warmup_batch = &regions[..BATCH_MAX];
    let warmup_names = protected_names(warmup_batch);
    let warmup_prompt = build_user_prompt(warmup_batch, &[], &warmup_names);
    let warmup_start = Instant::now();
    let _ = llm.generate_constrained(
        &warmup_prompt,
        &GenerateOptions::greedy(output_token_budget(warmup_batch)),
        Language::ChineseSimplified,
        &system_prompt(warmup_batch.len()),
        &AtomicBool::new(false),
    )?;
    let warmup_ms = duration_ms(warmup_start);

    let mut results = Vec::with_capacity(requested_batches.len() * 3);
    for batch_index in requested_batches {
        let start = (batch_index - 1) * BATCH_MAX;
        let batch = &regions[start..start + BATCH_MAX];
        let prior = regions[..start]
            .iter()
            .rev()
            .take(CONTEXT_MAX_UTTERANCES)
            .collect::<Vec<_>>();
        let prior = prior
            .into_iter()
            .rev()
            .map(|region| ContextItem {
                source_english: region.normalized_english.clone(),
                chinese: region.simplified_chinese.clone(),
            })
            .collect::<Vec<_>>();
        let context = bound_context(&llm, &prior)?;
        let names = protected_names(batch);

        for variant in [
            "legacy-records",
            "explicit-count-records",
            "release-readable-glossary",
        ] {
            let (system, user) = match variant {
                "legacy-records" => (
                    legacy_system_prompt(),
                    legacy_user_prompt(batch, &context, &names),
                ),
                "explicit-count-records" => (
                    system_prompt(batch.len()),
                    legacy_user_prompt(batch, &context, &names),
                ),
                "release-readable-glossary" => (
                    system_prompt(batch.len()),
                    build_user_prompt(batch, &context, &names),
                ),
                _ => unreachable!("fixed probe variant"),
            };
            let max_output_tokens = output_token_budget(batch);
            let started = Instant::now();
            let raw_output = llm.generate_constrained(
                &user,
                &GenerateOptions::greedy(max_output_tokens),
                Language::ChineseSimplified,
                &system,
                &AtomicBool::new(false),
            )?;
            let wall_ms = duration_ms(started);
            let (parsed, _) = parse_numbered_output(&raw_output, batch);
            results.push(ProtocolProbeResult {
                variant,
                batch_index: *batch_index,
                item_ids: batch.iter().map(|region| region.id.clone()).collect(),
                system_prompt: system,
                user_prompt: user,
                max_output_tokens,
                wall_ms,
                raw_output: raw_output.clone(),
                exact_ordered_shape: has_exact_ordered_shape(&raw_output, batch.len()),
                parsed_positions: parsed.iter().filter(|item| item.issue.is_none()).count(),
            });
        }
    }
    drop(llm);

    Ok(ProtocolProbeEvidence {
        model_id: candidate.model_id.to_string(),
        model_path: candidate.path.display().to_string(),
        expected_sha256: candidate.expected_sha256.to_owned(),
        prompt_revision: PROMPT_REVISION,
        prompt_hash: PROMPT_HASH,
        load_ms,
        warmup_ms,
        corrected_batch_count: batch_count,
        requested_batches: requested_batches.to_vec(),
        results,
    })
}

fn legacy_system_prompt() -> String {
    format!(
        "Translate only the numbered English manga lines after INPUT into concise, natural \
Simplified Chinese for a reader targeting cumulative HSK 2.0 level {REQUESTED_HSK_LEVEL}. Prefer \
vocabulary at or below that level and short grammar. Preserve meaning, tone, relationships, \
protected names, ASCII numbers, every negation, and question intent. C and N records are \
reference-only; never output them. The numbers are temporary positions, not application IDs. \
Output each requested position once in the same order, exactly \
`<position><TAB><Simplified Chinese>`. Never copy the English input or add a third field. Output no \
other text."
    )
}

fn legacy_user_prompt(
    batch: &[GoldRegion],
    context: &[ContextItem],
    names: &[NameMapping],
) -> String {
    let mut prompt = String::new();
    writeln!(
        &mut prompt,
        "REV\tdirect-hsk-en-zh-2026-07-26\nLEVEL\t{REQUESTED_HSK_LEVEL}"
    )
    .expect("String writes cannot fail");
    writeln!(&mut prompt, "CONTEXT\t{}", context.len()).expect("String writes cannot fail");
    for item in context {
        writeln!(
            &mut prompt,
            "C\t{}\t{}",
            compact(&item.source_english),
            compact(&item.chinese)
        )
        .expect("String writes cannot fail");
    }
    writeln!(&mut prompt, "NAMES\t{}", names.len()).expect("String writes cannot fail");
    for name in names {
        writeln!(
            &mut prompt,
            "N\t{}\t{}",
            compact(name.source),
            compact(name.chinese)
        )
        .expect("String writes cannot fail");
    }
    writeln!(&mut prompt, "INPUT\t{}", batch.len()).expect("String writes cannot fail");
    for (index, region) in batch.iter().enumerate() {
        writeln!(
            &mut prompt,
            "{}\t{}",
            index + 1,
            compact(&region.normalized_english)
        )
        .expect("String writes cannot fail");
    }
    prompt
}

fn has_exact_ordered_shape(output: &str, expected_count: usize) -> bool {
    let lines = output
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    lines.len() == expected_count
        && lines.iter().enumerate().all(|(index, line)| {
            let position = index + 1;
            let digits = position.to_string();
            line.starts_with(&digits)
                && line.as_bytes().get(digits.len()) == Some(&b'\t')
                && !line[digits.len() + 1..].trim().is_empty()
                && !line[digits.len() + 1..].contains('\t')
        })
}

async fn run_candidate(
    runtime: &RuntimeManager,
    backend: Arc<LlamaBackend>,
    candidate: &Candidate,
    regions: &[GoldRegion],
) -> Result<CandidateEvidence> {
    let load_start = Instant::now();
    let mut llm = Llm::load_file(
        runtime,
        candidate.model_id,
        false,
        candidate.path.clone(),
        backend,
    )
    .await
    .with_context(|| format!("failed to load {}", candidate.display_name))?;
    let load_ms = duration_ms(load_start);

    let warmup_batch = &regions[..BATCH_MAX];
    let warmup_names = protected_names(warmup_batch);
    let warmup_prompt = build_user_prompt(warmup_batch, &[], &warmup_names);
    let warmup_options = GenerateOptions::greedy(output_token_budget(warmup_batch));
    let warmup_start = Instant::now();
    let warmup_raw = llm.generate_constrained(
        &warmup_prompt,
        &warmup_options,
        Language::ChineseSimplified,
        &system_prompt(warmup_batch.len()),
        &AtomicBool::new(false),
    )?;
    let warmup_ms = duration_ms(warmup_start);
    let warmup_output_tokens = llm.token_count(&warmup_raw)?;

    let mut context = Vec::<ContextItem>::new();
    let mut batches = Vec::new();
    for (batch_index, batch) in regions.chunks(BATCH_MAX).enumerate() {
        let bounded_context = bound_context(&llm, &context)?;
        let context_tokens = llm.token_count(&render_context(&bounded_context))?;
        let names = protected_names(batch);
        let user_prompt = build_user_prompt(batch, &bounded_context, &names);
        let user_prompt_tokens = llm.token_count(&user_prompt)?;
        let max_output_tokens = output_token_budget(batch);
        let options = GenerateOptions::greedy(max_output_tokens);
        let start = Instant::now();
        let mut first_piece_ms = None;
        let mut first_complete_line_ms = None;
        let mut streamed = String::new();
        let raw = llm.generate_constrained_streaming(
            &user_prompt,
            &options,
            Language::ChineseSimplified,
            &system_prompt(batch.len()),
            &AtomicBool::new(false),
            |piece| {
                if first_piece_ms.is_none() && !piece.is_empty() {
                    first_piece_ms = Some(duration_ms(start));
                }
                streamed.push_str(piece);
                if first_complete_line_ms.is_none() && streamed.contains('\n') {
                    first_complete_line_ms = Some(duration_ms(start));
                }
                Ok(())
            },
        )?;
        let wall_ms = duration_ms(start);
        let output_tokens = llm.token_count(&raw)?;
        let (primary, ignored_output_lines) = parse_numbered_output(&raw, batch);
        let invalid = batch
            .iter()
            .zip(&primary)
            .enumerate()
            .filter_map(|(index, (region, parsed))| {
                let problems = primary_problems(region, parsed, &names);
                (!problems.is_empty()).then_some((index, problems))
            })
            .collect::<Vec<_>>();
        let mut final_parsed = primary.clone();
        let mut repair_parsed_by_index = HashMap::<usize, ParsedItem>::new();
        let mut repairs = Vec::with_capacity(invalid.len());
        for (index, problems) in &invalid {
            let region = &batch[*index];
            let repair_prompt =
                build_repair_prompt(region, &primary[*index], problems, &bounded_context, &names);
            let repair_user_prompt_tokens = llm.token_count(&repair_prompt)?;
            let repair_max_output_tokens = output_token_budget(std::slice::from_ref(region));
            let repair_options = GenerateOptions::greedy(repair_max_output_tokens);
            let repair_start = Instant::now();
            let mut repair_first_piece_ms = None;
            let mut repair_first_complete_line_ms = None;
            let mut repair_streamed = String::new();
            let repair_raw = llm.generate_constrained_streaming(
                &repair_prompt,
                &repair_options,
                Language::ChineseSimplified,
                &repair_system_prompt(),
                &AtomicBool::new(false),
                |piece| {
                    if repair_first_piece_ms.is_none() && !piece.is_empty() {
                        repair_first_piece_ms = Some(duration_ms(repair_start));
                    }
                    repair_streamed.push_str(piece);
                    if repair_first_complete_line_ms.is_none() && repair_streamed.contains('\n') {
                        repair_first_complete_line_ms = Some(duration_ms(repair_start));
                    }
                    Ok(())
                },
            )?;
            let repair_wall_ms = duration_ms(repair_start);
            let repair_output_tokens = llm.token_count(&repair_raw)?;
            let (repair_parsed, repair_ignored) =
                parse_repair_output(&repair_raw, &region.normalized_english);
            if deterministic_problems(region, &repair_parsed, &names).is_empty() {
                final_parsed[*index] = repair_parsed.clone();
            }
            repair_parsed_by_index.insert(*index, repair_parsed);
            repairs.push(RepairEvidence {
                item_id: region.id.clone(),
                user_prompt_tokens: repair_user_prompt_tokens,
                max_output_tokens: repair_max_output_tokens,
                output_tokens: repair_output_tokens,
                first_piece_ms: repair_first_piece_ms,
                first_complete_line_ms: repair_first_complete_line_ms,
                wall_ms: repair_wall_ms,
                raw_output: repair_raw,
                ignored_output_lines: repair_ignored,
            });
        }
        let items = build_item_evidence(
            batch,
            &final_parsed,
            &names,
            &primary,
            &repair_parsed_by_index,
        );
        for (region, parsed_item) in batch.iter().zip(&final_parsed) {
            if deterministic_problems(region, parsed_item, &names).is_empty()
                && let Some(chinese) = parsed_item.text.clone()
            {
                context.push(ContextItem {
                    source_english: region.normalized_english.clone(),
                    chinese,
                });
            }
        }
        batches.push(BatchEvidence {
            batch_index: batch_index + 1,
            item_ids: batch.iter().map(|region| region.id.clone()).collect(),
            item_count: batch.len(),
            context_count: bounded_context.len(),
            context_tokens,
            user_prompt_tokens,
            max_output_tokens,
            output_tokens,
            first_piece_ms,
            first_complete_line_ms,
            wall_ms,
            total_wall_ms: wall_ms + repairs.iter().map(|repair| repair.wall_ms).sum::<f64>(),
            output_tokens_per_second: rate(output_tokens, wall_ms),
            raw_output: raw,
            ignored_output_lines,
            repairs,
            items,
        });
    }
    let summary = summarize(&batches);
    drop(llm);
    Ok(CandidateEvidence {
        id: candidate.id.to_owned(),
        display_name: candidate.display_name.to_owned(),
        runtime_model_id: candidate.model_id.to_string(),
        model_path: candidate.path.display().to_string(),
        repository_revision: candidate.repository_revision.to_owned(),
        expected_bytes: candidate.expected_bytes,
        expected_sha256: candidate.expected_sha256.to_owned(),
        load_ms,
        warmup_ms,
        warmup_output_tokens,
        batches,
        summary,
    })
}

fn system_prompt(count: usize) -> String {
    primary_system_prompt(REQUESTED_HSK_LEVEL, count)
}

fn repair_system_prompt() -> String {
    shared_repair_system_prompt(REQUESTED_HSK_LEVEL)
}

fn build_user_prompt(
    batch: &[GoldRegion],
    context: &[ContextItem],
    names: &[NameMapping],
) -> String {
    let context = context
        .iter()
        .map(|item| DirectHskContext {
            source_english: &item.source_english,
            chinese: &item.chinese,
        })
        .collect::<Vec<_>>();
    let names = names
        .iter()
        .map(|name| DirectHskName {
            source_english: name.source,
            chinese: name.chinese,
        })
        .collect::<Vec<_>>();
    let sources = batch
        .iter()
        .map(|region| region.normalized_english.as_str())
        .collect::<Vec<_>>();
    primary_user_prompt(&context, &names, &sources)
}

fn build_repair_prompt(
    region: &GoldRegion,
    primary: &ParsedItem,
    problems: &[String],
    context: &[ContextItem],
    names: &[NameMapping],
) -> String {
    let problems = problems.iter().map(String::as_str).collect::<Vec<_>>();
    let names = names
        .iter()
        .map(|name| DirectHskName {
            source_english: name.source,
            chinese: name.chinese,
        })
        .collect::<Vec<_>>();
    // Match production: preceding answers aid primary generation but are
    // omitted from a singular repair so the model cannot copy a sibling.
    let _ = context;
    repair_user_prompt(
        &region.normalized_english,
        primary.text.as_deref(),
        &problems,
        &names,
    )
}

fn protected_names(batch: &[GoldRegion]) -> Vec<NameMapping> {
    let mut names = Vec::new();
    for region in batch {
        let lower = region.normalized_english.to_ascii_lowercase();
        let mut occupied = Vec::<(usize, usize)>::new();
        for mapping in NAME_MAPPINGS {
            let needle = mapping.source.to_ascii_lowercase();
            if let Some(start) = lower.find(&needle) {
                let range = (start, start + needle.len());
                if occupied
                    .iter()
                    .any(|existing| range.0 >= existing.0 && range.1 <= existing.1)
                {
                    continue;
                }
                occupied.push(range);
                if !names.iter().any(|existing: &NameMapping| {
                    existing.source == mapping.source && existing.chinese == mapping.chinese
                }) {
                    names.push(*mapping);
                }
            }
        }
    }
    names
}

fn validate_name_glossary(regions: &[GoldRegion]) -> Result<()> {
    for region in regions {
        let mappings = protected_names(std::slice::from_ref(region));
        for token in region
            .hsk_tokens
            .iter()
            .filter(|token| token.classification == "proper-name")
        {
            ensure!(
                mappings.iter().any(|mapping| {
                    mapping.chinese.contains(&token.text) || token.text.contains(mapping.chinese)
                }),
                "proper-name token `{}` in {} has no English-to-Chinese glossary mapping",
                token.text,
                region.id
            );
        }
    }
    Ok(())
}

fn bound_context(llm: &Llm, all: &[ContextItem]) -> Result<Vec<ContextItem>> {
    let start = all.len().saturating_sub(CONTEXT_MAX_UTTERANCES);
    let mut bounded = all[start..].to_vec();
    while !bounded.is_empty() && llm.token_count(&render_context(&bounded))? > CONTEXT_MAX_TOKENS {
        bounded.remove(0);
    }
    Ok(bounded)
}

fn render_context(context: &[ContextItem]) -> String {
    let context = context
        .iter()
        .map(|item| DirectHskContext {
            source_english: &item.source_english,
            chinese: &item.chinese,
        })
        .collect::<Vec<_>>();
    context_budget_text(&context)
}

fn output_token_budget(batch: &[GoldRegion]) -> usize {
    let source_chars = batch
        .iter()
        .map(|region| region.normalized_english.chars().count())
        .sum::<usize>();
    source_chars
        .div_ceil(2)
        .saturating_add(
            batch
                .len()
                .saturating_mul(OUTPUT_TOKENS_PER_UTTERANCE)
                .saturating_add(8),
        )
        .clamp(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
}

fn parse_numbered_output(output: &str, expected: &[GoldRegion]) -> (Vec<ParsedItem>, Vec<String>) {
    let mut slots = (0..expected.len())
        .map(|_| Vec::<ParsedLine>::new())
        .collect::<Vec<_>>();
    let mut ignored = Vec::new();
    for raw_line in output.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            continue;
        }
        if let Some((position, parsed)) = parse_output_line(line, expected.len()) {
            if matches!(
                &parsed,
                ParsedLine::Candidate(text)
                    if compact(text) == compact(&expected[position - 1].normalized_english)
            ) {
                ignored.push(line.to_owned());
                continue;
            }
            slots[position - 1].push(parsed);
        } else {
            ignored.push(line.to_owned());
        }
    }
    let items = slots
        .into_iter()
        .map(|mut lines| {
            if lines.is_empty() {
                return ParsedItem {
                    text: None,
                    issue: Some("missingLine".to_owned()),
                };
            }
            if lines.len() > 1 {
                let text = lines.drain(..).find_map(|line| match line {
                    ParsedLine::Candidate(text) if !text.trim().is_empty() => {
                        Some(normalize_full_width_digits(text.trim()))
                    }
                    ParsedLine::Candidate(_) | ParsedLine::Malformed(None) => None,
                    ParsedLine::Malformed(Some(text)) if !text.trim().is_empty() => {
                        Some(normalize_full_width_digits(text.trim()))
                    }
                    ParsedLine::Malformed(Some(_)) => None,
                });
                return ParsedItem {
                    text,
                    issue: Some("duplicateLine".to_owned()),
                };
            }
            match lines.pop().expect("one line") {
                ParsedLine::Malformed(text) => ParsedItem {
                    text: text.map(|text| normalize_full_width_digits(text.trim())),
                    issue: Some("malformedLine".to_owned()),
                },
                ParsedLine::Candidate(text) if text.trim().is_empty() => ParsedItem {
                    text: None,
                    issue: Some("emptyTranslation".to_owned()),
                },
                ParsedLine::Candidate(text) => ParsedItem {
                    text: Some(normalize_full_width_digits(text.trim())),
                    issue: None,
                },
            }
        })
        .collect();
    (items, ignored)
}

fn parse_repair_output(output: &str, source_english: &str) -> (ParsedItem, Vec<String>) {
    let mut lines = output
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .filter(|line| !line.trim().is_empty());
    let Some(line) = lines.next() else {
        return (
            ParsedItem {
                text: None,
                issue: Some("missingLine".to_owned()),
            },
            Vec::new(),
        );
    };
    let extra = lines.map(str::to_owned).collect::<Vec<_>>();
    if !extra.is_empty() || line.contains('\t') {
        return (
            ParsedItem {
                text: None,
                issue: Some("malformedLine".to_owned()),
            },
            extra,
        );
    }
    if compact(line) == compact(source_english) {
        return (
            ParsedItem {
                text: None,
                issue: Some("sourceEcho".to_owned()),
            },
            Vec::new(),
        );
    }
    let text = normalize_full_width_digits(line.trim());
    if text.is_empty() {
        return (
            ParsedItem {
                text: None,
                issue: Some("emptyTranslation".to_owned()),
            },
            Vec::new(),
        );
    }
    (
        ParsedItem {
            text: Some(text),
            issue: None,
        },
        Vec::new(),
    )
}

fn parse_output_line(line: &str, expected_count: usize) -> Option<(usize, ParsedLine)> {
    let digit_count = line
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 {
        return None;
    }
    let digits = &line[..digit_count];
    let position = digits.parse::<usize>().ok()?;
    if position == 0 || position > expected_count || position.to_string() != digits {
        return None;
    }
    let text_start = match line.as_bytes().get(digit_count) {
        Some(b'\t') => digit_count + 1,
        Some(b' ') => {
            line.as_bytes()[digit_count..]
                .iter()
                .take_while(|byte| **byte == b' ')
                .count()
                + digit_count
        }
        _ => return Some((position, ParsedLine::Malformed(None))),
    };
    let text = &line[text_start..];
    if text.contains('\t') {
        let mut fields = text.split('\t');
        let kind = fields.next();
        let candidate = fields.next();
        let has_more = fields.next().is_some();
        let salvage = match (kind, candidate, has_more) {
            (Some("D" | "C" | "T" | "S"), Some(candidate), false)
                if !candidate.trim().is_empty() =>
            {
                Some(candidate.to_owned())
            }
            _ => None,
        };
        return Some((position, ParsedLine::Malformed(salvage)));
    }
    Some((position, ParsedLine::Candidate(text.to_owned())))
}

fn deterministic_problems(
    region: &GoldRegion,
    parsed: &ParsedItem,
    names: &[NameMapping],
) -> Vec<String> {
    let mut problems = Vec::new();
    if let Some(issue) = parsed.issue.as_deref() {
        problems.push(
            match issue {
                "missingLine" => "missing numbered output line",
                "duplicateLine" => "duplicate numbered output line",
                "malformedLine" => "line must be exactly `<position><TAB><Simplified Chinese>`",
                "sourceEcho" => "translate the source instead of copying it",
                "emptyTranslation" => "translation is empty",
                _ => "invalid numbered output line",
            }
            .to_owned(),
        );
    }
    let candidate = parsed.text.as_deref().unwrap_or("");
    let expected_numbers = ascii_numbers(&region.normalized_english);
    let actual_numbers = normalized_numbers_for_source(&region.normalized_english, candidate);
    if expected_numbers != actual_numbers {
        problems.push(format!(
            "preserve ASCII numbers exactly: expected {expected_numbers:?}, got {actual_numbers:?}"
        ));
    }
    let source_lower = region.normalized_english.to_ascii_lowercase();
    for mapping in names {
        if source_lower.contains(&mapping.source.to_ascii_lowercase())
            && !candidate.contains(mapping.chinese)
        {
            problems.push(format!(
                "translate protected name `{}` exactly as `{}`",
                mapping.source, mapping.chinese
            ));
        }
    }
    if has_question_intent(&source_lower) && !has_chinese_question_intent(candidate) {
        problems.push("preserve the source question intent".to_owned());
    }
    problems
}

fn primary_problems(
    region: &GoldRegion,
    parsed: &ParsedItem,
    names: &[NameMapping],
) -> Vec<String> {
    let mut problems = deterministic_problems(region, parsed, names);
    let expected = ascii_numbers(&region.normalized_english);
    let actual = ascii_numbers(parsed.text.as_deref().unwrap_or(""));
    if actual != expected
        && !problems
            .iter()
            .any(|problem| problem.starts_with("preserve ASCII numbers exactly:"))
    {
        problems.push(format!(
            "preserve ASCII numbers exactly: expected {expected:?}, got {actual:?}"
        ));
    }
    problems
}

fn build_item_evidence(
    gold: &[GoldRegion],
    parsed: &[ParsedItem],
    names: &[NameMapping],
    primary: &[ParsedItem],
    repair_by_index: &HashMap<usize, ParsedItem>,
) -> Vec<ItemEvidence> {
    gold.iter()
        .zip(parsed)
        .enumerate()
        .map(|(index, (region, parsed))| {
            let candidate = parsed.text.as_deref().unwrap_or("");
            let expected_names = region
                .hsk_tokens
                .iter()
                .filter(|token| token.classification == "proper-name")
                .map(|token| token.text.clone())
                .collect::<Vec<_>>();
            let missing_names = expected_names
                .iter()
                .filter(|name| !candidate.contains(name.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            let expected_ascii_numbers = ascii_numbers(&region.normalized_english);
            let actual_ascii_numbers =
                normalized_numbers_for_source(&region.normalized_english, candidate);
            let question_required =
                has_question_intent(&region.normalized_english.to_ascii_lowercase());
            let question_preserved = !question_required || has_chinese_question_intent(candidate);
            let structured_success = parsed.issue.is_none() && parsed.text.is_some();
            let mut critical = Vec::new();
            if !missing_names.is_empty() {
                critical.push("protectedName".to_owned());
            }
            if actual_ascii_numbers != expected_ascii_numbers {
                critical.push("asciiNumber".to_owned());
            }
            if !question_preserved {
                critical.push("questionIntent".to_owned());
            }
            // Apply the actual prompt-protected names as an additional exact
            // check even when the gold token uses a shorter proper-name span.
            for mapping in names {
                if region
                    .normalized_english
                    .to_ascii_lowercase()
                    .contains(&mapping.source.to_ascii_lowercase())
                    && !candidate.contains(mapping.chinese)
                    && !critical.iter().any(|failure| failure == "protectedName")
                {
                    critical.push("protectedName".to_owned());
                }
            }
            ItemEvidence {
                id: region.id.clone(),
                source_english: region.normalized_english.clone(),
                approved_chinese: region.simplified_chinese.clone(),
                candidate_chinese: parsed.text.clone(),
                structured_success,
                parse_issue: parsed.issue.clone(),
                primary_candidate_chinese: primary[index].text.clone(),
                primary_structured_success: primary[index].issue.is_none()
                    && primary[index].text.is_some(),
                primary_parse_issue: primary[index].issue.clone(),
                repair_attempted: repair_by_index.contains_key(&index),
                repair_candidate_chinese: repair_by_index
                    .get(&index)
                    .and_then(|repair| repair.text.clone()),
                repair_structured_success: repair_by_index
                    .get(&index)
                    .map(|repair| repair.issue.is_none() && repair.text.is_some()),
                repair_parse_issue: repair_by_index
                    .get(&index)
                    .and_then(|repair| repair.issue.clone()),
                expected_names,
                missing_names,
                expected_ascii_numbers,
                actual_ascii_numbers,
                question_required,
                question_preserved,
                exact_reference_match: candidate == region.simplified_chinese,
                character_unigram_f1: ngram_f1(candidate, &region.simplified_chinese, 1),
                character_bigram_f1: ngram_f1(candidate, &region.simplified_chinese, 2),
                critical_proxy_failures: critical,
            }
        })
        .collect()
}

fn summarize(batches: &[BatchEvidence]) -> CandidateSummary {
    let items = batches
        .iter()
        .flat_map(|batch| batch.items.iter())
        .collect::<Vec<_>>();
    let regions = items.len();
    let primary_structured_successes = items
        .iter()
        .filter(|item| item.primary_structured_success)
        .count();
    let structured_successes = items.iter().filter(|item| item.structured_success).count();
    let repair_batches = batches
        .iter()
        .filter(|batch| !batch.repairs.is_empty())
        .count();
    let repaired_items = items.iter().filter(|item| item.repair_attempted).count();
    let repaired_items_passing_final_validation = items
        .iter()
        .filter(|item| item.repair_attempted && item_protocol_valid(item))
        .count();
    let protected_names = items.iter().map(|item| item.expected_names.len()).sum();
    let missing_names = items
        .iter()
        .map(|item| item.missing_names.len())
        .sum::<usize>();
    let protected_names_preserved = protected_names - missing_names;
    let ascii_numbers = items
        .iter()
        .map(|item| item.expected_ascii_numbers.len())
        .sum();
    let ascii_numbers_preserved = items
        .iter()
        .map(|item| {
            item.expected_ascii_numbers
                .iter()
                .zip(&item.actual_ascii_numbers)
                .take_while(|(expected, actual)| expected == actual)
                .count()
        })
        .sum();
    let names_and_numbers = protected_names + ascii_numbers;
    let names_and_numbers_preserved = protected_names_preserved + ascii_numbers_preserved;
    let questions = items.iter().filter(|item| item.question_required).count();
    let questions_preserved = items
        .iter()
        .filter(|item| item.question_required && item.question_preserved)
        .count();
    let items_with_critical_proxy_failures = items
        .iter()
        .filter(|item| !item.critical_proxy_failures.is_empty())
        .count();
    let critical_proxy_failure_count = items
        .iter()
        .map(|item| item.critical_proxy_failures.len())
        .sum();
    let exact_reference_matches = items
        .iter()
        .filter(|item| item.exact_reference_match)
        .count();
    let warm_total_ms = batches.iter().map(|batch| batch.total_wall_ms).sum::<f64>();
    let primary_output_tokens = batches.iter().map(|batch| batch.output_tokens).sum();
    let repair_output_tokens = batches
        .iter()
        .flat_map(|batch| &batch.repairs)
        .map(|repair| repair.output_tokens)
        .sum();
    let output_tokens = primary_output_tokens + repair_output_tokens;
    let mut latencies = batches
        .iter()
        .map(|batch| batch.total_wall_ms)
        .collect::<Vec<_>>();
    let first_line_latencies = batches
        .iter()
        .filter_map(|batch| batch.first_complete_line_ms)
        .collect::<Vec<_>>();
    let automated_zero_critical_proxy_failures = critical_proxy_failure_count == 0;
    let names_and_numbers_preservation_rate =
        fraction(names_and_numbers_preserved, names_and_numbers);
    CandidateSummary {
        regions,
        primary_structured_successes,
        primary_structured_success_rate: fraction(primary_structured_successes, regions),
        structured_successes,
        structured_success_rate: fraction(structured_successes, regions),
        repair_batches,
        repaired_items,
        repaired_items_passing_final_validation,
        protected_names,
        protected_names_preserved,
        protected_name_preservation_rate: fraction(protected_names_preserved, protected_names),
        ascii_numbers,
        ascii_numbers_preserved,
        ascii_number_preservation_rate: fraction(ascii_numbers_preserved, ascii_numbers),
        names_and_numbers,
        names_and_numbers_preserved,
        names_and_numbers_preservation_rate,
        questions,
        questions_preserved,
        items_with_critical_proxy_failures,
        critical_proxy_failure_count,
        exact_reference_matches,
        mean_character_unigram_f1: mean(
            items.iter().map(|item| item.character_unigram_f1),
            regions,
        ),
        mean_character_bigram_f1: mean(items.iter().map(|item| item.character_bigram_f1), regions),
        warm_batch_latency_p50_ms: nearest_rank(&mut latencies, 0.50),
        warm_batch_latency_p95_ms: nearest_rank(&mut latencies, 0.95),
        warm_first_line_p50_ms: (!first_line_latencies.is_empty()).then(|| {
            let mut values = first_line_latencies;
            nearest_rank(&mut values, 0.50)
        }),
        warm_total_ms,
        primary_output_tokens,
        repair_output_tokens,
        output_tokens,
        aggregate_output_tokens_per_second: rate(output_tokens, warm_total_ms),
        automated_zero_critical_proxy_failures,
        names_and_numbers_at_least_99_percent: names_and_numbers_preservation_rate >= 0.99,
        human_naturalness_matches_qwen4b: None,
        qualifies_as_smaller_replacement: false,
    }
}

fn item_protocol_valid(item: &ItemEvidence) -> bool {
    item.structured_success
        && !item
            .critical_proxy_failures
            .iter()
            .any(|failure| matches!(failure.as_str(), "protectedName" | "asciiNumber"))
}

fn write_review_packet(
    output: &Path,
    regions: &[GoldRegion],
    candidates: &[CandidateEvidence],
) -> Result<()> {
    let review_dir = output.join("blinded-review");
    fs::create_dir_all(&review_dir)?;
    // Deliberately permuted relative to benchmark order. The key is separate
    // from the reviewer-facing packet.
    let label_order = [
        ("Candidate A", "hy-mt2-1.8b-q4-k-m"),
        ("Candidate B", "qwen3.5-4b-q4-k-m"),
        ("Candidate C", "qwen3.5-2b-q4-k-m"),
    ];
    let key = label_order
        .iter()
        .map(|(label, id)| {
            let candidate = candidates
                .iter()
                .find(|candidate| candidate.id == *id)
                .expect("fixed candidate exists");
            serde_json::json!({
                "label": label,
                "candidateId": candidate.id,
                "displayName": candidate.display_name,
                "repositoryRevision": candidate.repository_revision,
                "sha256": candidate.expected_sha256,
            })
        })
        .collect::<Vec<_>>();
    fs::write(
        review_dir.join("candidate-key.json"),
        serde_json::to_vec_pretty(&key)?,
    )?;

    let outputs = candidates
        .iter()
        .map(|candidate| {
            let values = candidate
                .batches
                .iter()
                .flat_map(|batch| batch.items.iter())
                .map(|item| (item.id.clone(), item.candidate_chinese.clone()))
                .collect::<HashMap<_, _>>();
            (candidate.id.as_str(), values)
        })
        .collect::<HashMap<_, _>>();
    let mut packet = String::new();
    packet.push_str(
        "# Blinded naturalness review: 30 Years Since the Prologue chapter 5\n\n\
Reviewer instructions: read the English source and each anonymous Chinese candidate. \
For each candidate, enter a naturalness score from 1 (unusable Chinese) through 5 \
(native-quality manga dialogue), mark any critical meaning error as yes/no, and add \
brief notes. Do not open `candidate-key.json` until every row is complete. Blank \
cells are intentional; no human score has been inferred.\n\n",
    );
    for region in regions {
        writeln!(
            &mut packet,
            "## {}\n\nEnglish: {}\n",
            region.id, region.normalized_english
        )?;
        packet.push_str("| Anonymous candidate | Chinese output | Naturalness (1-5) | Critical meaning error (yes/no) | Notes |\n");
        packet.push_str("| --- | --- | --- | --- | --- |\n");
        for (label, id) in &label_order {
            let text = outputs
                .get(id)
                .and_then(|candidate| candidate.get(&region.id))
                .and_then(Option::as_deref)
                .unwrap_or("<missing>");
            writeln!(
                &mut packet,
                "| {label} | {} |  |  |  |",
                escape_markdown(text)
            )?;
        }
        packet.push('\n');
    }
    fs::write(review_dir.join("naturalness-review.md"), packet)?;
    Ok(())
}

fn compact(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_full_width_digits(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '０'..='９' => {
                char::from_u32(u32::from(character) - u32::from('０') + u32::from('0'))
                    .expect("full-width digit has an ASCII form")
            }
            _ => character,
        })
        .collect()
}

fn normalized_numbers_for_source(source_english: &str, text: &str) -> Vec<String> {
    let expected = ascii_numbers(source_english);
    let actual_ascii = ascii_numbers(text);
    if actual_ascii == expected || !actual_ascii.is_empty() {
        return actual_ascii;
    }
    expected
        .into_iter()
        .filter(|number| {
            chinese_number_variants(number)
                .iter()
                .any(|chinese| text.contains(chinese))
        })
        .collect()
}

fn chinese_number_variants(ascii: &str) -> Vec<String> {
    let digit_sequence = ascii
        .chars()
        .filter_map(|digit| match digit {
            '0' => Some('零'),
            '1' => Some('一'),
            '2' => Some('二'),
            '3' => Some('三'),
            '4' => Some('四'),
            '5' => Some('五'),
            '6' => Some('六'),
            '7' => Some('七'),
            '8' => Some('八'),
            '9' => Some('九'),
            _ => None,
        })
        .collect::<String>();
    let mut variants = vec![digit_sequence];
    if let Ok(value) = ascii.parse::<u16>()
        && value <= 9_999
    {
        let standard = chinese_integer_below_10_000(value);
        if !variants.contains(&standard) {
            variants.push(standard.clone());
        }
        if standard.starts_with("二百") || standard.starts_with("二千") {
            variants.push(format!("两{}", &standard['二'.len_utf8()..]));
        }
    }
    variants.sort_by_key(|variant| std::cmp::Reverse(variant.len()));
    variants
}

fn chinese_integer_below_10_000(value: u16) -> String {
    if value == 0 {
        return "零".to_owned();
    }
    let digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    let units = ["千", "百", "十", ""];
    let divisors = [1_000_u16, 100, 10, 1];
    let mut rendered = String::new();
    let mut zero_pending = false;
    for (index, divisor) in divisors.into_iter().enumerate() {
        let digit = usize::from(value / divisor % 10);
        if digit == 0 {
            zero_pending |= !rendered.is_empty() && value % divisor != 0;
            continue;
        }
        if zero_pending {
            rendered.push('零');
            zero_pending = false;
        }
        if !(digit == 1 && divisor == 10 && rendered.is_empty()) {
            rendered.push(digits[digit]);
        }
        rendered.push_str(units[index]);
    }
    rendered
}

fn ascii_numbers(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut numbers = Vec::new();
    let mut start = None;
    for (index, byte) in bytes.iter().copied().enumerate() {
        if byte.is_ascii_digit() {
            start.get_or_insert(index);
        } else if let Some(number_start) = start.take() {
            numbers.push(text[number_start..index].to_owned());
        }
    }
    if let Some(number_start) = start {
        numbers.push(text[number_start..].to_owned());
    }
    numbers
}

fn has_question_intent(source_lower: &str) -> bool {
    if source_lower.contains('?') {
        return true;
    }
    let words = source_lower
        .split(|character: char| !character.is_ascii_alphabetic())
        .filter(|word| !word.is_empty())
        .take(3)
        .collect::<Vec<_>>();
    let [first, rest @ ..] = words.as_slice() else {
        return false;
    };
    if matches!(
        *first,
        "how" | "what" | "when" | "where" | "who" | "whom" | "whose" | "why"
    ) {
        return true;
    }
    if *first == "which" {
        return rest.first().is_none_or(|second| *second != "means");
    }
    let Some(second) = rest.first() else {
        return false;
    };
    matches!(
        *first,
        "am" | "are"
            | "can"
            | "could"
            | "did"
            | "do"
            | "does"
            | "had"
            | "has"
            | "have"
            | "is"
            | "may"
            | "might"
            | "must"
            | "shall"
            | "should"
            | "was"
            | "were"
            | "will"
            | "would"
    ) && !matches!(*second, "not" | "going" | "begun")
}

fn has_chinese_question_intent(text: &str) -> bool {
    text.contains('?')
        || text.contains('？')
        || [
            "什么",
            "怎么",
            "为什么",
            "为何",
            "谁",
            "哪",
            "是否",
            "吗",
            "呢",
            "几",
            "多少",
            "难道",
        ]
        .iter()
        .any(|marker| text.contains(marker))
}

fn ngram_f1(candidate: &str, reference: &str, width: usize) -> f64 {
    let candidate = normalized_han_and_ascii(candidate);
    let reference = normalized_han_and_ascii(reference);
    let candidate_grams = ngrams(&candidate, width);
    let reference_grams = ngrams(&reference, width);
    if candidate_grams.is_empty() && reference_grams.is_empty() {
        return 1.0;
    }
    if candidate_grams.is_empty() || reference_grams.is_empty() {
        return 0.0;
    }
    let mut available = BTreeMap::<String, usize>::new();
    for gram in &reference_grams {
        *available.entry(gram.clone()).or_default() += 1;
    }
    let mut overlap = 0usize;
    for gram in &candidate_grams {
        if let Some(count) = available.get_mut(gram)
            && *count > 0
        {
            *count -= 1;
            overlap += 1;
        }
    }
    let precision = overlap as f64 / candidate_grams.len() as f64;
    let recall = overlap as f64 / reference_grams.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn normalized_han_and_ascii(value: &str) -> Vec<char> {
    value
        .chars()
        .filter(|character| {
            character.is_alphanumeric()
                || ('\u{3400}'..='\u{4dbf}').contains(character)
                || ('\u{4e00}'..='\u{9fff}').contains(character)
        })
        .flat_map(char::to_lowercase)
        .collect()
}

fn ngrams(characters: &[char], width: usize) -> Vec<String> {
    if characters.len() < width {
        return Vec::new();
    }
    characters
        .windows(width)
        .map(|window| window.iter().collect())
        .collect()
}

fn nearest_rank(values: &mut [f64], quantile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let rank = (quantile * values.len() as f64).ceil() as usize;
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(values: impl Iterator<Item = f64>, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        values.sum::<f64>() / count as f64
    }
}

fn rate(tokens: usize, milliseconds: f64) -> f64 {
    if milliseconds <= 0.0 {
        0.0
    } else {
        tokens as f64 / (milliseconds / 1_000.0)
    }
}

fn duration_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn escape_markdown(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\r', "")
        .replace('\n', "<br>")
}

#[cfg(test)]
mod tests {
    use super::{has_chinese_question_intent, has_question_intent};

    #[test]
    fn question_intent_ignores_imperatives_and_sentence_fragments() {
        assert!(!has_question_intent("do not mourn those who left."));
        assert!(!has_question_intent("which means this place is far away."));
        assert!(!has_question_intent("...am going to the university."));
        assert!(!has_question_intent("...has begun preparing."));
    }

    #[test]
    fn question_intent_accepts_punctuation_inversion_and_chinese_markers() {
        assert!(has_question_intent("you are leaving?"));
        assert!(has_question_intent("has it been seven years"));
        assert!(has_question_intent("what on earth..."));
        assert!(has_chinese_question_intent("到底怎么回事……"));
        assert!(has_chinese_question_intent("你要走吗"));
    }
}
