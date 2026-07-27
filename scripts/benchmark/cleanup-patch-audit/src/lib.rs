#![allow(dead_code)]

mod geometry;

// Compile the production implementation byte-for-byte in this small CPU-only
// harness. The main workspace's browser package pulls in unrelated native model
// build scripts, so this isolated manifest keeps correctness evaluation local.
#[path = "../../../../crates/browser-companion/src/pipeline_adapter/patch.rs"]
mod patch;
