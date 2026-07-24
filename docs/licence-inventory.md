# Licence inventory

This inventory covers new browser-product assets and externally hosted model
candidates. Koharu itself remains `GPL-3.0-only` at the pinned upstream
revision.

| Asset | Source/revision | Licence | Distribution decision |
| --- | --- | --- | --- |
| Gate 0 synthetic image and annotations | generated in this repository | GPL-3.0-only | committed |
| Qwen3.5 4B Q4_K_M GGUF | `unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523` | Apache-2.0, inherited from Qwen | remote candidate only; not bundled |
| Qwen3.5 2B Q4_K_M GGUF | `unsloth/Qwen3.5-2B-GGUF@f6d5376be1edb4d416d56da11e5397a961aca8ae` | Apache-2.0, inherited from Qwen | remote candidate only; not bundled |
| Hunyuan-MT 7B Q4_K_M GGUF | `Mungert/Hunyuan-MT-7B-GGUF@61e98ae605cc4fe9581fd3ff1052a271843e4d64` | Tencent Hunyuan Community License Agreement inherited from `tencent/Hunyuan-MT-7B@9305c78383f0bcc94358e08667ee2c76107877e3` | excluded; terms deny use in EU, UK, and South Korea |
| LFM2.5 1.2B Q4_K_M GGUF | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF@047e06635fbe71469926b35ea414537245218200` | LFM Open License v1.0 | remote research candidate only; commercial use is conditioned on revenue below US$10M |

Primary licence locations are recorded in the model manifest. Checksums refer
to the exact remote files, not to repository metadata.

HSK and dictionary runtime data have their own attribution and source-hash
files beside the generated data. They must be included here before a release
artifact is made.
