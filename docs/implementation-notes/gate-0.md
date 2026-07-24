# Gate 0 implementation note

Date: 2026-07-24

## Completed evidence

- Koharu is pinned to
  `2107843f0c7e2458de5a329980c78575401babb5`; the extraction map is recorded in
  ADR 0003.
- Protocol v1 has shared valid and malformed fixtures consumed by Rust and
  TypeScript.
- `cargo test -p browser-companion` passed the initial contract suite.
- Firefox contract parsing passed both TypeScript type checking and Vitest.
- `fixtures/images/gate0-source.png` is reproducibly generated, synthetic, and
  licence-recorded.
- The existing `koharu-app` pipeline executable calls the same
  `koharu_app::pipeline::run` path as the application and completed against the
  committed synthetic fixture.
- The permanent extension ID is
  `hsk-manga-translator@local.mangalations`; the native host is
  `local.mangalations.hsk_manga`.

## Development-machine audit

| Component | Observed |
| --- | --- |
| Rust | `rustc`/Cargo 1.95 |
| JavaScript | Node 24.15, npm 11.12; Bun invoked through pinned `bun@1.3.14` |
| Browser | Firefox 153 |
| Hardware | RTX 4080 SUPER (16 GiB), 32 GiB system RAM |
| Native build helpers | CMake and Ninja from Visual Studio 2019 Build Tools |
| Added for this check | official LLVM 22.1.0 extracted into ignored project cache; published installer SHA-256 verified |
| Still missing | CUDA Toolkit (`nvcc`), Playwright Firefox browser |

An initial `cargo check -p koharu-app --bin pipeline` failed before CMake was
added to the process path. With CMake/Ninja configured, a single-job retry
reached `koharu-llm`'s build script and failed deterministically because bindgen
could not find `clang.dll` or `libclang.dll`. The error explicitly requests a
valid `LIBCLANG_PATH`.

LLVM 22.1.0's official Windows installer was downloaded from the LLVM GitHub
release, verified as
`b31d5f54942e017cb878e594529723dd629cc7b54c9bf7a331e2dc01e8ea5e75`,
and extracted without system installation. After importing the existing Visual
Studio 2019 developer environment and pointing `LIBCLANG_PATH` at that
extraction, `cargo check -p koharu-app --bin pipeline -j 1` passed.

## Headless production-pipeline run

Input:

```text
fixtures/images/gate0-source.png
SHA-256 caee11b97a553fecdd51064c0218f6245a354b8a5794388d033f8dbcbf31abc5
900 x 1200
```

The generator produced the same byte hash in two independent reruns. The
unmodified pipeline ran these seven pinned stages and reported actual progress
at 0, 14, 28, 42, 57, 71, 85, and 100 percent:

```text
speech-bubble-segmentation
pp-doclayout-v3
paddle-ocr-vl-1.6
yuzumarker-font-detection
comic-text-detector-seg
lama-manga
koharu-renderer
```

The first build/setup/run completed in 418 seconds and populated 2.421 GiB of
Koharu's ignored runtime/model cache. A second run of the built executable with
that cache warm completed in 223 seconds. The CLI was invoked with `--cpu`, but
the downloaded llama.cpp runtime still discovered Vulkan and the PaddleOCR-VL
path reported using the RTX 4080 SUPER; this run must not be described as a
pure-CPU benchmark.

The scene contained four ordered text nodes:

| OCR text | Confidence |
| --- | ---: |
| `WE HAVE TO / LEAVE NOW!` | 0.6746 |
| `ARE YOU READY?` | 0.6932 |
| `YES. / LET'S GO!` | 0.6832 |
| `A synthetic Gate 0 fixture` | 0.3732 |

Visual inspection confirmed that `segment.png` covers all four text areas and
that `inpainted.png` removes their glyphs while retaining panel and bubble
outlines. The run also emitted `bubble.png`, `rendered.png`, and `scene.json`.

## Gate status

Spike 0A (Koharu extraction and a real headless pipeline run) and Spike 0D
(shared protocol fixtures) are complete. Gate 0 is not marked passed yet
because the remaining requirements must be measured in real Firefox:

- native launch from Firefox with detached-daemon survival and duplicate
  discovery;
- a representative large binary transfer to the extension.

Those checks are deliberately not replaced with mocks or inferred from unit
tests. Gate 1/2 fixture implementation may be integrated independently, but
Gate 3 remains blocked until the production pipeline evidence exists.
