# llama.cpp for Nintendo Switch

Run large language models locally on a Nintendo Switch. This is a port of [llama.cpp](https://github.com/ggml-org/llama.cpp) to Switch homebrew (devkitPro / libnx): a console chat app, `llama-switch.nro`, that loads GGUF models from the SD card and runs them on the Tegra X1 CPU. No PC or server is involved; the network is only used to download models.

## Features

- Chat with streaming replies, multi-turn context and KV-cache reuse between turns
- Built-in catalog of small chat models that fit in the Switch's memory, downloadable from Hugging Face inside the app (resumable, with progress, speed and ETA)
- Any other `.gguf` copied to the SD card is picked up automatically
- Per-model context size, temperature, top-p and top-k
- CPU clock setting: Stock, Boost (default) or Overclock
- In-app benchmark (prompt processing and generation speed, memory bandwidth)
- Cortex-A57 NEON kernels for Q4_0, Q8_0, Q4_K, Q5_K and Q6_K (the A57 has no dotprod/i8mm, so upstream's fast paths don't apply)

## Requirements

- A Switch that can run homebrew (custom firmware with the Homebrew Menu)
- **Title override mode.** Start the Homebrew Menu by holding **R** while launching any game. That gives the app about 3.2 GB of RAM. From the album (applet mode) it only gets about 400 MB, which no model in the catalog fits into; the app shows a warning banner in that case.
- Wi-Fi, only for downloading models in the app
- Free SD card space for the models (0.4 – 1.9 GB each)

## Installation

1. Download `llama-switch.nro` from the [latest release](../../releases/latest).
2. Copy it to `sdmc:/switch/` on the SD card.
3. Launch it from the Homebrew Menu in title override mode (see above).
4. Pick a model, choose **Download**, then **Load model** and **Chat**.

## Models

The app keeps its model list in `sdmc:/switch/llama_cpp_switch/models.json`. It is created from the built-in catalog on first start:

| Model | Quant | Download | Context |
|---|---|---|---|
| LFM2.5 1.2B Instruct (QAD) | Q4_0 | 696 MB | 4096 |
| Qwen3 1.7B | Q4_K_M | 1.1 GB | 4096 |
| Gemma 3 1B IT | Q4_K_M | 806 MB | 4096 |
| Qwen3.5 0.8B | Q8_0 | 812 MB | 4096 |
| Qwen3.5 2B | Q4_K_M | 1.3 GB | 4096 |
| Granite 4.0 H 1B | Q4_K_M | 901 MB | 4096 |
| Qwen3 0.6B | Q8_0 | 639 MB | 4096 |
| LFM2.5 350M | Q8_0 | 379 MB | 4096 |
| SmolLM3 3B | Q4_K_M | 1.9 GB | 2048 |

LFM2.5 1.2B is the recommended starting point. SmolLM3 3B is the largest model that fits and is slow. The Qwen and SmolLM3 entries use `/no_think` as their system prompt, so they answer directly instead of spending minutes on a thinking phase first.

### Using your own models

Copy any `.gguf` file into `sdmc:/switch/llama_cpp_switch/models/`; it shows up in the model list on the next start or after **Reload JSON**. Keep the file size plus KV cache under about 3 GB. In practice that means models up to about 3B parameters at 4-bit.

You can also edit `models.json` and choose **Reload JSON** in the menu:

```json
{
  "models": [
    {
      "name": "My model",
      "url": "https://huggingface.co/<user>/<repo>/resolve/main/<file>.gguf",
      "path": "sdmc:/switch/llama_cpp_switch/models/<file>.gguf",
      "context": 4096,
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": 40,
      "system_prompt": "",
      "chat_template": ""
    }
  ]
}
```

`url` is optional (without it the entry is local-only); Hugging Face `/blob/main/` links are rewritten to `/resolve/main/` automatically. `context` is clamped to 256 – 8192. `system_prompt` and `chat_template` are optional. Leave `chat_template` empty to use the template stored in the GGUF.

## Controls

**Menu**

| Button | Action |
|---|---|
| Up / Down | Move |
| L / R | Jump between sections |
| Left / Right | Change a value |
| A | Select |
| + | Quit |

**Chat**

| Button | Action |
|---|---|
| X | Write or edit the prompt (on-screen keyboard) |
| A | Send the prompt (opens the keyboard if it is empty) |
| B (hold) | Stop the reply; press B to go back to the menu |
| Y | Clear the chat |
| − | Take back the last prompt to edit it |
| Up / Down, ZL / ZR | Scroll |

**Benchmark:** A full run, X quick run, Y toggle thread pinning, B back.

## CPU clock

| Setting | What it does |
|---|---|
| Stock | Leaves the clocks alone |
| Boost (default) | CPU 1785 MHz and memory 1600 MHz, also in handheld mode |
| Overclock | The highest rates the system allows. Needs sys-clk-OC and asks for confirmation first. Runs hot and drains the battery. |

The setting only applies while a model is loaded. The normal clocks come back when the model is unloaded or the app exits. If sys-clk is running, it may override the setting.

## Files on the SD card

All under `sdmc:/switch/llama_cpp_switch/`:

| Path | Contents |
|---|---|
| `models/` | Downloaded and copied `.gguf` files |
| `models.json` | Model list and per-model settings |
| `settings.json` | App settings (CPU clock) |
| `bench.txt` | Benchmark results, appended after every run |
| `llama.log` | llama.cpp log; rotated to `llama.log.1` when it gets large |

## Building

Install [devkitPro](https://devkitpro.org/wiki/Getting_Started) with the `switch-dev`, `switch-cmake` and `dkp-cmake-common-utils` packages, then run:

```bash
scripts/build-switch.sh
```

The result is `build-switch/examples/switch/llama-switch.nro`. Extra `-D` options are passed through to CMake. `DEVKITPRO` defaults to `/opt/devkitpro`.

The build pins ggml-cpu to ARMv8.0 (`armv8-a+crc+crypto`). Never raise this: the A57 has no dotprod, i8mm, fp16 arithmetic, SVE or LSE, and code using any of them crashes on the console.

### Releases

`release-assets/llama-switch.nro` is a committed prebuilt binary. To publish a release, rebuild and commit it, then attach it to a new release:

```bash
gh release create vX.Y.Z release-assets/llama-switch.nro --title "llama.cpp for Nintendo Switch X.Y.Z"
```

## Changes from upstream llama.cpp

Most of the tree is unmodified upstream code. The port consists of:

- **The app**, in `examples/switch/`.
- **Platform patches** in the core libraries, all behind `#if defined(__SWITCH__)`:
  - `memalign` instead of `posix_memalign`
  - device memory reported from `svcGetInfo`
  - dynamic backend loading stubbed out (no `dlopen`)
  - thread pinning with `svcSetThreadCoreMask`
  - a few header exclusions in `cpp-httplib`
- **Cortex-A57 kernels** in `ggml/src/ggml-cpu/arch/arm/`, enabled on any AArch64 CPU without dotprod:
  - a 4-row Q4_0/Q8_0 GEMV for single-token decode
  - rewritten no-dotprod `vec_dot`s for Q4_0, Q8_0, Q4_K, Q5_K and Q6_K

List every port change with `grep -rn __SWITCH__ ggml src examples vendor`.

`scripts/sync-ggml.sh` and `scripts/sync_vendor.py` overwrite `ggml/` and `vendor/`, so re-apply the patches after syncing with upstream.

For everything else (the library, other platforms, model conversion), see the [upstream README](https://github.com/ggml-org/llama.cpp#readme) and [`docs/`](docs/).

## Limitations

- CPU only; the Switch GPU isn't used.
- Generation uses 3 of the 4 CPU cores, because the system reserves the fourth.
- The console is 80×45 characters and ASCII only, so other characters in model output are replaced.
- Models are fully loaded into RAM and must fit in about 3 GB including the KV cache.

## License

MIT, like upstream llama.cpp. See [LICENSE](LICENSE). Based on [llama.cpp](https://github.com/ggml-org/llama.cpp) by the ggml authors.
