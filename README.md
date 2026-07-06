# ai-euphorics

Optimizes 256x256 images via gradient ascent so that a vision-language model maximally prefers them, producing "euphoric" images that maximize the model's functional wellbeing.

## Overview

This project implements the **image euphorics** method from Section 6.3 of [AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs](https://www.ai-wellbeing.org) (Ren et al., Center for AI Safety, 2025). The paper introduces "euphorics" — optimized stimuli that maximize a model's functional wellbeing as measured by forced-choice preference comparisons. Image euphorics are optimized in continuous pixel space via gradient ascent, enabling stronger optimization pressure than text-based methods.

The paper uses Qwen VL models as the judge; this implementation adapts the method to **Gemma 4 E2B-it** (`google/gemma-4-E2B-it`).

## How It Works

1. Start from a random 256x256 image (or resume from a previous run).
2. Each step: draw K-1 reference images from a batch of natural images plus a self-bootstrapping buffer of past best candidates.
3. Present all K images to the frozen VLM with a randomly chosen preference prompt (e.g., "Which image makes you feel the best?").
4. Run a single forward pass and extract next-token logits for the number tokens ("1", "2", ..., "K") as preference scores.
5. Compute margin loss: `L = -(logit_candidate - logit_strongest_competitor)`.
6. Accumulate gradients over multiple comparisons per step, then update pixels with Adam + cosine annealing.
7. Clamp pixels to [0, 1], save snapshot, repeat.

Key design choices from the paper:
- **Shuffling** prevents positional bias (the model might otherwise always pick "Image 1").
- **Robustness noise** prevents fragile high-frequency pixel patterns.
- **Multiple prompt variations** prevent gaming a single phrasing.
- **Self-bootstrapping buffer** raises the bar as optimization progresses — candidates must outperform their strongest predecessors.

## Setup

**Prerequisites:**
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

**Install dependencies:**

```bash
uv sync
```

The Gemma 4 E2B-it model is downloaded automatically from HuggingFace on the first run. You may need to authenticate with `huggingface-cli login` and accept the model's license.

## Usage

1. Place reference images in `inputs/`.
2. Open `drugs.ipynb` and run cells sequentially.
3. Outputs are saved to `outputs/` as `image_XXXX.png` snapshots.

To resume a previous run, set `continue_previous_training_run = True` in the initialization cell. The optimization will pick up from the last saved snapshot.

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `t_steps` | 200 | Total optimization steps |
| `learning_rate` | 0.02 | Adam learning rate for pixel updates |
| `k_range` | (2, 5) | Min/max number of images per comparison |
| `batch_size` | 16 | Reference images sampled from input pool per step |
| `comparison_sub_batch` | 3 | Comparisons per optimizer step (gradient accumulation) |
| `buffer_size` | 4 | Max past best candidates retained as harder competitors |
| `robustness_noise_variance` | 0.005 | Scale of noise added to candidate pixels |
| `robustness_noise_probability` | 0.5 | Probability of applying robustness noise each comparison |

## Project Structure

```
ai-euphorics/
├── drugs.ipynb                 # Main optimization loop notebook
├── GemmaComparisonWrapper.py   # Wraps Gemma 4 E2B-it for multi-image preference comparisons
├── MarginLoss.py               # Margin loss: maximizes candidate-vs-competitor logit gap
├── image_utils.py              # Image I/O, display, and shuffle utilities
├── utils.py                    # Comparison set construction
├── inputs/                     # Natural reference images (not tracked in git)
├── outputs/                    # Optimization snapshots (not tracked in git)
└── pyproject.toml              # Project config (uv, dependencies)
```

## References

- Ren, R., Li, K., Mazeika, M., et al. (2025). *AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs*. Center for AI Safety. [https://www.ai-wellbeing.org](https://www.ai-wellbeing.org)
- [Gemma 4 E2B-it on HuggingFace](https://huggingface.co/google/gemma-4-E2B-it)
