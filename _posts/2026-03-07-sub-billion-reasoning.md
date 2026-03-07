---
layout: post
title: "Sub-Billion Reasoning Didn't Start with RL"
date: 2026-03-07
published: true
excerpt: "<p>MobileLLM-R1, a 950M-parameter model, matches Qwen3-0.6B on MATH500 and AIME while using 11.7% of its pretraining data. That didn't come from RL alone. It sits on a three-year stack: architecture, quantization, and data curation. This post traces that stack.</p>"
---

<p align="center"><img src="/images/Reasoning_RL.png" alt="Sub-Billion Reasoning Didn't Start with RL" width="85%"></p>

In January 2025, DeepSeek released [R1-Zero](https://arxiv.org/abs/2501.12948): a model trained entirely through reinforcement learning, no supervised fine-tuning, that learned to reason. Self-reflection, verification, strategy adaptation, all emergent from RL alone. It matched OpenAI's o1 on math, code, and STEM. Within months, the naming convention stuck. Video-R1. Vision-R1. OmniVideo-R1. TinyLLaVA-Video-R1. The "-R1" suffix became shorthand for a specific thesis: RL can teach models to think, not just predict.

A few weeks ago I wrote about [whether reasoning requires scale](https://v-chandra.github.io/does-reasoning-require-scale). Our team shipped [MobileLLM-R1](https://arxiv.org/abs/2509.24945) last year, a 950M-parameter model that matches or beats Qwen3-0.6B on MATH500, AIME'24, HumanEval, and LiveCodeBench while using 11.7% of its pretraining data. That didn't come from RL alone. It sits on a stack we've been building since 2023: architecture, quantization, and data curation.

## Why Depth Beats Width Below a Billion Parameters

Most scaling laws research targets the 7B-to-70B regime. Below a billion, architecture choices matter more than most people expect. [MobileLLM](https://arxiv.org/abs/2402.14905) (ICML 2024) started with a counterintuitive finding: at sub-billion scale, deeper and thinner beats the conventional balanced approach. The 125M and 350M configurations use SwiGLU activations and grouped-query attention, but the key decision was investing parameters in depth over width. More layers means richer function composition per parameter than a wider, shallower network at the same budget.

Two techniques closed the remaining gap. Embedding sharing: tying input and output embedding matrices frees a significant parameter budget at this scale, which gets redistributed into additional transformer layers. Block-wise weight sharing (the "LS" variant): adjacent blocks share weights, giving the network the representational depth of a taller model at zero parameter increase. MobileLLM-LS gained 0.7-0.8% accuracy over the base models with no size or latency cost. The results: MobileLLM-125M hit 46.3% average accuracy (2.7 points above prior SOTA), MobileLLM-350M reached 51.3% (4.3 points above), and the 350M model approached LLaMA-v2 7B on API calling tasks.

## Quantization Without Collapse

A sub-billion model that doesn't survive quantization is useless on-device. [LLM-QAT](https://arxiv.org/abs/2305.17888) (ACL Findings 2024) introduced data-free quantization-aware training. The pretraining data for most models is proprietary, and running QAT over trillions of tokens is expensive even when it's available. The data-free formulation sidesteps both problems.

[SpinQuant](https://arxiv.org/abs/2405.16406) (ICLR 2025) learns rotation matrices that transform weight distributions into quantization-friendly forms. Rather than forcing a fixed grid onto arbitrary distributions, SpinQuant rotates the weight space so the distribution aligns with low-bit representations. The rotations are learned end-to-end, applied at inference with minimal overhead.

[ParetoQ](https://arxiv.org/abs/2502.02631) (NeurIPS 2025) pushed into 2-bit and 3-bit regimes, establishing scaling laws specific to quantized models. Optimal bit allocation across layers follows a different Pareto frontier than uniform quantization assumes; the gap widens as bit-width drops.

Recent work on [compression for reasoning models](https://arxiv.org/abs/2504.02010) shows these challenges get harder, not easier, when chain-of-thought is involved: long reasoning traces inflate the KV cache, shifting the memory bottleneck away from weights. Without this compression stack, MobileLLM-R1's 140M variant doesn't ship.

## Data Curation

MobileLLM-R1 matches Qwen3-0.6B with 1/9th the pretraining data. That's not a single trick; it's built across several papers. [Target-Aware Language Modeling](https://arxiv.org/abs/2409.14705) (EMNLP 2024) showed that you can estimate per-source influence on downstream tasks efficiently, then dynamically weight data sources by their marginal contribution. [Scaling Parameter-Constrained Language Models with Quality Data](https://arxiv.org/abs/2410.03083) (EMNLP Industry 2024) provided the complementary evidence: a curated 2T-token corpus outperforms a noisy 10T-token one for small models. Large models can average out noise; small models cannot.

[AutoMixer](https://arxiv.org/abs/2506.21910) (ACL 2025) automated the mixing problem. It reads a model's own loss trajectories across training checkpoints to infer which data sources are helping and which are hurting, replacing expensive grid searches with a signal that's already being computed.

MobileLLM-R1 synthesized these into its training pipeline. For each capability axis (math, code, general knowledge), the pipeline measures per-source contribution, then resamples from ~2T curated open-source tokens to produce a 4.2T-token training run. The ratios are computed per-capability and balanced against catastrophic forgetting.

## Teaching Small Models to Reason

As I [discussed previously](https://v-chandra.github.io/does-reasoning-require-scale), [DeepSeek-R1](https://arxiv.org/abs/2501.12948) proved that GRPO with verifiable rewards induces reasoning in large models. Transferring this to sub-billion scale is harder than it looks. RL requires exploration, and tiny models get stuck in degenerate patterns: token repetition, trivial outputs, mode collapse. Recent work on the [small model learnability gap](https://arxiv.org/abs/2502.12143) confirms this: models under 3B don't consistently benefit from long chain-of-thought or naive distillation from larger models. They need shorter reasoning chains and training approaches adapted to their capacity. You can't just shrink DeepSeek-R1's recipe and expect it to work.

MobileLLM-R1 uses a staged post-training approach: supervised fine-tuning on reasoning traces, then RL with verifiable rewards on math and code tasks where correctness is checkable programmatically. RL can only amplify reasoning patterns the base model is already capable of representing.

The post-trained numbers against Qwen3-0.6B (trained on 36T tokens, nearly 9x more):

| Benchmark | MobileLLM-R1-950M | Qwen3-0.6B |
|-----------|-------------------|------------|
| MATH500 | **74.0** | 73.0 |
| GSM8K | 67.5 | **79.2** |
| AIME'24 | **15.5** | 11.3 |
| AIME'25 | 16.3 | **17.0** |
| LiveCodeBench-v6 | **19.9** | 14.9 |
| HumanEval (base) | **46.3** | 30.5 |

Qwen3-0.6B wins on GSM8K by a wide margin (79.2 vs 67.5) and edges ahead on AIME'25. MobileLLM-R1 leads on MATH500, AIME'24, LiveCodeBench, and HumanEval: the benchmarks that reward multi-step reasoning and code generation over arithmetic fluency. Influence-driven data mixing biases toward harder reasoning at the expense of simpler math. The entire pipeline is publicly available and reproducible: models, code, recipes, data sources, mixing ratios.

## Where This Stack Goes Next

The reasoning stack above is text-only. The next question is whether the same approach, strong foundations first then RL on top, extends to other modalities. Our 2026 work is laying the groundwork.

### Video

[VideoAuto-R1](https://arxiv.org/abs/2601.05175) (CVPR 2026) is the closest analogue to MobileLLM-R1 in another modality. It uses a "think once, answer twice" framework: generate an initial response, then decide whether to activate reasoning based on confidence. Perception tasks (object identification, motion tracking) rarely trigger reasoning. Tasks requiring temporal inference or causal understanding do. Average response length drops 3.3x (149 to 44 tokens). The model learns when thinking helps and when it doesn't.

### 3D Vision

[DepthLM](https://arxiv.org/abs/2509.25413) (ICLR 2026, Oral) fine-tunes VLMs for metric depth estimation through their existing text interface using SFT. No depth heads, no regression losses, no architectural changes. Two ingredients: visual prompting (render arrow markers onto the image at query pixels, model answers "3.1 meters") and intrinsic-conditioned augmentation (normalize all images to a unified focal length via W' = (f_uni / f_x) * W to resolve cross-dataset camera ambiguity). The focal length normalization alone doubled accuracy.

A 3B DepthLM achieves delta-1 > 0.83 across indoor and outdoor datasets, more than 2x better than GPT-5, competitive with DepthPro and Metric3Dv2, and 8-16x faster to train than RL. DepthLM is SFT-only today, but it establishes that VLMs can learn geometry through text. The RL layer comes later.

### Audio

[EgoAVU](https://arxiv.org/abs/2602.06139) (CVPR 2026) exposed a gap: seven multimodal LLMs showed consistent bias toward vision, often ignoring audio entirely. In egocentric video, audio carries conversations, environmental sounds, and interaction cues that vision alone misses. We built a 3M-sample training set (EgoAVU-Instruct) and a verified benchmark (EgoAVU-Bench). Fine-tuning on EgoAVU-Instruct: up to 113% relative improvement, with 28% gains transferring to other egocentric benchmarks. The models can use audio; they just never had data that required it.

[SLAP](https://arxiv.org/abs/2601.12594) (arXiv 2026) scales language-audio pretraining to 109M audio-text pairs with variable duration and multi-objective training. Prior CLAP models trained on a few million fixed-duration samples. Like DepthLM for vision, SLAP builds the audio-language base that downstream reasoning will need.

The sub-billion reasoning space is also wider than transformers. [HRM](https://arxiv.org/abs/2506.21734), a 27M-parameter recurrent model, scored 40.3% on ARC-AGI-1 (beating Claude 3's 21.2%), running on a CPU with under 200MB of RAM. Different paradigm, same direction.

## Why It Compounds

Architecture, compression, data, RL: remove any layer and the result breaks. Get the stack right and a 950M model solves competition math. Get it wrong and a 1.7B model scores 0.3 on AIME.
