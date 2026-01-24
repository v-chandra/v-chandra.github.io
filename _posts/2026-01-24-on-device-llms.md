---
layout: post
title: "On-Device LLMs: State of the Union, 2026"
date: 2026-01-24
published: true
excerpt: "<p>Three years ago, running a language model on a phone meant a toy demo. Today, billion-parameter models run in real time on flagship devices. This shift came not from faster chips alone, but from rethinking how we build, compress, and deploy models.</p>"
---

**Vikas Chandra** and **Raghuraman Krishnamoorthi**

Three years ago, running a language model on a phone meant a toy demo. Today, billion-parameter models run in real time on flagship devices. This shift came not from faster chips alone, but from rethinking how we build, compress, and deploy models.

This post covers the latest: what's changed, what works, and where things are headed. We'll focus on techniques that have proven useful in practice, not just what looks good in papers.

<p align="center"><img src="/images/on_device_llms.png" alt="On-Device LLMs" width="80%"></p>

## **Why On-Device LLMs?**

The case for on-device LLMs comes down to four things:

**Latency.** Cloud round-trips add 200-500ms before you see the first token. For AR overlays, real-time translation, or voice assistants, that delay breaks the experience. On-device inference can generate tokens in under 20ms each, particularly for short context lengths.

**Privacy.** Data that never leaves the device can't be breached in transit or logged on a server. For health data, financial information, or anything personal, this matters. It's also becoming a regulatory requirement in some domains.

**Cost.** Cloud inference at scale is expensive. Every query costs money. On-device shifts that cost to hardware the user already owns. For high-volume applications, the economics are compelling.

**Availability.** On-device LLMs are always available. Access to the cloud depends on connectivity, which is often not available with high reliability.

The catch has always been capability. Edge devices have limited memory, limited compute, and limited power budgets. If your use case requires frontier reasoning, broad world knowledge, or long multi-turn conversations, cloud is still the better choice. But for latency-sensitive, privacy-critical, or high-volume applications, on-device is increasingly viable.

## **The Constraints**

Before diving into solutions, it helps to understand what we're fighting against.

### **Memory is the Bottleneck**

The common assumption is that edge devices lack compute. They don't. Mobile NPUs now deliver serious TOPS, getting close to the capability of data-center GPUs in 2017 (for example, V100 is 125 TOPS)\!

- Apple A19 Pro Neural Engine: \~35 TOPS  
- Qualcomm Snapdragon 8 Elite Gen 5: \~60 TOPS  
- MediaTek Dimensity 9400+: \~50 TOPS

But TOPS alone doesn't tell you much. Can the NPU run the ops your model needs? Many have limited support for attention, dynamic shapes, or certain activations. Is the toolchain mature enough to deploy without heroic engineering? Real-world models run far from peak utilization.

Another constraint is the availability of RAM. Available RAM is typically limited to \<4GB even on high-end devices due to the need to co-exist with other services and the overhead of the operating system. This limits both maximum model size and the suitability of approaches like MoE (Mixture of Experts).

The deeper constraint is memory bandwidth. Mobile devices have 50-90 GB/s; data center GPUs have 2-3 TB/s. That's a 30-50x gap. For LLM inference, this gap is decisive because decode is memory-bound: you load the entire model weights for each token generated, so the compute units sit idle waiting for memory. This is why model compression and techniques for predicting multiple tokens have such an outsized impact on mobile. Going from 16-bit to 4-bit isn't just 4x less storage; it's 4x less memory traffic per token, which directly translates to throughput. Similarly, predicting multiple tokens at each step is “free”, there is no latency penalty.

### **Power Budget**

Mobile devices run on batteries, and sustained inference drains them fast. A model that drains your battery or triggers thermal throttling isn't practical, regardless of how fast it runs.

This creates pressure toward:

- Smaller models (fewer operations)  
- Quantized models (simpler arithmetic)  
- Sparse models (skip unnecessary computation)  
- Efficient scheduling (burst when needed, sleep otherwise)  
- Parallel generation of output (burst output faster and go to sleep)

The always-on use case (continuous listening, ambient sensing) is especially constrained. You need single-digit milliwatts, not hundreds.

## **Efficient Language Models**

With constraints understood, how do you build models that work within them?

### **How Small Can You Go?**

The first question everyone asks: how small can a language model be and still be useful?

The answer has shifted dramatically. In 2022, conventional wisdom said you needed at least 7B parameters for coherent text generation. Today, sub-billion parameter models handle many practical tasks.

MobileLLM found something counterintuitive: at small scale, architecture matters more than parameter count. The standard scaling recipe (wider layers as you grow) doesn't apply below 1B parameters. Deep-thin architectures (more layers, smaller hidden dimensions) consistently outperform wide-shallow ones. A 125M parameter model with the right architecture runs at 50 tokens/second on an iPhone and handles basic tasks surprisingly well.

The major labs have since converged on this insight:

<table>
<thead>
<tr><th>Model</th><th>Sizes</th><th>Key Strength</th></tr>
</thead>
<tbody>
<tr><td><strong>Llama 3.2</strong> (Meta, 2024)</td><td>1B, 3B</td><td>128K context, Qualcomm/MediaTek optimized</td></tr>
<tr><td><strong>Gemma 3</strong> (Google, 2025)</td><td>270M - 27B</td><td>Extreme efficiency at small sizes</td></tr>
<tr><td><strong>Phi-4</strong> (Microsoft, 2025)</td><td>3.8B (mini), 14B</td><td>Phi-4-reasoning rivals o1-mini on math</td></tr>
<tr><td><strong>SmolLM2</strong> (HuggingFace, 2025)</td><td>135M, 360M, 1.7B</td><td>11T training tokens, outperforms Llama 3.2 1B</td></tr>
<tr><td><strong>Qwen2.5</strong> (Alibaba, 2024)</td><td>0.5B, 1.5B</td><td>Very strong general small-model performance; good multilingual coverage</td></tr>
</tbody>
</table>

The pattern across all of these: data quality and training methodology matter as much as architecture. Phi-4 uses high-quality synthetic datasets. SmolLM2 introduces specialized math and code datasets (FineMath, Stack-Edu). Gemma 3 uses knowledge distillation from larger models. You're not just fighting for parameter efficiency; you're fighting for every capability point at a fixed size.

Don't assume you need a big model. For many applications (summarization, simple Q\&A, text formatting, basic code assistance), sub-1B models work. Start small and scale up only if you need to.

### **Reasoning at the Edge**

Some on-device use cases need more than pattern matching: analyzing personal documents, reasoning about health data, triaging messages. Can small models actually work through these multi-step problems? Early evidence says yes, with caveats.

**Distillation from reasoning models** works well. DeepSeek-R1 distillation produced models from 1.5B to 70B parameters that retain strong reasoning capabilities, with distilled 8B models surpassing much larger base models on math benchmarks. The approach: generate chain-of-thought data from a strong reasoning model, then fine-tune smaller models on that data.

**Qwen3's small models** show similar results. Qwen3-4B rivals the performance of Qwen2.5-72B-Instruct on reasoning tasks. The Qwen3-30B-A3B MoE model (activating only 3B parameters) outperforms QwQ-32B despite 10x fewer active parameters.

**MobileLLM-R1** and **MobileLLM-R1.5** demonstrated this at the extreme edge: 2-5x better performance on reasoning benchmarks compared to models twice the size, running entirely on mobile CPU.

**Liquid AI’s LFM 2.5** models have also shown very strong performance, driven by a larger training set and RL-based post-training.

What these results show: reasoning isn't purely a function of parameter count. It's about training methodology. Distillation from strong reasoning models coupled with RL-based post-training is crucial.

But there are limits. Small models still struggle with long chains of reasoning, novel problem types, and tasks requiring broad world knowledge. For on-device, this means being thoughtful about which tasks you route to local models versus the cloud.

### **Data for Small Models**

At a small scale, data strategy matters as much as architecture. You can't brute-force your way to capability with more parameters, so every training token needs to count.

Small models benefit disproportionately from high-quality, targeted data. Our work on scaling parameter-constrained language models showed that data quality improvements yield larger gains for smaller models than for larger ones. A 1B model trained on curated data can match a 3B model trained on web scrapes.

**Data mixing** is where much of the leverage lies. The ratio of code to text to math to instruction data dramatically affects downstream capability. AutoMixer (Meta, ACL 2025\) discovered that checkpoint artifacts during training encode information about optimal data mixtures, enabling automatic mixture adjustment without expensive ablations. Hand-tuning data ratios is expensive and doesn't transfer across model sizes.

**Granular sampling** goes further. Target-Aware Language Modeling (Meta, EMNLP 2024\) showed that sampling strategy at the document and passage level affects what the model learns. Not all documents contribute equally; selectively upweighting high-signal content improves efficiency.

If you're training a small model for a specific domain, invest in data curation. The marginal hour spent on data quality often beats the marginal hour spent on architecture search. SmolLM2's specialized datasets (FineMath, Stack-Edu) and Phi-4's synthetic data pipelines reflect this insight.

## **Architectures**

### **Mixture of Experts**

MoE activates only a subset of parameters per token, and since early 2025, over 60% of frontier model releases have adopted MoE designs. DeepSeek-V3, for example, uses 256 experts with fine-grained routing.

The memory challenge: despite sparse computation, you still load all experts into memory. MoBiLE addresses this for consumer GPUs by using "mixture of big-little experts," reducing expert count for unimportant tokens while maintaining full experts for important ones. This achieves 1.6-1.7x speedup with negligible accuracy loss.

For edge, MoE's appeal is clear: you get large-model capability with small-model compute. The challenge is fitting all experts in memory, which is where quantization and offloading techniques become essential. Current techniques help but don't fully solve the problem; more on this in "What's Next."

### **Novel Building Blocks**

LLM architectures are still dominated by attention + FFN layers, but that is changing. In addition to MoE, several variants of attention mechanisms have been proposed in the literature, with several key directions:

1. **Improve performance:** Architectures claiming improvements over attention started emerging in 2025, with Gated Delta-Net from Qwen and Manifold-Constrained Hyper-Connections (mHC) from DeepSeek showing improvements.
2. **Long context support with reduced latency:** Hybrid approaches combining Mamba and attention have gained traction (Qwen3 Next, Nvidia-Nemotron3) as a way to deal with long context efficiently. In parallel, alternative approaches with a focus on latency reduction like LIV convolutions and linear attention have also emerged.

## **Quantization**

If architecture determines your baseline capability, quantization determines whether it actually fits on device.

### **4-Bit is the New Default**

The standard recipe for deployment has converged: train in 16-bit, quantize to 4-bit for deployment. GPTQ (2022) and AWQ (2023) showed that 4-bit post-training quantization preserves most model quality with 4x memory reduction. This is now standard practice. AWQ alone has over 19 million downloads on HuggingFace.

The challenge is edge cases. Naïve quantization blows up on outlier activations, which large language models produce regularly. QAT with range learning (ParetoQ) works well and is suitable for accelerators, which often have additional constraints on quantization. If that is not possible, post training techniques below are promising:

**SmoothQuant** (MIT HAN Lab) smooths outliers by migrating the quantization difficulty from activations to weights. The insight: activations have outliers in specific channels, while weights are relatively uniform. By applying a mathematically equivalent per-channel scaling, you can make activations easier to quantize without changing the model's behavior. This enables 8-bit quantization of both weights and activations with minimal loss.

**SpinQuant** (Meta) learns rotation matrices that reshape activation distributions before quantization. Rotations are orthogonal transformations that don't change model outputs but can dramatically reduce outliers. The result: 4-bit quantization of weights, activations, and KV-cache together, with under 3% accuracy loss on tasks where previous methods degraded 25%+.

For serving at scale, **QServe** (MIT HAN Lab) takes this further with W4A8KV4 quantization: 4-bit weights, 8-bit activations, and 4-bit KV cache. This requires careful co-design of the quantization scheme and the serving system, but delivers substantial throughput improvements.

For practitioners: start with AWQ or GPTQ for a quick baseline. If you're seeing quality degradation, look into outlier-aware methods. SmoothQuant is training-free and works well for 8-bit. SpinQuant handles the harder 4-bit case for activations and KV-cache.

Recently, hardware support for mxfp4 is starting to show up in edge hardware (Apple A19 Pro), reducing quantization loss thanks to the superior format.

### **Going Lower: 2-Bit and Beyond**

4-bit is practical, but can you go lower? Yes, though the rules change.

BitNet (Microsoft) showed that models trained natively at 1.58 bits can work. A 2B parameter model fits in 400MB and runs efficiently on CPU. But you can't just quantize an existing model to 1.58 bits and expect it to work. You have to train from scratch at that precision.

ParetoQ (our work) mapped the full quantization Pareto frontier and found something surprising: the relationship between bits and accuracy isn't smooth. At 3-4 bits, quantization acts like compression. At 2 bits and below, the model learns fundamentally different representations. If you have a fixed budget for the model size, it is better to have a larger model quantized down to 2-bits, rather than a model with half the number of parameters quantized to 4-bits. This matters for the future. If low-bit training works at scale, we're not just compressing models; we're finding new efficiency frontiers that high-precision training can't reach.

You can also mix and match: with mixed precision quantization, different layers can be at different precisions, preserving quality while compressing even further.

### **When to Use What**

<table>
<thead>
<tr><th>Bits</th><th>Memory</th><th>Quality</th><th>Use Case</th></tr>
</thead>
<tbody>
<tr><td>8-bit</td><td>2x smaller</td><td>~same</td><td>Server, no constraints</td></tr>
<tr><td>4-bit</td><td>4x smaller</td><td>1-3% drop</td><td>Server/Mobile/edge, QAT</td></tr>
<tr><td>Sub 4-bit</td><td>4x-8x smaller</td><td>3% drop</td><td>Mobile/edge, best tradeoff, QAT</td></tr>
<tr><td>Vector Quantization</td><td>8x smaller</td><td>~3% drop</td><td>Hardware accelerators, Apple Neural Engine</td></tr>
</tbody>
</table>

## **Inference Optimization**

Beyond compression, how you run inference matters as much as what you're running.

### **Attention Efficiency**

Attention is the bottleneck for long sequences. FlashAttention (Tri Dao et al.) made attention IO-aware, reducing memory reads/writes between GPU HBM and SRAM through tiling. FlashAttention-2 improved parallelism and achieved up to 72% model FLOPs utilization on A100s. FlashAttention-3 targets H100s with up to 75% utilization (740 TFLOPs/s). FlashAttention-4, presented at Hot Chips 2025, optimizes for Blackwell with another 20% speedup.

For on-device, the principles matter more than the specific implementations: minimize memory traffic, tile computations to fit in fast memory, parallelize across what you have. At the architecture level, local-global attention and grouped query attention are now standard for on-device models, with newer architectures often skipping attention for certain layers in the model, drastically reducing the KV cache size and complexity.

### **KV Cache Management**

The KV cache grows linearly with sequence length and can dominate memory usage during long-context inference, often exceeding the model weights themselves. For edge deployment, KV cache compression is often more impactful than weight quantization for long-context applications, with research showing that KV cache can be quantized down to 3 bits, with negligible drop in quality.

MIT HAN Lab's work showed that you don't need to cache everything; you need to cache the right things. **StreamingLLM** discovered that preserving "attention sinks" (initial tokens) enables infinite-length generation with fixed memory. **DuoAttention** found that different attention heads serve different purposes (retrieval vs. streaming) and can be treated differently to reduce both memory and latency.

Compression strategies have evolved beyond simple eviction. **ChunkKV** treats semantic chunks rather than individual tokens as compression units, preserving linguistic structure while improving throughput by 26% over token-level methods. **EvolKV** uses evolutionary search to find optimal per-layer cache budgets, achieving better performance than full KV cache on some tasks while using only 1.5% of the original budget.

### **Speculative Decoding**

Autoregressive decoding is inherently sequential, generating one token at a time. Speculative decoding breaks this bottleneck by using a small draft model to propose multiple tokens, then verifying them in parallel with the target model.

Two approaches dominate. **Medusa** (Princeton, 2024\) adds extra decoding heads to predict multiple future tokens simultaneously, achieving 2.2-3.6x speedup over vanilla decoding. The original model stays untouched; only the new heads are fine-tuned. **EAGLE** (SafeAI Lab) extrapolates hidden state features to predict draft tokens without any fine-tuning of the target model, achieving similar speedups with better acceptance rates. EAGLE-3 fuses low-, mid-, and high-level semantic features for better draft quality.

Both are now integrated into major serving frameworks (vLLM, TensorRT-LLM). Intel and Weizmann Institute (ICML 2025\) showed that any small draft model can accelerate any LLM regardless of vocabulary differences, delivering up to 2.8x faster inference. Online Speculative Decoding (UC Berkeley, 2025\) adapts draft models continuously during serving.

For on-device, speculative decoding is particularly attractive because you often have a smaller model available anyway. The draft model can be a quantized or pruned version of the target, or a separate tiny model trained for speculation.

### **Diffusion LLMs**

A different approach to breaking the sequential bottleneck: diffusion LLMs (LLaDA, SBD and TiDAR) predict multiple tokens per step by treating text generation as a denoising process. Rather than generating left-to-right, these models iteratively refine all tokens in parallel. Combined with speculative decoding, diffusion approaches promise speedups of 4-6x over autoregressive decoding. The technique is still maturing, but early results suggest it could be particularly valuable for on-device scenarios where latency matters more than raw throughput.

### **Pruning**

Pruning removes weights to reduce model size and computation. Two flavors:

**Unstructured pruning** (SparseGPT, Wanda) removes individual weights, achieving high sparsity ratios but requiring sparse matrix support for actual speedups. SparseGPT can prune to 50% sparsity in one shot without retraining.

**Structured pruning** (LLM-Pruner, SlimLLM) removes entire channels, heads, or layers. The resulting models run fast on standard hardware but typically need more careful handling to preserve quality. SlimLLM evaluates importance at the channel/head level rather than aggregating individual elements. For edge, structured pruning is usually more practical since most mobile hardware doesn't efficiently support sparse operations.

**Co-design** approaches (Nemotron-Flash, Liquid AI) trade off latency against model accuracy to determine model hyperparameters. The search space can extend to include pruning, quantization and even building blocks.

## **Inference Frameworks**

With optimization techniques covered, the question becomes: what software actually runs these models? The stack has matured considerably. You're no longer building everything from scratch.

**ExecuTorch** (Meta) hit 1.0 GA in October 2025, marking production readiness. The runtime has a 50KB base footprint and runs on everything from microcontrollers to high-end smartphones. It supports 12+ hardware backends (Apple, Qualcomm, Arm, MediaTek, Vulkan) and over 80% of the most popular edge LLMs on HuggingFace work out of the box. Meta now uses ExecuTorch across Instagram, WhatsApp, Messenger, and Facebook, serving billions of users. If you're in the PyTorch ecosystem, this is the natural choice.

**llama.cpp** remains the go-to for CPU inference. It's simple, portable, and continuously optimized. For running LLMs on laptops, desktops, or servers without GPUs, it's hard to beat. The community has added support for many model architectures beyond LLaMA, and the GGUF format has become a de facto standard for quantized model distribution.

**MLX** (Apple) is optimized for Apple Silicon. If you're targeting Macs or have a Mac-based development workflow, it offers good performance with a familiar NumPy-like API. Unified memory makes CPU/GPU coordination efficient.

**MLC-LLM** compiles models for deployment across diverse hardware. It's useful when you need to target multiple platforms from a single source.

Our recommendation: pick based on your deployment target and existing stack. Don't over-engineer the choice; they all work. ExecuTorch for mobile production, llama.cpp for desktop/prototyping, MLX for Apple ecosystem.

If you're just starting out, grab a quantized model from HuggingFace (Llama 3.2 or Gemma 3 in GGUF format), run it with llama.cpp to validate your use case works, then move to ExecuTorch when you're ready for production mobile deployment. Profile on real hardware early; emulators and simulators are not accurate on performance.

## **Beyond Text**

The techniques above apply beyond language models. Vision and multimodal models face the same constraints and benefit from the same solutions.

**Vision-language models** have shrunk dramatically. SmolVLM-256M uses under 1GB memory and outperforms models 300x its size by optimizing which visual tokens matter. MiniCPM-V achieves frontier-level performance while running on phones. FastVLM (Apple) optimizes the visual encoder specifically for on-device latency. The winning approach: co-optimize vision encoder, language backbone, and fusion mechanism together.

**Image generation models** can now run on-device. SnapFusion and MobileDiffusion now enable image creation on high-end phones in under a second. Coupled with efficient vision language models, image editing is now possible on-device, though still expensive.

The techniques from earlier sections (quantization, pruning, efficient attention, KV cache optimization) transfer directly. A quantized VLM benefits from the same outlier handling as a quantized LLM. Speculative decoding works for any autoregressive model.

**Native multi-modal models.** Multi-modal model architectures are migrating to a native approach, where all modalities are converted to tokens using a lightweight tokenizer/patchifying layer with a common LM backbone. This approach is already popular for frontier models (Qwen3 Omni, Gemini3). For on-device, native multimodal architectures simplify deployment by requiring a single model rather than separate encoders, and the shared backbone means compression techniques apply uniformly across modalities.

## **Training for On-Device**

On-device inference gets the attention, but training efficiency determines who can build these models in the first place.

### **Democratizing Pre-Training**

Full pre-training was thought to require massive GPU clusters. That assumption is breaking down.

APOLLO (our work) showed that Adam's per-parameter adaptation is overkill. Learning rate scaling at the channel or tensor level captures most of the benefit. By projecting into a low-rank auxiliary space, APOLLO achieves AdamW-level performance with SGD-level memory. GaLore (UC Berkeley) independently discovered a similar approach. The practical impact: training LLaMA-7B from scratch on a single 12GB GPU, a task that required eight A100s two years ago. For on-device, this expands who can create efficient models. More teams training means more exploration of the architecture space for edge deployment.

### **Fine-Tuning**

If you're adapting an existing model rather than training from scratch, LoRA and its variants are standard practice. Train only low-rank adapters, freeze the base model, and you can fine-tune on consumer hardware.

The variants have multiplied. **QLoRA** keeps the base model in 4-bit while training LoRA adapters in higher precision, enabling fine-tuning of 7B+ models on a single GPU. **DoRA** (2024) decomposes weights into magnitude and direction, fine-tuning both while using LoRA for the directional component. DoRA consistently outperforms LoRA across rank settings, with larger gains at lower ranks. **RoRA** (January 2025\) optimizes the scaling factor, replacing α/r with α/√r for better performance as rank increases.

For most practitioners, fine-tuning is the path. The base models are good enough; adaptation to your domain or task is where you add value.

## **What's Next**

### **MoE on Edge**

Mixture of Experts offers large-model capability with small-model compute, but edge deployment remains challenging. The problem: even with sparse activation, you still need to store all experts. For models like Mixtral-8x7B, expert loading dominates inference time on consumer hardware. The compute is fast; the memory shuffling isn't.

EdgeMoE partitions experts to external storage and fetches them only when activated, reducing memory 5-18% while improving inference 1.2-2.7x. Collaborative compression has shrunk DeepSeek-V3 from 1.3TB to 103GB through expert pruning and mixed-precision quantization. But these are early solutions. The architecture that makes MoE truly practical on mobile (sub-10W, sub-8GB) doesn't exist yet.

### **Test-Time Compute for Small Models**

A counterintuitive finding: small models can outperform large models by spending more compute at inference time. HuggingFace demonstrated that Llama 3.2 1B with Diverse Verifier Tree Search outperforms the 8B model. Llama 3.2 3B outperforms 70B. The key is compute-optimal inference strategies: tree search, self-verification, and adaptive sampling.

For on-device, you're constrained on model size but not necessarily on inference budget for high-value queries. A 1B model that thinks longer might beat a 7B model that answers immediately. The field is actively developing, but the implication is significant: the capability ceiling for small models may be higher than their parameter count suggests.

### **On-Device Personalization**

Fine-tuning on-device would enable personalization without sending data to the cloud for training or providing extensive context via a prompt to the model. The appeal is obvious: your device learns your preferences, writing style, and domain vocabulary without that data ever leaving. An interesting direction here is test-time training, which allows the model to move user context into weights, by optimizing on data at test time on a self-supervised task.

### **Novel Architectures**

In addition to MoEs and improvements to attention mechanisms, novel architectures are emerging, providing improved performance without needing additional parameters. Recent innovations like ManifoldHC, HyperConnections, and Conditional Memory via Scalable Lookup show promise for improving model quality at fixed parameter budgets.

---

## **References by Topic**

### **Efficient Language Models**

- **MobileLLM**: [Liu et al., 2024](https://arxiv.org/abs/2402.14905) \- Deep-thin architectures for sub-billion parameter models  
- **Llama 3.2**: [Meta, 2024](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) \- 1B/3B models for on-device deployment  
- **Gemma 3**: [Google, 2025](https://blog.google/technology/developers/gemma-3/) \- 270M to 27B with extreme efficiency  
- **Phi-4**: [Microsoft, 2025](https://huggingface.co/microsoft/phi-4) \- 14B flagship, Phi-4-mini 3.8B, Phi-4-reasoning  
- **SmolLM2**: [HuggingFace, 2025](https://arxiv.org/abs/2502.02737) \- 135M-1.7B trained on 11T tokens  
- **MobileLLM-R1**: [Meta, 2025](https://huggingface.co/collections/facebook/mobilellm-r1-68c4597b104fac45f28f448e) \- Reasoning distillation for mobile  
- **DeepSeek-R1 Distillation**: [DeepSeek, 2025](https://huggingface.co/deepseek-ai/DeepSeek-R1) \- Reasoning distillation to 1.5B-70B models  
- **Qwen3**: [Qwen Team, 2025](https://qwenlm.github.io/blog/qwen3/) \- Small models with strong reasoning (4B rivals 72B)
- **Liquid Foundation Models**: [Liquid AI, 2025](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai) \- Hybrid architectures for edge deployment and reasoning

### **Data for Small Models**

- **AutoMixer**: [Chang et al., 2025](https://aclanthology.org/2025.acl-long.979.pdf) \- Automatic data mixing via checkpoint artifacts
- **Scaling with Quality Data**: [Chang et al., 2024](https://aclanthology.org/2024.emnlp-industry.8.pdf) \- Data quality for parameter-constrained models
- **Target-Aware Language Modeling**: [Chang et al., 2024](https://aclanthology.org/2024.emnlp-main.719.pdf) \- Granular data sampling

### **Architectures**

**Mixture of Experts:**

- **MoE Survey**: [Liu et al., 2025](https://arxiv.org/abs/2412.14219) \- Inference optimization for MoE
- **MoBiLE**: [Zhao et al., 2025](https://arxiv.org/abs/2510.12357) \- Consumer GPU MoE inference
- **MoE Comprehensive Survey**: [Mu et al., 2025](https://arxiv.org/abs/2503.07137) \- Comprehensive MoE overview

**Novel Building Blocks:**

- **Mamba**: [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) \- State space models with linear scaling in sequence length
- **Gated Delta-Net**: [Yang et al., 2025](https://arxiv.org/abs/2412.06464) \- Gated linear attention with delta rule

### **Quantization**

- **GPTQ**: [Frantar et al., 2022](https://arxiv.org/abs/2210.17323) \- Post-training quantization via approximate second-order  
- **AWQ**: [Lin et al., 2023](https://arxiv.org/abs/2306.00978) \- Activation-aware weight quantization
- **SmoothQuant**: [Xiao et al., 2023](https://arxiv.org/abs/2211.10438) \- Smooth activations for 8-bit quantization
- **SpinQuant**: [Liu et al., 2024](https://arxiv.org/abs/2405.16406) \- Rotation-based outlier handling for 4-bit
- **QServe**: [Lin et al., 2024](https://arxiv.org/abs/2405.04532) \- W4A8KV4 serving system
- **ParetoQ**: [Wang et al., 2024](https://arxiv.org/abs/2502.02631) \- Full quantization Pareto frontier  
- **BitNet**: [Microsoft, 2024](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) \- 1.58-bit models trained from scratch  
- **SlimLLM**: [Huang et al., 2026](https://openreview.net/pdf?id=PO9bBEPNWy) \- Mixed precision quantization  
- **KV Quant**: [Hooper et al., 2024](https://arxiv.org/abs/2401.18079) \- KV cache quantization

### **Inference Optimization**

**Attention:**

- **FlashAttention**: [Dao et al., 2022](https://arxiv.org/abs/2205.14135) \- IO-aware exact attention  
- **FlashAttention-2**: [Dao, 2023](https://arxiv.org/abs/2307.08691) \- Improved parallelism  
- **FlashAttention-3**: [Dao et al., 2024](https://tridao.me/publications/flash3/flash3.pdf) \- Hopper optimization

**KV Cache:**

- **StreamingLLM**: [Xiao et al., 2023](https://arxiv.org/abs/2309.17453) \- Attention sinks for infinite context
- **DuoAttention**: [Xiao et al., 2024](https://arxiv.org/abs/2410.10819) \- Retrieval vs streaming heads  
- **ChunkKV**: [Liu et al., 2025](https://arxiv.org/abs/2502.00299) \- Semantic chunk compression
- **EvolKV**: [Yu et al., 2025](https://aclanthology.org/2025.findings-emnlp.88/) \- Evolutionary cache optimization
- **KV Cache Survey**: [Li et al., 2024](https://arxiv.org/abs/2412.19442) \- Comprehensive survey

**Speculative Decoding:**

- **EAGLE**: [Li et al., 2024](https://github.com/SafeAILab/EAGLE) \- Feature extrapolation for draft tokens
- **Medusa**: [Cai et al., 2024](https://arxiv.org/abs/2401.10774) \- Multiple decoding heads, 2.2-3.6x speedup
- **Online Speculative Decoding**: [Liu, 2025](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) \- Adaptive draft models
- **Intel/Weizmann Research**: [Mamou et al., 2025](https://newsroom.intel.com/artificial-intelligence/intel-weizmann-institute-speed-ai-with-speculative-decoding-advance) \- Universal speculative decoding

**Diffusion LLMs:**

- **LLaDA**: [Nie et al., 2025](https://arxiv.org/abs/2502.09992) \- Large language diffusion with masking
- **SBD**: [Gat et al., 2025](https://arxiv.org/abs/2509.04185) \- Score-based diffusion for LLMs
- **TiDAR**: [Liu et al., 2025](https://arxiv.org/abs/2511.08923) \- Time-aware diffusion for autoregressive generation

**Pruning:**

- **SparseGPT**: [Frantar & Alistarh, 2023](https://arxiv.org/abs/2301.00774) \- One-shot unstructured pruning  
- **Wanda**: [Sun et al., 2023](https://arxiv.org/abs/2306.11695) \- Pruning by weights and activations  
- **LLM-Pruner**: [Ma et al., 2023](https://arxiv.org/abs/2305.11627) \- Structured pruning for LLMs  
- **SlimLLM**: [Guo et al., 2025](https://proceedings.mlr.press/v267/guo25a.html) \- Accurate structured pruning  
- **Awesome LLM Pruning**: [GitHub Repository](https://github.com/liyunqianggyn/Awesome-LLMs-Pruning)

### **Beyond Text (Vision, Multimodal)**

- **SmolVLM**: [HuggingFace, 2025](https://arxiv.org/abs/2504.05299) \- Efficient VLM design  
- **MiniCPM-V**: [Yao et al., 2025](https://www.nature.com/articles/s41467-025-61040-5) \- Edge-efficient VLMs
- **FastVLM**: [Apple, 2024](https://machinelearning.apple.com/research/fast-vision-language-models) \- Fast visual encoding
- **Small VLM Survey**: [Ahmed et al., 2025](https://www.sciencedirect.com/science/article/abs/pii/S156625352500867X) \- Comprehensive small VLM overview
- **SnapFusion**: [Li et al., 2023](https://snap-research.github.io/SnapFusion/) \- Text-to-image on mobile
- **MobileDiffusion**: [Zhao et al., 2023](https://arxiv.org/abs/2311.16567) \- Efficient diffusion for mobile  
- **Scaling laws for native multi-modal models**: [Aghajanyan et al., 2023](https://arxiv.org/abs/2301.03728) \- Unified scaling for multimodal training

### **Training Efficiency**

- **APOLLO**: [Zhu et al., 2024](https://arxiv.org/abs/2412.05270) \- Memory-efficient pre-training
- **GaLore**: [Zhao et al., 2024](https://arxiv.org/abs/2403.03507) \- Gradient low-rank projection  
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685) \- Low-rank adaptation  
- **QLoRA**: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) \- Quantized LoRA  
- **DoRA**: [Liu et al., 2024](https://arxiv.org/abs/2402.09353) \- Weight-decomposed low-rank adaptation  
- **RoRA**: [Liu et al., 2025](https://arxiv.org/abs/2501.04315) \- Rank-optimized low-rank adaptation

### **Inference Frameworks**

- **ExecuTorch 1.0**: [Meta, Oct 2025](https://pytorch.org/blog/introducing-executorch-1-0/) \- PyTorch edge deployment, 50KB footprint  
- **llama.cpp**: [ggerganov](https://github.com/ggerganov/llama.cpp) \- CPU inference, GGUF format  
- **MLX**: [Apple](https://github.com/ml-explore/mlx) \- Apple Silicon optimization  
- **MLC-LLM**: [MLC](https://github.com/mlc-ai/mlc-llm) \- Cross-platform compilation

### **Future Directions**

**MoE on Edge:**

- **OLMoE**: [Muennighoff et al., 2024](https://arxiv.org/abs/2409.02060) \- Sparse Mixture-of-Experts (MoE)  
- **EdgeMoE**: [Yi et al., 2024](https://www.researchgate.net/publication/389411273_EdgeMoE_Empowering_Sparse_Large_Language_Models_on_Mobile_Devices) \- Expert partitioning for mobile  
- **Collaborative MoE Compression**: [Chen et al., 2025](https://arxiv.org/abs/2509.25689) \- DeepSeek-V3 compression for edge  
- **MoE for Mobile Edge**: [Li et al., 2024](https://arxiv.org/abs/2412.15690) \- Theory of MoE in edge computing

**Test-Time Compute:**

- **Scaling Test-Time Compute**: [Snell et al., 2025](https://arxiv.org/abs/2408.03314) \- Optimal test-time scaling
- **Inference Scaling Laws**: [Wu et al., 2025](https://arxiv.org/abs/2408.00724) \- Compute-optimal inference
- **Test-Time Scaling Survey**: [Agarwal et al., 2025](https://arxiv.org/abs/2512.02008) \- Comprehensive TTS overview

**On-Device Personalization:**

- **Test Time Training**: [Tandon et al., 2025](https://arxiv.org/abs/2512.23675) \- On-device adaptation via self-supervised learning

**Novel Architectures:**

- **ManifoldHC**: [Xie et al., 2026](https://arxiv.org/abs/2512.24880) \- Manifold-based hyperconnections
- **HyperConnections**: [Zhu et al., 2025](https://openreview.net/pdf?id=9FqARW7dwB) \- Improved residual connections
- **Conditional Memory via Scalable Lookup**: [Cheng et al., 2026](https://arxiv.org/abs/2601.07372) \- Efficient conditional computation

### **Surveys and Curated Resources**

- **On-Device AI Survey**: [Wang et al., 2025](https://arxiv.org/abs/2503.06027) \- Comprehensive edge intelligence survey
- **Efficient LLM Survey**: [Zhou et al., 2024](https://arxiv.org/abs/2404.14294) \- Comprehensive inference survey  
- **Awesome Efficient LLM**: [GitHub](https://github.com/horseee/Awesome-Efficient-LLM) \- Curated paper list  
- **Awesome LLM Inference**: [GitHub](https://github.com/xlite-dev/Awesome-LLM-Inference) \- Inference papers with code  
- **MIT HAN Lab**: [hanlab.mit.edu](https://hanlab.mit.edu/) \- Song Han's efficient AI research
