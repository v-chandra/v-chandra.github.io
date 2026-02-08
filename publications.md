---
layout: page
title: Selected Recent Publications
permalink: /publications/
---

<style>
.view-toggle {
  margin-bottom: 1.5em;
}
.view-toggle select {
  padding: 8px 12px;
  font-size: 1em;
  border-radius: 4px;
  border: 1px solid #ccc;
  background: #fff;
  cursor: pointer;
}
.view-toggle select:hover {
  border-color: #888;
}
.publication-view {
  display: none;
}
.publication-view.active {
  display: block;
}
</style>

<div class="view-toggle">
  <label for="view-select"><strong>View by:</strong></label>
  <select id="view-select" onchange="toggleView(this.value)">
    <option value="chronological">Chronological</option>
    <option value="topic">Topic</option>
  </select>
</div>

<div id="topic-view" class="publication-view" markdown="1">

## Language Models

1. MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes,
**ICLR (2026)** [[PDF]](https://arxiv.org/pdf/2509.24945)

1. SpinQuant: LLM Quantization with Learned Rotations,
**ICLR (2025)**. [[PDF]](https://www.arxiv.org/pdf/2405.16406)

1. AutoMixer: Checkpoint Artifacts as Automatic Data Mixers,
**ACL (2025)** [[PDF]](https://aclanthology.org/2025.acl-long.979.pdf)

1. Streamlining Language Models via Semantic Basis Analysis,
**TMLR (2025)** [[PDF]](https://openreview.net/pdf?id=qq7NNAXvuv)

1. Target-Aware Language Modeling via Granular Data Sampling,
**EMNLP (2024)**. [[PDF]](https://aclanthology.org/2024.emnlp-main.719.pdf)

1. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases,
**ICML (2024)**. [[PDF]](https://arxiv.org/pdf/2402.14905)

1. Towards Zero-Shot Multilingual Transfer for Code-Switched Responses,
**ACL (2023)**. [[PDF]](https://aclanthology.org/2023.acl-long.417.pdf)

1. Self-Vocabularizing Training for Neural Machine Translation,
**NAACL SRW (2025)**. [[PDF]](https://aclanthology.org/2025.naacl-srw.16.pdf)

1. Scaling Parameter-Constrained Language Models with Quality Data,
**EMNLP Industry (2024)**. [[PDF]](https://aclanthology.org/2024.emnlp-industry.8.pdf)

1. LLM-QAT: Data-Free Quantization Aware Training for Large Language Models,
**ACL Findings (2024)**. [[PDF]](https://arxiv.org/pdf/2305.17888.pdf)

1. MobileLLM-Pro Technical Report,
**arXiv (2025)** [[PDF]](https://arxiv.org/pdf/2511.06719)

1. ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization,
**arXiv (2025)** [[PDF]](https://arxiv.org/pdf/2502.02631?)

1. Agent-as-a-Judge: Evaluate Agents with Agents,
**arXiv (2024)**. [[PDF]](https://arxiv.org/pdf/2410.10934)

1. An Introduction to Vision-Language Modeling,
**arXiv (2024)**. [[PDF]](https://arxiv.org/pdf/2405.17247)

1. MiniGPT-v2: Large Language Model As a Unified Interface for Vision-Language Multi-task Learning,
**arXiv (2023)**. [[PDF]](https://arxiv.org/pdf/2310.09478.pdf)

1. Revisiting Sample Size Determination in Natural Language Understanding,
**arXiv (2023)**. [[PDF]](https://arxiv.org/pdf/2307.00374.pdf)

---

## Efficient AI & Model Compression

1. CPT: Efficient Deep Neural Network Training via Cyclic Precision,
**ICLR (2021) (Spotlight)**. [[PDF]](https://arxiv.org/pdf/2101.09868.pdf)

1. AlphaNet: Improved Training of Supernet with Alpha-Divergence,
**ICML (2021) (Long Presentation)**. [[PDF]](https://arxiv.org/pdf/2102.07954.pdf)

1. APOLLO: SGD-like Memory, AdamW-level Performance,
**MLSys (2025)** [[PDF]](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)

1. NASViT: Neural Architecture Search for Efficient Vision Transformers with Gradient Conflict aware Supernet Training,
**ICLR (2022)**. [[PDF]](https://openreview.net/pdf?id=Qaw16njk6L)

1. DepthShrinker: A New Compression Paradigm Towards Boosting Real-Hardware Efficiency of Compact Neural Networks,
**ICML (2022)**. [[PDF]](https://arxiv.org/pdf/2206.00843.pdf)

1. AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling,
**CVPR (2021)**. [[PDF]](https://arxiv.org/pdf/2011.09011.pdf)

1. Double-win Quant: Aggressively Winning Robustness of Quantized Deep Neural Networks via Random Precision Training and Inference,
**ICML (2021)**. [[PDF]](http://proceedings.mlr.press/v139/fu21c/fu21c.pdf)

1. One weight bitwidth to rule them all,
**Embedded Vision Workshop, ECCV (2020) (Best Paper Award)**. [[PDF]](https://arxiv.org/pdf/2008.09916.pdf)

1. Mixture-of-Supernets: Improving Weight-Sharing Supernet Training with Architecture-Routed Mixture-of-Experts,
**ACL Findings (2024)**. [[PDF]](https://arxiv.org/pdf/2306.04845.pdf)

1. ScaleNAS: Multi-Path One-Shot NAS for Scale-Aware High-Resolution Representation,
**AutoML (2022)**. [[PDF]](https://openreview.net/pdf?id=BWfeZ6SIlq)

1. Contrastive Quant: Quantization makes Stronger Contrastive Learning,
**DAC (2022)**. [[PDF]](https://dl.acm.org/doi/abs/10.1145/3489517.3530419)

1. NASGEM: Neural Architecture Search via Graph Embedding Method,
**AAAI (2021)**. [[PDF]](https://arxiv.org/pdf/2007.04452.pdf)

1. Energy-Aware Neural Architecture Optimization With Splitting Steepest Descent,
**NeurIPS Workshop (2019)**. [[PDF]](https://arxiv.org/pdf/1910.03103.pdf)

1. Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations,
**arXiv (2024)**. [[PDF]](https://arxiv.org/pdf/2411.17713)

1. Basis Selection: Low-Rank Decomposition of Pretrained Large Language Models for Target Applications,
**arXiv (2024)**. [[PDF]](https://arxiv.org/pdf/2405.15877)

1. Low-Rank+ Sparse Tensor Compression for Neural Networks,
**arXiv (2021)**. [[PDF]](https://arxiv.org/pdf/2111.01697.pdf)

1. CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs,
**arXiv (2018)**. [[PDF]](https://arxiv.org/pdf/1801.06601.pdf)

1. Deep Convolutional Neural Network Inference with Floating-point Weights and Fixed-point Activations,
**arXiv (2017)**. [[PDF]](https://arxiv.org/pdf/1703.03073.pdf)

---

## Computer Vision & 3D

1. DepthLM: Metric Depth from Vision Language Models,
**ICLR (2026) (Oral Presentation)** [[PDF]](https://arxiv.org/pdf/2509.25413)

1. EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything,
**CVPR (2024) (Highlight)**. [[PDF]](https://arxiv.org/pdf/2312.00863)

1. EdgeTAM: On-Device Track Anything Model,
**CVPR (2025)** [[PDF]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_EdgeTAM_On-Device_Track_Anything_Model_CVPR_2025_paper.pdf)

1. MVDiffusion++: A Dense High-resolution Multi-view Diffusion Model for Single or Sparse-view 3D Object Reconstruction,
**ECCV (2025)** [[PDF]](https://arxiv.org/pdf/2402.12712)

1. CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians,
**ECCV (2024)**. [[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04306.pdf)

1. Taming Mode Collapse in Score Distillation for Text-to-3D Generation,
**CVPR (2024)**. [[PDF]](https://arxiv.org/pdf/2401.00909.pdf)

1. MVDiffHD: A Dense High-resolution Multi-view Diffusion Model for Single or Sparse-view 3D Object Reconstruction,
**ECCV (2024)**. [[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02446.pdf)

1. Fast Point Cloud Generation with Straight Flows,
**CVPR (2023)**. [[PDF]](https://arxiv.org/pdf/2212.01747.pdf)

1. Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation,
**CVPR (2022)**. [[PDF]](http://128.84.4.34/pdf/2111.01236)

1. KeepAugment: A Simple Information-Preserving Data Augmentation Approach,
**CVPR (2021)**. [[PDF]](https://arxiv.org/pdf/2011.11778.pdf)

1. Feature-Align Network with Knowledge Distillation for Efficient Denoising,
**WACV (2022)**. [[PDF]](https://openaccess.thecvf.com/content/WACV2022W/WACI/papers/Young_Feature-Align_Network_With_Knowledge_Distillation_for_Efficient_Denoising_WACVW_2022_paper.pdf)

1. PathFusion: Path-consistent Lidar-Camera Deep Feature Fusion,
**3DV (2024)**. [[PDF]](https://arxiv.org/pdf/2212.06244.pdf)

1. EVRNet: Efficient Video Restoration on Edge Devices,
**ACM MM (2021)**. [[PDF]](https://arxiv.org/pdf/2012.02228.pdf)

1. VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice,
**arXiv (2026)** [[PDF]](https://arxiv.org/pdf/2601.05175)

1. Efficient Track Anything,
**arXiv (2024)**. [[PDF]](https://arxiv.org/pdf/2411.18933)

1. LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding,
**arXiv (2024)**. [[PDF]](https://arxiv.org/pdf/2410.17434)

1. SteinDreamer: Variance Reduction for Text-to-3D Score Distillation via Stein Identity,
**arXiv (2023)**. [[PDF]](https://arxiv.org/pdf/2401.00604.pdf)

1. SqueezeSAM: User Friendly Mobile Interactive Segmentation,
**arXiv (2023)**. [[PDF]](https://arxiv.org/pdf/2312.06736)

1. Vision Transformers with Patch Diversification,
**arXiv (2021)**. [[PDF]](https://arxiv.org/pdf/2104.12753.pdf)

1. Can Temporal Information Help with Contrastive Self-Supervised Learning?,
**arXiv (2020)**. [[PDF]](https://arxiv.org/pdf/2011.13046.pdf)

---

## Speech & Audio

1. Breaking Down Power Barriers in On-Device Streaming ASR: Insights and Solutions,
**NAACL (2025)** [[PDF]](https://aclanthology.org/2025.naacl-long.1.pdf)

1. TODM: Train Once Deploy Many Efficient Supernet-Based RNN-T Compression For On-Device ASR Models,
**ICASSP (2024)**. [[PDF]](https://arxiv.org/pdf/2309.01947)

1. Stack-and-Delay: A New Codebook Pattern for Music Generation,
**ICASSP (2024)**. [[PDF]](https://arxiv.org/pdf/2309.08804)

1. In-Context Prompt Editing for Conditional Audio Generation,
**ICASSP (2024)**. [[PDF]](https://arxiv.org/pdf/2311.00895)

1. On the Open Prompt Challenge in Conditional Audio Generation,
**ICASSP (2024)**. [[PDF]](https://arxiv.org/pdf/2311.00897)

1. Folding Attention: Memory and Power Optimization for On-Device Transformer-based Streaming Speech Recognition,
**ICASSP (2024)**. [[PDF]](https://arxiv.org/pdf/2309.07988)

1. Omni-sparsity DNN: Fast Sparsity Optimization for On-Device Streaming E2E ASR via Supernet,
**ICASSP (2022)**. [[PDF]](https://arxiv.org/pdf/2110.08352.pdf)

1. Memory-efficient Speech Recognition on Smart Devices,
**ICASSP (2021)**. [[PDF]](https://arxiv.org/pdf/2102.11531.pdf)

1. Streaming Parallel Transducer Beam Search with Fast-Slow Cascaded Encoders,
**INTERSPEECH (2022)**. [[PDF]](https://www.isca-speech.org/archive/pdfs/interspeech_2022/mahadeokar22_interspeech.pdf)

1. Collaborative Training of Acoustic Encoders for Speech Recognition,
**INTERSPEECH (2021)**. [[PDF]](https://arxiv.org/pdf/2106.08960.pdf)

1. Data Efficient Reflow for Few Step Audio Generation,
**SLT (2024)**. [[PDF]](https://ieeexplore.ieee.org/document/10832165/)

1. Towards Temporally Synchronized Visually Indicated Sounds Through Scale-Adapted Positional Embeddings,
**NeurIPS Workshop (2024)**. [[PDF]](https://openreview.net/forum?id=HZq8Gakf6e)

1. SLAP: Scalable Language-Audio Pretraining with Variable-Duration Audio and Multi-Objective Training,
**arXiv (2026)** [[PDF]](https://arxiv.org/pdf/2601.12594)

1. SyncFlow: Toward Temporally Aligned Joint Audio-Video Generation from Text,
**arXiv (2024)**. [[PDF]](https://arxiv.org/pdf/2412.15220)

1. High Fidelity Text-Guided Music Generation and Editing via Single-Stage Flow Matching,
**arXiv (2024)**. [[PDF]](https://arxiv.org/pdf/2407.03648)

1. Enhance Audio Generation Controllability Through Representation Similarity Regularization,
**arXiv (2023)**. [[PDF]](https://arxiv.org/pdf/2309.08773.pdf)

1. Exploring Speech Enhancement for Low-resource Speech Synthesis,
**arXiv (2023)**. [[PDF]](https://arxiv.org/pdf/2309.10795.pdf)

1. FoleyGen: Visually-Guided Audio Generation,
**arXiv (2023)**. [[PDF]](https://arxiv.org/pdf/2309.10537.pdf)

1. LiCo-Net: Linearized Convolution Network for Hardware-efficient Keyword Spotting,
**arXiv (2022)**. [[PDF]](https://arxiv.org/pdf/2211.04635.pdf)

1. Noisy Training Improves E2E ASR for the Edge,
**arXiv (2021)**. [[PDF]](https://arxiv.org/pdf/2107.04677.pdf)

1. Hello Edge: Keyword Spotting on Microcontrollers,
**arXiv (2017)**. [[PDF]](https://arxiv.org/pdf/1711.07128.pdf)

---

## Systems ML

1. DREAM: A Dynamic Scheduler for Dynamic Real-time Multi-model ML Workloads,
**ASPLOS (2024)**. [[PDF]](https://arxiv.org/pdf/2212.03414)

1. XRBench: An Extended Reality (XR) Machine Learning Benchmark Suite for the Metaverse,
**MLSys (2023)**. [[PDF]](https://arxiv.org/pdf/2211.08675.pdf)

1. Heterogeneous Dataflow Accelerators for Multi-DNN Workloads,
**HPCA (2021)**. [[PDF]](https://arxiv.org/pdf/1909.07437.pdf)

1. Mind Mappings: Enabling Efficient Algorithm-Accelerator Mapping Space Search,
**ASPLOS (2021)**. [[PDF]](https://arxiv.org/pdf/2103.01489.pdf)

1. RecNMP: Accelerating Personalized Recommendation with Near-Memory Processing,
**ISCA (2020)**. [[PDF]](https://arxiv.org/pdf/1912.12953.pdf)

1. Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Networks,
**ISCA (2018)**. [[PDF]](https://arxiv.org/pdf/1712.01507.pdf)

1. Co-Exploration of Neural Architectures and Heterogeneous ASIC Accelerator Designs Targeting Multiple Tasks,
**DAC (2020)**. [[PDF]](https://arxiv.org/pdf/2002.04116.pdf)

1. Improving Efficiency in Neural Network Accelerator using Operands Hamming Distance Optimization,
**NeurIPS Workshop (2019)**. [[PDF]](https://arxiv.org/pdf/2002.05293.pdf)

1. Not All Ops are Created Equal!,
**SysML (2018)**. [[PDF]](https://arxiv.org/pdf/1801.04326.pdf)

1. Throughput-optimized OpenCL-based FPGA Accelerator for Large-scale Convolutional Neural Networks,
**FPGA (2016)**. [[PDF]](https://dl.acm.org/citation.cfm?id=2847276)

1. DNA: Differentiable Network-Accelerator Co-Search,
**arXiv (2020)**. [[PDF]](https://arxiv.org/pdf/2010.14778.pdf)

1. Federated Learning with Non-IID Data,
**arXiv (2018)**. [[PDF]](https://arxiv.org/pdf/1806.00582.pdf)

1. PrivyNet: A Flexible Framework for Privacy-Preserving Deep Neural Network Training,
**arXiv (2018)**. [[PDF]](https://arxiv.org/pdf/1709.06161.pdf)

</div>

<div id="chronological-view" class="publication-view active" markdown="1">

## 2026

1. DepthLM: Metric Depth from Vision Language Models,
**ICLR (Oral Presentation)** [[PDF]](https://arxiv.org/pdf/2509.25413)

1. MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes,
**ICLR** [[PDF]](https://arxiv.org/pdf/2509.24945)

1. SLAP: Scalable Language-Audio Pretraining with Variable-Duration Audio and Multi-Objective Training,
**arXiv** [[PDF]](https://arxiv.org/pdf/2601.12594)

1. VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice,
**arXiv** [[PDF]](https://arxiv.org/pdf/2601.05175)

---

## 2025

1. SpinQuant: LLM Quantization with Learned Rotations,
**ICLR** [[PDF]](https://www.arxiv.org/pdf/2405.16406)

1. AutoMixer: Checkpoint Artifacts as Automatic Data Mixers,
**ACL** [[PDF]](https://aclanthology.org/2025.acl-long.979.pdf)

1. Breaking Down Power Barriers in On-Device Streaming ASR: Insights and Solutions,
**NAACL** [[PDF]](https://aclanthology.org/2025.naacl-long.1.pdf)

1. Streamlining Language Models via Semantic Basis Analysis,
**TMLR** [[PDF]](https://openreview.net/pdf?id=qq7NNAXvuv)

1. EdgeTAM: On-Device Track Anything Model,
**CVPR** [[PDF]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_EdgeTAM_On-Device_Track_Anything_Model_CVPR_2025_paper.pdf)

1. MVDiffusion++: A Dense High-resolution Multi-view Diffusion Model for Single or Sparse-view 3D Object Reconstruction,
**ECCV** [[PDF]](https://arxiv.org/pdf/2402.12712)

1. APOLLO: SGD-like Memory, AdamW-level Performance,
**MLSys** [[PDF]](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)

1. Self-Vocabularizing Training for Neural Machine Translation,
**NAACL SRW** [[PDF]](https://aclanthology.org/2025.naacl-srw.16.pdf)

1. MobileLLM-Pro Technical Report,
**arXiv** [[PDF]](https://arxiv.org/pdf/2511.06719)

1. ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization,
**arXiv** [[PDF]](https://arxiv.org/pdf/2502.02631?)

---

## 2024

1. EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything,
**CVPR (Highlight)** [[PDF]](https://arxiv.org/pdf/2312.00863)

1. Target-Aware Language Modeling via Granular Data Sampling,
**EMNLP** [[PDF]](https://aclanthology.org/2024.emnlp-main.719.pdf)

1. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases,
**ICML** [[PDF]](https://arxiv.org/pdf/2402.14905)

1. Taming Mode Collapse in Score Distillation for Text-to-3D Generation,
**CVPR** [[PDF]](https://arxiv.org/pdf/2401.00909.pdf)

1. CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians,
**ECCV** [[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04306.pdf)

1. MVDiffHD: A Dense High-resolution Multi-view Diffusion Model for Single or Sparse-view 3D Object Reconstruction,
**ECCV** [[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02446.pdf)

1. DREAM: A Dynamic Scheduler for Dynamic Real-time Multi-model ML Workloads,
**ASPLOS** [[PDF]](https://arxiv.org/pdf/2212.03414)

1. TODM: Train Once Deploy Many Efficient Supernet-Based RNN-T Compression For On-Device ASR Models,
**ICASSP** [[PDF]](https://arxiv.org/pdf/2309.01947)

1. Stack-and-Delay: A New Codebook Pattern for Music Generation,
**ICASSP** [[PDF]](https://arxiv.org/pdf/2309.08804)

1. In-Context Prompt Editing for Conditional Audio Generation,
**ICASSP** [[PDF]](https://arxiv.org/pdf/2311.00895)

1. On the Open Prompt Challenge in Conditional Audio Generation,
**ICASSP** [[PDF]](https://arxiv.org/pdf/2311.00897)

1. Folding Attention: Memory and Power Optimization for On-Device Transformer-based Streaming Speech Recognition,
**ICASSP** [[PDF]](https://arxiv.org/pdf/2309.07988)

1. Scaling Parameter-Constrained Language Models with Quality Data,
**EMNLP Industry** [[PDF]](https://aclanthology.org/2024.emnlp-industry.8.pdf)

1. Data Efficient Reflow for Few Step Audio Generation,
**SLT** [[PDF]](https://ieeexplore.ieee.org/document/10832165/)

1. Towards Temporally Synchronized Visually Indicated Sounds Through Scale-Adapted Positional Embeddings,
**NeurIPS Workshop** [[PDF]](https://openreview.net/forum?id=HZq8Gakf6e)

1. LLM-QAT: Data-Free Quantization Aware Training for Large Language Models,
**ACL Findings** [[PDF]](https://arxiv.org/pdf/2305.17888.pdf)

1. Mixture-of-Supernets: Improving Weight-Sharing Supernet Training with Architecture-Routed Mixture-of-Experts,
**ACL Findings** [[PDF]](https://arxiv.org/pdf/2306.04845.pdf)

1. PathFusion: Path-consistent Lidar-Camera Deep Feature Fusion,
**3DV** [[PDF]](https://arxiv.org/pdf/2212.06244.pdf)

1. Efficient Track Anything,
**arXiv** [[PDF]](https://arxiv.org/pdf/2411.18933)

1. SyncFlow: Toward Temporally Aligned Joint Audio-Video Generation from Text,
**arXiv** [[PDF]](https://arxiv.org/pdf/2412.15220)

1. Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations,
**arXiv** [[PDF]](https://arxiv.org/pdf/2411.17713)

1. LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding,
**arXiv** [[PDF]](https://arxiv.org/pdf/2410.17434)

1. Agent-as-a-Judge: Evaluate Agents with Agents,
**arXiv** [[PDF]](https://arxiv.org/pdf/2410.10934)

1. High Fidelity Text-Guided Music Generation and Editing via Single-Stage Flow Matching,
**arXiv** [[PDF]](https://arxiv.org/pdf/2407.03648)

1. An Introduction to Vision-Language Modeling,
**arXiv** [[PDF]](https://arxiv.org/pdf/2405.17247)

1. Basis Selection: Low-Rank Decomposition of Pretrained Large Language Models for Target Applications,
**arXiv** [[PDF]](https://arxiv.org/pdf/2405.15877)

---

## 2023

1. Towards Zero-Shot Multilingual Transfer for Code-Switched Responses,
**ACL** [[PDF]](https://aclanthology.org/2023.acl-long.417.pdf)

1. XRBench: An Extended Reality (XR) Machine Learning Benchmark Suite for the Metaverse,
**MLSys** [[PDF]](https://arxiv.org/pdf/2211.08675.pdf)

1. Fast Point Cloud Generation with Straight Flows,
**CVPR** [[PDF]](https://arxiv.org/pdf/2212.01747.pdf)

1. SteinDreamer: Variance Reduction for Text-to-3D Score Distillation via Stein Identity,
**arXiv** [[PDF]](https://arxiv.org/pdf/2401.00604.pdf)

1. MiniGPT-v2: Large Language Model As a Unified Interface for Vision-Language Multi-task Learning,
**arXiv** [[PDF]](https://arxiv.org/pdf/2310.09478.pdf)

1. Revisiting Sample Size Determination in Natural Language Understanding,
**arXiv** [[PDF]](https://arxiv.org/pdf/2307.00374.pdf)

1. Enhance Audio Generation Controllability Through Representation Similarity Regularization,
**arXiv** [[PDF]](https://arxiv.org/pdf/2309.08773.pdf)

1. Exploring Speech Enhancement for Low-resource Speech Synthesis,
**arXiv** [[PDF]](https://arxiv.org/pdf/2309.10795.pdf)

1. FoleyGen: Visually-Guided Audio Generation,
**arXiv** [[PDF]](https://arxiv.org/pdf/2309.10537.pdf)

1. SqueezeSAM: User Friendly Mobile Interactive Segmentation,
**arXiv** [[PDF]](https://arxiv.org/pdf/2312.06736)

---

## 2022

1. NASViT: Neural Architecture Search for Efficient Vision Transformers with Gradient Conflict aware Supernet Training,
**ICLR** [[PDF]](https://openreview.net/pdf?id=Qaw16njk6L)

1. Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation,
**CVPR** [[PDF]](http://128.84.4.34/pdf/2111.01236)

1. DepthShrinker: A New Compression Paradigm Towards Boosting Real-Hardware Efficiency of Compact Neural Networks,
**ICML** [[PDF]](https://arxiv.org/pdf/2206.00843.pdf)

1. Feature-Align Network with Knowledge Distillation for Efficient Denoising,
**WACV** [[PDF]](https://openaccess.thecvf.com/content/WACV2022W/WACI/papers/Young_Feature-Align_Network_With_Knowledge_Distillation_for_Efficient_Denoising_WACVW_2022_paper.pdf)

1. Omni-sparsity DNN: Fast Sparsity Optimization for On-Device Streaming E2E ASR via Supernet,
**ICASSP** [[PDF]](https://arxiv.org/pdf/2110.08352.pdf)

1. Streaming Parallel Transducer Beam Search with Fast-Slow Cascaded Encoders,
**INTERSPEECH** [[PDF]](https://www.isca-speech.org/archive/pdfs/interspeech_2022/mahadeokar22_interspeech.pdf)

1. ScaleNAS: Multi-Path One-Shot NAS for Scale-Aware High-Resolution Representation,
**AutoML** [[PDF]](https://openreview.net/pdf?id=BWfeZ6SIlq)

1. Contrastive Quant: Quantization makes Stronger Contrastive Learning,
**DAC** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3489517.3530419)

1. LiCo-Net: Linearized Convolution Network for Hardware-efficient Keyword Spotting,
**arXiv** [[PDF]](https://arxiv.org/pdf/2211.04635.pdf)

---

## 2021

1. CPT: Efficient Deep Neural Network Training via Cyclic Precision,
**ICLR (Spotlight)** [[PDF]](https://arxiv.org/pdf/2101.09868.pdf)

1. AlphaNet: Improved Training of Supernet with Alpha-Divergence,
**ICML (Long Presentation)** [[PDF]](https://arxiv.org/pdf/2102.07954.pdf)

1. AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling,
**CVPR** [[PDF]](https://arxiv.org/pdf/2011.09011.pdf)

1. KeepAugment: A Simple Information-Preserving Data Augmentation Approach,
**CVPR** [[PDF]](https://arxiv.org/pdf/2011.11778.pdf)

1. Double-win Quant: Aggressively Winning Robustness of Quantized Deep Neural Networks via Random Precision Training and Inference,
**ICML** [[PDF]](http://proceedings.mlr.press/v139/fu21c/fu21c.pdf)

1. Heterogeneous Dataflow Accelerators for Multi-DNN Workloads,
**HPCA** [[PDF]](https://arxiv.org/pdf/1909.07437.pdf)

1. Mind Mappings: Enabling Efficient Algorithm-Accelerator Mapping Space Search,
**ASPLOS** [[PDF]](https://arxiv.org/pdf/2103.01489.pdf)

1. NASGEM: Neural Architecture Search via Graph Embedding Method,
**AAAI** [[PDF]](https://arxiv.org/pdf/2007.04452.pdf)

1. Collaborative Training of Acoustic Encoders for Speech Recognition,
**INTERSPEECH** [[PDF]](https://arxiv.org/pdf/2106.08960.pdf)

1. Memory-efficient Speech Recognition on Smart Devices,
**ICASSP** [[PDF]](https://arxiv.org/pdf/2102.11531.pdf)

1. EVRNet: Efficient Video Restoration on Edge Devices,
**ACM MM** [[PDF]](https://arxiv.org/pdf/2012.02228.pdf)

1. Noisy Training Improves E2E ASR for the Edge,
**arXiv** [[PDF]](https://arxiv.org/pdf/2107.04677.pdf)

1. Low-Rank+ Sparse Tensor Compression for Neural Networks,
**arXiv** [[PDF]](https://arxiv.org/pdf/2111.01697.pdf)

1. Vision Transformers with Patch Diversification,
**arXiv** [[PDF]](https://arxiv.org/pdf/2104.12753.pdf)

---

## 2020

1. One weight bitwidth to rule them all,
**ECCV Workshop (Best Paper Award)** [[PDF]](https://arxiv.org/pdf/2008.09916.pdf)

1. RecNMP: Accelerating Personalized Recommendation with Near-Memory Processing,
**ISCA** [[PDF]](https://arxiv.org/pdf/1912.12953.pdf)

1. Co-Exploration of Neural Architectures and Heterogeneous ASIC Accelerator Designs Targeting Multiple Tasks,
**DAC** [[PDF]](https://arxiv.org/pdf/2002.04116.pdf)

1. Can Temporal Information Help with Contrastive Self-Supervised Learning?,
**arXiv** [[PDF]](https://arxiv.org/pdf/2011.13046.pdf)

1. DNA: Differentiable Network-Accelerator Co-Search,
**arXiv** [[PDF]](https://arxiv.org/pdf/2010.14778.pdf)

---

## 2019

1. Energy-Aware Neural Architecture Optimization With Splitting Steepest Descent,
**NeurIPS Workshop** [[PDF]](https://arxiv.org/pdf/1910.03103.pdf)

1. Improving Efficiency in Neural Network Accelerator using Operands Hamming Distance Optimization,
**NeurIPS Workshop** [[PDF]](https://arxiv.org/pdf/2002.05293.pdf)

---

## 2018

1. Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Networks,
**ISCA** [[PDF]](https://arxiv.org/pdf/1712.01507.pdf)

1. Not All Ops are Created Equal!,
**SysML** [[PDF]](https://arxiv.org/pdf/1801.04326.pdf)

1. Federated Learning with Non-IID Data,
**arXiv** [[PDF]](https://arxiv.org/pdf/1806.00582.pdf)

1. CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs,
**arXiv** [[PDF]](https://arxiv.org/pdf/1801.06601.pdf)

1. PrivyNet: A Flexible Framework for Privacy-Preserving Deep Neural Network Training,
**arXiv** [[PDF]](https://arxiv.org/pdf/1709.06161.pdf)

---

## 2017

1. Hello Edge: Keyword Spotting on Microcontrollers,
**arXiv** [[PDF]](https://arxiv.org/pdf/1711.07128.pdf)

1. Deep Convolutional Neural Network Inference with Floating-point Weights and Fixed-point Activations,
**arXiv** [[PDF]](https://arxiv.org/pdf/1703.03073.pdf)

---

## 2016

1. Throughput-optimized OpenCL-based FPGA Accelerator for Large-scale Convolutional Neural Networks,
**FPGA** [[PDF]](https://dl.acm.org/citation.cfm?id=2847276)

</div>

<script>
function toggleView(view) {
  document.getElementById('topic-view').classList.remove('active');
  document.getElementById('chronological-view').classList.remove('active');
  document.getElementById(view + '-view').classList.add('active');
}
</script>
