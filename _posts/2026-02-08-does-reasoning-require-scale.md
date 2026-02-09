---
layout: post
title: "Does Reasoning Require Scale?"
date: 2026-02-08
published: true
excerpt: "<p>A 950M parameter model solves more competition math problems than models nearly twice its size. The gap isn't parameter count, it's training methodology and inference strategy. But cheap reasoning shifts the bottleneck to reliability: small models can reason, they just don't know when they're wrong.</p>"
---

<p align="center"><img src="/images/reasoning_scale.png" alt="Does Reasoning Require Scale?" width="85%"></p>

[MobileLLM-R1](https://arxiv.org/abs/2509.24945), a 950M parameter model, scores 15.5% on AIME. [OLMo-2](https://arxiv.org/abs/2501.00656), 50% larger at 1.48B parameters, scores 0.6%. [SmolLM-2](https://arxiv.org/abs/2502.02737), larger still at 1.7B, scores 0.3%. The smallest model solves roughly 2 out of 15 competition math problems. The larger ones solve essentially none.

Now, this comparison isn't quite fair. MobileLLM-R1 was specifically trained for reasoning through distillation; OLMo-2 and SmolLM-2 are general-purpose base models without that targeted approach. But that's exactly the point: the gap comes from training methodology, not parameter count. Whatever reasoning capability exists in large models can be compressed into much smaller ones more efficiently than the parameter ratio would suggest.

This pattern isn't isolated. [DeepSeek's distilled 32B model](https://arxiv.org/abs/2501.12948) outperforms OpenAI's o1-mini on reasoning benchmarks despite being a fraction of the size. Research on [test-time compute scaling](https://arxiv.org/abs/2408.03314) shows that smaller models with proper search strategies can outperform models over 10x larger in FLOPs-matched evaluations. Parameter count and reasoning capability are far less correlated than most people assume.

## What Actually Drives Reasoning

Scale is one lever, but two others matter as much or more, and we've been underweighting them.

The first is training methodology. The DeepSeek R1 work showed that RL-based post-training dramatically improves reasoning in ways supervised fine-tuning doesn't replicate. Distillation from strong reasoning models transfers capability with surprising efficiency. MobileLLM-R1's results come from distilling reasoning patterns into sub-billion parameter architectures using high-quality data, and this works better than running RL directly on small models.

The second is inference strategy. Test-time compute scaling lets models "think longer" through search or self-verification. A small model that explores multiple paths and checks its work can beat a larger model that generates a single answer. Reasoning capability isn't fixed at training time; it's partially a function of how much compute you spend at inference.

## Where Scale Still Wins

To be clear: scale still matters for some things. Frontier capabilities on novel problem types, tasks requiring broad world knowledge, very long reasoning chains where small models accumulate errors: these still favor larger models. You can't distill what the teacher doesn't know. The claim isn't that scale is irrelevant, but that it's not the only path to reasoning, and for many practical tasks it's not the most efficient one.

Most of the evidence here comes from math benchmarks. Whether these results generalize to code reasoning, multi-step planning, or common-sense inference is still an open question. Math is where distillation and test-time compute have been studied most, so that's where the strongest data exists. But math reasoning is also unusually verifiable, which makes it easier to train and evaluate. Other domains may not compress as cleanly.

## Cheap Reasoning, Unreliable Reasoning

If training and inference matter as much as scale, the most immediate consequence is cost. A capable 1B model is 10-100x cheaper to run than a 70B, which changes the math for any application that chains multiple reasoning calls together. Agents are the obvious case: an agent that plans, acts, reflects, and replans might make dozens of reasoning calls per task. At 70B inference costs, that's expensive. At 1B costs, it's nearly free.

But once reasoning gets cheap, the bottleneck moves to reliability. Small models can reason through problems; they just have no idea when they're wrong. They produce confident answers whether they've nailed the logic or hallucinated a step. In practice, most production systems will probably need fast and slow paths, using cheap models for routine decisions and reserving heavier verification for anything high-stakes.

This points to what I think is the real open problem: calibrated uncertainty. A 1B model that can solve a problem and flag "I'm not confident here, escalate this" would be more useful than a 70B model you call for everything. We don't have good ways to do this yet. Current small models are confidently wrong at roughly the same rate they're confidently right, and we lack reliable training signals for teaching a model to know the boundary of its own competence. Getting calibration right matters more than another 10x in parameters, because it determines whether cheap reasoning is actually deployable.

There's also early work on coordinating multiple small models instead of scaling up, with models cross-checking each other's reasoning or exploring solution paths in parallel. Whether coordination overhead kills the gains is still unclear, but worth watching.

So does reasoning require scale? Less than we assumed. The harder question now is not whether small models can reason, but whether they can know when to stop trusting themselves.
