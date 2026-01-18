---
layout: post
title: "The Personal Context Graph: Why On-Device AI will capture the layer that cloud models can't"
date: 2026-01-14
---

There's a growing consensus in AI that the next trillion-dollar platform won't be another chatbot or copilot. It'll be the system that captures ["context graphs"](https://foundationcapital.com/context-graphs-ais-trillion-dollar-opportunity/): the decision traces, exceptions, and precedents that currently live in Slack threads, deal desk conversations, and people's heads.

The thesis is compelling. But it's focused on the wrong scale.

The real context graph opportunity isn't in the enterprise. It's in your pocket.

![Personal Context Graph](/images/personal_context_graph.png)

## The Decision Trace Problem, Personalized

Traditional systems capture *what* happened, but not *why*. A CRM stores "20% discount applied." It doesn't store that Finance approved it because the customer had a similar deal last quarter and the VP made an exception based on expansion plans mentioned in a call.

The same problem exists at the personal level, and it's even more acute.

Your phone knows you ordered Thai food. It doesn't know you ordered it because you were stressed, it was raining, and your partner was working late. Your calendar shows you declined a meeting. It doesn't capture that you declined because you've been in back-to-backs all week and the agenda was vague.

These personal decision traces are everywhere:
- Why you swiped left or right
- Why you chose that route instead of the faster one
- Why you replied to one email immediately but let another sit for days
- Why you bought the cheaper option this time but splurged last time

No cloud service can capture this context. Not because of technical limitations, but because **you would never upload it.**

## The Privacy Paradox

The most valuable decision traces are the ones you'd never share: your financial anxieties, relationship dynamics, health concerns, work frustrations, daily habits.

This is exactly the data that would make AI genuinely useful. It's also exactly the data that [creates massive privacy risk](https://www.techpolicy.press/the-privacy-challenges-of-emerging-personalized-ai-services/) when it leaves your device.

Cloud providers have built "private cloud compute" architectures as workarounds. But the real solution isn't better cloud privacy. It's moving the model to where the context already lives.

## The MobileLLM Thesis: Architecture Over Scale

The conventional wisdom in AI: bigger is better. But for on-device applications, **architecture matters more than scale.**

[MobileLLM](https://arxiv.org/abs/2402.14905) (ICML 2024) proved this with a 350M parameter model that outperformed prior state-of-the-art by 4.3% through smarter design: deep-thin architectures, embedding sharing, and grouped-query attention. The 125M model runs at 50 tokens/second on an iPhone, compared to 3-6 for LLaMA 7B.

## From Chat to Reasoning: MobileLLM-R1

If 2024 proved small models could be useful, 2025 proved they could *think*.

[MobileLLM-R1](https://arxiv.org/abs/2509.24945) achieves 5× higher accuracy on MATH compared to OLMo-1.24B, and scores 15.5 on AIME versus 0.6 for comparable models, despite training on 88% fewer tokens.

The line continued with [MobileLLM-Pro](https://arxiv.org/abs/2511.06719), a 1B model with 128k context that outperforms Gemma 3 1B and Llama 3.2 1B while achieving int4 quantization with less than 1.3% quality loss.

These aren't toy models. They can reason about context and make decisions.

## Why This Matters for Context Graphs

The economics are compelling: sub-billion parameter models are [10-30× cheaper](https://developer.nvidia.com/blog/how-small-language-models-are-key-to-scalable-agentic-ai/) than 405B models. Fine-tuning takes hours instead of weeks. Latency drops from seconds to milliseconds.

But the real advantage isn't cost or speed. It's **context access**.

An on-device model has secure access to your emails, messages, photos, calendar, and app usage. It can build a unique model of *you* that understands your relationships, priorities, and context, without ever exposing your data to third parties.

This isn't a privacy workaround. It's a structural advantage that cloud models cannot replicate.

## What This Enables

With a personal context graph, AI moves from reactive to proactive. Instead of answering questions, it anticipates needs. Instead of generic suggestions, it offers ones grounded in your history, preferences, and patterns. The difference between "here are some restaurants nearby" and "you usually order Thai when you're stressed, and you seem stressed."

Phones are uniquely positioned to build this. Beyond apps and messages, they have sensors: GPS for location patterns, accelerometer for activity, microphone for ambient context, and connection to wearables for sleep and heart rate. Combined with on-device models that can reason about this multimodal stream, phones can infer not just where you are, but how you're doing.

## The Execution Path Argument

Enterprise AI startups have an advantage when they "sit in the execution path," seeing full context at decision time. On-device models have the same advantage for personal decisions:

<table style="width:100%; border-collapse: collapse; margin: 1.5em 0;">
  <thead>
    <tr style="background-color: #f5f5f5;">
      <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Enterprise Context Graph</th>
      <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Personal Context Graph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 10px;">Sits in deal flow</td>
      <td style="border: 1px solid #ddd; padding: 10px;">Sits in daily decisions</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 10px;">Sees CRM + Slack + Zoom</td>
      <td style="border: 1px solid #ddd; padding: 10px;">Sees calendar + messages + location + apps</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 10px;">Cloud-deployed</td>
      <td style="border: 1px solid #ddd; padding: 10px;">Device-deployed</td>
    </tr>
  </tbody>
</table>

When you decide whether to respond to a message, an on-device model sees the full picture: message content, conversation history, your location, calendar, and response patterns. A cloud model sees the message content, maybe.

## What a Personal Context Graph Stores

A personal context graph would capture entities (people, places, topics, decisions), decision traces (deviations from patterns, surrounding context, outcomes), and precedents ("stressed + raining + partner working late → Thai food").

This graph lives on-device, updates continuously, and never leaves your phone. It gives the model access to *why* you made past decisions, not just *what* you did.

Building this isn't just pattern matching. It requires inference about *why* decisions were made. A model that scores 15.5 on AIME can reason about why you skipped lunch or chose to walk instead of drive.

## The Competitive Moat

Whoever builds the personal context graph captures an unprecedented moat. Cloud providers can't replicate it because users won't share the data. Competitors can't transfer it since the graph is local and personal. The switching cost is your entire behavioral history.

This is stronger than a traditional data moat. The user's data is *about how they make decisions*, which is more valuable and more private than transactional data.

## The Stakes

Enterprise context graphs are widely considered a trillion-dollar opportunity. For personal AI, the market is arguably larger, as consumer devices outnumber enterprise deployments by orders of magnitude.

But beyond market size, the personal context graph represents AI that actually knows you. Not AI trained on internet text that can simulate knowing you. AI that has observed your decisions, captured your reasoning, and built a model of *how you think*.

This can only be built on-device. It requires models that can reason, not just respond. And the trajectory from MobileLLM through R1 and Pro shows that such models now exist.

---

*The infrastructure is shipping now. What's missing is the intentional design of systems that capture decision traces, not just actions, but the context that explains them. Who will build it first?*

---

## References

- Foundation Capital. ["Context Graphs: AI's Trillion-Dollar Opportunity"](https://foundationcapital.com/context-graphs-ais-trillion-dollar-opportunity/). 2025.
- Liu et al. ["MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"](https://arxiv.org/abs/2402.14905). ICML 2024.
- Meta AI. ["MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners"](https://arxiv.org/abs/2509.24945). 2025.
- Meta AI. ["MobileLLM-Pro"](https://arxiv.org/abs/2511.06719). 2025.
- NVIDIA Research. ["Small Language Models are the Future of Agentic AI"](https://arxiv.org/abs/2506.02153). 2025.
