---
layout: post
title: "The Personal Context Graph: Why On-Device AI Will Capture the Layer That Cloud Models Can't"
---

There's a growing consensus in AI that the next trillion-dollar platform won't be another chatbot or copilot. It'll be the system that captures ["context graphs"](https://foundationcapital.com/context-graphs-ais-trillion-dollar-opportunity/)—the decision traces, exceptions, precedents, and cross-system context that currently live in Slack threads, deal desk conversations, and people's heads.

The thesis is compelling. But it's focused on the wrong scale.

The real context graph opportunity isn't in the enterprise. It's in your pocket.

## The Decision Trace Problem, Personalized

Traditional systems of record capture *what* happened, but not *why it was allowed to happen*. A CRM stores "20% discount applied." It doesn't store that Finance approved it because the customer had a similar deal last quarter, reminded the rep via Slack, and the VP made an exception based on the customer's expansion plans mentioned in a Zoom call.

This same problem exists at the personal level—but it's even more acute.

Your phone knows you ordered Thai food. It doesn't know you ordered it because you were stressed, it was raining, your partner was working late, and you remembered that restaurant from a recommendation last month. Your calendar shows you declined a meeting. It doesn't capture that you declined because you've been in back-to-backs all week, you noticed the agenda was vague, and similar meetings from that organizer have historically run over.

These personal decision traces are everywhere:
- Why you swiped left or right
- Why you chose that route, not the faster one
- Why you replied to that email immediately but let the other one sit
- Why you bought the cheaper option this time but splurged last time

No cloud service can capture this context. Not because of technical limitations, but because of an architectural impossibility: **you would never upload it.**

## The Privacy Paradox of Cloud Context

Here's the fundamental problem with cloud-based context graphs for consumers: the most valuable decision traces are the ones you'd never share.

Consider what a truly useful personal AI assistant would need to know:
- Your financial situation and spending anxieties
- Your relationship dynamics and social priorities
- Your health concerns and coping mechanisms
- Your work frustrations and career ambitions
- Your daily routines and private habits

This is exactly the data that would make AI genuinely useful—and exactly the data that [creates massive privacy risk](https://www.techpolicy.press/the-privacy-challenges-of-emerging-personalized-ai-services/) when it leaves your device. As researchers have noted, emerging AI services emphasize "extreme personalization" which sets off a race for "massive amounts of detailed user information, much of which is highly sensitive."

Cloud providers know this. Google's response was [Private Cloud Compute](https://thehackernews.com/2025/11/google-launches-private-ai-compute.html)—a "secure, fortified space" that processes sensitive data with "on-device-level privacy." Apple routes personal context through [Private Cloud Compute](https://machinelearning.apple.com/research/introducing-apple-foundation-models) architecture specifically designed to never expose user data.

But these are workarounds to a fundamental architectural mismatch: cloud models trying to access local context. The real solution isn't better cloud privacy. It's moving the model to where the context already lives.

## The MobileLLM Thesis: Architecture Over Scale

The conventional wisdom in AI has been simple: bigger is better. More parameters, more data, more compute. But a different thesis has been quietly proven correct: **for on-device applications, architecture matters more than scale.**

[MobileLLM](https://arxiv.org/abs/2402.14905), introduced at ICML 2024, demonstrated this definitively. By focusing on architectural innovations rather than parameter count, the MobileLLM team achieved something counterintuitive: a 350M parameter model that outperformed prior state-of-the-art models of similar size by 4.3%—not through more data or compute, but through smarter design.

The key insights:

**Deep and thin beats wide and shallow.** At the sub-billion scale, increasing depth while keeping width constrained captures more abstract concepts than the reverse. This defies traditional scaling laws but makes physical sense: mobile devices have limited memory bandwidth, and deep-thin architectures are more cache-friendly.

**Embedding sharing is critical at small scale.** In a 125M parameter model, embeddings account for over 20% of total parameters (versus just 3.7% in LLaMA-70B). MobileLLM's weight sharing between input and output layers reduced parameters by 11.8% with only 0.2 points accuracy drop—and that drop was recovered by reallocating saved parameters to additional layers.

**Grouped-query attention enables efficiency without sacrifice.** Combined with block-wise weight sharing, MobileLLM achieved production-ready inference: the 350M model consumes only 0.035 J/token, allowing conversational use for an entire day on a single iPhone charge. The 125M model runs at 50 tokens/second—compared to 3-6 tokens/second for LLaMA 7B on the same device.

This wasn't just an academic exercise. It was the foundation for what came next.

## From Chat to Reasoning: MobileLLM-R1

If 2024 proved small models could be useful, 2025 proved they could *think*.

[MobileLLM-R1](https://arxiv.org/abs/2509.24945) extended the MobileLLM architecture to reasoning tasks—math, code, and scientific problems. The results challenged assumptions about what sub-billion parameter models could achieve:

- **MobileLLM-R1-950M achieves 5× higher accuracy on MATH** compared to OLMo-1.24B, and 2× higher than SmolLM2-1.7B
- On AIME, it scores 15.5 compared to 0.6 for OLMo-2-1.48B—a 25× improvement
- Despite training on only 11.7% of the tokens used by Qwen3's 36T-token corpus, MobileLLM-R1-950M matches or surpasses Qwen3-0.6B across multiple reasoning benchmarks

The architectural innovations continued: 32k context length (up from 4k in base models), grouped-query attention with 24 attention heads and 6 KV heads, and block-wise weight sharing that reduces parameter count without latency penalties.

MobileLLM-R1.5 pushed further with on-policy knowledge distillation, using larger models as teachers while maintaining the sub-billion parameter footprint required for edge deployment.

Then came [MobileLLM-Pro](https://howaiworks.ai/blog/mobilellm-pro-announcement): a 1B parameter model with 128k context that outperforms Gemma 3 1B by 5.7% and Llama 3.2 1B by 7.9% across reasoning, knowledge, and long-context retrieval. It achieves int4 quantization with less than 1.3% quality degradation, 1.8× faster prefill through 3:1 local-global attention, and reduces KV cache from 117MB to 40MB for 8k context.

This progression—from MobileLLM to R1 to R1.5 to Pro—represents the maturation of on-device AI from "good enough" to "genuinely capable." These aren't toy models. They're models that can reason about your context and make decisions.

## Why This Matters for Context Graphs

NVIDIA's June 2025 paper ["Small Language Models are the Future of Agentic AI"](https://arxiv.org/abs/2506.02153) makes the theoretical case that SLMs are "sufficiently powerful, inherently more suitable, and necessarily more economical" for specialized tasks. They argue for "a federation of smaller, faster, privacy-friendly agents—running on the edge, in your browser, or even offline."

The MobileLLM family is the empirical proof.

The economics are compelling: running a sub-billion parameter model can be [10-30× cheaper](https://developer.nvidia.com/blog/how-small-language-models-are-key-to-scalable-agentic-ai/) than running a 405B model. Fine-tuning takes hours instead of weeks. Latency drops from seconds to milliseconds.

But the real advantage isn't cost or speed. It's **context access**.

An on-device model has secure access to what Apple calls "the rich tapestry of your personal data: your emails, text messages, photo library, calendar appointments, and even your app usage patterns." It can learn locally and build "a unique neural model of *you*"—understanding "your relationships, your priorities, and your personal context."

A MobileLLM-R1 model running locally can:
- Access your full message history without sending it to a server
- See your calendar, location, and app usage patterns in real-time
- Reason about multi-step decisions (thanks to 32k context and genuine reasoning capability)
- Run continuously in the background, building context over time
- Never expose your data to third parties

This isn't a privacy workaround. It's a structural advantage that cloud models cannot replicate.

## The Execution Path Argument

Enterprise AI startups have a structural advantage when they "sit in the execution path"—seeing full context at decision time, capturing what inputs were gathered, what policy was evaluated, what exception route was invoked.

On-device models have the same structural advantage for personal decisions:

<table>
  <thead>
    <tr>
      <th>Enterprise Context Graph</th>
      <th>Personal Context Graph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sits in deal flow</td>
      <td>Sits in daily decisions</td>
    </tr>
    <tr>
      <td>Sees CRM + Slack + Zoom</td>
      <td>Sees calendar + messages + location + apps</td>
    </tr>
    <tr>
      <td>Captures B2B exceptions</td>
      <td>Captures personal preferences</td>
    </tr>
    <tr>
      <td>Persists approval chains</td>
      <td>Persists behavioral patterns</td>
    </tr>
    <tr>
      <td>Cloud-deployed</td>
      <td>Device-deployed</td>
    </tr>
  </tbody>
</table>

When you open your phone to decide whether to respond to a message, an on-device model sees: the message content, your recent conversation history, your current location, your calendar for today, the time since you last texted that person, and the pattern of when you typically respond. A cloud model sees: the message content, maybe.

The device is already in the execution path. With models like MobileLLM-Pro offering 128k context, the question isn't whether there's enough capability to capture decision traces—it's whether anyone is building systems to do so.

## What a Personal Context Graph Actually Looks Like

Drawing from [research on AI memory systems](https://mem0.ai/blog/ai-memory-layer-guide) and [local knowledge graph implementations](https://www.marktechpost.com/2025/04/26/implementing-persistent-memory-using-a-local-knowledge-graph-in-claude-desktop/), a personal context graph would store:

**Entities:**
- People (relationships, communication patterns, shared history)
- Places (frequency, time-of-day patterns, activities associated)
- Topics (interests, evolving preferences, knowledge level)
- Decisions (what you chose, what you rejected, why)

**Decision Traces:**
- When you deviated from your usual pattern
- What context surrounded that deviation
- What the outcome was
- Whether you repeated the decision

**Precedents:**
- "Last time I was stressed + raining + partner working late → Thai food"
- "Vague meeting agendas from X → decline or ask for details first"
- "Products recommended by Y → higher trust, less research"

This graph lives on-device. It updates continuously from your actual behavior. It never leaves your phone. And it makes the on-device model dramatically more useful because the model has access to *why* you made past decisions, not just *what* you did.

The reasoning capabilities of MobileLLM-R1 are particularly relevant here: building a personal context graph isn't just pattern matching—it requires inference about *why* decisions were made, connecting disparate signals into coherent explanations. A model that can score 15.5 on AIME can certainly reason about why you skipped lunch or chose to walk instead of drive.

## The Competitive Moat

If context graphs become the real source of truth for autonomy, then whoever builds the personal context graph captures an unprecedented moat.

Cloud providers can't replicate it—users won't share the data. Competitors can't transfer it—the graph is local and personal. The switching cost is your entire behavioral history and preference model.

This is stronger than a traditional data moat. It's not just that you have the user's data—it's that the user's data is *about how they make decisions*, which is both more valuable and more private than transactional data.

Apple seems to understand this. Their entire Apple Intelligence positioning is "personal context awareness"—AI that "understands your calendar, messages, location, and habits." Samsung has made it [explicit which Galaxy AI features run on vs. off device](https://www.androidcentral.com/apps-software/ai/ai-2025-report-card), letting users disable cloud features entirely.

The race is on to own the personal context layer. And the teams building capable on-device models—models that can actually reason about context rather than just pattern-match—have a significant head start.

## What This Means for On-Device Model Development

The MobileLLM progression offers a roadmap for building personal context graphs:

**Architecture-first thinking pays off.** The deep-thin architecture, embedding sharing, and grouped-query attention in MobileLLM weren't just optimizations—they were prerequisites for running capable models on constrained hardware. The same architectural discipline applies to context graph systems: efficient storage, incremental updates, and cache-friendly access patterns matter more than raw capability.

**Reasoning capability unlocks new use cases.** The jump from MobileLLM to MobileLLM-R1 wasn't just benchmark improvement—it enabled fundamentally new applications. A model that can reason about math can also reason about decisions: "You usually respond within an hour to messages from close friends, but you've left this one for three hours. Your calendar shows back-to-back meetings and you're in a location you don't usually visit. Should I draft a quick reply explaining you're busy?"

**Long context enables decision traces.** MobileLLM-Pro's 128k context window isn't just for document summarization—it's for maintaining the full history of a decision sequence. Why did you book that flight? To understand, the model needs to see your initial search, the options you considered, the price alerts you set, the calendar conflicts you resolved, and the final booking. That's a long context problem.

**Quantization preserves capability.** MobileLLM-Pro achieves int4 quantization with less than 1.3% quality degradation. This means the context graph system can run continuously—always capturing, always reasoning—without draining the battery. The 0.035 J/token efficiency of MobileLLM-350M translates to all-day operation, which is exactly what continuous context capture requires.

**Open training recipes enable ecosystem development.** The MobileLLM-R1 release included complete training recipes and data sources. This isn't just academic transparency—it's ecosystem building. Personal context graphs will require models fine-tuned on specific decision domains (health, finance, relationships), and open recipes make that possible without starting from scratch.

## The Stakes

Enterprise context graphs are widely considered a trillion-dollar opportunity. For personal AI, the market is arguably larger—Gartner predicts [75% of enterprise data will be processed at the edge by 2025](https://blog.premai.io/small-language-models-slms-for-efficient-edge-deployment/), and consumer devices outnumber enterprise deployments by orders of magnitude.

But beyond market size, the personal context graph represents something more fundamental: AI that actually knows you.

Not AI that has been trained on internet text and can simulate knowing you. Not AI that has access to your last ten messages and can guess at your context. AI that has observed your decisions, captured your reasoning, and built a model of *how you think*.

This can only be built on-device. It requires models that can reason, not just respond. And the trajectory from MobileLLM through R1, R1.5, and Pro shows that such models now exist.

---

*The infrastructure for personal context graphs is shipping now. MobileLLM-Pro offers 128k context and outperforms larger models on reasoning benchmarks. MobileLLM-R1 achieves 5× the accuracy of comparable open-source models on mathematical reasoning. The architectural patterns are understood: efficient attention, weight sharing, knowledge distillation. What's missing is the intentional design of systems that capture decision traces—not just actions, but the context that explains them.*

*The trillion-dollar question: who will build it first?*

---

## References

- Liu et al. ["MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"](https://arxiv.org/abs/2402.14905). ICML 2024.
- Meta AI. ["MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes"](https://arxiv.org/abs/2509.24945). September 2025.
- Meta AI. ["MobileLLM-Pro: 1B On-Device Model with 128k Context"](https://howaiworks.ai/blog/mobilellm-pro-announcement). October 2025.
- NVIDIA Research. ["Small Language Models are the Future of Agentic AI"](https://arxiv.org/abs/2506.02153). June 2025.
- Google. ["Introducing Gemma 3n: The Developer Guide"](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/). 2025.
- Apple Machine Learning Research. ["Updates to Apple's On-Device and Server Foundation Language Models"](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates). 2025.
- Mem0. ["AI Memory Layer Guide"](https://mem0.ai/blog/ai-memory-layer-guide). December 2025.
- TechPolicy.Press. ["The Privacy Challenges of Emerging Personalized AI Services"](https://www.techpolicy.press/the-privacy-challenges-of-emerging-personalized-ai-services/).
