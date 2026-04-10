---
title: "Fine-Tuning vs. RAG: When to Teach the Model and When to Show It the Answer"
date: 2026-04-09
tags:
  - fine-tuning
  - rag
description: "Fine-tuning changes how the model thinks. RAG changes what it sees. A practical decision framework for when to use each — and when to use both."
---

## A Practical Guide for Agentic Programmers

You have domain-specific data and you want your LLM to use it. Two paths diverge. You can **fine-tune** the model — train it on your data so the knowledge becomes part of its parameters. Or you can use **RAG** — retrieve relevant data at query time and inject it into the prompt. Most teams frame this as a binary choice. It isn't. They solve different problems, and the best systems use both.

## What Each Actually Does

### Fine-Tuning: Changing How the Model Thinks

Fine-tuning takes a pre-trained model and continues training it on your dataset. The model's weights change. Your data becomes part of the model's "memory" — encoded in its parameters, not passed in the prompt.

What fine-tuning gives you:

- **Behavioral patterns.** The model learns *how* to respond, not just *what* to respond with. Tone, format, reasoning style, domain-specific conventions.
- **Implicit knowledge.** The model internalizes patterns it can apply to new inputs it hasn't seen — generalization, not just recall.
- **Latency reduction.** No retrieval step needed. The knowledge is in the model.
- **Smaller prompts.** You don't need to stuff the context window with examples and instructions — the model already knows.

What fine-tuning costs you:

- **Training compute and data.** You need curated, high-quality training examples — typically hundreds to thousands. Garbage in, garbage out.
- **Staleness.** The model's knowledge is frozen at training time. When your data changes, you retrain. That's expensive and slow.
- **Opacity.** You can't cite sources. The model "just knows" but can't point to where it learned it.
- **Catastrophic forgetting.** If done poorly, fine-tuning can degrade the model's general capabilities while improving narrow task performance.

### RAG: Changing What the Model Sees

RAG leaves the model untouched. At query time, you retrieve relevant documents from your data and inject them into the prompt as context. The model reads and responds based on this external evidence.

What RAG gives you:

- **Dynamic knowledge.** Update the data, update the answers. No retraining needed.
- **Citations.** The model can point to the specific document or passage it used. Auditable, verifiable.
- **Data freshness.** New documents are available as soon as they're indexed.
- **No model modification.** Works with any model — proprietary, open-source, swappable.

What RAG costs you:

- **Retrieval latency.** The search step adds time to every query.
- **Context window budget.** Retrieved documents compete for tokens with instructions, conversation history, and other context.
- **Retrieval failures.** If the search returns irrelevant documents, the model generates answers from bad context — potentially worse than no context at all.
- **No behavioral change.** RAG doesn't change how the model reasons or what style it uses. It only changes what information is available.

## The Decision Framework

The question isn't "fine-tuning or RAG?" It's "what problem am I actually solving?"

### Use RAG When the Problem Is Knowledge

If the model needs access to specific, citable facts that change over time — product catalogs, policy documents, customer data, regulatory text, knowledge bases — RAG is the right tool. The model doesn't need to internalize this knowledge. It needs to read it and reason about it.

WHOOP Coach is a RAG system. The LLM doesn't "know" your heart rate variability from training. It retrieves your biometric data at query time, reads it, and generates personalized coaching. When your data changes (every night, after every workout), the answers update automatically. Fine-tuning a model on one person's health data would be absurd.

**RAG is the right choice when:**
- Your data changes frequently (daily, weekly, monthly)
- You need to cite specific sources
- The data is per-user or per-tenant (can't be baked into one model)
- Compliance requires auditability of where answers came from
- You want model-agnostic architecture (swap models without retraining)

### Use Fine-Tuning When the Problem Is Behavior

If the model needs to reason differently, write in a specific style, follow domain-specific conventions, or handle specialized formats — fine-tuning changes the model's "instincts." RAG can't do this. You can paste a style guide into every prompt, but the model will always be fighting its default tendencies.

Cursor fine-tuned Llama-3-70B specifically for code application (the Fast Apply model). The task — taking a semantic diff and producing the correct full file — requires a specific behavioral pattern that's hard to prompt into a general model. The fine-tuned model achieves 1,000 tokens/second because it's optimized for this one task. RAG would add latency and wouldn't help — the model needs to know *how* to apply code changes, not *what* changes to apply.

**Fine-tuning is the right choice when:**
- You need consistent style, tone, or format that prompting can't reliably achieve
- The model needs domain-specific reasoning patterns (medical diagnosis logic, legal analysis conventions, code transformation rules)
- You want to reduce prompt size (and therefore cost and latency) by internalizing repeated instructions
- You have a well-defined, stable task that doesn't change frequently

### Use Both When the Problem Is Complex

Most real-world systems need both. Fine-tuning handles behavior; RAG handles knowledge.

Duolingo fine-tuned GPT-4 for its specific interaction patterns (gamified feedback, pedagogical scaffolding, Duolingo's voice) while using retrieval to access lesson content, grammar rules, and learner-specific data. The fine-tuning ensures the model acts like a Duolingo tutor. The retrieval ensures it teaches the right content for the right learner.

Cursor uses a fine-tuned model (Composer) for agentic coding behavior and RAG (the context engine with embeddings and reranking) for codebase knowledge. The fine-tuning teaches the model how to write code, use tools, and apply edits. The RAG gives it access to the specific codebase it's working on.

**Use both when:**
- The model needs specialized behavior (fine-tune) AND access to dynamic data (RAG)
- You're building a domain-specific assistant that should both sound right and know the right things
- You want the efficiency of internalized patterns with the freshness of retrieved knowledge

## Practical Comparison

| Dimension | Fine-Tuning | RAG |
|-----------|-------------|-----|
| **What changes** | Model weights | Prompt content |
| **Knowledge freshness** | Frozen at training time | As fresh as the index |
| **Setup cost** | High (data curation, training) | Medium (chunking, embedding, indexing) |
| **Per-query cost** | Lower (no retrieval step) | Higher (retrieval + larger prompts) |
| **Latency** | Lower | Higher (retrieval round-trip) |
| **Citability** | No — the model "just knows" | Yes — can cite source documents |
| **Data changes** | Requires retraining | Re-index and serve |
| **Model portability** | Locked to one model | Works with any model |
| **Best for** | Behavior, style, reasoning patterns | Facts, data, domain knowledge |

## Common Mistakes

### Fine-Tuning on Facts

Teams fine-tune models to "know" their product documentation, FAQ, or policy data. This works — until the documentation changes. Then you retrain, which takes time and money, and the old answers linger in production until the new model is deployed. RAG handles dynamic factual knowledge more gracefully.

Fine-tune for behavior. RAG for facts. This is the simplest heuristic and it's right most of the time.

### RAG Without Evaluation

Teams build a RAG pipeline, test it on a handful of questions, and ship it. Without systematic evaluation of retrieval quality (are you getting the right chunks?) and generation quality (is the model using them correctly?), you're flying blind. WHOOP built an evaluation framework that tests retrieval, generation, and end-to-end quality separately. Your RAG system needs the same.

### Fine-Tuning as a Substitute for Good Prompting

Before you fine-tune, make sure you've exhausted what prompting can do. Fine-tuning is expensive and irreversible (in the sense that you're now maintaining a custom model). Many "fine-tuning" use cases can be solved with better system prompts, few-shot examples, or structured output formats.

The rule of thumb: if you can describe the behavior you want in natural language and the model can follow it with good prompting, you don't need fine-tuning. If the behavior requires hundreds of examples to demonstrate because it's too subtle or complex for instructions, that's when fine-tuning earns its keep.

### Ignoring the Middle Ground: Prompt Caching

Many API providers now offer prompt caching — reusing the processed context from previous requests when the prompt prefix is identical. This gives you some of fine-tuning's latency benefits (the model has already "read" your context) with RAG's flexibility (the context can be updated). If your system prompt and retrieved context are relatively stable across requests, prompt caching can significantly reduce latency and cost without any training.

## A Decision Tree

```
Q: Does the model need to access specific, changing data?
├── Yes → You need RAG (at minimum)
│   Q: Does it also need specialized behavior/style?
│   ├── Yes → RAG + Fine-tuning
│   └── No → RAG + Good prompting
└── No → 
    Q: Does the model need to reason or behave differently than its defaults?
    ├── Yes → Fine-tuning (or very strong prompting first)
    └── No → You might not need either — just prompt well
```

## The Takeaway

Fine-tuning and RAG are complementary, not competing. Fine-tuning changes the model's instincts — how it reasons, writes, and behaves. RAG changes the model's knowledge — what it can reference when answering. Most real-world agentic systems need both: a model that thinks the right way about the specific information it retrieves at query time.

Start with RAG. It's cheaper, faster to iterate, and doesn't lock you into a specific model. Add fine-tuning when you've identified a behavioral gap that prompting can't close. And always — always — build your evaluation framework before you invest in either.

## Further Reading

- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning) — Official documentation on when and how to fine-tune
- [WHOOP — Delivering LLM-powered Health Solutions](https://openai.com/index/whoop/) — RAG in production for personalized health coaching
- [Cursor Composer 2](https://anthemcreation.com/en/artificial-intelligence/cursor-composer-2-proprietary-coding-ai-model/) — Fine-tuning + RL for specialized coding behavior
- [Duolingo's AI Evolution](https://openai.com/index/duolingo/) — Fine-tuning GPT-4 for pedagogical interaction patterns combined with content retrieval
