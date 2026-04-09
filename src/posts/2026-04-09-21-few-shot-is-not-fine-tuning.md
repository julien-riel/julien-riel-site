---
title: "21. Few-Shot Is Not Fine-Tuning"
date: 2026-04-09
tags:
  - agentic-programming
  - prompting-as-engineering
description: "Few-shot prompting — providing examples in the context window to shape model behavior — is powerful and widely used."
---

Few-shot prompting — providing examples in the context window to shape model behavior — is powerful and widely used. It's also widely misunderstood. Developers who get good results from few-shot examples sometimes conclude they've effectively customized the model. They haven't. They've influenced a single inference. The difference matters enormously when you're designing a system that needs to be reliable at scale.

Fine-tuning changes the model's weights. The learned behavior is baked in — it generalizes across inputs, persists across sessions, and doesn't consume context window space. Few-shot prompting changes nothing about the model. It provides examples that influence the current generation, and when the context window closes, the influence closes with it. Every new call starts from the base model again.

This means few-shot examples have to travel with every request. In a high-volume system, that's a real cost — tokens spent on examples are tokens not spent on task-relevant content. It also means the examples are subject to context window dynamics: in a long conversation, early examples can lose influence as later content pushes them further from the generation point.

The more consequential misunderstanding is about generalization. Few-shot examples teach the model a pattern for the cases you showed it. Fine-tuning teaches the model something more durable — a behavior that generalizes across the distribution of inputs it will encounter. If your use case requires consistent behavior across a wide variety of inputs, few-shot prompting may give you false confidence: it works on the examples you tested and degrades on inputs that don't closely resemble them.

None of this means few-shot prompting isn't valuable — it's often the right tool, especially for format control, style matching, and tasks where you have a few representative examples. But it's a prompting technique, not a training technique. Expecting it to behave like one will lead you to invest in examples when you should be investing in evaluation, or to skip fine-tuning when the task actually warrants it.

Know what the tool does. Use it for what it's for.
