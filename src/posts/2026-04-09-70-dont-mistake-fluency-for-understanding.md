---
title: "70. Ne confondez pas la fluidité avec la compréhension"
date: 2026-04-09
tags:
  - mindset
description: "You can become fluent with agentic systems without understanding them."
---

You can become fluent with agentic systems without understanding them. Fluency means you know the patterns — how to structure a prompt, which frameworks to reach for, what the common failure modes look like and how to patch them. Understanding means you know why the patterns work, what they're actually doing, and what to do when the pattern breaks down. The gap between these two is invisible until you hit a problem that falls outside the patterns you know.

The fluency trap is particularly easy to fall into in a field that moves fast and rewards people who can ship things quickly. You learn the practices that work, you apply them reliably, you build a reputation for knowing what you're doing. And you do know what you're doing — within the space of problems that resemble the ones you've already solved. The novel problem reveals the gap.

Understanding in this field means having a mental model of what's actually happening when an agent processes a prompt. Not the mathematical details of transformer attention, but the functional understanding: what the model is doing when it generates text, why context placement matters, why examples outperform instructions, why the same prompt behaves differently across models. This understanding is what lets you reason about new problems rather than pattern-match to old solutions.

The test is whether you can explain why something works. Not just that it works — why. If you can't explain why your prompt structure produces better results than the alternative, you're operating on superstition. It might be reliable superstition — the pattern works consistently enough that the lack of understanding doesn't hurt you today. But superstition doesn't generalize. Understanding does.

The path from fluency to understanding runs through deliberate examination of the things you do automatically. Why does this section of the system prompt come first? What would happen if it came last? Why do you use three examples rather than one? What does each additional example add? These questions feel pedantic when the system is working. They're essential when it isn't.

Know that it works. Know why it works. The second is harder and worth more.
