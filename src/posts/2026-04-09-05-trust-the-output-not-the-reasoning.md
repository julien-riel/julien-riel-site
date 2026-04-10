---
title: "5. Faites confiance au résultat, pas au raisonnement"
date: 2026-04-09
tags:
  - working-with-agents
description: "Chain-of-thought reasoning is genuinely useful — it improves output quality, makes the agent's process more legible, and gives you something to debug when things go wrong."
---

Chain-of-thought reasoning is genuinely useful — it improves output quality, makes the agent's process more legible, and gives you something to debug when things go wrong. But it creates a trap: the reasoning looks so coherent that you start trusting it. You read through the agent's step-by-step logic, it makes sense, and you conclude the output must be correct. This is backwards.

The reasoning is not a window into what the model is doing. It's another output. The model generates the reasoning the same way it generates everything else — by predicting likely tokens given the context. That reasoning can be internally consistent, logically structured, and completely disconnected from the actual computation that produced the final answer. Models have been shown to produce confident, coherent justifications for answers that are flat wrong. The explanation sounds good. The answer is still wrong.

This matters because the reasoning creates false confidence in a specific way. When an agent produces a bare answer and it's wrong, you see a wrong answer. When it produces a beautifully reasoned wrong answer, you see a convincing argument. The second is harder to catch and easier to defer to. Especially when you're moving fast, when the domain is unfamiliar, when the reasoning covers ground you'd have to think hard to verify independently.

The discipline is to evaluate outputs on their own terms — does the output match reality, meet the spec, pass the tests — not on the quality of the reasoning that preceded them. Treat the reasoning as a useful debugging artifact, not as evidence of correctness. If the output is wrong, the reasoning tells you where to look. If the output is right, the reasoning is interesting but not the point.

There's a related mistake in the other direction: distrusting an output because the reasoning seems wrong. Sometimes models arrive at correct answers through reasoning chains that look odd or take unnecessary detours. The reasoning is a sketch, not a proof. What matters is whether the answer holds up when you check it against ground truth.

Verify outputs. Use reasoning to understand failures. Don't let a good argument substitute for a correct answer.

The model is not showing its work. It's generating the appearance of showing its work. Keep that distinction close.
