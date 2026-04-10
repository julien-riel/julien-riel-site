---
title: "58. La vitesse d'itération est votre avantage compétitif"
date: 2026-04-09
tags:
  - mindset
description: "The developers who improve fastest in agentic programming are not the ones who think most carefully before they act — they're the ones who act, observe, and adjust in the shortest cycles."
---

The developers who improve fastest in agentic programming are not the ones who think most carefully before they act — they're the ones who act, observe, and adjust in the shortest cycles. The field is too new and the systems too complex for pure reasoning to substitute for empirical feedback. You have to run things to know how they behave.

This sounds obvious but cuts against habits that are well-established in software engineering. In conventional software, thinking carefully before writing code is usually right — the cost of refactoring is real and the compiler will tell you if you're wrong anyway. In agentic systems, the failure modes are probabilistic, the behavior is context-dependent, and the only way to know if something works is to run it against enough inputs to see the distribution of outputs. Careful thinking before running is useful. It doesn't substitute for running.

The practical implication is to instrument your iteration loop. How quickly can you change a prompt and see results? How quickly can you run your eval suite and get a quality signal? How quickly can you get a representative sample of outputs from a new configuration? The faster this loop, the more experiments you can run, the more quickly you converge on what works. Teams with slow iteration loops tend to make big bets because small experiments are too expensive. Teams with fast loops make many small bets and let the evidence guide them.

There's a related point about the size of changes. Large prompt rewrites make it hard to know which change produced the observed effect. Small, targeted changes — one variable at a time — produce cleaner signal. The discipline of changing one thing and measuring the effect is the discipline of scientific thinking applied to system improvement. It's slower per experiment and faster overall, because you're accumulating understanding rather than just accumulating changes.

The field moves fast. Your ability to move with it depends on how quickly you can learn from what you build.

Act. Observe. Adjust. Repeat faster than everyone else.
