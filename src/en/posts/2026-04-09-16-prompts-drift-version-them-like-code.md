---
title: "16. Prompts Drift — Version Them Like Code"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "A prompt that works today will not necessarily work tomorrow."
---

A prompt that works today will not necessarily work tomorrow. Models get updated. Your application evolves. The edge cases your prompt was tuned to handle give way to new ones. Someone tweaks the wording to fix one behavior and inadvertently breaks another. Six months later, nobody can explain why the prompt says what it says, and changing it feels risky because nobody knows what it's holding together.

This is software decay, and it happens to prompts for the same reasons it happens to code: they accumulate changes without documentation, they become load-bearing without anyone declaring them so, and the context that made them sensible at the time evaporates along with the people who wrote them.

Version control is the obvious fix. Prompts belong in repositories, with commit messages that explain not just what changed but why. "Made the tone more formal" is a poor commit message. "Made the tone more formal after customer feedback indicated the previous register felt too casual for enterprise users" is a design document. Future you — or the colleague who inherits this system — needs the why, not just the what.

Beyond version control, prompts benefit from the same review culture as code. Changes to system prompts should go through review, especially for production systems. The reviewer isn't checking grammar — they're asking whether this change could affect behavior in ways the author didn't anticipate. A one-line prompt change can have broad effects that aren't obvious until they surface in production.

The more invisible problem is the prompt that drifts without anyone noticing. Nobody changed the file. The model changed. A system prompt that was calibrated against one version of a model may behave differently against the next — subtly, in ways that don't trigger obvious errors but shift the output distribution in directions nobody intended. Catching this requires evaluation: running your prompts against a test set and comparing outputs across model versions.

Treat the prompt as source code. It has the same fragility, the same need for documentation, and the same capacity to become unmaintainable if you don't take care of it from the start.

What's not versioned is already lost.
