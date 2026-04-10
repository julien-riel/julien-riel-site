---
title: "86. Break Large Tasks into Prompts, Not Just Steps"
date: 2026-04-09
tags:
  - developer-as-user
description: "A prompt asking for five hundred lines of code is asking the assistant to make dozens of design decisions without knowing which ones you've already made, which ones are constrained by the rest of t..."
---

A prompt asking for five hundred lines of code is asking the assistant to make dozens of design decisions without knowing which ones you've already made, which ones are constrained by the rest of the codebase, and which ones you care about. The output will be technically coherent but architecturally disconnected from your intentions. You'll spend more time editing it into shape than if you'd broken the task into smaller pieces.

Smaller prompts produce better outputs for the same reason small functions are better than large ones: each unit has a single, clear responsibility. A prompt that asks for one thing — "implement the validation logic for this form, returning a typed result for each field" — can be precise about the constraints and can produce something you can evaluate completely. A prompt that asks for the entire form handling layer produces something you can only partially evaluate until it's all there, at which point changing the foundation is expensive.

The decomposition should follow the natural seams of the problem: the layers of the architecture, the separation between data transformation and side effects, the boundary between business logic and infrastructure. These are the same seams you'd use to decompose the task for a junior developer. The assistant responds well to the same structure.

There's also an attention economy argument. A large prompt asks the model to hold many constraints in attention simultaneously, and at the edges of that window things drift — later code doesn't fully respect constraints established early in the output. Smaller prompts reset that window cleanly and let you validate at each boundary before moving forward.

Decompose before you prompt. The prompt is the specification of one unit of work.
