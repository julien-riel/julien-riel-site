---
title: "55. Agentic Programming Rewards the Lazy Thinker"
date: 2026-04-09
tags:
  - mindset
description: "Lazy, here, is a technical term."
---

Lazy, here, is a technical term. The lazy thinker is the one who asks: what's the simplest version of this that could work? What can I not build and still solve the problem? Where am I adding complexity that isn't earning its keep? This is the disposition that produces clean systems, and it's unusually valuable in agentic programming because the temptation toward unnecessary complexity is unusually strong.

Agents make complexity cheap to add. You can wire in another tool, extend the system prompt, add another agent to the pipeline — all without writing much code. The cost of adding capability feels low. The cost of the complexity you've added doesn't show up until you're debugging a production failure and you can't tell which of the seven components contributed to it.

The lazy approach starts with the smallest possible system. One agent, minimal tools, a prompt that does the least it needs to. Run it. See where it fails. Add exactly what's needed to address the failure — nothing more. This isn't iterative development as a methodology; it's iterative development as a discipline against the impulse to anticipate problems that haven't occurred yet.

The industrious thinker builds the system they imagined. The lazy thinker builds the system the problem actually requires, which is almost always smaller. Imagined requirements are generous. Real requirements are constrained. The gap between them is waste — complexity that consumes maintenance time, introduces failure modes, and makes the system harder to reason about without making it better at the actual task.

There's also a cognitive economy argument. Agentic systems require genuine mental effort to understand — the probabilistic behavior, the context dynamics, the interaction between components. Every unnecessary component is more surface area your brain has to hold. The lazy system is easier to debug, easier to explain, easier to hand off, and easier to improve because you can actually see all of it.

The lazy thinker asks what can be removed. The answer is usually more than expected.
