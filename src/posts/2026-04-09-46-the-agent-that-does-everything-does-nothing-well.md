---
title: "46. L'agent qui fait tout ne fait rien de bien"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "There's a fantasy version of an agentic system where one agent handles everything — any question, any task, any domain, with equal competence across all of them."
---

There's a fantasy version of an agentic system where one agent handles everything — any question, any task, any domain, with equal competence across all of them. It's a compelling fantasy because it's simple. One system to build, one system to maintain, one system to explain to stakeholders. The reality is an agent that's mediocre across the board and excellent nowhere, because excellence requires specificity and specificity requires limits.

The mechanism is straightforward. A general-purpose agent needs a system prompt broad enough to cover every case it might encounter. Broad prompts produce broad behavior — the agent has no strong prior about what good looks like in any particular context, so it produces the average of everything it's seen. That average is coherent and fluent and consistently underwhelming. It lacks the sharpness that comes from a system that knows exactly what it's optimizing for.

The evidence shows up in user behavior. Users of general-purpose agents develop workarounds — elaborate prompt rituals designed to push the agent toward the specific behavior they actually want. They learn to specify the role, the format, the constraints, the tone — all the things that a purpose-built agent would have encoded from the start. The user is doing the work of specialization at interaction time, every time, because the system didn't do it at design time.

There's an organizational dimension too. A general-purpose agent has no clear owner. When it fails at a legal task, is that a legal problem or an agent problem? When it underperforms on code review, is the prompt wrong or the model wrong or the use case wrong? Without a defined scope, there's no clear accountability, and without accountability, quality doesn't improve — it just drifts.

The alternative isn't necessarily a proliferation of single-purpose agents with no overlap. It's a portfolio of agents with clear, distinct scopes, each optimized for its domain, orchestrated by something that routes tasks to the right specialist. More complex to build, much better to use.

Generality is a feature in a model. In an agent, it's usually a design gap.
