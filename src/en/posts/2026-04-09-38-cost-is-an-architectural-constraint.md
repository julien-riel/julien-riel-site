---
title: "38. Cost Is an Architectural Constraint"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Token costs have a way of surprising teams that didn't plan for them."
---

Token costs have a way of surprising teams that didn't plan for them. The prototype runs cheaply because it processes a handful of requests a day. Production runs ten thousand. The context windows are large because someone decided more context was always better. The retry logic runs three attempts by default. Nobody added up what that means at scale, and the first billing cycle is educational in a way nobody wanted.

Cost is not an operational concern you address after the architecture is set. It's a constraint that shapes the architecture from the start — as real as latency, reliability, or correctness. A system that produces great outputs but costs ten dollars per user interaction is not a viable system, regardless of how impressive the demo looks.

The levers are well-defined once you know where to look. Model selection is the biggest one: the difference in cost between a large frontier model and a smaller, faster model can be an order of magnitude, and for many tasks the smaller model is good enough. Context window size is the second: every token in the window costs money, and bloated contexts — full conversation histories, over-retrieved documents, verbose system prompts — add up quickly. Task decomposition is the third: a large agent that handles everything in one call may cost more than a pipeline of smaller, cheaper agents where only the final step uses the expensive model.

The discipline is to instrument cost from day one. Know what each agent call costs. Know what each tool call costs. Know what the total cost per task is across the pipeline. Without this instrumentation, you're optimizing blind — you can't make good tradeoffs between capability and cost because you don't know what anything costs.

There's also a design smell worth watching for: the system where cost is invisible to the people making design decisions. When developers can experiment freely without seeing the cost of their experiments, they optimize for capability and convenience. When cost is visible — in dashboards, in per-request breakdowns, in monthly summaries — tradeoffs get made more carefully.

Build for what it costs to run, not just what it costs to demo.
