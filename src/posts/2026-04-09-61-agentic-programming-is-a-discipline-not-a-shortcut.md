---
title: "61. La programmation agentique est une discipline, pas un raccourci"
date: 2026-04-09
tags:
  - mindset
description: "The pitch for agentic programming often sounds like a promise of less work."
---

The pitch for agentic programming often sounds like a promise of less work. You describe what you want, the agent does it, you review and ship. Less code, less debugging, less of the tedious work that slows everything down. There's truth in this — agents do reduce certain kinds of work significantly. The mistake is concluding that less of one kind of work means less work overall.

The work that agents reduce is largely execution work — the translation of a well-understood specification into working code or content. This is real labor and agents handle it well. The work that agents don't reduce — and in some ways increase — is the thinking work: understanding the problem clearly enough to specify it, designing the system thoughtfully enough to be maintainable, reviewing outputs carefully enough to catch what went wrong, building the evaluation infrastructure to know if things are improving or degrading.

In fact, agentic programming raises the bar on thinking work. When execution is cheap, the bottleneck moves to specification. The developer who could get away with a fuzzy mental model of the problem — because the implementation would reveal the gaps quickly and cheaply — now needs a sharper model upfront, because the agent will faithfully execute the fuzzy specification and produce something that looks complete but isn't right. The tax on unclear thinking is higher, not lower.

The discipline shows up in the practices that distinguish teams that ship reliable agentic systems from teams that ship impressive demos. Evals. Versioned prompts. Observability infrastructure. Careful scope definition. Human checkpoints for consequential actions. None of these are shortcuts — they're the engineering rigor that makes the system trustworthy rather than just functional.

Developers who approach agentic programming as a shortcut tend to accumulate technical debt in exactly the places where agent systems are most fragile: prompt management, output validation, failure handling. The demo worked. The production system doesn't, not reliably, and now they're doing the engineering work they deferred, under pressure, with a live system that's already behaving badly.

The shortcut is a loan. The discipline is what makes it worth taking.
