---
title: "2. Le prompt est l'architecture"
date: 2026-04-09
tags:
  - working-with-agents
description: "Most developers treat the prompt as an afterthought — a thing you write once, probably badly, then tweak when something breaks."
---

Most developers treat the prompt as an afterthought — a thing you write once, probably badly, then tweak when something breaks. That's the wrong mental model. The prompt is the architecture. Change the prompt and you change the system. Get it wrong and no amount of clever infrastructure will save you.

This is counterintuitive because prompts look like text, and text feels informal. It doesn't feel like you're making a structural decision when you write one. But you are. You're defining what the agent knows about its role, what it pays attention to, what it ignores, how it formats its output, and what it does when things get ambiguous. A poorly structured prompt doesn't just produce worse outputs — it produces unpredictable outputs, which is worse.

The analogy that holds up is interface design. A well-designed API is explicit about its contracts: what inputs are valid, what outputs to expect, how errors are communicated. A well-designed prompt does the same work. It tells the agent what context it's operating in, what good output looks like, and what to do at the edges. A vague prompt is a leaky interface — it works when conditions are ideal and fails in ways you won't anticipate when they aren't.

Consider what happens when you add a new tool to an agent without updating the system prompt. The tool is available, but the agent has no mental model for when to use it, or why, or how it fits into the broader task. You've changed the capability of the system without changing the architecture that governs it. This is the agentic equivalent of adding a column to a database without updating the schema — technically possible, reliably problematic.

The best practitioners treat prompt writing as a first-class engineering activity. They version their prompts. They test changes systematically. They document the reasoning behind design decisions — not just what the prompt says, but why it says it that way. When something breaks, they look at the prompt first, not the infrastructure.

There's a discipline here that most teams arrive at late: separating what the agent is supposed to do (the task definition) from how it should do it (the behavioral constraints) from what it should know (the context). Conflating these produces prompts that are hard to maintain and harder to debug. Separating them produces prompts that can be reasoned about — and changed — with confidence.

The field moves fast, and the temptation is to think about which model to use, which orchestration framework to adopt, which vector database to wire in. Those choices matter. But a mediocre prompt on good infrastructure still gives you a mediocre agent. A precise, thoughtful prompt makes every other part of the system work harder.

Write the prompt like you're writing the spec. Because you are.
