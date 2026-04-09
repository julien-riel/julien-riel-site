---
title: "8. Know When to Use an Agent and When to Use a Function"
date: 2026-04-09
tags:
  - agentic-programming
  - working-with-agents
description: "Agents are impressive enough that it's tempting to use them for everything."
---

Agents are impressive enough that it's tempting to use them for everything. They handle ambiguity, generalize across tasks, and can do things no deterministic system could. But they're also slow, expensive, and non-deterministic. A function that parses a date string doesn't need a language model. Using one anyway isn't clever — it's waste dressed up as sophistication.

The distinction is simpler than it looks. If the task has a correct answer that can be computed reliably with code, use code. If the task requires judgment, language understanding, or generalization across cases you can't enumerate, use an agent. The line isn't always clean, but it's cleaner than most teams draw it.

Where teams go wrong is in the middle cases — tasks that feel like they require intelligence but actually don't. Extracting structured data from a consistent format. Routing requests to one of three known categories. Validating that output meets a well-defined spec. These look like agent tasks because they involve text and interpretation. They're actually just pattern matching, and pattern matching is what code is for.

The cost of misclassifying in the agent direction is real. Every agent call has latency — typically seconds, not milliseconds. Every call costs tokens. Every call introduces variance: the same input might produce slightly different outputs on different runs. For a task that a regex or a simple classifier would handle deterministically in microseconds, that tradeoff is never worth it.

There's also a reliability argument. A deterministic function either works or it doesn't, and when it doesn't, you know immediately. An agent that handles the easy cases correctly but drifts on edge cases gives you the illusion of reliability while failing in ways that are hard to catch. Complexity has a way of hiding in the gap between "works most of the time" and "works reliably."

The practical test: could you write a unit test that covers all the cases this task will encounter? If yes, write the function. If the case space is too wide, too ambiguous, or too dependent on context to enumerate — that's when you reach for the agent.

Use the right tool. The agent is a powerful one. Don't reach for it when a screwdriver will do.
