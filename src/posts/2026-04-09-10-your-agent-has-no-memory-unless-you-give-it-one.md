---
title: "10. Your Agent Has No Memory Unless You Give It One"
date: 2026-04-09
tags:
  - agentic-programming
  - working-with-agents
description: "Every time you call a language model, it starts fresh."
---

Every time you call a language model, it starts fresh. It has no recollection of the last conversation, the last task, the last mistake it made or the correction you gave it. The context window is the entirety of what it knows. When the window closes, everything in it is gone.

This is the part of agent architecture that surprises people the most, and keeps surprising them even after they know it intellectually. The agent seemed to understand the project. It seemed to have a feel for the codebase. Then you start a new session and it's a stranger again, asking questions you already answered last week.

Memory in agentic systems is an engineering problem you have to solve explicitly. There are a few approaches, each with tradeoffs. You can extend the context window — keep adding conversation history until it fits. This works until it doesn't: context windows have limits, long contexts slow inference down, and models tend to lose track of information from the early parts of a long context. You can use retrieval — store past interactions in a vector database and pull in the relevant pieces at the start of each new session. This scales better but requires you to get retrieval right, which is its own problem. You can maintain structured state — a document or database that captures the key facts you want the agent to carry forward, updated explicitly after each session.

The right approach depends on what kind of memory you need. There's a difference between episodic memory — what happened in past sessions — and semantic memory — what facts the agent should know about the domain. There's a difference between memory that needs to be exact and memory that just needs to be approximately right. Designing for memory means being specific about what needs to persist, why, and at what fidelity.

The mistake teams make is assuming memory will emerge from the model. It won't. The model is stateless by design. If your agent needs continuity across sessions, you have to build it, maintain it, and pass it in explicitly every time.

The agent remembers nothing. You decide what it gets to keep.
