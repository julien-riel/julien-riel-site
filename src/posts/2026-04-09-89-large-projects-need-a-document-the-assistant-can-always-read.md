---
title: "89. Large Projects Need a Document the Assistant Can Always Read"
date: 2026-04-09
tags:
  - developer-as-user
description: "On a small task, the context you need fits in a prompt."
---

On a small task, the context you need fits in a prompt. On a large project, it doesn't — and every new session starts without the accumulated understanding that makes the assistant useful. The conventions you've established, the architectural decisions you've made, the constraints that apply across the codebase — all of it is gone when the window closes. Without a solution, you spend the first ten minutes of every session re-establishing context you already established yesterday.

The solution is a persistent document — a `ARCHITECTURE.md`, an `AGENT.md`, a `CONTEXT.md` — that you include at the start of every session with the assistant. Not a full specification, but the condensed version of what the assistant needs to know to work effectively in this project: the architectural patterns you're using, the conventions for error handling and naming, the decisions that have been made and shouldn't be revisited, the parts of the codebase that are stable and the parts that are actively changing.

This document is worth maintaining carefully because it pays dividends on every session. Each time you establish a new convention or make a significant architectural decision, add it. Each time the assistant produces something that violates a project constraint you forgot to mention, add that constraint. The document grows as the project grows, and the quality of assistance improves as the document improves.

There's a secondary benefit: the process of writing the document forces clarity about what you actually know about your own project. Constraints that live implicitly in your head are easy to violate. Constraints written down in a document become legible — to the assistant, to new team members, and to yourself when you come back to the project after a month away.

The persistent context document is the memory your assistant doesn't have. Build it early and maintain it.
