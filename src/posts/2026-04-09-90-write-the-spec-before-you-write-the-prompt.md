---
title: "90. Write the Spec Before You Write the Prompt"
date: 2026-04-09
tags:
  - agentic-programming
  - developer-as-user
description: "For a small task — fix this bug, add this field — the prompt can be the spec."
---

For a small task — fix this bug, add this field — the prompt can be the spec. For anything significant — a new module, a new API, a substantial refactoring — prompting without a spec produces code that implements what you asked for rather than what you needed. Those are the same thing only when you've thought carefully about what you need, which is the work that writing a spec requires.

A spec doesn't have to be a formal document. It can be a short prose description: what this thing does, what it doesn't do, how it fits into the existing system, what the edge cases are, what success looks like. The discipline of writing it down — before the code exists — forces the decisions that prompting tries to skip. Where does the data come from? How are errors surfaced? What happens when the dependency is unavailable? Writing the spec surfaces these questions. Prompting buries them.

The spec also serves as the reference point for review. When the assistant produces an implementation, the question isn't "does this look reasonable?" — it's "does this implement the spec?" Reasonable-looking code that doesn't implement the spec is a failure. Spec-driven review is faster and more reliable than intuition-driven review because the target is explicit.

On a collaborative team, the spec is also communication — it's how you establish alignment on what's being built before code exists, when changing direction is cheap. A prompt sent directly to an assistant before the team has aligned on the spec is a way of generating code you might have to throw away.

Write the spec. The prompt is how you hand it to the assistant.
