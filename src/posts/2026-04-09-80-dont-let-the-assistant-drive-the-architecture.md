---
title: "80. Don't Let the Assistant Drive the Architecture"
date: 2026-04-09
tags:
  - agentic-programming
  - developer-as-user
description: "The assistant is excellent at implementing decisions."
---

The assistant is excellent at implementing decisions. It is not the right entity to make them. This distinction collapses quickly in practice — you ask for an implementation, the implementation implies an architecture, you accept the implementation, and the architecture is now in your codebase without having been deliberately chosen.

The failure mode is subtle because the assistant's architectural choices are usually reasonable. It picks well-known patterns, uses standard abstractions, makes conventional decisions. The problem isn't that the choices are bad in the abstract — it's that they might not be right for your specific context, your team's conventions, your system's constraints, your long-term direction. The assistant doesn't know any of that unless you've told it, and architectural decisions are exactly the kind of thing that's hard to fully specify in a prompt.

The practical rule is to make the architectural decision before you prompt, not after. Before asking the assistant to implement a new feature, decide how it should fit into the existing structure. Before asking it to add a new module, decide where that module belongs and how it communicates with its neighbors. The prompt should specify the architecture. The assistant should implement it.

When you're not sure what the right architecture is — which is often — that's a signal to do the design work first, not to delegate it to the assistant. Ask the assistant to help you think through options, describe the tradeoffs, identify the constraints you might be missing. Use it as a thinking partner in the design process. But make the decision yourself, explicitly, before the code exists.

The assistant builds what you give it to build. Make sure you're the one deciding what to build.
