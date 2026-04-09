---
title: "28. The Tool Is the Interface"
date: 2026-04-09
tags:
  - agentic-programming
  - building-agentic-systems
description: "When you give an agent a tool, you're not just extending its capabilities — you're defining the boundary between what the agent decides and what the world does."
---

When you give an agent a tool, you're not just extending its capabilities — you're defining the boundary between what the agent decides and what the world does. That boundary is the most consequential design decision in an agentic system, and most teams make it without realizing they're making it at all.

A tool is an interface in the fullest sense. It has a contract: inputs it accepts, outputs it returns, errors it can produce. It has semantics: what it means to call it, what state it changes, what it assumes about the world before the call and guarantees about the world after. A well-designed tool makes the agent's job clearer. A poorly designed one introduces ambiguity that the agent will resolve unpredictably.

The most common tool design mistake is making tools too broad. A tool called `execute_action` that takes a free-form string and does whatever it parses out of that string is not a tool — it's a delegation of interface design to the model. The model will use it inconsistently because there's no contract to be consistent with. A tool called `send_email` with explicit parameters for recipient, subject, and body is a real interface. The model knows what to provide and what to expect back.

Narrow tools compose better than broad ones. An agent with ten specific tools — each doing one thing well — is more reliable and more debuggable than an agent with two omnibus tools. When something goes wrong, you can ask which tool was called and with what parameters. The failure is localized. With broad tools, the failure is somewhere inside the tool's interpretation of a free-form input, which is much harder to find.

Tool design also determines blast radius. A read-only tool that fetches data can be called freely — if it fails or returns wrong data, the damage is limited to the current task. A tool that modifies state — writes to a database, sends a message, executes a payment — carries real-world consequences that can't be undone. These tools deserve extra care in their design: explicit confirmation parameters, idempotency guarantees, clear error states that the agent can reason about.

The agent is only as good as the tools you gave it. Design them like the interfaces they are.
