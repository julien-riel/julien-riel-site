---
title: "72. The First Version Should Be Embarrassingly Simple"
date: 2026-04-09
tags:
  - mindset
description: "Every lasting principle in software has a version of this at its core."
---

Every lasting principle in software has a version of this at its core. Start simple. Ship early. Learn from real use. The specific failure mode it's preventing is always the same: the system designed in the absence of evidence, built to handle requirements that turned out not to be real, complex in ways that cost maintenance time without adding user value.

In agentic programming, this principle is more important and more frequently violated than almost anywhere else. The tooling makes complexity cheap to add. The demos of sophisticated multi-agent systems make simple single-agent solutions feel inadequate. The field moves fast and there's a pressure to use the latest techniques, to build the architecture that will scale, to solve problems you don't have yet. The result is first versions that are several versions ahead of what the evidence justifies.

The embarrassingly simple first version is a single agent with a minimal prompt, a small number of tools, no complex orchestration, and a human review step for anything consequential. It probably handles only the most common case. It probably fails on inputs outside that case in ways that are obvious and recoverable. It probably doesn't impress anyone who sees it. It also runs, produces real outputs, and generates the evidence that every subsequent design decision should be based on.

The evidence you get from the simple version is irreplaceable. Real users interact with it in ways you didn't anticipate. The common case turns out to be slightly different from what you assumed. The failure modes are different from the ones you designed around. The thing you thought would be the hard problem isn't, and something you didn't think about at all is. You can't know any of this without running the system, and the simple version runs sooner, cheaper, and with less to unwind when you need to change direction.

There's also something clarifying about the constraint of simplicity. When you allow yourself to build the complex version immediately, you defer the hard question of what the system actually needs to do. Simplicity forces the answer. One agent means one job. One prompt means one clear scope. One human review step means one explicit judgment about what can and can't be trusted to the machine. The constraints reveal the design.

The embarrassingly simple version isn't the final version. It's the version that earns the right to the next one.

Start there. The sophistication will come when it's deserved.

---

---

## Part 6 — The Developer as User

*For developers who work alongside AI coding assistants — Claude Code, GitHub Copilot, Cursor, and their successors.*

---

### Working with the Assistant
