---
title: "27. Les évaluations sont votre suite de tests"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Every serious software project has tests."
---

Every serious software project has tests. Agentic systems need tests too — they're just harder to write, which is exactly why most teams skip them and then wonder why they can't tell if a change made things better or worse.

The difficulty is that agent outputs aren't always right or wrong in a binary sense. A code function either passes its tests or it doesn't. A generated summary either captures the key points or it doesn't — but "captures the key points" isn't a predicate you can evaluate automatically. This ambiguity is real, and it causes teams to throw up their hands and rely on vibes. Vibes don't scale.

Evaluations — evals — are the testing infrastructure for probabilistic systems. They consist of a set of inputs with known-good outputs or quality criteria, a method for scoring agent outputs against those criteria, and a process for running the eval whenever something changes. The scoring doesn't have to be fully automated; human evaluation is legitimate and often necessary. What matters is that the process is systematic, repeatable, and runs before you ship.

Building a good eval suite starts with collecting failures. Every time the agent produces a bad output in production or testing, that input goes into the eval set. Over time you accumulate a collection of hard cases — the inputs that break things, the edge cases that weren't anticipated, the scenarios where the agent does something plausible but wrong. That collection is more valuable than any synthetic test suite, because it represents the actual distribution of ways your system fails.

The second component is golden outputs — examples of what good looks like for a representative range of inputs. These define your quality bar concretely. When you change a prompt or upgrade a model, you run the eval and check how many golden outputs you still match. Regressions are visible. Improvements are measurable.

Teams that build evals early ship with more confidence and improve faster. Teams that don't build evals are always guessing — about whether the new model is better, about whether the prompt change helped, about whether the system is degrading in production.

You wouldn't ship code without tests. Don't ship agents without evals.
