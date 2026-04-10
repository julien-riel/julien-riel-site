---
title: "60. Apprenez à lire l'échec comme un détective, pas comme un juge"
date: 2026-04-09
tags:
  - mindset
description: "When an agent fails, the instinct is to assign blame."
---

When an agent fails, the instinct is to assign blame. The model hallucinated. The prompt was bad. The retrieval missed. You pick the culprit, fix it, and move on. This feels like debugging. It's actually just pattern matching with a verdict attached.

A detective doesn't start with a verdict. A detective starts with evidence and works backward. What actually happened? What does the log show? What was in the context window when things went wrong? What did the agent do just before it failed? The questions are specific and the answers are descriptive before they're evaluative.

This distinction matters because agent failures are usually overdetermined. The model did hallucinate, and the retrieval did miss, and the prompt was ambiguous, and the user's input was unusual, and all four of those things together produced the failure. If you pick one culprit and fix it, you may not have actually fixed anything — you've just changed which combination of factors will cause the next failure.

The judge mindset also creates a subtle organizational problem. If blame lands on the model, the response is to switch models. If blame lands on the prompt, the response is to rewrite it. These interventions are sometimes right, but they're often premature, made before you actually understand what happened. A team that regularly misdiagnoses failures builds a codebase full of fixes to problems they didn't have.

Diagnosis before intervention. Evidence before conclusion. The discipline is to stay curious for longer than feels comfortable — to resist the pull toward the fix until you're confident you understand what you're fixing.

Practically, this means logging more than you think you need to. It means building tools that let you replay agent runs with different inputs. It means writing up failure post-mortems that describe what happened, not just what was changed. The goal is a team that accumulates real understanding of how their systems fail, not just a growing list of patches.

Agent systems fail in combinations. The developers who get better at them are the ones who develop a taste for the whole picture — who can look at a failure and see not one broken thing, but a set of conditions that aligned badly, and then design against the conditions rather than just the symptom.

The culprit is rarely who you thought. The case is always more interesting than it first appears.
