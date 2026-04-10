---
title: "88. Utilisez l'assistant pour mettre vos idées à l'épreuve"
date: 2026-04-09
tags:
  - developer-as-user
description: "Before you commit to an implementation approach, describe it to the assistant and ask what could go wrong."
---

Before you commit to an implementation approach, describe it to the assistant and ask what could go wrong. Not "implement this" — "here's what I'm thinking, what are the failure modes?" The assistant has no attachment to your idea, no social incentive to protect your feelings, and a broad knowledge of how similar approaches have failed in similar contexts. It will find things you missed.

This is adversarial prompting applied to your own work, and it's one of the highest-value uses of a coding assistant. The ego-free dynamic that makes the assistant reluctant to criticize — if you don't ask for criticism — becomes a powerful asset when you explicitly invite it. Ask it to steelman the alternative you rejected. Ask it for the three most likely ways this design fails under load. Ask it what a skeptical code reviewer would say about the approach.

The feedback is most useful before the code exists. Once you've written the implementation, sunk-cost dynamics kick in and critical feedback becomes harder to act on even when it's right. Before the code exists, the feedback is pure information — it costs nothing to update your design in response to a critique you can't immediately dismiss.

There's a specific version of this that's especially valuable: ask the assistant to propose an alternative approach and explain the tradeoffs. Not because the alternative is necessarily better, but because understanding why you're not taking it sharpens your understanding of why you are. The best justification for an architectural choice is one you've articulated explicitly, not one that lives in your head as "this seemed right."

Your attachment to your own ideas is the biggest obstacle to improving them. The assistant has none.

---

### Building at Scale with an Assistant
