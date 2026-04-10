---
title: "51. Know What Your Agent Cannot Know"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Every agent has an epistemic boundary — a line between what it can know and what it cannot."
---

Every agent has an epistemic boundary — a line between what it can know and what it cannot. On one side: everything in its training data, everything in its context window, everything returned by its tools. On the other side: everything else. The developers who work most reliably with agents have mapped that boundary carefully. The ones who get burned have assumed the boundary is further out than it is.

The training cutoff is the most discussed limitation and the least subtle. The agent doesn't know about events after its training data ends. This is well understood, frequently forgotten in practice, and easy to check — ask the agent about something recent and see what it says. The more dangerous epistemic gaps are the ones that aren't obvious.

The agent doesn't know your organization. It doesn't know your codebase, your customers, your internal processes, your historical decisions and why they were made. It can reason about these things if you put them in the context, but it has no access to them otherwise. Teams that have worked with an agent long enough sometimes forget this — the agent has been helpful for so long that it starts to feel like it knows the company. It doesn't. It knows what was in the context window of the sessions it participated in, which is a small and curated slice of institutional knowledge.

The agent doesn't know what it doesn't know. This is the most operationally important gap. A human expert who encounters the edge of their knowledge usually knows they're at the edge — there's a felt sense of uncertainty that triggers caution. Agents don't have this. They generate the most likely response given their inputs, and if their inputs don't contain the information needed to answer correctly, they generate the most likely plausible-sounding response instead. The output looks the same whether the agent knows the answer or is confabulating one.

Designing around epistemic limits means building in verification for the claims that matter, restricting the agent's scope to domains where its knowledge is reliable, and being explicit with users about what the agent can and cannot be trusted to know.

The agent doesn't know what it doesn't know. You have to know it for both of you.
