---
title: "14. Révisez la sortie de l'agent comme vous révisez la PR d'un junior"
date: 2026-04-09
tags:
  - working-with-agents
description: "The right mental model for reviewing agent output isn't proofreading — it's code review."
---

The right mental model for reviewing agent output isn't proofreading — it's code review. Not a quick scan for typos, but a careful read for correctness, edge cases, hidden assumptions, and the things that look right but aren't.

A junior developer's pull request deserves real attention not because juniors are bad at their jobs, but because they're working with less context than you have. They might not know about the edge case you've seen before. They might have solved the stated problem while missing the unstated constraint. The code might work today and fail under conditions they didn't think to test. The review isn't a formality — it's where the knowledge transfer happens and where the errors get caught before they matter.

Agent output has the same profile. The agent is capable, often impressively so, but its knowledge of your specific context is limited to what you gave it. It doesn't know what you've learned from three years on this codebase. It doesn't know about the customer who does the unusual thing that breaks the obvious implementation. It doesn't know that the last person who took this approach regretted it. It knows what you told it, and it generalized from there.

Reviewing with this frame changes what you look for. You stop asking "is this grammatically correct" or "does this generally make sense" and start asking "is this actually right for our situation." You look for places where the agent made a reasonable assumption that happens to be wrong for your context. You check the edges — what happens with empty input, with unusually large input, with the user who does the thing nobody anticipated.

The failure mode of treating agent output as finished work is subtle because the output often looks finished. It's fluent, well-structured, internally consistent. These are surface properties. Correctness for your specific situation is a deeper property, and it doesn't come from the model — it comes from you.

The agent drafted it. You're responsible for it. Review accordingly.
