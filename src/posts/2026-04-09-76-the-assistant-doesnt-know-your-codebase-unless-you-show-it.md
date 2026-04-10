---
title: "76. The Assistant Doesn't Know Your Codebase Unless You Show It"
date: 2026-04-09
tags:
  - developer-as-user
description: "Every session starts fresh."
---

Every session starts fresh. The assistant has no memory of the refactoring you did last week, the convention you established last month, the architectural decision you made last year and the reasons behind it. It knows what you put in the context window, and nothing else. This is the same constraint that applies to any agent — but it surprises developers who have been working productively with an assistant for months and start to feel like it knows the project.

The feeling is understandable. When you've had hundreds of good interactions, when the assistant consistently produces code that fits your patterns, it starts to feel like shared context has accumulated. It hasn't. What's happened is that you've gotten better at providing context implicitly — you've learned to phrase requests in ways that encode your conventions, to paste the right reference code, to describe constraints you used to leave unstated. The assistant hasn't learned your codebase. You've learned to carry it with you.

This distinction matters when something goes wrong. If the assistant produces code that violates a project convention, the failure isn't the assistant forgetting — it's you not providing. The mental model of a forgetful colleague leads you to feel frustrated at the assistant. The correct mental model of a stateless system leads you to fix the context.

The practical response is to develop habits of context provision: paste a representative example of the pattern you want to follow, include the interface the new code must conform to, add a comment about the constraint that isn't obvious. These habits don't just help the assistant — they document the things that are currently implicit in your head, which makes the codebase easier to maintain regardless of who's writing it.

The assistant is as good as the context you provide. That's entirely within your control.
