---
title: "19. Les system prompts sont des contrats"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "A system prompt isn't instructions — it's a contract."
---

A system prompt isn't instructions — it's a contract. It defines what the agent is, what it does, and what it refuses to do. The moment you treat it as a suggestion, you've lost control of the system.

Contracts have specific properties. They're explicit, not implied. They're stable — you don't change a contract mid-transaction without both parties agreeing. They have edge cases spelled out, not left to interpretation. And crucially, they create expectations: downstream systems, users, and other agents all behave based on what the contract promises. Break the contract silently and everything downstream breaks in ways that are hard to trace.

Most system prompts are written like rough drafts. Vague on scope, silent on failure modes, inconsistent about format. They work fine in the happy path and fall apart the moment something unexpected happens. That's not a prompt problem — it's a contract problem. The contract didn't cover the case.

Writing a system prompt as a contract means being explicit about the things you'd rather not think about. What does the agent do when the user asks something outside its scope? What does it do when tool calls fail? When the context is ambiguous, does it ask for clarification or make its best guess? These aren't edge cases you can defer — they're the cases that define the system's actual behavior in production.

There's also the stability requirement. Teams that iterate quickly on system prompts often create a subtler problem: the contract changes, but nothing downstream is notified. An agent that used to return structured JSON now returns prose because someone improved the system prompt. The pipeline that was parsing that JSON breaks. This is why prompt versioning isn't just good hygiene — it's contract management.

The hardest part of writing a good system prompt is the negative space: what the agent won't do. It's tempting to only specify the positive behavior. But an agent without explicit constraints will fill ambiguity with something, and that something might not be what you wanted. Negative constraints are often where the real contract lives.

Treat a changed system prompt the way you'd treat a changed API contract — with tests, with versioning, and with the assumption that something downstream is depending on the old behavior.

The agent will honor the contract you gave it. Write one worth honoring.
