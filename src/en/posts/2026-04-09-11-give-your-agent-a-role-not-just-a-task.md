---
title: "11. Give Your Agent a Role, Not Just a Task"
date: 2026-04-09
tags:
  - working-with-agents
description: "There's a difference between telling an agent what to do and telling an agent what it is."
---

There's a difference between telling an agent what to do and telling an agent what it is. "Summarize this document" is a task. "You are a senior technical writer summarizing internal documentation for a non-technical audience" is a role. The role produces better results — not because of magic words, but because it loads a coherent set of behaviors, constraints, and priorities that the model can apply consistently across everything the task requires.

Roles work because language models are trained on human-generated text, which is full of role-specific behavior. The way a lawyer reads a contract is different from the way an engineer reads one. The way a copy editor approaches a paragraph is different from the way a developer does. When you assign a role, you're not just setting a tone — you're activating a cluster of domain-specific behaviors that the model has learned from examples of people in that role doing that kind of work.

The practical difference is visible in edge cases. Give an agent the task "review this code for bugs" and it will find bugs. Give it the role "you are a senior engineer doing a security-focused code review before a production deployment" and it will find different bugs — it will weight differently, flag differently, and explain its findings in a way that's calibrated to what a security-conscious senior engineer would care about. The task is the same. The lens is different.

Roles also provide a consistent fallback for situations the task specification didn't anticipate. If you've told the agent it's a technical writer for a non-technical audience, and the document contains jargon you didn't explicitly tell it to simplify, it has a principle for handling that case. Without the role, it has to guess. Guessing is where inconsistency lives.

The failure mode is roles that are too vague to be useful — "you are a helpful assistant who is good at many things" doesn't give the model anything to work with. Useful roles are specific about domain, audience, and the values that should guide tradeoffs. Not just what the agent is, but how it thinks.

Tell the agent what it is. The task follows from the role.
