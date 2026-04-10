---
title: "43. Never Let an Agent Send an Email It Cannot Unsend"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "The irreversibility of actions is the most important dimension of agentic system design, and it's the one that gets the least attention before something goes wrong."
---

The irreversibility of actions is the most important dimension of agentic system design, and it's the one that gets the least attention before something goes wrong. Reading data is reversible — if the agent reads the wrong thing, nothing changes. Writing data is usually recoverable — records can be corrected, state can be restored. But sending an email, posting a message, executing a transaction, publishing content — these are actions that exist in the world the moment they're taken, and taking them back is either impossible or expensive.

The principle is simple: the more irreversible an action, the more confirmation it deserves before execution. An agent that can autonomously send emails on your behalf needs a higher bar of confidence before it acts than an agent that drafts emails for you to review. Not because autonomous action is inherently bad, but because the cost of getting it wrong is asymmetric — a sent email you didn't mean to send can damage a relationship, violate a privacy expectation, or create a legal obligation that can't be undone by an apology.

Teams underestimate this risk in the early stages of building because they're testing with their own accounts, on their own data, with recipients who know the system is in development. The stakes feel low. In production, with real users, with real recipients, with real consequences — the calculus is different.

The design pattern is a confirmation layer between agent decision and real-world action. For low-stakes, high-reversibility actions, the confirmation can be implicit — the agent acts and logs what it did. For high-stakes, low-reversibility actions, the confirmation should be explicit — the agent presents what it's about to do, waits for approval, then acts. The boundary between these categories should be drawn conservatively and reviewed as you learn how the system behaves in production.

There's also a transparency requirement. When an agent acts on someone's behalf, the recipient of that action often deserves to know. An email from an agent should probably indicate it's from an agent, or at minimum be reviewed by a human who takes responsibility for its content. The alternative — agents acting seamlessly as humans — creates a trust problem that extends beyond your system.

Review before you send. Some things you can't take back.
