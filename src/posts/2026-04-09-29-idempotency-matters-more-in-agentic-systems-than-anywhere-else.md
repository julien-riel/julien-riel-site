---
title: "29. Idempotency Matters More in Agentic Systems Than Anywhere Else"
date: 2026-04-09
tags:
  - agentic-programming
  - building-agentic-systems
description: "Idempotency — the property that calling something multiple times produces the same result as calling it once — is a good practice in any distributed system."
---

Idempotency — the property that calling something multiple times produces the same result as calling it once — is a good practice in any distributed system. In agentic systems, it's close to a requirement. Agents retry. They loop. They lose track of what they've already done. Without idempotent operations, these behaviors turn into duplicated actions, inconsistent state, and failures that are very hard to untangle.

The reason agents create more idempotency pressure than traditional software is that their control flow is probabilistic. A conventional program retries a failed operation because an explicit retry loop told it to. An agent retries because it generated a token sequence that included trying the operation again — perhaps because it didn't register the first attempt, perhaps because the first attempt returned an ambiguous result, perhaps because something in the conversation context made it seem like the action hadn't been taken yet. You often can't predict when a retry will happen or why.

The practical consequence is that any tool your agent can call that has side effects should be designed to be called multiple times safely. Creating a record: check if it exists first, or use a client-supplied idempotency key. Sending a notification: track what's been sent and deduplicate. Charging a payment: require an idempotency key that prevents double-charges. These aren't exotic engineering patterns — they're standard practice in distributed systems. The reason to apply them more aggressively in agentic contexts is that the retry behavior is less predictable and less controllable than in systems you wrote yourself.

The failure mode is memorable when it occurs. A user receives the same email five times. A database record gets created in duplicate. A financial transaction processes twice. These failures are embarrassing at best and costly at worst, and they often occur in production long after your test suite gave you a false sense of security, because the conditions that trigger unexpected retries are hard to reproduce in testing.

Design every state-changing tool as if it will be called twice. Because eventually it will be.
