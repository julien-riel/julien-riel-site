---
title: "47. La sécurité commence par ce que vous mettez dans la fenêtre de contexte"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "The context window is the most sensitive surface in an agentic system."
---

The context window is the most sensitive surface in an agentic system. Everything the agent knows, everything it can act on, everything that shapes its behavior — it all passes through the context. That makes it the primary attack surface, the primary data leakage risk, and the primary place where security decisions either get made correctly or get deferred until something goes wrong.

The data leakage risk is the most immediate. Developers building retrieval systems pull documents into the context to give the agent relevant information. If those documents contain sensitive data — personal information, credentials, internal business data — and the agent's output surfaces that data to users who shouldn't see it, the retrieval system has become a data exposure mechanism. The agent doesn't know what's sensitive. It knows what it was given and what it was asked. If it was given sensitive data and asked a question whose answer involves that data, it will use it.

The fix requires thinking carefully about what goes into retrieval. Not just what's relevant, but what's appropriate for the agent to see given the identity and permissions of the user making the request. Access control at the retrieval layer — ensuring the agent only sees documents the user is authorized to see — is not optional in any system that handles data with meaningful sensitivity differences between users.

Credentials deserve special attention. System prompts that contain API keys, database passwords, or authentication tokens are common in early-stage development and catastrophically wrong in production. The context window is logged. It's passed through APIs. It ends up in places you didn't intend. Credentials belong in environment variables and secrets managers, accessed at runtime, never embedded in prompts.

There's a broader principle here about least exposure. The agent should see the minimum information necessary to do its job. Not everything that might be useful — the minimum that's actually necessary. Every additional piece of context is an additional piece of information that can be misused, leaked, or manipulated.

What you put in the context is what you're trusting the agent with. Choose carefully.
