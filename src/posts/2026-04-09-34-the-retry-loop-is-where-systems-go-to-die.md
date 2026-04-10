---
title: "34. La boucle de retry est là où les systèmes vont mourir"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Retry logic is necessary."
---

Retry logic is necessary. Every system that calls external services needs it — networks fail, services time out, transient errors happen. But in agentic systems, retry logic has a particular failure mode that's worth understanding before you build it: the agent that retries indefinitely, convinced it's making progress, consuming tokens and time and money while producing nothing useful.

The problem is that agents generate their own reasons to retry. A conventional retry loop has a fixed condition: the operation failed, wait and try again. An agent can construct reasons to keep going from the content of the conversation — the tool returned an ambiguous result, so try again with a different approach; the output didn't match expectations, so try a different formulation; the last attempt was almost right, so iterate once more. Each of these is individually reasonable. Together they produce a loop that can run for a very long time before anyone notices.

This is especially dangerous when the retries have side effects. An agent retrying a database write, a message send, or an API call that charges per request can cause real damage before the loop terminates. The retry logic that seemed like a safety feature becomes the failure mode itself.

The fix requires explicit limits at multiple levels. A maximum number of attempts per operation. A maximum number of steps per task. A maximum wall-clock time before the task is abandoned and flagged for human review. These limits should be set conservatively and adjusted based on observed behavior — not left open-ended because the task might genuinely need more attempts.

There's also a design question about what the agent does when it hits a limit. Failing silently is the worst outcome — the task appears to complete while having done nothing. Failing loudly, with a clear error state and enough context to understand what was attempted, is the foundation of any meaningful retry strategy at the human level.

Retry logic without exit conditions isn't reliability. It's optimism without a plan.
