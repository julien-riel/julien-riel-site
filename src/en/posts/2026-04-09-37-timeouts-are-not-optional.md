---
title: "37. Timeouts Are Not Optional"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Every external call your agent makes — to a model API, to a tool, to a database, to a third-party service — needs a timeout."
---

Every external call your agent makes — to a model API, to a tool, to a database, to a third-party service — needs a timeout. This is true in conventional software and doubly true in agentic systems, where a hanging call doesn't just block the current operation; it can freeze an entire task, consume context window budget, and leave the agent in an ambiguous state that's hard to recover from.

The reason timeouts get skipped is optimism. The service is reliable. The network is fast. The tool has never hung before. These observations are accurate right up until they're not, and systems without timeouts are systems that discover their dependencies' failure modes in production, under load, at the worst possible time.

In agentic systems, the timeout problem has an additional dimension: the agent itself can be a source of unbounded execution. A model call with no timeout can hang indefinitely if the API is under load. A tool that makes a network request with no timeout can block the agent's entire reasoning loop. A retry strategy without a total time limit can extend a task far beyond any reasonable expectation. Each of these is a place where "it usually works" becomes "it sometimes hangs forever."

Setting good timeouts requires knowing what normal looks like. If a tool call typically completes in under a second, a five-second timeout is conservative and appropriate. If a model call typically takes three seconds, a thirty-second timeout leaves room for slow responses without waiting forever. These numbers come from observation — instrument your calls, understand the distribution of response times, and set timeouts that cover the legitimate tail without covering the infinite tail.

What happens when a timeout fires is as important as the timeout itself. The agent needs a defined behavior for each timeout case — retry, fail the task, escalate to a human, or skip the step and proceed with degraded capability. Undefined timeout behavior produces undefined agent behavior, which is the thing you were trying to avoid.

Every call needs a deadline. The system that will eventually hang is the one without one.
