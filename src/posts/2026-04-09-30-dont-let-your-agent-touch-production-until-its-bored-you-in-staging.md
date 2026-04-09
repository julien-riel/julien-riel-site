---
title: "30. Don't Let Your Agent Touch Production Until It's Bored You in Staging"
date: 2026-04-09
tags:
  - agentic-programming
  - building-agentic-systems
description: "There's a moment in every agentic project where the system is working well enough in testing that the temptation to deploy becomes almost irresistible."
---

There's a moment in every agentic project where the system is working well enough in testing that the temptation to deploy becomes almost irresistible. The demos are clean. The obvious cases all pass. The team has been looking at it for weeks and nobody can find a new way to break it. Ship it.

Don't. Not yet.

The gap between "working in testing" and "working in production" is wider for agentic systems than for most software, because agentic systems encounter a much more diverse distribution of inputs in the real world than any test suite captures. Users do unexpected things. They provide context in unusual formats. They ask questions at the boundary of scope. They combine capabilities in ways you never anticipated. The agent that handles your test cases gracefully can still fail badly on the inputs you didn't think to test.

The discipline is to run the system in staging — against real-world-like inputs, with real-world-like variability — until it stops surprising you. Not until it handles everything perfectly, but until the failure modes are familiar. Until you've seen the edge cases and decided how to handle them. Until the behavior feels predictable not because it never fails, but because when it fails, it fails in ways you recognize and have accounted for.

The "bored you" standard is deliberately subjective. It means the system has been running long enough that you're no longer discovering new failure modes. You've stopped being surprised. The last interesting failure was a while ago. That's when you have enough confidence in the system's behavior to trust it with real users.

This requires patience that's genuinely hard to maintain when stakeholders are eager and the system looks ready. The argument for waiting is asymmetric: a premature deployment that fails badly costs more — in user trust, in debugging time, in reputation — than a careful deployment that takes a few more weeks.

Let it bore you first. Production is not a testing environment.
