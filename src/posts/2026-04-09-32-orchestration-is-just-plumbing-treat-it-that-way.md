---
title: "32. L'orchestration n'est que de la plomberie — traitez-la ainsi"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Orchestration frameworks have a way of becoming the center of attention in agentic systems."
---

Orchestration frameworks have a way of becoming the center of attention in agentic systems. The framework is new, it has opinions, it introduces abstractions, and pretty soon you're writing code that's more about satisfying the framework than solving the problem. This is a familiar trap in software — it happens with ORMs, with microservice meshes, with frontend frameworks — and it happens with agent orchestration too.

The purpose of orchestration is to move data between agents, manage state across steps, handle retries, and wire up tools. These are real needs. They're also fundamentally boring infrastructure concerns. The value of your system lives in the agents themselves — in the prompts, the tools, the evals, the domain knowledge you've encoded. The orchestration is the pipes. Nobody cares about the pipes as long as they work.

Treating orchestration as plumbing has practical implications. It means choosing the simplest orchestration approach that meets your needs, not the most sophisticated one available. It means keeping your orchestration logic thin — routing, sequencing, error handling — and your agent logic fat. It means being willing to swap orchestration frameworks without rewriting your agents, which requires keeping them decoupled.

The teams that over-invest in orchestration often do so because it feels like progress. You're building infrastructure, designing systems, making technical choices. It has the texture of real engineering work. But orchestration that doesn't serve agent capability is overhead. The question to ask of every orchestration decision is: does this make my agents better, or does it make my orchestration more elaborate?

Framework churn is also real in this space. The orchestration framework that's popular today may be superseded in a year. Agents that are tightly coupled to their orchestration framework are hard to migrate. Agents that treat orchestration as interchangeable infrastructure move much more freely.

Know where the value is. It's not in the pipes.
