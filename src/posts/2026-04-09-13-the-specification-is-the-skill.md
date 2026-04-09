---
title: "13. The Specification Is the Skill"
date: 2026-04-09
tags:
  - agentic-programming
  - working-with-agents
description: "The developers who get the most out of agents aren't the ones who know the most about models."
---

The developers who get the most out of agents aren't the ones who know the most about models. They're the ones who can write a precise specification. That skill — breaking a task down into exactly what's needed, no more, no less — turns out to be the bottleneck, not the technology.

This surprises people because the pitch for agents is that they reduce the need for precision. You don't have to write exact code anymore — you describe what you want and the agent figures it out. And that's true, up to a point. For simple tasks, loose descriptions work fine. As tasks get more complex, loose descriptions produce outputs that are approximately right, which in software is another way of saying wrong.

A good specification does several things at once. It defines the goal clearly enough that success is recognizable. It establishes the constraints — what the output must include, what it must not include, what format it needs to be in, what edge cases matter. It anticipates the places where the agent will have to make a judgment call and tells it how to make that call. It defines what done looks like before the work starts.

That last part is where most specifications fail. "Write a function that processes user input" is not a specification — it's the beginning of a conversation. A specification says what inputs are valid, what the function should return for each, what it should do when input is invalid, and what performance characteristics matter. Writing that down forces clarity that the vague version defers.

The connection to agent quality is direct. An agent working from a vague specification fills the gaps with its own judgment, which may be reasonable but won't be consistent and won't always match yours. An agent working from a precise specification has less room to wander and more signal for where to go when it does.

The deeper point is that writing good specifications is valuable regardless of whether an agent is involved. It's what senior developers do when they break down a problem before writing code. Agents just make the skill more visible — and the absence of it more costly.

You don't need to learn a new skill. You need to take an old one more seriously.
