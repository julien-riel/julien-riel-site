---
title: "40. The Best Agents Have a Narrow Personality"
date: 2026-04-09
tags:
  - agentic-programming
  - building-agentic-systems
description: "A general-purpose agent sounds like the goal."
---

A general-purpose agent sounds like the goal. One agent, any task, maximum flexibility. In practice, general-purpose agents are mediocre at everything and excellent at nothing. The agents that work best in production have a sharply defined personality — a coherent sense of what they are, what they do, and how they do it — and that specificity is a feature, not a limitation.

Personality here means more than tone. It means a consistent set of values that govern how the agent makes tradeoffs. A code review agent that prioritizes correctness over readability will behave differently than one that prioritizes readability over correctness — not just in what it says, but in which issues it flags, which it lets pass, and how it explains its reasoning. Those are different agents, appropriate for different contexts. An agent that tries to balance both without a clear priority will be inconsistent in ways that frustrate the people relying on it.

Narrow personality also makes agents more predictable, which makes them more trustworthy. Users who interact with an agent repeatedly develop a mental model of how it behaves. When the behavior is consistent — when the agent reliably does the same kind of thing in the same kind of way — that mental model is accurate and useful. When the behavior is variable — when the same question gets different treatment depending on subtle context differences — the mental model breaks down and users stop trusting their intuitions about the system.

The design process for a narrow personality is the same as the design process for a good system prompt: figure out what this agent is for, who it's for, what it values, and what it won't do — and then encode all of that explicitly. The agent's personality is a design artifact, not an emergent property. Left unspecified, it will be inconsistent. Specified precisely, it becomes a reliable characteristic of the system.

The temptation to make agents broader comes from wanting to avoid building multiple agents. That's the wrong optimization. Three narrow agents that each do their job well are better than one wide agent that does everything passably.

Know what the agent is. Make it that, completely.

---

## Part 4 — Agents in the Real World
