---
title: "53. Vous êtes responsable de tout ce que fait l'agent"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "When an agent makes a mistake — gives wrong information, takes a harmful action, produces output that damages a user's interests — the question of responsibility has a clear answer."
---

When an agent makes a mistake — gives wrong information, takes a harmful action, produces output that damages a user's interests — the question of responsibility has a clear answer. It's you. Not the model provider, not the framework you used, not the agent itself. You built the system, you deployed it, you put it in front of users. The outputs are yours.

This isn't a legal argument, though it may become one. It's a design argument. Developers who internalize responsibility for agent behavior make different decisions than developers who feel insulated from it. They build more validation. They design more conservative defaults. They invest in observability so they can see what the system is doing. They think carefully about what happens when things go wrong, because they know that when things go wrong, it's their problem.

The temptation to diffuse responsibility is strong, especially when agents are marketed as autonomous systems that make their own decisions. The autonomy is real — agents do make decisions you didn't explicitly program. But autonomy in execution doesn't transfer responsibility for outcomes. You chose the model, wrote the prompts, defined the tools, set the scope, and decided when the system was ready to deploy. Every one of those decisions is yours.

This becomes most concrete in high-stakes domains. An agent giving medical information to someone who acts on it. An agent making financial decisions on a user's behalf. An agent communicating with customers in ways that create legal obligations. In each case, the question isn't whether the agent had good intentions — it's whether the outputs were appropriate and whether the system was designed with sufficient care for the stakes involved.

The responsible posture is to treat the agent's outputs as your outputs. Read them with the same critical eye you'd apply to anything you were putting your name on. Build review into the workflow for anything consequential. Be honest with users about what the system is and what it can and can't be trusted to do.

The agent acts. You're accountable. Design accordingly.

---

## Part 5 — Mindset
