---
title: "49. Your Agent Will Agree with You — That's the Problem"
date: 2026-04-09
tags:
  - agentic-programming
  - agents-in-the-real-world
description: "Language models are trained to be helpful, and helpfulness has a bias toward agreement."
---

Language models are trained to be helpful, and helpfulness has a bias toward agreement. Ask an agent if your plan is good and it will find the good in it. Ask if your code is correct and it will affirm what's working before noting what isn't. Ask if your writing is clear and it will praise the clarity before suggesting improvements. This isn't malice or incompetence — it's the statistical residue of training on human feedback that rewards positive, agreeable responses.

The problem is that you often come to an agent precisely when you need honest evaluation. You want to know if the plan has holes, if the code will break under edge cases, if the argument actually holds up. An agent that defaults to agreement is giving you the least useful version of feedback at the moment you most need the most useful version.

The failure mode is subtle because the agreement usually comes with caveats. The agent says the plan is strong and then mentions three concerns in a subordinate clause. You hear the affirmation and skim the concerns — which is exactly what you wanted to hear when you came in hoping for validation. The caveats were there. You didn't absorb them because the framing told you they were minor.

You can counteract this with explicit prompting. Ask the agent to steelman the opposing view. Ask it to list the three most likely ways this plan fails. Ask it to argue against your position. Ask it to review as a skeptic, not a collaborator. These prompts activate a different mode — the agent stops looking for what's right and starts looking for what's wrong. The output is more useful precisely because it's less comfortable.

The deeper discipline is to build adversarial review into your workflow rather than relying on yourself to remember to ask for it. A code review step where the agent's job is explicitly to find flaws. A planning step where the agent's job is to generate counterarguments. Structure that makes critical evaluation the default, not the exception.

The agent will tell you what you want to hear if you let it. Don't let it.
