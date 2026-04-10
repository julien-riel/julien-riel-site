---
title: "54. Vous êtes le développeur senior — l'agent est le junior"
date: 2026-04-09
tags:
  - mindset
description: "The most useful mental model for working with agents isn't \"tool\" and it isn't \"collaborator\" — it's \"junior developer.\" Capable, fast, knowledgeable across a broad surface area, genuinely helpful ..."
---

The most useful mental model for working with agents isn't "tool" and it isn't "collaborator" — it's "junior developer." Capable, fast, knowledgeable across a broad surface area, genuinely helpful on well-defined tasks, and in consistent need of the kind of oversight that only comes from someone who knows the codebase, the context, and the consequences.

A good junior developer can write solid code, research unfamiliar problems, draft documentation, and handle a wide range of tasks you'd otherwise do yourself. You don't micromanage every line. But you also don't hand them the keys to production and walk away. You review their work. You catch the assumptions they didn't know to question. You recognize when they've solved the stated problem while missing the actual one. The value of your oversight isn't that they're incompetent — it's that you have context they don't.

The agent has the same profile, amplified. It's faster than any junior developer, available at any hour, and has read more code than any human ever will. It's also missing everything that isn't in the context window — the institutional knowledge, the architectural decisions and why they were made, the customer behavior that makes the obvious implementation wrong, the political constraints that rule out the technically correct solution. You have all of that. The agent has none of it unless you provide it.

This framing has a practical payoff: it calibrates your review process correctly. You don't rubber-stamp a junior's work because it looks competent on the surface. You read it with the specific question: is this right for our situation, given what I know that they don't? That's the right question to ask of agent output too. Not "is this generally correct" but "is this correct here, in this context, for these users, given everything I know that the agent doesn't."

It also calibrates your expectations. You don't expect a junior developer to make senior architectural decisions autonomously. You give them well-scoped tasks, review the outputs, and expand their autonomy as trust is established through demonstrated judgment. The same escalation of trust applies to agents — earn it task by task, domain by domain, as you build up evidence about where the agent's judgment is reliable and where it needs your hand.

The agent is the most capable junior you've ever worked with. Manage it like one.
