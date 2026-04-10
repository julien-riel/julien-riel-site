---
title: "6. Les agents échouent gracieusement ou pas — il n'y a pas de milieu"
date: 2026-04-09
tags:
  - working-with-agents
description: "Most systems fail on a spectrum."
---

Most systems fail on a spectrum. A web server under load starts dropping requests slowly, giving you time to notice and respond. A database running low on disk space degrades gracefully, warning you before it stops. The failure is visible, incremental, and recoverable. You build monitoring for exactly this kind of decay.

Agents fail differently. They don't degrade — they drift. The outputs get subtly worse over time, in ways that are hard to detect unless you're looking specifically for them. The agent starts making slightly different assumptions. Its tone shifts. It begins handling edge cases in new ways. Nothing breaks loudly. The system is still running. The outputs are still coming. They're just not right anymore.

This makes graceful degradation in agentic systems a design problem you have to solve on purpose, not a property you get for free. You have to decide, in advance, what failure looks like and how you want the system to behave when it gets there. An agent that hits a tool failure — does it retry silently, surface the error to the user, or attempt a workaround? An agent that receives contradictory information — does it flag the contradiction, pick the most recent source, or ask for clarification? Each of these is a design decision. Leave them unspecified and the agent will make them for you, inconsistently.

The ungraceful failure is easier to design for, perversely. If your agent is going to fail badly, fail loudly. Surface the error. Stop the process. Make noise. A loud failure is debuggable. A silent drift that corrupts your data or misleads your users for three weeks before someone notices — that's the failure mode you actually can't afford.

The practical question to ask for every tool your agent uses, every external dependency it touches, every edge case in its task: what do I want to happen when this goes wrong? Write that down. Build it. Test it. Don't leave it to chance and don't assume the model will handle it sensibly, because sensible and consistent are not the same thing.

Failure will come. The only variable is whether you designed for it.
