---
title: "31. Small Agents Beat Big Agents"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "The instinct when building agentic systems is to make the agent capable of everything."
---

The instinct when building agentic systems is to make the agent capable of everything. One agent, one prompt, all the tools, all the tasks. It seems efficient. It's actually a trap.

Big agents are hard to reason about. When a single agent is responsible for understanding the user's intent, retrieving relevant information, calling external APIs, formatting output, and handling errors, you've created a system where any failure could be caused by anything. Debugging becomes archaeology. You dig through logs trying to figure out which part of the agent's reasoning went wrong, and often you can't tell, because the failure is somewhere in the middle of a long chain of decisions the agent made without explaining itself.

Small agents have a narrower job. A classifier that determines task type. A retriever that pulls relevant context. A generator that drafts output. A validator that checks it. Each one does one thing and is testable in isolation. When something breaks, you know where to look. When you want to improve performance, you know what to change without worrying about breaking something else.

This mirrors everything we already know about software design. Small, focused functions are easier to test than large, sprawling ones. The same principle applies here — the unit of composition in an agentic system is the agent, and small units compose better than large ones.

The practical objection is latency: multiple agents in sequence means multiple model calls, and model calls are slow. That's real. But it's often overweighted. A pipeline of three small agents that reliably produces correct output is usually better than one big agent that's fast but wrong fifteen percent of the time and opaque when it fails. Reliability compounds in ways latency doesn't.

There's also a context window argument for small agents. A focused agent needs focused context — a smaller, more precise slice of information. A big agent accumulates context across multiple sub-tasks, burns through the window, and starts losing important information from earlier in the conversation. Small agents reset cleanly between tasks.

Start with the smallest agent that could possibly work. Make it bigger only when the seams start to show.
