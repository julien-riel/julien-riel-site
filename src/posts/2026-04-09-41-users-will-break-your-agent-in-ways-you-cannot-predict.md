---
title: "41. Users Will Break Your Agent in Ways You Cannot Predict"
date: 2026-04-09
tags:
  - agentic-programming
  - agents-in-the-real-world
description: "You can spend weeks testing an agent against every scenario you can imagine, and a user will break it on day one with an input you never considered."
---

You can spend weeks testing an agent against every scenario you can imagine, and a user will break it on day one with an input you never considered. This isn't a failure of imagination — it's a property of the gap between the people who build systems and the people who use them. Users bring their own mental models, their own vocabulary, their own assumptions about what the system can do. Those models don't match yours, and the mismatch produces inputs your agent has never seen.

The inputs that break agents aren't usually malicious or even unusual from the user's perspective. They're the natural expression of how that person thinks about the problem. A user who pastes an entire email thread into a field designed for a single question. A user who asks the agent to do something adjacent to its purpose, assuming it will figure out what they mean. A user who types in their native language when the agent was designed for English. A user who asks the same question five different ways, convinced that the right phrasing will unlock the answer they want. Each of these is a normal human behavior. None of them are in your test suite.

The response to this isn't to test more exhaustively — you can't enumerate the space of human behavior. The response is to design for graceful handling of unexpected inputs. What does the agent do when it receives something it doesn't understand? Does it fail helpfully, explaining what it can and can't do? Does it make a reasonable attempt and flag its uncertainty? Does it silently produce something plausible but wrong? The last option is the one to design away from, because it creates the worst user experience: the user thinks they got an answer, acts on it, and discovers the problem later.

The first month of production is your most valuable testing period. The inputs that arrive in that month represent the actual distribution of how your users think about the problem — which is always different from how you do. Collect them. Analyze the failures. Use them to build the eval set that your pre-launch testing couldn't have produced.

Design for the users you have, not the users you imagined.
