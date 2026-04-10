---
title: "42. Latency Is a UX Problem, Not Just an Infrastructure Problem"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "A model call takes time."
---

A model call takes time. Usually seconds. Sometimes more. For a developer running a batch job, that's fine — you kick it off and come back. For a user waiting for a response in an interactive interface, three seconds feels long and ten seconds feels broken. The latency characteristics that are acceptable in one context are dealbreakers in another, and conflating the two is how you ship a technically functional system that users abandon.

The infrastructure response to latency is optimization: smaller models, caching, streaming, parallel calls. These matter and you should pursue them. But they have limits, and the more important response is often design — shaping the user experience so that the wait feels shorter, or so that the user is doing something useful while the agent works.

Streaming is the most impactful design intervention available. Showing the agent's response as it generates, rather than waiting for the complete output, fundamentally changes how latency feels. A ten-second response that streams progressively feels faster than a three-second response that appears all at once, because the user has something to read almost immediately. The cognitive experience of waiting is much worse than the cognitive experience of reading something that's still arriving.

Progress indicators help for longer operations — not generic spinners, but specific signals about what's happening. "Searching your documents" is better than a rotating circle. "Drafting a response based on three sources" is better than "thinking." These signals give users a mental model of what the agent is doing, which makes the wait feel purposeful rather than opaque.

There's also a product question underneath the infrastructure question: should this be an interactive experience at all? Some agent tasks are too long to make users wait for synchronously. A task that takes thirty seconds probably belongs in an async workflow — start it, do something else, get notified when it's done — rather than a chat interface where the user stares at a spinner. Choosing the wrong interaction model creates a latency problem that no amount of optimization will fully solve.

Fast enough for the task. Designed for the wait. Both matter.
