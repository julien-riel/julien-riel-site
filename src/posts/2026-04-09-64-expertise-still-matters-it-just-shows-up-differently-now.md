---
title: "64. Expertise Still Matters — It Just Shows Up Differently Now"
date: 2026-04-09
tags:
  - mindset
description: "There's a version of the agentic future where expertise is devalued — where the gap between the expert and the novice closes because both can prompt an agent to do the work."
---

There's a version of the agentic future where expertise is devalued — where the gap between the expert and the novice closes because both can prompt an agent to do the work. This version is wrong, but it's wrong in a way that requires explanation, because the surface evidence for it is real. A developer with two years of experience using agents can produce outputs that would have required ten years of experience without them. The gap closes for execution. It doesn't close for judgment.

Expertise in an agentic context shows up in the quality of the specification, the precision of the review, and the accuracy of the failure diagnosis. The expert knows what a good outcome looks like well enough to recognize when the agent has produced something that looks good but isn't. The novice, lacking that reference point, accepts the plausible output. The agent amplifies both of them, which means it amplifies the difference between them — the expert gets more leverage from their expertise, the novice gets more confident about their mistakes.

This plays out concretely in code review. A senior developer reviewing agent-generated code brings the same knowledge they'd bring to reviewing human-written code: the architectural patterns that cause problems at scale, the edge cases that the obvious implementation misses, the performance characteristics that only matter under load. The agent can write the code. It can't review it with the knowledge that comes from having seen that pattern fail in production three times.

Domain expertise also determines the quality of the context the agent receives. An expert knows which details matter and which don't — which constraints are essential to specify and which the agent can be trusted to handle reasonably. They write prompts that are precise where precision matters and flexible where flexibility is appropriate. A novice either over-specifies — burying the important constraints in noise — or under-specifies — leaving gaps the agent fills with reasonable but wrong assumptions.

The experts who thrive are the ones who redirect their expertise toward the things agents can't do: judgment, evaluation, specification, diagnosis. The experts who struggle are the ones who compete with agents on execution, trying to stay relevant by doing the work faster rather than doing the thinking better.

Expertise didn't become less valuable. Its expression changed.
