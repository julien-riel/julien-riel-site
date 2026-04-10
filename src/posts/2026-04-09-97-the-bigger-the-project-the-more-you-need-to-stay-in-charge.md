---
title: "97. The Bigger the Project, the More You Need to Stay in Charge"
date: 2026-04-09
tags:
  - developer-as-user
description: "The temptation scales with the capability."
---

The temptation scales with the capability. On a small task, handing the assistant full autonomy and reviewing the output feels like a reasonable tradeoff. On a large project, the same approach applied across dozens of sessions produces a codebase that reflects the assistant's judgment more than yours — one where the architectural coherence you didn't specify has been replaced by the assistant's defaults.

This isn't a failure of the assistant. It's a failure of oversight at the scale where oversight matters most. Small tasks have small blast radii. Large projects accumulate decisions across many sessions, and decisions made without your guidance in session three constrain what's possible in session thirty. The autonomy that was productive on the small task becomes drift on the large project.

The response isn't to do more of the work yourself — it's to increase the frequency and depth of review, not decrease it. More sessions means more checkpoints, not fewer. More generated code means more careful reading, not less. The overhead of oversight scales with the stakes, not with the volume of output.

The developers who maintain control of large AI-assisted projects are the ones who stay close to the architectural decisions — who review not just whether the code works but whether it reflects the design they intended. They treat each session as a collaboration where their judgment governs the direction and the assistant contributes the execution. They don't let the momentum of fast generation substitute for the deliberateness of good design.

The assistant is faster than you. You're responsible for where it's going. Both of those things are true at the same time.
