---
title: "71. Construisez pour l'agent que vous avez, pas celui que vous aimeriez avoir"
date: 2026-04-09
tags:
  - mindset
description: "Every developer working with agents has a gap between the current capabilities of the tools they're using and the capabilities they wish those tools had."
---

Every developer working with agents has a gap between the current capabilities of the tools they're using and the capabilities they wish those tools had. The current model is almost good enough for the task but not quite. The context window is almost large enough but fills up at the wrong moment. The reasoning is almost reliable enough to trust autonomously but still needs a checkpoint. Almost, almost, almost.

Building for the agent you wish you had means designing systems that depend on capabilities that don't quite exist yet — then wondering why they don't work. The system assumes a level of instruction-following reliability that the current model doesn't achieve. The workflow assumes context retention across a session length that exceeds what the model handles well. The architecture assumes tool use precision that the current model achieves in testing but not consistently in production. Each assumption is individually reasonable given where the field is heading. Together they produce a system that works in the demo and fails in deployment.

Building for the agent you have means taking current capabilities seriously as constraints, not temporary obstacles to be designed around. If the model struggles with tasks that require tracking more than five variables simultaneously, design the task to require fewer. If the model is unreliable at long-horizon planning, add human checkpoints rather than hoping this run will be the reliable one. If context length causes degradation, build summarization into the workflow rather than assuming the model will handle long contexts as well as short ones.

This is not pessimism about the field. It's the pragmatism that produces systems that actually work. The capabilities will improve — they always have. When they do, you remove the constraints you built around the old limitations. But you can only remove constraints that you acknowledged. You can't fix a system that was built on assumptions that were never true.

There's also a compounding benefit to designing within current constraints: it forces clarity about what actually needs to happen for the system to work. The constraints reveal the essential complexity. Systems designed within tight constraints are often better systems than ones designed with the assumption of unlimited capability, because the constraints force the hard thinking that unconstrained design defers.

The agent you have is the one you're shipping with. Build for it honestly.
