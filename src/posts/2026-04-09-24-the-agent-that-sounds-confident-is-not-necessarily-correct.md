---
title: "24. The Agent That Sounds Confident Is Not Necessarily Correct"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "Language models are fluent by default."
---

Language models are fluent by default. They produce text that reads as assured, coherent, and authoritative regardless of whether the underlying content is accurate. This is not a bug the developers forgot to fix — it's a consequence of how these models are trained. Fluency and correctness are different properties, and the training process optimizes heavily for the former.

The problem is that humans read confidence as a signal of reliability. We've evolved to do this — in most human communication, someone who speaks with conviction has usually checked their facts, or at least believes they have. That heuristic breaks badly with language models, which produce confident prose about things they have no reliable basis for asserting.

The practical effect is that agent outputs require skepticism proportional to the stakes, not proportional to how the output reads. An agent that summarizes a document with calm authority might have missed a key nuance. An agent that provides a step-by-step technical procedure might have fabricated a step that sounds plausible. The text gives you no reliable signal about which is happening.

Calibration is the skill you're developing here — the ability to assess how likely an agent output is to be correct given the type of task, the quality of the context, and your knowledge of where this model tends to fail. In domains you know well, calibration comes naturally: you can spot the wrong answer because you know what right looks like. In domains where you're relying on the agent precisely because you don't know the domain — which is common, and legitimate — calibration requires external verification. Check the claims. Follow the citations. Test the code.

Some prompting techniques can reduce unwarranted confidence — asking the agent to express uncertainty explicitly, asking it to identify the parts of its response it's least sure about, asking it to distinguish between what it knows and what it's inferring. These help. They don't solve the problem.

Read agent output like you'd read a smart intern's first draft: with appreciation for the effort and independent judgment about the content.
