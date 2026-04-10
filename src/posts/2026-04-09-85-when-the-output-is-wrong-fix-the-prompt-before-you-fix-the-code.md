---
title: "85. When the Output Is Wrong, Fix the Prompt Before You Fix the Code"
date: 2026-04-09
tags:
  - developer-as-user
description: "When the assistant produces code that isn't quite right, the instinct is to edit the code directly — it's faster, it's familiar, it produces the result you need immediately."
---

When the assistant produces code that isn't quite right, the instinct is to edit the code directly — it's faster, it's familiar, it produces the result you need immediately. But editing the code manually is a solution to one instance. Fixing the prompt is a solution to the class of cases the prompt represents, and the assistant will encounter that class again.

The habit of prompt-first debugging pays off more as the codebase grows and the same tasks recur. If the assistant consistently produces functions with insufficient error handling, and you consistently fix them by hand, you've established a workflow where the assistant does the easy parts and you clean up the hard ones on every pass. If instead you identify the pattern, add a clear constraint to your prompt — "always handle the case where the input is null and return a typed error" — you change the output going forward.

This requires pausing before editing, which is the hard part. When you're tired and the code needs to be right, reaching for the keyboard is faster than thinking about why the prompt failed. But the accumulation of manual fixes without prompt improvement is technical debt of a different kind — you're compensating for a known gap in your workflow without addressing it.

The question to ask before editing: why did the prompt produce this? Usually the answer is one of a small set: the constraint was unstated, the example showed something different from what you wanted, the outcome wasn't specified clearly enough. Identifying which one takes thirty seconds and produces a better prompt for next time.

Edit the code when you need to ship now. Fix the prompt so you don't have to edit next time.
