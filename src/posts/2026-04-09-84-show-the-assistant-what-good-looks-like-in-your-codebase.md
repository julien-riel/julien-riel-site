---
title: "84. Show the Assistant What Good Looks Like in Your Codebase"
date: 2026-04-09
tags:
  - developer-as-user
description: "Abstract instructions produce generic code."
---

Abstract instructions produce generic code. "Follow our error handling conventions" produces something the assistant invented based on common patterns. Pasting three examples of how your codebase actually handles errors produces code that fits. The model is fundamentally learning from examples — give it the right ones.

This is few-shot prompting applied to code generation, and the principle is the same as in any prompting context: examples outperform instructions. You can spend a paragraph describing your naming conventions or you can paste a well-named module and say "follow this style." The second approach is faster to write, harder to misinterpret, and produces better output.

The examples you choose matter. A single well-chosen example from the actual codebase is worth more than three synthetic examples you wrote for the prompt. Real code carries implicit information — the level of abstraction you favor, the way you structure error paths, how much you comment, how you name things at the boundaries of a module. A synthetic example can only carry what you explicitly put in it.

There's also a calibration benefit. When you paste an example and ask the assistant to follow its style, you're establishing a concrete reference point for the conversation. If the output drifts from the style, you can point to the example and say "more like this." Without the example, "more like this" has no referent.

Build a personal library of good examples from your codebase — the functions, the modules, the test files that represent the standard you're aiming for. They're worth more in a prompt than any description you could write.
