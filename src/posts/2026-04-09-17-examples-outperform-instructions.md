---
title: "17. Les exemples surpassent les instructions"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "If you want an agent to produce output in a particular format, style, or structure, showing it an example is almost always more effective than describing what you want."
---

If you want an agent to produce output in a particular format, style, or structure, showing it an example is almost always more effective than describing what you want. This isn't intuition — it's consistent with how these models work. They're trained on examples. They generalize from examples. When you give them an example, you're speaking their language.

The failure mode is writing elaborate instructions where a single example would do the job in a third of the words. "Please format the output as a JSON object with a 'summary' key containing a string no longer than two sentences, a 'tags' key containing an array of strings, and a 'confidence' key containing a float between 0 and 1" is twelve words longer and less clear than just showing the output you want.

Examples are also more robust to edge cases. Instructions describe the cases you thought of. Examples, especially multiple examples, encode the implicit logic that would take paragraphs to fully specify. A model that sees three examples of how to handle ambiguous input has learned something about your intent that couldn't easily be written down.

The number of examples matters, but not linearly. Going from zero examples to one example is the biggest jump in quality. Going from one to three is significant. Going from five to ten is marginal for most tasks. The first example sets the template. Subsequent examples refine the edges. At some point you're adding examples to handle cases that rarely occur and the return diminishes.

There's a selection effect worth paying attention to: the examples you choose encode your values. If all your examples are clean, well-formed inputs, the agent learns to handle clean inputs well and may struggle with messy ones. Including an example of a difficult or edge-case input — and showing how to handle it — is often worth more than several additional happy-path examples.

Instructions tell the agent what you want. Examples show it. Show it.
