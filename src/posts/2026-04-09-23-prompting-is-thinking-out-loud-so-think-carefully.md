---
title: "23. Prompter c'est penser à voix haute — alors pensez soigneusement"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "There's a reason bad prompts produce bad outputs: they're usually the product of fuzzy thinking."
---

There's a reason bad prompts produce bad outputs: they're usually the product of fuzzy thinking. The prompt is where your understanding of the problem gets externalized. If that understanding is incomplete, the prompt will be too — and the agent will faithfully execute your confusion.

This is uncomfortable because it removes a convenient excuse. When the agent produces something wrong, it's tempting to attribute it to the model — its limitations, its quirks, its tendency to go off in unexpected directions. Sometimes that's true. More often, the prompt was doing the work of deferring a decision you hadn't made yet. The agent hit the unresolved question and answered it without you.

Writing a good prompt is an act of thinking, not transcription. It requires knowing what you actually want — at the level of detail necessary to act on it. That's harder than it sounds for complex tasks, because the gap between "I'll know it when I see it" and "I can specify it precisely enough for an agent to produce it" is often wider than expected. Closing that gap is the work. The prompt is the record of having closed it.

One useful habit is to write the prompt, then read it as if you've never seen the problem before. Does it contain everything a competent person would need to do this task well? Are the constraints clear? Are the priorities explicit when things conflict? Is success defined well enough that you'd recognize it? If any of these answers is no, the prompt isn't done — your thinking isn't done.

Another useful habit is to write the prompt before you build the system. Trying to specify exactly what an agent should do forces you to confront the parts of the problem you haven't fully designed yet. Ambiguities in the prompt are ambiguities in the system design. Better to find them in a text editor than in production logs.

The agent doesn't make your thinking clearer. It makes the quality of your thinking visible.
