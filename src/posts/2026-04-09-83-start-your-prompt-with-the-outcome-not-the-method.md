---
title: "83. Commencez votre prompt par le résultat, pas la méthode"
date: 2026-04-09
tags:
  - developer-as-user
description: "\"Refactor this function\" is a method instruction."
---

"Refactor this function" is a method instruction. "Make this function testable in isolation without changing its public interface" is an outcome instruction. The difference is significant: the method instruction delegates the entire design decision to the assistant, while the outcome instruction specifies what success looks like and leaves the implementation path open.

Outcome-first prompts produce better results because they give the assistant a target to optimize toward rather than a procedure to execute. When you specify the outcome, the assistant can evaluate whether its approach achieves it and adjust. When you specify the method, the assistant executes the method whether or not it achieves what you actually needed.

The discipline of outcome-first prompting also forces you to clarify your own goals. "Refactor this function" often means several different things — improve readability, reduce complexity, improve performance, make it testable — and you might not have decided which one you actually want. Writing the outcome forces the decision. What, specifically, should be true about the code when you're done that isn't true now?

This doesn't mean you can never specify the method. Sometimes you know the method is correct and you want the assistant to implement it. But even then, adding the outcome as a check — "implement X approach so that Y is achieved" — gives the assistant a way to flag when the method doesn't serve the outcome. That feedback is often worth more than the implementation itself.

Know what done looks like before you describe how to get there.
