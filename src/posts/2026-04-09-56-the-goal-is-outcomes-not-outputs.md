---
title: "56. L'objectif est les résultats, pas les sorties"
date: 2026-04-09
tags:
  - mindset
description: "An agent that produces a beautiful summary of a document hasn't succeeded."
---

An agent that produces a beautiful summary of a document hasn't succeeded. It's succeeded if the person who reads the summary understands something they needed to understand, makes a better decision, saves time they would have spent reading the full document. The output is the means. The outcome is the point. Conflating them is how you build technically impressive systems that don't actually help anyone.

This distinction matters most in evaluation. Teams that evaluate agent quality by output quality — is the summary well-written, is the code syntactically correct, is the response grammatically fluent — are measuring the wrong thing. These properties correlate with quality but don't define it. A well-written summary of the wrong content fails the user. Syntactically correct code that doesn't solve the actual problem fails the developer. Fluent responses to the wrong question fail everyone.

Outcome-focused evaluation requires knowing what the user was actually trying to accomplish and whether the agent helped them accomplish it. That's harder to measure than output quality, which is probably why teams measure output quality instead. But hard to measure doesn't mean optional. You can measure outcomes through user behavior — did they take the action the information was meant to enable? Through follow-up rates — did they come back with clarifying questions that suggest the first response missed the mark? Through direct feedback — did the output help?

The output focus also distorts what gets built. Teams optimizing for output quality invest in making outputs look better — more polished prose, better formatting, more comprehensive coverage. Teams optimizing for outcomes invest in understanding the user's actual goal, which sometimes means shorter outputs, less comprehensive coverage, and more direct answers that don't showcase the agent's capability but actually address the need.

There's a design question underneath this: do you know what your users are trying to accomplish? Not what they're asking for — what they're trying to accomplish. These are often different. The user who asks for a summary of a legal document is trying to make a decision, not collect information. The agent that helps them make the decision has succeeded. The agent that summarizes the document beautifully while leaving the decision as hard as before has produced a good output and a bad outcome.

Measure what matters. The output is evidence. The outcome is the verdict.
