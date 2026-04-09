---
title: "81. Context Is a Skill You Can Improve"
date: 2026-04-09
tags:
  - agentic-programming
  - developer-as-user
description: "Knowing what context to provide — and how to provide it — is the most leveraged skill in working with an AI coding assistant."
---

Knowing what context to provide — and how to provide it — is the most leveraged skill in working with an AI coding assistant. Two developers giving the same assistant the same task will get different results, and the difference often comes down to context quality. One pastes the relevant interface and a representative example. The other writes a one-line request. The outputs are not comparable.

Context quality has several dimensions. Relevance: the assistant works better with the specific file it needs to understand than with the entire repository. Precision: a concrete example of the pattern you want to follow is more useful than an abstract description of it. Completeness: the constraints that seem obvious to you — the error handling style, the naming conventions, the dependencies you want to avoid — need to be stated explicitly. Format: structured context is easier for the model to use than a wall of pasted text.

The skill develops through deliberate attention to failure. When the assistant produces something wrong, ask: what was missing from the context that would have prevented this? Usually something was — a constraint you forgot to mention, an example you didn't paste, a convention you assumed was obvious. Add it to your mental checklist for the next prompt.

Over time, you develop a sense for what context a given type of task needs. Code refactoring needs the existing code and the target interface. Test writing needs the function signature and an example of how the module's tests are structured. Bug fixing needs the error message, the stack trace, and the code path that produced them. These patterns become intuitive with practice.

The assistant's capability is fixed. Your ability to use it isn't. Context is where the improvement lives.
