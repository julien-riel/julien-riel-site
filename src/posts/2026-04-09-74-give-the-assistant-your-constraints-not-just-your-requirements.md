---
title: "74. Give the Assistant Your Constraints, Not Just Your Requirements"
date: 2026-04-09
tags:
  - agentic-programming
  - developer-as-user
description: "\"Write a function that parses this config file\" produces something."
---

"Write a function that parses this config file" produces something. "Write a function that parses this config file, handles malformed input by returning a typed error rather than throwing, uses only the standard library, and follows the error handling conventions in the rest of this codebase" produces something useful. The gap between those two requests is the gap between a requirement and a specification — and it's the developer's job to close it.

This is harder than it sounds because constraints are often tacit. You know that this project doesn't use external dependencies without having to think about it. You know that errors are returned, not thrown, because you wrote the convention yourself. You know that this function will be called in a hot path and performance matters. None of this is in the requirement. You have to make it explicit, which requires first making it conscious.

The exercise of listing your constraints before prompting is valuable independent of the assistant. It's a form of design thinking — forcing you to articulate the requirements that aren't in the spec because everyone on the team already knows them. The assistant needs them stated. Writing them down is often the moment you discover that you don't agree on them as clearly as you thought.

The constraints that matter most are the ones about what the code must not do: must not throw, must not mutate the input, must not make network calls, must not break the existing interface. Positive requirements describe the target. Negative constraints define the boundaries. Both are necessary. Most prompts only contain one.

Tell the assistant what the code can't do. That's where the real specification lives.
