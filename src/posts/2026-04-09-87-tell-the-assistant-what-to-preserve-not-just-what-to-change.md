---
title: "87. Tell the Assistant What to Preserve, Not Just What to Change"
date: 2026-04-09
tags:
  - agentic-programming
  - developer-as-user
description: "Every prompt implicitly asks the assistant to optimize for the goal you stated."
---

Every prompt implicitly asks the assistant to optimize for the goal you stated. If you ask it to improve performance, it will improve performance — and it might change the function signature, remove a validation step, or restructure the error handling to do so, because none of those were mentioned as constraints. The assistant doesn't know what matters to you beyond what you said. It fills the rest with reasonable judgment that may not match yours.

The missing half of most prompts is the preservation constraint: what must stay the same. The public interface. The existing tests. The error handling contract. The behavior for edge cases that are already handled correctly. These are the load-bearing parts of the existing code that a new optimization might inadvertently break. Stating them explicitly makes the assistant treat them as fixed points rather than variables.

This is especially important for refactoring tasks, where the whole point is to change the implementation while preserving the behavior. "Refactor this function to reduce cyclomatic complexity" without specifying that all existing tests must continue to pass is an open invitation to change what the function does. The assistant might produce something simpler and wrong.

The discipline is to think about what you're not trying to change before you describe what you are. Make a list, even a mental one: the interface is fixed, the test coverage must not regress, the logging behavior must stay the same. Then include those constraints in the prompt. The output will be better and the review will be faster, because you'll know exactly what to check.

State what can't move. The assistant will work around it.
