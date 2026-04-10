---
title: "91. Let the Assistant Write the Plan, Then Edit It"
date: 2026-04-09
tags:
  - developer-as-user
description: "When you're starting a substantial piece of work, ask the assistant to write an implementation plan before writing any code."
---

When you're starting a substantial piece of work, ask the assistant to write an implementation plan before writing any code. Describe what you're trying to build, provide the relevant context, and ask: what are the steps, what are the dependencies between them, what are the decisions that need to be made before implementation starts?

The plan the assistant produces will be imperfect. It will miss constraints specific to your codebase, make assumptions about your preferences that may not hold, and propose an ordering that might not match your priorities. These imperfections are exactly why the exercise is valuable. Editing a bad plan is much faster than writing a good one from scratch, and the imperfections reveal the decisions you hadn't consciously made yet.

The planning conversation also surfaces ambiguities in your spec before they become bugs in your code. If you describe a feature and the assistant's plan reveals three different interpretations of what "user settings" means in your system, you want to know that now, not after implementing the wrong one.

Once you've edited the plan into something you're confident in, it becomes the document that guides the implementation prompts. Each step in the plan becomes a prompt. The dependencies between steps tell you what context to carry forward. The decisions you made during editing become explicit constraints in the prompts that need them.

The plan is cheap to produce and expensive to skip. Let the assistant write the first draft.
