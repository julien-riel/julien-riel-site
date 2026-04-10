---
title: "75. Lisez chaque ligne qu'il écrit"
date: 2026-04-09
tags:
  - developer-as-user
description: "The speed of generation is the trap."
---

The speed of generation is the trap. The assistant produces fifty lines of code in three seconds, it looks plausible, the tests pass, you commit. Two days later you're debugging a failure that traces back to a subtle logic error in those fifty lines — an error that would have been obvious if you'd read carefully, which you didn't because it arrived fast and looked right.

Speed of production and correctness of production are independent variables. The assistant generates code at a rate that creates a psychological pressure to accept it at the same rate. Resist this. The code deserves the same reading you'd give to a pull request from a competent colleague — not suspicious, not line-by-line word parsing, but genuine comprehension. Do you understand what each part does? Does it do what you intended? Are there edge cases the implementation doesn't handle?

The cases where careful reading matters most are exactly the cases where it's hardest to maintain: when you're tired, when you're on deadline, when the task feels routine, when you've asked for something similar many times before and it's always been fine. The assistant doesn't get tired. It doesn't get sloppy under pressure. But it also doesn't know what you actually need — it knows what you asked for. Those are sometimes different things, and only you can catch the gap.

There's a specific failure mode worth naming: the code that is technically correct but wrong for your situation. It compiles, the tests pass, the logic is sound — but it solves a subtly different problem than the one you have. This failure is invisible if you're only checking that the code runs. It's visible if you're reading to understand.

The assistant writes the first draft. You're responsible for every line that ships.
