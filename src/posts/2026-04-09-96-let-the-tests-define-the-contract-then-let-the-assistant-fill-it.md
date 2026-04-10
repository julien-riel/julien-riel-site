---
title: "96. Laissez les tests définir le contrat, puis laissez l'assistant le remplir"
date: 2026-04-09
tags:
  - developer-as-user
description: "Writing tests before implementation isn't just a quality practice in an AI-assisted workflow — it's a communication protocol."
---

Writing tests before implementation isn't just a quality practice in an AI-assisted workflow — it's a communication protocol. A well-written test suite describes precisely what the code must do: the inputs it accepts, the outputs it produces, the edge cases it handles, the errors it returns. Given that specification, the assistant can implement against it, and you have an objective criterion for whether the implementation is complete.

This changes what "done" means in a way that's particularly valuable when working with an AI. Without tests, "done" is a judgment call — does this look right, does it seem to handle the cases I care about, does it follow the patterns I wanted. With tests, "done" is verifiable: the tests pass or they don't. The assistant can verify its own work rather than requiring you to.

The tests you write before implementation are also better tests than the ones written after. Post-implementation tests tend to reflect the implementation — they test what the code does rather than what it should do. Pre-implementation tests reflect the specification — they test the contract. When the implementation drifts from the contract, pre-implementation tests catch it. Post-implementation tests often don't, because they were written to match the implementation that produced the drift.

There's a practical workflow: write the test file first, describe what each test is checking in a comment, and let the assistant implement the code that makes them pass. Review the implementation against the tests. The tests are the acceptance criteria; the review is checking whether the implementation satisfies them.

Define done before you start. The assistant knows when it's there.
