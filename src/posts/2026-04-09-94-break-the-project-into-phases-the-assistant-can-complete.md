---
title: "94. Break the Project into Phases the Assistant Can Complete"
date: 2026-04-09
tags:
  - developer-as-user
description: "A project described as a single continuous flow is hard to work on with an AI assistant."
---

A project described as a single continuous flow is hard to work on with an AI assistant. The context shifts across sessions, the state of the work is hard to communicate, and it's not clear at any given moment what the assistant should be doing or how to know when a piece is done.

A project broken into phases — each with a defined scope, clear deliverables, and explicit completion criteria — maps naturally onto how the assistant works. Each phase fits in a session or a small number of sessions. The deliverable for each phase is testable before the next phase starts. The completion criteria define what context the next phase should start with.

The phases should follow the natural dependencies in the project: foundation before features, interfaces before implementations, happy path before edge cases. This is the same sequencing you'd use for any well-planned project — the assistant doesn't change the logic of good project structure, it just makes the structure more important because each phase needs to be independently verifiable.

Phase boundaries are also the right place for human review. At the end of each phase, before starting the next, review what was produced. Does it meet the completion criteria? Are the interfaces as designed? Are there decisions embedded in the implementation that will constrain future phases in ways you didn't intend? Catching these at phase boundaries is cheap. Catching them after three phases have been built on top of them is not.

Structure the project for checkpoints. The assistant does the work between them. You do the work at them.
