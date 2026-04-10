---
title: "92. Use Markdown, Not Prose, for Specifications"
date: 2026-04-09
tags:
  - developer-as-user
description: "A specification written as flowing prose is hard to reference, hard to update, and hard to provide as context."
---

A specification written as flowing prose is hard to reference, hard to update, and hard to provide as context. A specification written in structured Markdown — with headers, lists, code examples, and explicit sections for requirements, constraints, and edge cases — is easy to navigate, easy to maintain, and easy to drop into a prompt context.

The structure does work that prose can't. A list of acceptance criteria is unambiguous in a way that a paragraph describing the feature isn't. A code example showing the expected interface is clearer than three sentences explaining it. An explicit section called "Out of Scope" prevents the assistant from helpfully adding features you didn't ask for. The format enforces a discipline of specificity that prose tends to undermine.

Markdown specifications also compose well. You can include the relevant section of the spec in a prompt rather than the whole document. You can update a single section without rewriting the entire spec. You can link between sections when one constraint depends on another. These properties matter more as the project grows and the spec becomes a living document rather than a one-time artifact.

There's a template worth developing for your own use: a standard structure for feature specifications that works well for you and the assistant. Something like: Problem Statement, Proposed Solution, Acceptance Criteria, Edge Cases, Out of Scope, Open Questions. The specific sections matter less than the habit of using them consistently — consistency means you always know where to look for the constraint you need.

The format of the spec determines how well you can use it. Structure it for the work you're going to do with it.
