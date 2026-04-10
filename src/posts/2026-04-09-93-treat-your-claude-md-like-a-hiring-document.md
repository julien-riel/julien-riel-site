---
title: "93. Traitez votre CLAUDE.md comme un document d'embauche"
date: 2026-04-09
tags:
  - developer-as-user
description: "Claude Code reads a `CLAUDE.md` file at the start of every session."
---

Claude Code reads a `CLAUDE.md` file at the start of every session. Most developers who use it treat it as a list of rules — don't use this library, follow these conventions, run these commands before committing. This is underusing it significantly. The `CLAUDE.md` is the document that onboards your AI collaborator to your project, and it deserves the care you'd put into onboarding a new team member.

A good onboarding document for a human developer would tell them: what this project is and why it exists, the architectural decisions that define its structure and the reasoning behind them, the conventions that are non-negotiable and the ones that are preferences, the parts of the codebase that are fragile and need careful handling, the things that have been tried and didn't work, the tools and workflows the team uses. All of this is relevant to the assistant too.

The reasoning behind decisions is especially valuable. "We use X library" is a rule. "We use X library because Y library had performance problems at our scale and Z library didn't support the authentication model we need" is a decision with context — and an assistant that understands the context can make better judgment calls on adjacent decisions you haven't specified.

The `CLAUDE.md` also serves as documentation for human team members. The process of writing it — articulating the project's conventions and decisions clearly enough for an AI to act on them — produces exactly the kind of documentation that new team members need and that rarely gets written because it feels obvious to the people who already know it.

Write the `CLAUDE.md` as if you're explaining the project to a new hire who is very capable but knows nothing about your specific context. That's exactly what you're doing.
