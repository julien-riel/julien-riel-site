---
title: "95. Gardez un journal de décisions que l'assistant peut consulter"
date: 2026-04-09
tags:
  - developer-as-user
description: "Why did you choose this database over the alternatives? Why is the authentication layer structured this way? Why does this module have this interface rather than the more obvious one? If these deci..."
---

Why did you choose this database over the alternatives? Why is the authentication layer structured this way? Why does this module have this interface rather than the more obvious one? If these decisions live only in your head, you'll relitigate them — with yourself, with your team, and with the assistant, which will propose the alternatives you already rejected every time it encounters the decision point.

A decision log is a lightweight document that records significant technical decisions, the alternatives that were considered, and the reasoning behind the choice made. It doesn't need to be elaborate — a few sentences per decision is enough to provide the context that prevents the question from being re-opened unproductively.

The value for the assistant is specific: when you include the decision log in context, the assistant stops suggesting the approaches you've already ruled out. It builds on decisions rather than questioning them. The conversations become more productive because they start from the current state rather than relitigating the journey that got you there.

The value for the team is broader. Decisions documented with reasoning are decisions that can be revisited deliberately when circumstances change, rather than undone accidentally when someone doesn't know they were made. The log is also institutional memory — when the person who made the decision leaves, the reasoning stays.

Write down the decisions that felt hard to make. Those are the ones that will be questioned again.
