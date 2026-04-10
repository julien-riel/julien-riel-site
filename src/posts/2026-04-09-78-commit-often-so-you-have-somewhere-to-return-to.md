---
title: "78. Committez souvent, pour avoir un endroit où revenir"
date: 2026-04-09
tags:
  - developer-as-user
description: "Working with an AI coding assistant changes the rhythm of development."
---

Working with an AI coding assistant changes the rhythm of development. Changes arrive in larger chunks, faster. A task that used to take two hours of incremental work now arrives in twenty minutes of generation and review. The acceleration is real. So is the risk: when something goes wrong — and it will — you want a recent stable state to return to, not a two-hour hole to climb out of.

Frequent commits aren't just version control hygiene in this context. They're the mechanism that makes it safe to move fast. Each commit is a checkpoint: the system was in a known good state here. If the next batch of generated code breaks something subtle, you can bisect, compare, and recover. Without commits, you have a long history of fast changes and no clean way to understand when the problem was introduced.

The commit message matters too. "WIP" is not useful when you're debugging three days later. "Add input validation per spec section 3.2" is. The assistant can draft commit messages — and will produce better ones if you describe what the change accomplishes rather than what it does. Take thirty seconds to write a real commit message. Future you will use it.

There's also a psychological benefit. Frequent commits create a sense of stable ground — you know where you've been and where you can return. Working without them, especially at the pace an AI assistant enables, creates a kind of vertigo. You're moving fast but you're not sure where you are, and backing up feels harder than it should.

The assistant makes it easy to move fast. Commits make it safe to.
