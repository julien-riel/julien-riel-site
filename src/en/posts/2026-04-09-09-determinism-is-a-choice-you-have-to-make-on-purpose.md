---
title: "9. Determinism Is a Choice You Have to Make on Purpose"
date: 2026-04-09
tags:
  - working-with-agents
description: "By default, language models are non-deterministic."
---

By default, language models are non-deterministic. Run the same prompt twice and you'll get similar outputs, not identical ones. For some tasks that's fine — even desirable. For others, it's a hidden bug waiting to surface in production.

The problem isn't non-determinism itself. The problem is non-determinism you didn't choose. When you build a system without thinking about whether it needs to be deterministic, you get a system whose behavior you can't fully reason about, can't fully test, and can't fully explain to users when they ask why they got a different result today than they did yesterday.

Most APIs expose a temperature parameter for exactly this reason. Temperature zero — or close to it — makes the model pick the most likely token at each step, which produces near-deterministic outputs for most inputs. Higher temperatures introduce more randomness, which produces more varied outputs. This is a dial you can turn. Turning it intentionally is part of the architecture; leaving it at the default is a decision by omission.

The cases where determinism matters most are the ones where your system's output feeds into something else. If the agent's output is parsed by downstream code, variability in format breaks the parser. If the agent makes a decision that's logged and audited, you need to be able to reproduce it. If the agent's output is shown to a user and they come back the next day expecting consistency, non-determinism is a UX problem.

The cases where non-determinism is an asset are creative tasks, brainstorming, and any situation where you want variety across multiple runs. Generating five alternative headlines benefits from variability. Extracting a structured address from a form submission does not.

This is a decision worth making explicitly, per task, per system. Not once at the top level — different agents in the same pipeline might have different determinism requirements. The classifier that routes tasks probably wants temperature near zero. The agent that drafts responses might want a little more range.

Know what you need. Set it on purpose. The default is not a design decision — it's a deferred one.
