---
title: "18. Negative Space Matters — Tell Your Agent What Not to Do"
date: 2026-04-09
tags:
  - agentic-programming
  - prompting-as-engineering
description: "Most prompts describe what the agent should do."
---

Most prompts describe what the agent should do. Few describe what it shouldn't. That asymmetry is where a surprising number of production failures live.

The reason is simple: a language model fills gaps with probability. When you don't specify a behavior, the model defaults to whatever response is most likely given its training. Usually that's fine. Sometimes it's exactly what you didn't want — the agent that adds unsolicited caveats to every answer, the one that reformats output in a way that breaks downstream parsing, the one that apologizes extensively before delivering bad news when you needed it to just deliver the news. None of these are unreasonable behaviors in the abstract. They're just wrong for your system, and you never told the agent that.

Negative constraints are harder to write than positive ones because they require you to anticipate failure modes before they occur. You have to ask: what would a reasonable agent do here that I wouldn't want? That question is uncomfortable because it forces you to imagine the system going wrong, which feels pessimistic when you're in the optimistic phase of building something new. Do it anyway.

Some negative constraints are universal enough to belong in every system prompt. Don't fabricate citations. Don't assume information that wasn't provided. Don't continue past the scope of the task. Others are specific to your use case and your users. A customer service agent probably shouldn't speculate about competitor products. A code review agent probably shouldn't rewrite code it wasn't asked to rewrite. A summarization agent probably shouldn't editorialize.

The discipline of writing negative constraints forces a useful clarity about what the agent is actually for. When you sit down to enumerate what the agent shouldn't do, you often discover that you hadn't fully articulated what it should do either. The negative space illuminates the positive.

There's a balance. A prompt that's mostly prohibitions is brittle and confusing — the agent spends its cognitive budget navigating restrictions rather than doing the work. Negative constraints should be targeted: the specific behaviors that would be plausible without them and problematic if they occurred.

Define the shape by describing the edges. The middle takes care of itself.
