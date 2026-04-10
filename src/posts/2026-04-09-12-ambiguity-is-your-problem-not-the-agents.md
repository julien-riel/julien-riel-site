---
title: "12. L'ambiguïté est votre problème, pas celui de l'agent"
date: 2026-04-09
tags:
  - working-with-agents
description: "When an agent produces an output that isn't what you wanted, the temptation is to say the prompt was ambiguous."
---

When an agent produces an output that isn't what you wanted, the temptation is to say the prompt was ambiguous. This is usually true. It's also deflection. The ambiguity was there before the agent saw it. You put it there, or you failed to remove it. The agent didn't create the problem — it just made it visible.

Ambiguity in instructions is normal. Natural language is imprecise by design; it relies on shared context, common sense, and conversational repair to fill gaps. When you talk to a colleague, they can ask what you meant. They can infer from your tone. They can draw on weeks of shared project history to interpret an underspecified request. Agents have none of that unless you explicitly provide it. What reads as clear to you — because you're filling in all the gaps from your own knowledge — reads as genuinely ambiguous to the model, which has only the context window.

The discipline is to read your prompts as if you know nothing beyond what's written. Not as the author who knows what they meant, but as a reader encountering the text cold. Better still: give the prompt to a colleague and ask them what they think it's asking for. If they hesitate, or give a different answer than you expected, you've found your ambiguity before the agent does.

There's a specific kind of ambiguity that's especially costly: conflicting constraints. "Be concise but thorough." "Be direct but diplomatic." "Summarize for a general audience but preserve technical accuracy." Each of these pairs contains real tension, and the agent will resolve it somehow — just not necessarily the way you'd want it to. When you have conflicting constraints, prioritize them explicitly. Tell the agent which one wins when they can't both be satisfied.

Removing ambiguity is harder than it sounds because it requires you to know what you actually want — specifically, at the level of detail the agent needs to act on. That's often where the real work is. Vague instructions are frequently a sign of vague thinking.

Clarify your thinking first. The prompt is just the transcript.
