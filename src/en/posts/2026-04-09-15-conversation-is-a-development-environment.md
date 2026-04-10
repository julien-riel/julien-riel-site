---
title: "15. Conversation Is a Development Environment"
date: 2026-04-09
tags:
  - working-with-agents
description: "The conversational interface to a language model isn't just a way to get answers — it's a place to think."
---

The conversational interface to a language model isn't just a way to get answers — it's a place to think. Developers who treat it as a search engine ask one question and evaluate the response. Developers who treat it as a development environment iterate, push back, explore alternatives, and use the agent as a thinking partner across a whole problem.

The difference in output quality is significant. A single-turn interaction with an agent produces whatever the model thinks is the most likely good response given the initial prompt. A multi-turn conversation produces something shaped by your feedback, your corrections, your domain knowledge injected at the right moments. The first is the agent's best guess. The second is a collaboration.

This reframes what skill means in working with agents. It's not just about writing better initial prompts — it's about knowing how to steer a conversation productively. That means recognizing when the agent has gone in the wrong direction early, before you've built on top of a flawed foundation. It means knowing when to ask for alternatives rather than accepting the first response. It means understanding when to inject context mid-conversation — "actually, there's a constraint I didn't mention" — rather than starting over.

The conversation also serves as a record of your thinking. The questions you asked, the directions you explored, the dead ends you identified — that's a log of a design process. Teams that treat conversational development as throwaway work lose that record. Teams that preserve it, even informally, build up a picture of how decisions got made.

There's a practical limit: long conversations accumulate context that can drift. The agent's early understanding of the problem shapes everything that follows, and if that early understanding was wrong, correction gets harder as the conversation grows. The skill is knowing when to start fresh with better inputs versus when to keep building on what's there.

The blank prompt box isn't a query field. It's where the work starts.

---

## Part 2 — Prompting as Engineering
