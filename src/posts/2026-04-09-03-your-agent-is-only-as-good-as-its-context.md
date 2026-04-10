---
title: "3. Your Agent Is Only as Good as Its Context"
date: 2026-04-09
tags:
  - working-with-agents
description: "Garbage in, garbage out is one of the oldest principles in computing."
---

Garbage in, garbage out is one of the oldest principles in computing. With agents, it's more insidious: sophisticated in, sophisticated-sounding garbage out. The agent will work with whatever context you give it, and it will do so fluently, which means bad context doesn't produce obvious errors — it produces plausible-looking wrong answers.

Context is everything the agent knows when it starts working: the system prompt, the conversation history, the documents you've retrieved, the tool outputs you've passed back in. The agent has no access to anything outside that window. It cannot check. It cannot ask (unless you've built that in). It reasons from what it has, and if what it has is incomplete, stale, or subtly wrong, the reasoning will be too.

The failure mode developers hit most often isn't missing context — it's assumed context. You know the codebase, the business rules, the edge cases that matter. The agent doesn't, unless you've told it. When you run a task and get a result that's technically correct but obviously wrong for your situation, that's usually why. The agent solved the problem it was given. You gave it the wrong problem.

Retrieval-augmented systems make this concrete. You build a pipeline that pulls relevant documents into the context before the agent runs. It works beautifully in testing, where your retrieval hits the right documents. In production, retrieval misses. The agent gets adjacent documents — related enough to seem right, wrong enough to matter. And because the agent doesn't know what it doesn't know, it proceeds confidently with what it has.

The discipline is to audit your context before you audit your prompt. When an agent fails, the first question isn't "did the model get confused?" — it's "what did the model actually see?" Log the full context. Read it like the agent would. Often the failure is obvious the moment you do this: a key piece of information wasn't there, or something contradictory was.

Designing good context is an underrated skill. It means knowing what to include, what to exclude, and how to structure information so the agent can use it. Too much context is its own problem — the agent buries the signal in noise, or hits the context window limit and loses the early parts of the conversation. Too little and you're expecting inference where you need facts.

The agent is doing its best with what you gave it. Give it better things.
