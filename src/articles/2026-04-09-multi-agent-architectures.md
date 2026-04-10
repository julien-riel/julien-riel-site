---
title: "Architectures multi-agents : quand un seul agent ne suffit pas"
date: 2026-04-09
tags:
  - architecture
  - multi-agent
description: "A practical guide to multi-agent patterns — orchestrator-workers, pipelines, ensembles, and swarms — and where they break."
---

## A Practical Guide for Agentic Programmers

A single agent with the right tools can do a lot. But at some point, you'll hit a wall: the task is too complex for one context window, requires different expertise at different stages, or needs parallel execution paths. That's when you need multiple agents working together.

Multi-agent systems are powerful. They're also where complexity multiplies fastest. This guide covers when to use them, how to architect them, and where they break.

## Why Multiple Agents?

The case for multi-agent architectures comes from three constraints:

**Context window limits.** A single agent handling a complex workflow needs to hold instructions, tool definitions, conversation history, intermediate results, and retrieved documents — all in one context window. As tasks grow, the context budget runs out. Splitting responsibilities across agents means each one operates with a focused, manageable context.

**Specialization.** Different subtasks benefit from different models, prompts, and tools. A coding agent needs code execution tools and a model optimized for code. A research agent needs web search and a model good at synthesis. A planning agent needs reasoning depth but doesn't need tools at all. Trying to make one agent do everything means every subtask gets a mediocre configuration.

**Parallelism.** Some subtasks are independent and can run simultaneously. A single agent executes sequentially by nature — it generates one token at a time. Multiple agents can work in parallel, dramatically reducing latency for tasks with independent sub-problems.

WHOOP operates over 500 specialized agents across their app — Memory, Daily Outlook, Day in Review, Activity Insights, onboarding, and dozens more. Each has a defined role, its own prompt, and its own set of tools. This isn't accidental complexity. It's a deliberate architecture that lets each agent be excellent at one thing.

## The Core Patterns

### Pattern 1: Orchestrator-Workers

One agent (the orchestrator) receives the user's request, decomposes it into subtasks, and delegates each subtask to a specialized worker agent. The orchestrator collects the results and synthesizes the final response.

```
User request → Orchestrator → [Worker A, Worker B, Worker C] → Orchestrator → Final response
```

This is the most common pattern. Notion rebuilt their AI architecture around it — replacing task-specific prompt chains with a central reasoning model that coordinates modular sub-agents. The orchestrator handles planning and synthesis. The workers handle execution.

**When to use it:** Complex tasks that naturally decompose into subtasks (research + analysis + formatting, or data retrieval + computation + explanation).

**The hard part:** The orchestrator needs to be smart enough to decompose the task well and to know when a worker's result is good enough. A bad decomposition leads to wasted work or missing context between subtasks.

### Pattern 2: Pipeline (Sequential Handoff)

Agents process information in sequence, each one refining or transforming the output of the previous stage. Like an assembly line.

```
User request → Agent A (extract) → Agent B (analyze) → Agent C (format) → Final response
```

Cursor's original Bugbot used a variation: eight parallel instances of the same agent, each processing the code diff in a different order, with a voting step at the end. That's a hybrid between pipeline and ensemble.

**When to use it:** Tasks with clear sequential stages where each stage has different requirements — extraction → validation → transformation → generation.

**The hard part:** Information loss between stages. Each handoff is a potential point where critical context gets dropped. Design your inter-agent communication format carefully — structured data with explicit fields, not freeform text.

### Pattern 3: Debate / Ensemble

Multiple agents independently tackle the same problem, then their outputs are compared, combined, or voted on. This increases reliability at the cost of latency and compute.

```
User request → [Agent A, Agent B, Agent C] → Aggregator → Final response
```

**When to use it:** High-stakes decisions where accuracy matters more than speed — medical diagnosis, legal analysis, code review. Bugbot's eight-pass majority voting was exactly this pattern.

**The hard part:** Defining how to aggregate disagreements. Majority voting is simple but loses nuance. A separate judge agent can resolve conflicts but adds another failure point. And cost scales linearly with the number of agents.

### Pattern 4: Autonomous Swarm

Agents dynamically spawn sub-agents based on what they discover during execution. The orchestrator doesn't pre-plan all subtasks — it adapts as new information emerges. Cursor's Composer model (Kimi K2.5-based) uses Agent Swarm, where the model learns through RL to dynamically decompose tasks and dispatch parallel sub-agents.

**When to use it:** Exploratory tasks where the full scope isn't known upfront — research, debugging, data investigation.

**The hard part:** Everything. Control, observability, cost management, and preventing runaway execution are all significantly harder when agent creation is dynamic. This pattern requires mature tooling and strong kill switches.

## Communication Between Agents

How agents pass information to each other is as important as what each agent does. Three approaches, in order of increasing structure:

### Natural Language Messages

Agents communicate via freeform text. Simple to implement, but lossy. The receiving agent has to parse unstructured text, which can miss critical details or misinterpret ambiguous phrasing.

Use this when: agents are handling inherently unstructured tasks (creative writing, open-ended research).

### Structured Data

Agents exchange JSON, XML, or typed objects with defined schemas. The sending agent outputs structured data; the receiving agent knows exactly what fields to expect.

```json
{
  "task_id": "extract_metrics",
  "status": "complete",
  "results": {
    "heart_rate_avg": 54,
    "hrv_avg": 62,
    "sleep_score": 78
  },
  "confidence": 0.92,
  "sources": ["sleep_data_2024_03_15"]
}
```

Use this whenever agents feed into programmatic downstream steps. The structure acts as a contract between agents, making failures explicit rather than silent.

### Shared State / Blackboard

All agents read from and write to a shared state object (sometimes called a blackboard). Each agent can see the full context of what other agents have done and add its own contributions.

WHOOP's Memory nuggets function as a shared state: any agent can write a memory, and all agents can read relevant memories. Notion's block-based architecture serves a similar purpose — agents operate on a shared graph of structured data.

Use this when: agents need to be aware of each other's work without explicit point-to-point communication. The shared state provides coordination without coupling.

## Where Multi-Agent Systems Break

### Failure Cascades

When Agent A's bad output feeds into Agent B, the error compounds. Agent B doesn't know Agent A made a mistake — it treats the input as authoritative. By Agent C, the error has been amplified and embedded in a confident-sounding response.

**Mitigation:** Validate at every handoff. Add lightweight checks between agents — type validation, assertion checks, or a quick LLM-as-a-judge pass that flags obviously wrong intermediate results. Don't assume upstream agents are reliable.

### Context Loss

Each agent handoff is a potential context bottleneck. The orchestrator summarizes, and the summary misses a critical detail. The worker completes its subtask perfectly — except it didn't have the one piece of information that changes everything.

**Mitigation:** Be explicit about what context each agent needs. Don't rely on implicit understanding. Include relevant metadata in inter-agent messages. When in doubt, pass more context than seems necessary.

### Cost Explosion

Multiple agents mean multiple LLM calls. An orchestrator-workers pattern with 5 workers and a synthesis step means at least 7 LLM calls per request. If each worker does retrieval and multi-step reasoning, you might be at 20+ calls. At production scale, this gets expensive fast.

**Mitigation:** Use smaller, cheaper models for simple subtasks. Not every agent needs a frontier model. Route based on task complexity — the same way Cursor uses a custom model for Tab, a 70B for code application, and frontier models for reasoning.

### Observability Collapse

When a multi-agent system produces a bad result, you need to know which agent failed, what it saw, and what it produced. Without structured logging of every inter-agent message and every agent's reasoning, debugging is impossible.

**Mitigation:** Log everything. Every agent call, every input, every output, every tool invocation. WHOOP's eval framework provides trace-level details for every agent interaction — not just the final output, but the intermediate chain. This is non-negotiable for production multi-agent systems.

### Coordination Overhead

As you add agents, the coordination cost grows. The orchestrator spends more tokens managing the workflow than the workers spend on actual work. At some point, the overhead exceeds the benefit of specialization.

**Mitigation:** Keep the number of agents as small as possible. Don't split into agents for architectural elegance — split only when a single agent genuinely can't handle the task due to context limits, specialization needs, or parallelism requirements. Three well-designed agents usually outperform ten poorly-designed ones.

## When to Use Multi-Agent (And When Not To)

**Use multi-agent when:**

- The task genuinely requires different tools, models, or expertise at different stages
- The context for the full task exceeds what one agent can hold
- Independent subtasks can be parallelized for latency gains
- You need reliability through redundancy (ensemble patterns)

**Don't use multi-agent when:**

- A single agent with good tools can handle the task (most tasks)
- You're splitting agents for organizational rather than technical reasons
- You don't have the observability infrastructure to debug multi-agent failures
- The coordination overhead exceeds the benefit of specialization

The essay in this book is right: small agents beat big agents. But the corollary is equally true — one good agent beats three unnecessary ones. Add agents when you have a concrete reason. Remove them when you can.

## A Starting Architecture

If you're building your first multi-agent system, start here:

1. **Build a single agent that handles the full task.** Push it until it fails — context overflow, tool confusion, quality degradation.
2. **Identify the failure mode.** Is the context too large? Does the agent struggle with one specific subtask? Is latency unacceptable?
3. **Split only at the failure point.** Extract the problematic subtask into a specialized worker agent. Keep everything else in the main agent.
4. **Add structured communication.** Define the contract between agents with schemas, not freeform text.
5. **Add evaluation at each boundary.** Test the orchestrator's decomposition. Test each worker's output independently. Test the final synthesis.
6. **Add observability from day one.** If you can't trace a failure through the agent chain, you can't fix it.

This approach gives you the simplicity of a single agent where it works and the power of multi-agent where it's needed. Don't design a multi-agent architecture. Grow into one.

## Further Reading

- [From Idea To Agent In Less Than Ten Minutes](https://engineering.prod.whoop.com/ai-studio) — How WHOOP manages 500+ specialized agents with AI Studio
- [Notion's GPT-5 Rebuild](https://openai.com/index/notion/) — Notion's shift from prompt chains to orchestrator-workers architecture
- [How Kimi, Cursor, and Chroma Train Agentic Models with RL](https://www.philschmid.de/kimi-composer-context) — Agent Swarm pattern and dynamic sub-agent dispatch
- [Cursor's Bugbot Evolution](https://medium.com/data-science-collective/how-cursor-actually-works-c0702d5d91a9) — Pipeline to agent transition in production code review
- [Anthropic: Building Effective Agents](https://docs.anthropic.com/en/docs/build-with-claude/agent-patterns) — Patterns and anti-patterns for agent architectures
