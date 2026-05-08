---
title: "Context Is Finite. Program Accordingly."
date: 2026-05-08
tags:
  - context-engineering
  - agents
  - architecture
description: "An inventory of the techniques that fill the window, the phenomena that degrade it, the heuristics to master it. And along the way, the most expensive anti-pattern in production agents."
---

<p class="deck">
An inventory of the techniques that fill the window, the phenomena that degrade it, the heuristics to master it. And along the way, the most expensive anti-pattern in production agents.
</p>

<div class="notice-prereq">
  <strong>Prerequisite.</strong> This article assumes you know what a <em>token</em> is, roughly how a transformer works, and why the model receives the entire history on every turn. If those notions aren't already in place, the companion article <a href="/en/articles/understanding-llms/">What's Really Happening When You Talk to an AI</a> sets the stage in fifteen minutes.
</div>

<div class="section-num">§ 01 — Inventory</div>

## Everything we've invented to <span class="accent">tame</span> a token predictor.

A transformer on its own does just one thing: predict the next token from what it has in front of it. To make it useful in production — so it *answers*, *remembers*, *acts*, *holds up over time* — we've invented a dozen techniques. Each one fills a specific gap. Each one *inhabits* the context window in some way. Here's the inventory, framed by what it **costs** and what it **unlocks**.

### Frame the behavior · the system prompt

The instruction text placed at the head of every request. Defines role, tone, rules, guardrails, output format, sometimes examples. It's what turns a text predictor into an assistant. **Cost:** permanent, and paid on every turn. Often 5,000 to 25,000 tokens for a consumer product, more for an agent with lots of tools.

### Personalize without duplicating · user preferences

A small extra block specific to the user, injected before the conversation — language, tone, expertise, current projects. **Cost:** low in tokens but high in priority — these lines weigh heavily on prediction.

### Grant capabilities · tools and MCP

A model can't read a file, query a database, or send an email — it just produces text. The fix: declare tools the model invokes by writing a structured call (function calling, tool use), which the application executes on its behalf. The **Model Context Protocol** (MCP) standardizes how tools are declared and exposed, letting you plug in third-party servers (Asana, Gmail, GitLab, internal databases…) without rewriting the pipeline. **Cost:** every declared tool occupies the window — JSON schema, description, parameters — *even when it's never called*. Wire up ten MCP servers and you're paying that bill ten times over.

### Teach procedures · skills

`SKILL.md` files containing procedural recipes injected only when a trigger matches. Instead of bloating the system prompt with every possible recipe, you store them separately and load on demand. **Cost:** zero until they're activated; modest when they are. The big trap — a poorly designed skill can pull data into the window that should have been processed elsewhere. That's the subject of § 04.

### Keep the thread · conversation history

The model is stateless. To make a conversation feel continuous, the application reconstructs the full history on every turn. **Cost:** linear in the number of exchanges. By turn 40, you're paying the same price 40 times over.

### Compress the old stuff · automatic summarization

When you're approaching the limit, the application replaces older turns with a condensed summary produced by the model itself. **Cost:** compression is *irreversible* — a detail erased doesn't come back.

### Persist across conversations · memory

A separate store from the history, holding durable facts (preferences, projects, professional context) reinjected into the window when relevant. **Cost:** low in tokens, but demands discipline — what to remember, what to forget, what to suggest.

### Retrieve instead of loading everything · RAG

A document corpus (hundreds of docs, thousands of pages) doesn't fit in the window. *Retrieval-Augmented Generation* indexes the corpus separately, and at query time, only fetches the relevant passages for injection. The recent evolution — *agentic* RAG — lets the agent decide *when* and *what* to retrieve rather than imposing a frozen pre-LLM step. **Cost:** indexing infrastructure on the side, and answer quality depends on retrieval quality.

### Cut the cost of stable prefixes · prompt caching

Every request recomputes the system prompt and tool definitions — even when nothing has changed. Providers now cache the attention computation (*KV cache*) for stable portions. On subsequent requests, those tokens cost a fraction of their normal price and latency drops. **Cost:** zero in tokens — it's pure optimization — but it requires keeping the prefix identical from one request to the next, byte-for-byte.

### Isolate the noise · sub-agents

Some tasks demand reading large volumes (web, files, multiple searches) that would saturate the parent's window. Delegate to a sub-agent that has its own window, processes the noise on its side, and returns only a compact summary. Also enables parallelization. **Cost:** every sub-agent pays for its own system prompt and its own tools; summary compression remains irreversible. See § 06.

### Compact the context · the background operation

Over a long agentic session — tools called, files read, sub-agents invoked — the window fills with material that's no longer relevant. **Compaction** prunes or summarizes peripheral portions to free up space. It's the more general idea that summarization is just one instance of. **Cost:** like any compression, you lose something. The challenge is to lose *the right thing*.

### The typical allocation

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 1</span><span>Typical allocation in a production agent</span></div>
  <svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="How the window splits between artifacts">
    <rect x="20" y="40" width="560" height="220" fill="none" stroke="#3d3525" stroke-width="1.5"/>
    <text x="20" y="30" class="svg-text svg-text-faint">┌─ WINDOW</text>
    <text x="580" y="30" class="svg-text svg-text-faint" text-anchor="end">~200,000 tokens ─┐</text>
    <rect x="20" y="40" width="140" height="220" fill="#c2553a" opacity="0.85"/>
    <text x="90" y="155" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">SYSTEM</text>
    <text x="90" y="170" class="svg-text" text-anchor="middle" fill="#0f0d0a">~15-25k</text>
    <rect x="160" y="40" width="68" height="220" fill="#e8a04b" opacity="0.85"/>
    <text x="194" y="148" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">TOOLS</text>
    <text x="194" y="162" class="svg-text" text-anchor="middle" fill="#0f0d0a">defs</text>
    <rect x="228" y="40" width="46" height="220" fill="#ffc26b" opacity="0.85"/>
    <text x="251" y="148" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700" font-size="9">SKILLS</text>
    <rect x="274" y="40" width="18" height="220" fill="#7a8b5c" opacity="0.85"/>
    <rect x="292" y="40" width="170" height="220" fill="#4d8a8a" opacity="0.85"/>
    <text x="377" y="148" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">TOOL RESULTS</text>
    <text x="377" y="164" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-style="italic">the main vector of saturation</text>
    <rect x="462" y="40" width="80" height="220" fill="#a89c84" opacity="0.85"/>
    <text x="502" y="148" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">HISTORY</text>
    <rect x="542" y="40" width="22" height="220" fill="#5a5040" opacity="0.85"/>
    <rect x="564" y="40" width="16" height="220" fill="#f4ecdc" opacity="0.95"/>
    <text x="572" y="280" class="svg-text" text-anchor="middle" font-size="8">↑ user</text>
    <text x="20" y="290" class="svg-text svg-text-dim">└─ each active solution = a block of tokens you pay for</text>
    <text x="300" y="312" class="svg-label-big" text-anchor="middle">all solutions share the same pool</text>
  </svg>
  <figcaption class="diagram-caption">Every technique leaves a footprint here. None are free.</figcaption>
</figure>

<div class="section-num">§ 02 — Phenomena</div>

## Six things that happen <span class="accent">in</span> the window, and that we don't really control.

The previous solutions are levers you pull. There are also phenomena you endure — properties of the model, properties of attention, properties of the data — and you have to integrate them as constraints. These six show up in nearly every production agent. Having a name for them is the first step to handling them.

<div class="glossary">

  <div class="gl-item">
    <div class="gl-term">Lost in the middle</div>
    <div class="gl-def">The model's attention <strong>is not uniform</strong> across the window. The beginning and end get priority; the middle is underused. It's an empirically documented architectural effect (the <em>Lost in the Middle</em> paper, Liu et al., 2023), softened in recent models but not gone.</div>
    <div class="gl-signal">→ SIGNAL · the agent ignores an instruction you know is there, but buried in the middle of a long context. Move it to the start or the end.</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Context rot</div>
    <div class="gl-def">The fuller the window, the more reasoning quality <strong>tends to drop</strong>, even well below the theoretical limit. An agent at 150,000 tokens isn't equivalent to the same agent at 30,000. Compaction isn't only about space — it's also about performance.</div>
    <div class="gl-signal">→ SIGNAL · your agent's first actions are precise, the later ones drift. Compact at 50-60% fill, not at 95%.</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Attention dilution</div>
    <div class="gl-def">A specific case of <em>context rot</em>: even when the model has the theoretical capacity to look at everything, adding irrelevant content <strong>reduces the relative weight</strong> of the relevant content. Noise doesn't just cost tokens — it dilutes signal.</div>
    <div class="gl-signal">→ SIGNAL · adding "just-in-case" documentation degrades performance instead of improving it. Cut what's not useful, never load it "as a precaution".</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Tool soup</div>
    <div class="gl-def">Past a certain number of declared tools (in practice, around fifteen to twenty depending on the model), the agent <strong>starts choosing badly</strong> — close tools confused, missing tools ignored, complex tools mis-parameterized. The bigger it gets, the slower and more wrong it gets.</div>
    <div class="gl-signal">→ SIGNAL · the agent invokes the wrong tool, or forgets one you know was available. Activate tools by phase, not all of them all the time.</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Runaway agent</div>
    <div class="gl-def">Without an explicit cap, an agent can enter a loop where every tool call produces a result that justifies another tool call. The window swells, quality drops, and the bill climbs in silence. Particularly common when the agent searches, doesn't find, and rephrases.</div>
    <div class="gl-signal">→ SIGNAL · a "simple" session burns ten times the tokens you expected. Set a cap on tool calls, add checkpoints, and trigger compaction or stop at fill thresholds.</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Prompt injection</div>
    <div class="gl-def">Any external content — web page, email, file, tool result — can carry <strong>hidden instructions</strong> that hijack the agent. The model doesn't naturally distinguish <em>data</em> from <em>orders</em>. The more powerful the agent's tools, the more serious the risk. Mandatory mental hygiene: treat third-party content as potentially hostile.</div>
    <div class="gl-signal">→ SIGNAL · the agent does something you didn't ask for after reading external content. Mark third-party content, restrict the tools usable after a read, require human validation for irreversible actions.</div>
  </div>

</div>

<div class="section-num">§ 03 — Heuristics</div>

## Eleven principles for <span class="accent">arbitrating</span> competing appetites.

Knowing the solutions and the phenomena isn't enough: you have to know how to compose them. Here are the heuristics I use, and that I see used in production agents. None is revolutionary on its own; their value comes from the *discipline* of applying them together. For each, an alarm signal that triggers it, and a case where it doesn't apply.

<div class="heuristics-list">

  <div class="heuristic">
    <div class="heur-name">Measure before you optimize</div>
    <p class="heur-body">Before trying to compress or rewrite, know <strong>how much each artifact actually weighs</strong>. Every modern API exposes a token count per message. Count first, target the biggest line item, then optimize.</p>
    <div class="heur-signal"><strong>Signal</strong> You "feel" the agent is dragging but you don't know where. Open the logs, count tokens by category (system, tools, history, results).</div>
    <div class="heur-counter"><strong>Except when</strong> Quick prototype to validate an idea. Don't optimize what isn't stable yet.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Precision over exhaustiveness in the system prompt</div>
    <p class="heur-body">The reflex is to stuff the system prompt with examples "just in case". A long system prompt fatigues the model (see <em>context rot</em>) and inflates the cost of every request. Better a <strong>tight</strong> framing that delegates the details to skills loaded on demand.</p>
    <div class="heur-signal"><strong>Signal</strong> System prompt &gt; 30k tokens, or with sections that are never triggered, or rewritten every sprint.</div>
    <div class="heur-counter"><strong>Except when</strong> The business context is so specialized that no skill can replace it (strict regulation, non-negotiable brand voice).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Only wire up the tools you need</div>
    <p class="heur-body">Every declared tool occupies the window <em>even when it's never used</em>. Wiring up ten MCP servers "for the future" means spending thousands of tokens permanently and feeding the <em>tool soup</em>. Activating tools by <strong>task profile or by phase</strong> produces noticeably better agents.</p>
    <div class="heur-signal"><strong>Signal</strong> More than fifteen tools declared, or an agent that hesitates between two close tools.</div>
    <div class="heur-counter"><strong>Except when</strong> You measure and you know that no tool is superfluous. In that case, document the reason for each.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Never load a raw file when you can process it with code</div>
    <p class="heur-body">This is the single most important principle — § 04 is dedicated to it. Asking the model to "look at" a 100,000-line CSV or a fifty-page PDF is the most common cause of saturation. Giving the model a way to <strong>write code that operates on the data</strong> and only bring back the result is the fundamental pivot.</p>
    <div class="heur-signal"><strong>Signal</strong> A single tool call brings back more than 5,000 tokens of context.</div>
    <div class="heur-counter"><strong>Except when</strong> The file is small (&lt; 2k tokens) and the model needs to grasp its entirety (a nuanced re-read of a short text, for example).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Put the essentials at the edges</div>
    <p class="heur-body">Given the <em>lost in the middle</em> effect, critical instructions belong at the start or end of the window. The business rule you don't want ignored? At the end of the system prompt. The most important immediate instruction? In the last user message.</p>
    <div class="heur-signal"><strong>Signal</strong> A documented instruction isn't being followed. Before deciding "the model is dumb", check its position in the window.</div>
    <div class="heur-counter"><strong>Except when</strong> You have little content and everything fits in a short horizon. The rule only kicks in beyond a few thousand tokens.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Stabilize the prefix to enable the KV cache</div>
    <p class="heur-body"><em>Prompt caching</em> only works if the leading portion is <strong>identical from one request to the next, byte-for-byte</strong>. Putting today's date or a session ID right at the top invalidates the cache on every turn. Keeping the prefix immutable and placing variable elements further down is a free optimization — typically 80-90% reduction on stable-prefix cost, and latency cut by two or three times.</p>
    <div class="heur-signal"><strong>Signal</strong> Your Anthropic / OpenAI calls don't show a <em>cache hit</em> even though the system prompt is "identical".</div>
    <div class="heur-counter"><strong>Except when</strong> Your requests are rare or irregular — the cache has a limited lifetime (5 min on Anthropic by default).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Compact early, not in a panic</div>
    <p class="heur-body">Waiting for the window to be full to compact means compacting in a hurry — and badly. Well-built agents trigger compaction <strong>by threshold</strong> (60% fill is a good starting point), with a deliberate strategy: what to summarize, what to prune, what to keep verbatim.</p>
    <div class="heur-signal"><strong>Signal</strong> Compaction kicks in at 95%, or worse, doesn't exist and long sessions crash.</div>
    <div class="heur-counter"><strong>Except when</strong> You're in a session that's short by construction (single-turn, or with a hard cap on calls).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Delegate noisy work to sub-agents</div>
    <p class="heur-body">Any task that involves <strong>reading a lot to produce a little</strong> — web exploration, large file reading, multi-source research — is a natural candidate for a sub-agent. The parent keeps its window light; the sub-agent absorbs the noise in its own and only returns a summary.</p>
    <div class="heur-signal"><strong>Signal</strong> The main agent's context is 70% filled with search results or raw content.</div>
    <div class="heur-counter"><strong>Except when</strong> The task requires the parent to see the detail (audit, traceability, multi-step reasoning over specific items).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Treat all external content as hostile</div>
    <p class="heur-body">A web page, an email, a tool result are data — and they can carry hidden instructions (see <em>prompt injection</em>). For agents with sensitive tools (sending emails, accessing internal systems, executing code), this is non-negotiable. Mark third-party content, restrict the tools usable after a read, require human validation for irreversible actions — disciplines, not options.</p>
    <div class="heur-signal"><strong>Signal</strong> Your agent has access to email, a browser, or external data AND can execute side-effecting actions.</div>
    <div class="heur-counter"><strong>Except when</strong> The agent is purely read-only and has no side-effecting tools. The risk becomes theoretical.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Remember what lasts, not what passes</div>
    <p class="heur-body">Persistent memory is precious but treacherous. You put durable facts in there (preferences, ongoing projects, professional context), not micro-details from a conversation. Useful rule: <strong>if the information isn't relevant in at least three future conversations, it has no business being in memory</strong>.</p>
    <div class="heur-signal"><strong>Signal</strong> Memory contains "the user said X on Tuesday" for X's that will never come back. Or worse, accumulated contradictions.</div>
    <div class="heur-counter"><strong>Except when</strong> It's explicitly a note-taking or personal-journal agent — granular retention is then the feature.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Iterate with evals, not by gut feel</div>
    <p class="heur-body">Context optimization is like performance tuning: intuition is often wrong. Building <strong>a few reproducible tests</strong> — here's a question, here's the expected answer — and measuring the impact of each change prevents silent regressions. Adding a tool or a skill without measuring degrades things surprisingly fast.</p>
    <div class="heur-signal"><strong>Signal</strong> You add a feature and another behavior, with no apparent connection, becomes unstable.</div>
    <div class="heur-counter"><strong>Except when</strong> You're in pure exploration and performance isn't yet a criterion. Once in production, no more excuses.</div>
  </div>

</div>

<div class="section-num">§ 04 — The anti-pattern</div>

## Skills that <span class="accent">read</span> vs. skills that <span class="accent">execute</span>.

This is the most poorly understood distinction in agent engineering. A skill isn't a place where you drop data for the model to contemplate: it's an instruction manual for *operating on it outside the context*. It's also the optimization with the most spectacular gains — often **two orders of magnitude on token consumption**.

<div class="compare">
  <div class="compare-card bad">
    <span class="compare-tag">↯ Anti-pattern</span>
    <h4>The skill that reads</h4>
    <p>Loads the raw file into the window, asks the model to look at everything and then summarize it. Expensive, slow, fragile, capped by file size, and subject to <em>context rot</em>.</p>
  </div>
  <div class="compare-card good">
    <span class="compare-tag">✓ Good pattern</span>
    <h4>The skill that executes</h4>
    <p>Teaches the model to write code that operates on the data — analyze, filter, aggregate, validate. Only the <em>compact result</em> comes back into context. Code sees the bytes, the model sees the aggregate.</p>
  </div>
</div>

### The real cost, in numbers

Concrete case: "How many transactions over $1,000 are there in this 100,000-row CSV?" The file is roughly 8 MB of text, which is roughly **2 million tokens**. Let's compare the two trajectories:

<div class="showcase">
  <div class="lbl">A · The skill that reads (anti-pattern)</div>
  <div class="row head"><span class="lhs">item</span><span class="rhs">tokens</span></div>
  <div class="row bad"><span class="lhs">→ Attempt to load the whole thing</span><span class="rhs">2,000,000</span></div>
  <div class="row bad"><span class="lhs">→ Window limit exceeded (200k)</span><span class="rhs">failure</span></div>
  <div class="row bad"><span class="lhs">→ Fallback strategy: chunking + summaries</span><span class="rhs">~180,000</span></div>
  <div class="row bad"><span class="lhs">→ Result: approximation, no exact count</span><span class="rhs">imprecise</span></div>
  <div class="row total"><span class="lhs">TOTAL · 1 approximate answer</span><span class="rhs">~180,000 tk</span></div>
</div>

<div class="showcase">
  <div class="lbl">B · The skill that executes (good pattern)</div>
  <div class="row head"><span class="lhs">item</span><span class="rhs">tokens</span></div>
  <div class="row"><span class="lhs">→ Skill loaded into context</span><span class="rhs">~400</span></div>
  <div class="row"><span class="lhs">→ Model writes a Python script</span><span class="rhs">~200</span></div>
  <div class="row good"><span class="lhs">→ Script reads the CSV outside context (pandas)</span><span class="rhs">0</span></div>
  <div class="row good"><span class="lhs">→ Script output in context: "47,322"</span><span class="rhs">~5</span></div>
  <div class="row total"><span class="lhs">TOTAL · 1 exact answer</span><span class="rhs">~605 tk</span></div>
</div>

Ratio **~300×**. And along the way: answer B is *exact* whereas A is necessarily approximate. The good pattern is faster, cheaper, and more precise. It's not a tradeoff — it's just a better architecture.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 2</span><span>Two data trajectories</span></div>
  <svg viewBox="0 0 600 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Comparing the two approaches: read vs. execute">
    <text x="20" y="20" class="svg-text" font-weight="700" fill="#c2553a">A · DIRECT READ</text>
    <rect x="20" y="34" width="80" height="50" fill="none" stroke="#c2553a" stroke-width="1.5"/>
    <text x="60" y="58" class="svg-text" text-anchor="middle">file</text>
    <text x="60" y="72" class="svg-text svg-text-dim" font-size="9">8 MB</text>
    <line x1="100" y1="59" x2="180" y2="59" stroke="#c2553a" stroke-width="6"/>
    <polygon points="180,55 192,59 180,63" fill="#c2553a"/>
    <text x="140" y="48" class="svg-text" text-anchor="middle" fill="#c2553a" font-size="9">everything flows in</text>
    <rect x="195" y="34" width="180" height="50" fill="#c2553a" opacity="0.4" stroke="#c2553a" stroke-width="1.5"/>
    <text x="285" y="58" class="svg-text" text-anchor="middle" fill="#f4ecdc" font-weight="700">WINDOW SATURATED</text>
    <text x="285" y="72" class="svg-text" text-anchor="middle" fill="#f4ecdc" font-size="9">the model 'looks at' everything</text>
    <line x1="375" y1="59" x2="455" y2="59" stroke="#c2553a" stroke-width="2"/>
    <polygon points="455,55 467,59 455,63" fill="#c2553a"/>
    <rect x="470" y="34" width="80" height="50" fill="none" stroke="#c2553a" stroke-width="1.5"/>
    <text x="510" y="58" class="svg-text" text-anchor="middle">summary</text>
    <text x="510" y="72" class="svg-text svg-text-dim" font-size="9">approximate</text>
    <line x1="20" y1="120" x2="580" y2="120" stroke="#3d3525" stroke-width="1" stroke-dasharray="2,4"/>
    <text x="20" y="150" class="svg-text" font-weight="700" fill="#7a8b5c">B · CODE EXECUTION</text>
    <rect x="20" y="164" width="80" height="50" fill="none" stroke="#7a8b5c" stroke-width="1.5"/>
    <text x="60" y="188" class="svg-text" text-anchor="middle">file</text>
    <text x="60" y="202" class="svg-text svg-text-dim" font-size="9">8 MB</text>
    <line x1="100" y1="189" x2="180" y2="189" stroke="#7a8b5c" stroke-width="2"/>
    <polygon points="180,185 192,189 180,193" fill="#7a8b5c"/>
    <text x="140" y="178" class="svg-text" text-anchor="middle" fill="#7a8b5c" font-size="9">stays on disk</text>
    <rect x="195" y="164" width="100" height="50" fill="none" stroke="#7a8b5c" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="245" y="184" class="svg-text" text-anchor="middle" fill="#7a8b5c">SKILL.md</text>
    <text x="245" y="200" class="svg-text" text-anchor="middle" font-size="9" fill="#7a8b5c" font-style="italic">→ writes code</text>
    <line x1="295" y1="189" x2="345" y2="189" stroke="#7a8b5c" stroke-width="2"/>
    <polygon points="345,185 357,189 345,193" fill="#7a8b5c"/>
    <rect x="360" y="164" width="80" height="50" fill="#7a8b5c" opacity="0.2" stroke="#7a8b5c" stroke-width="1.5"/>
    <text x="400" y="184" class="svg-text" text-anchor="middle" font-weight="700">exec</text>
    <text x="400" y="200" class="svg-text" text-anchor="middle" font-size="9" fill="#7a8b5c">outside the window</text>
    <line x1="440" y1="189" x2="490" y2="189" stroke="#7a8b5c" stroke-width="2"/>
    <polygon points="490,185 502,189 490,193" fill="#7a8b5c"/>
    <rect x="505" y="164" width="60" height="50" fill="#7a8b5c" opacity="0.7"/>
    <text x="535" y="188" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">data</text>
    <text x="535" y="202" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-size="9">exact</text>
    <text x="300" y="260" class="svg-label-big" text-anchor="middle">code sees the bytes; the model sees the result</text>
    <line x1="60" y1="290" x2="540" y2="290" stroke="#3d3525"/>
    <text x="160" y="312" class="svg-text" text-anchor="middle" fill="#c2553a">A · ~180,000 tk · imprecise · capped</text>
    <text x="440" y="312" class="svg-text" text-anchor="middle" fill="#7a8b5c">B · ~600 tk · exact · scalable</text>
  </svg>
  <figcaption class="diagram-caption">A well-designed skill keeps the data on disk and only brings back the result.</figcaption>
</figure>

This idea — *code execution as context compression* — is the most cost-effective pattern in contemporary agent engineering. When you design a skill, always ask yourself: **does the model need to see the data, or just the result of processing it?** The answer is almost always "the result".

<div class="section-num">§ 05 — Audit</div>

## How to <span class="accent">measure</span> what's really happening in your window.

The rest of this article assumes you know what your agent is consuming. Most teams I meet only have an intuition. The audit isn't complicated; it just demands you do it once and instrument cleanly.

### The four base metrics

For every model call, log four numbers. **Total input tokens** — the full size sent to the model. **Output tokens** — what the model generated. **Cached tokens** (cache hit) — what cost the fraction. **Tokens billed at full price** — the difference. Every serious API (Anthropic, OpenAI, Google) exposes these counters in the response; you just need to capture and aggregate them.

### The breakdown by category

Once the totals are known, split the input. How much for the **system prompt**? How much for **tool definitions**? How much for **history**? How much for **tool results** in the current session? How much for loaded **skills**? At this stage, most production agents discover that *tool results devour 40-60% of the window* and nobody knew. That's typically where you should pull.

### The health indicators

Three indicators are worth tracking over time. The **cache hit rate** — under 70%, your prefix isn't stable. The **average window fill at end of session** — above 70%, you're in *context rot* territory. The **average number of tool calls per session** — if it drifts upward without quality gains, you have a *runaway agent* in formation.

### Practical tools

At minimum, a middleware that captures API counters and writes them to a database or structured log file. To go further: providers offer dashboards (Anthropic Console, OpenAI Usage), giving a global view but without the per-category breakdown. For Claude Code specifically, the `/context` command displays the current window's breakdown in real time — it's the most valuable read to learn. More on this in § 07.

<div class="section-num">§ 06 — Architecture</div>

## Sub-agents: <span class="accent">isolated</span> windows.

When a parent delegates to a sub-agent, it opens a clean window for it. The sub-agent absorbs the noise — raw reading, searches, exploration — then returns only a **compact summary**. The parent receives a telegram, not a flood. It's the pattern that lets an orchestration agent handle problems that exceed its own window by a wide margin.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 3</span><span>Parallel delegation</span></div>
  <svg viewBox="0 0 600 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Parent and sub-agent architecture with isolated windows">
    <rect x="180" y="20" width="240" height="80" fill="#16130e" stroke="#e8a04b" stroke-width="2"/>
    <text x="300" y="45" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700">PARENT</text>
    <text x="300" y="62" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="9">main window</text>
    <text x="300" y="80" class="svg-text" text-anchor="middle" font-size="9" font-style="italic">lightweight, orchestrates</text>
    <line x1="240" y1="100" x2="100" y2="160" stroke="#a89c84" stroke-width="1.5"/>
    <line x1="300" y1="100" x2="300" y2="160" stroke="#a89c84" stroke-width="1.5"/>
    <line x1="360" y1="100" x2="500" y2="160" stroke="#a89c84" stroke-width="1.5"/>
    <text x="170" y="130" class="svg-text svg-text-faint" text-anchor="middle" font-size="9">delegates</text>
    <text x="430" y="130" class="svg-text svg-text-faint" text-anchor="middle" font-size="9">in parallel</text>
    <rect x="30" y="160" width="140" height="120" fill="#0f0d0a" stroke="#4d8a8a" stroke-width="1.5"/>
    <text x="100" y="180" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-weight="700" font-size="9">SUB-AGENT 1</text>
    <rect x="40" y="190" width="120" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="40" y="200" width="100" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="40" y="210" width="115" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="40" y="220" width="90" height="6" fill="#4d8a8a" opacity="0.25"/>
    <rect x="40" y="230" width="120" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="40" y="240" width="105" height="6" fill="#4d8a8a" opacity="0.3"/>
    <text x="100" y="262" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-size="9" font-style="italic">absorbs the noise</text>
    <text x="100" y="274" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="8">file · web · search</text>
    <rect x="230" y="160" width="140" height="120" fill="#0f0d0a" stroke="#4d8a8a" stroke-width="1.5"/>
    <text x="300" y="180" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-weight="700" font-size="9">SUB-AGENT 2</text>
    <rect x="240" y="190" width="120" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="240" y="200" width="115" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="240" y="210" width="105" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="240" y="220" width="120" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="240" y="230" width="95" height="6" fill="#4d8a8a" opacity="0.4"/>
    <text x="300" y="262" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-size="9" font-style="italic">isolated window</text>
    <text x="300" y="274" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="8">own context</text>
    <rect x="430" y="160" width="140" height="120" fill="#0f0d0a" stroke="#4d8a8a" stroke-width="1.5"/>
    <text x="500" y="180" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-weight="700" font-size="9">SUB-AGENT 3</text>
    <rect x="440" y="190" width="120" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="440" y="200" width="100" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="440" y="210" width="115" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="440" y="220" width="105" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="440" y="230" width="118" height="6" fill="#4d8a8a" opacity="0.4"/>
    <text x="500" y="262" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-size="9" font-style="italic">parallelizable</text>
    <text x="500" y="274" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="8">independent</text>
    <line x1="100" y1="280" x2="240" y2="328" stroke="#e8a04b" stroke-width="1"/>
    <polygon points="238,323 246,330 234,332" fill="#e8a04b"/>
    <line x1="300" y1="280" x2="300" y2="328" stroke="#e8a04b" stroke-width="1"/>
    <polygon points="296,322 300,332 304,322" fill="#e8a04b"/>
    <line x1="500" y1="280" x2="360" y2="328" stroke="#e8a04b" stroke-width="1"/>
    <polygon points="362,323 354,330 366,332" fill="#e8a04b"/>
    <rect x="220" y="330" width="160" height="32" fill="#e8a04b" opacity="0.2" stroke="#e8a04b" stroke-width="1.5"/>
    <text x="300" y="350" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700" font-size="10">COMPACT SUMMARIES</text>
    <text x="300" y="376" class="svg-text" text-anchor="middle" fill="#c2553a" font-size="9" font-style="italic">⚠ compression is irreversible</text>
  </svg>
  <figcaption class="diagram-caption">Each sub-agent opens its own window, handles the noise, returns a telegram.</figcaption>
</figure>

### Advantages

**Isolation**: a sub-agent that saturates its own window doesn't affect the parent. **Parallelization**: several sub-agents can work simultaneously, which a monolithic agent's single window forbids. **Specialization**: each sub-agent can have its own system prompt and its own tools, finely tuned to its task.

### Limit

Compression is *irreversible*. If the sub-agent omits a detail in its summary, the parent has no way to recover it — short of re-running a delegation, which costs a full new cycle. That's why sub-agents demand particular care in defining their *return contract*: what *must* it surface, even if it lengthens the summary?

<div class="section-num">§ 07 — Practical focus</div>

## How this plays out in <span class="accent">Claude Code</span> and friends.

You're probably using Claude Code, Cursor, Cline, or a homegrown agent built on the Anthropic or OpenAI API. Here's how the previous principles show up in those tools — and where to look to diagnose them.

### Read the window in real time

In Claude Code, the `/context` command displays the exact breakdown of your current window: system prompt, MCP tools, loaded skills, history, tool results. It's the most useful read to learn. Run it regularly during a long session; you'll quickly identify which item is eating space. Most of the time, it's tool results — typically `Read`s of large files or `Bash`es returning bulky JSON.

### Automatic compaction

Claude Code triggers automatic compaction when the window approaches its limit. Older turns are replaced by a summary. You can also trigger it manually with `/compact`, adding instructions on what compaction must preserve ("keep the list of files I modified, the Bash commands run and their result"). Compacting early and with explicit instructions almost always gives better results than letting auto-compaction decide alone at the edge of the cliff.

### MCP arbitration

When you wire up several MCP servers (GitHub, Linear, database, Sentry, etc.), each adds its own tool definitions permanently. Measure the cost: `/context` gives it to you. If you see 20-30k tokens in MCP tools that only get used occasionally, consider activating servers *per project* via configuration rather than globally. It's one of the highest-yield levers on Claude Code.

### Skills, in practice

`SKILL.md` files aren't loaded by default: they're described in the system prompt as an index, and the agent opens them via their `view` tool when a trigger matches. This design is *the direct application of § 04*: the procedure only occupies the window on demand, and only when it serves. When you write your own skills, follow the same principle: short instructions, references to code, never raw data packaged into the markdown.

### The Task sub-agent

Claude Code exposes a `Task` tool that launches a sub-agent with its own context. Excellent application of § 06: delegate multi-file searches, large-directory exploration, code audits to a sub-agent. You'll get back a summary instead of flooding your main context.

### Cursor, Cline, Copilot, and the others

The principles are the same, the instrumentation differs. Cursor exposes less visibility into the window's composition; you often have to go through the API logs. Cline and the open-source agents based on the Model Context Protocol generally expose more detail. Whatever the tool, the question to ask stays the same: *what's filling my window, and why?*

<div class="section-num">§ 08 — State of play</div>

## Where we are, in <span class="accent">May 2026</span>.

The terrain shifts fast. This section is dated for that reason: what's true at the time of publication may not be six months from now. A few notable trends you can fold into your engineering thinking.

**Standard windows have stalled around 200k**, but experimental offerings at 1M tokens exist (Claude Sonnet in beta, Gemini for a while now). The per-token cost in "long context" mode stays meaningfully higher, and degradation at large window is more pronounced — in other words, the "1M" option is useful for singular cases (a large document to process at once) but remains a poor default reflex.

**The KV cache has become a universal given**. Anthropic, OpenAI, and Google all expose prompt caching mechanisms with explicit pricing. If you're not using them, you're leaving money on the table. Stable-prefix discipline is no longer an advanced optimization: it's the baseline expectation.

**MCP has become the de facto standard** for declaring third-party tools. The ecosystem now includes hundreds of public servers, which is both an opportunity (huge capabilities accessible quickly) and a trap (the *tool soup* temptation). The 2026 challenge is less about *plugging in* and more about *judiciously choosing what to plug in*.

**Skills have left the margins**. Anthropic popularized them in 2025 with Claude Code; the pattern has spread. Agents without an explicit skill system tend to accumulate everything in the system prompt — meaning they pay permanently for what they could load on demand.

**The "code execution as context compression" pattern** — the idea from § 04 — has become a topic in the agent engineering community and the subject of technical articles from Anthropic and others. If you haven't applied it in your architecture yet, it's probably the highest-priority item for your next iteration.

**Systematic evaluation remains under-practiced**. It's the discipline I see least often in place at teams building agents; and paradoxically it's the one that lets you apply all the others with confidence. Things are moving — tools like Promptfoo, Inspect, and Anthropic's evals are spreading — but the gap between teams that evaluate and teams that don't remains considerable.

<aside class="pull-quote">
  <p>Every token has a cost, every artifact has a failure mode, and agent engineering is largely about arbitrating these competing appetites.</p>
</aside>

<div class="further">
  <div class="further-label">★ Further reading</div>
  <ul>
    <li>
      <a href="https://www.anthropic.com/news/context-engineering" target="_blank" rel="noopener">Anthropic · Effective context engineering for AI agents</a>
      <span class="desc">The founding article on the discipline, by Anthropic's applied AI team.</span>
    </li>
    <li>
      <a href="https://arxiv.org/abs/2307.03172" target="_blank" rel="noopener">Liu et al. · Lost in the Middle (2023)</a>
      <span class="desc">The paper that empirically documented attention's non-uniformity across the window.</span>
    </li>
    <li>
      <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener">Model Context Protocol · specification</a>
      <span class="desc">The open standard for declaring and exposing tools to agents.</span>
    </li>
    <li>
      <a href="https://docs.claude.com/en/docs/claude-code/overview" target="_blank" rel="noopener">Anthropic · Claude Code documentation</a>
      <span class="desc">The reference for the <code>/context</code> and <code>/compact</code> commands, and the skill system.</span>
    </li>
    <li>
      <a href="https://docs.claude.com/en/docs/build-with-claude/prompt-caching" target="_blank" rel="noopener">Anthropic · Prompt caching</a>
      <span class="desc">How to enable the KV cache and structure your prefix to get the most out of it.</span>
    </li>
    <li>
      <a href="/en/articles/understanding-llms/">Companion article · What's Really Happening When You Talk to an AI</a>
      <span class="desc">The conceptual foundations, for sharing with a less technical audience.</span>
    </li>
  </ul>
</div>
