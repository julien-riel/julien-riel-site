---
title: "What's Really Happening When You Talk to an AI"
date: 2026-05-08
tags:
  - fundamentals
  - primer
description: "Tokens, transformers, context window, system prompt, tools: the conceptual foundations for really understanding how ChatGPT, Claude, or Gemini work. No equations."
---

<p class="deck">
Three ideas are enough to understand — and demystify — ChatGPT, Claude, Gemini, and all their cousins: the token, the transformer, and the context window. No formulas, just analogies that hold up.
</p>

<div class="section-num">§ 01 — The raw material</div>

## The AI doesn't read words. It reads <span class="accent">tokens</span>.

First surprise: when you type "Hello, how are you?" to an AI, it doesn't see your sentence, your words, or even your letters. It sees a sequence of **tokens** — fragments produced by an automatic split of your text.

A token isn't a whole word or a single letter: it's a fragment, somewhere in between. In English, a token is typically **3 to 4 characters**, or about three quarters of a word. The word `window` may fit in a single token because it's frequent. `windows` might split in two (`window` + `s`). A rare proper noun or a technical word can shatter into four or five pieces.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 1</span><span>A sentence, as the AI sees it</span></div>
  <svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A sentence broken down into tokens">
    <text x="20" y="30" class="svg-text svg-text-faint">┌─ WHAT YOU WRITE</text>
    <text x="20" y="58" class="svg-label-big" font-size="18">"La fenêtre de contexte est finie."</text>
    <line x1="300" y1="72" x2="300" y2="98" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="296,92 300,102 304,92" fill="#a89c84"/>
    <text x="312" y="89" class="svg-text svg-text-dim" font-size="9" font-style="italic">automatic tokenization</text>
    <text x="20" y="120" class="svg-text svg-text-faint">└─ WHAT THE AI SEES</text>
    <g font-family="JetBrains Mono, monospace" font-size="11">
      <rect x="20" y="130" width="38" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="39" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">La</text>
      <rect x="62" y="130" width="58" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="91" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">fenêtre</text>
      <rect x="124" y="130" width="34" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="141" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">de</text>
      <rect x="162" y="130" width="62" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="193" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">contexte</text>
      <rect x="228" y="130" width="36" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="246" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">est</text>
      <rect x="268" y="130" width="38" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="287" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">fin</text>
      <rect x="310" y="130" width="22" height="28" fill="#ffc26b" opacity="0.85"/>
      <text x="321" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">ie</text>
      <rect x="336" y="130" width="20" height="28" fill="#5a5040" opacity="0.85"/>
      <text x="346" y="148" text-anchor="middle" fill="#f4ecdc" font-weight="700">.</text>
    </g>
    <g font-family="JetBrains Mono, monospace" font-size="8" fill="#5a5040">
      <text x="39" y="174" text-anchor="middle">1</text>
      <text x="91" y="174" text-anchor="middle">2</text>
      <text x="141" y="174" text-anchor="middle">3</text>
      <text x="193" y="174" text-anchor="middle">4</text>
      <text x="246" y="174" text-anchor="middle">5</text>
      <text x="287" y="174" text-anchor="middle">6</text>
      <text x="321" y="174" text-anchor="middle">7</text>
      <text x="346" y="174" text-anchor="middle">8</text>
    </g>
    <text x="380" y="148" class="svg-text svg-text-dim" font-style="italic">8 tokens · "finie" splits in two</text>
  </svg>
  <figcaption class="diagram-caption">Tokenization favors frequent fragments. The French word *finie* splits as *fin* + *ie* (French example kept for layout).</figcaption>
</figure>

Why does this matter to you? Because everything in AI tools is measured in tokens: the bill if you pay by usage, the maximum length of a conversation, the size of documents you can analyze. When a provider advertises "200,000 tokens of context," that's roughly **500 pages of book**. When you paste a document in, it gets sliced into tokens before the model looks at it.

<div class="section-num">§ 02 — The mechanics</div>

## One operation, repeated thousands of times: predict the <span class="accent">next token</span>.

Here's the most counterintuitive idea in the field, and the one that changes everything: however sophisticated it gets, a large language model fundamentally does only one thing. **Given a sequence of tokens, predict the one that comes next.**

No global planning. No upfront thinking about the whole answer. No hidden plan. One token at a time, in a loop that only stops when the model decides it's done.

How does it do it? The architecture that performs this prediction is called a **transformer**. What you need to remember, without diving into the machinery, is its central principle — *attention*. For each token to produce, the model weighs the relative importance of every token already there. Each word looks at all the others and decides which ones matter. A kind of full re-read at every step.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 2</span><span>The loop, one step at a time</span></div>
  <svg viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Next-token prediction loop">
    <text x="20" y="24" class="svg-text svg-text-faint" font-weight="700">STEP t</text>
    <g font-family="JetBrains Mono, monospace" font-size="10">
      <rect x="20" y="38" width="42" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="41" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">La</text>
      <rect x="66" y="38" width="62" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="97" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">fenêtre</text>
      <rect x="132" y="38" width="36" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="150" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">de</text>
      <rect x="172" y="38" width="62" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="203" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">contexte</text>
      <rect x="238" y="38" width="36" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="256" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">est</text>
      <rect x="278" y="38" width="20" height="26" fill="none" stroke="#a89c84" stroke-dasharray="2,2"/>
      <text x="288" y="55" text-anchor="middle" fill="#a89c84">?</text>
    </g>
    <line x1="160" y1="74" x2="160" y2="96" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="156,90 160,100 164,90" fill="#a89c84"/>
    <rect x="60" y="100" width="200" height="40" fill="#16130e" stroke="#e8a04b" stroke-width="1.5"/>
    <text x="160" y="118" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700">TRANSFORMER</text>
    <text x="160" y="132" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="9" font-style="italic">attention over all tokens</text>
    <line x1="160" y1="142" x2="160" y2="164" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="156,158 160,168 164,158" fill="#a89c84"/>
    <text x="20" y="180" class="svg-text svg-text-faint" font-size="9">PROBABILITIES · top 4 candidates</text>
    <g font-family="JetBrains Mono, monospace" font-size="9">
      <rect x="20" y="186" width="100" height="14" fill="#7a8b5c" opacity="0.85"/>
      <text x="124" y="197" fill="#a89c84">finie · 0.62</text>
      <rect x="20" y="204" width="48" height="14" fill="#7a8b5c" opacity="0.55"/>
      <text x="124" y="215" fill="#a89c84">limitée · 0.18</text>
      <rect x="20" y="222" width="22" height="14" fill="#7a8b5c" opacity="0.4"/>
      <text x="124" y="233" fill="#a89c84">large · 0.07</text>
      <rect x="20" y="240" width="14" height="14" fill="#7a8b5c" opacity="0.3"/>
      <text x="124" y="251" fill="#a89c84">vaste · 0.04</text>
    </g>
    <line x1="240" y1="193" x2="320" y2="193" stroke="#e8a04b" stroke-width="1.5"/>
    <polygon points="318,189 328,193 318,197" fill="#e8a04b"/>
    <text x="280" y="184" class="svg-text" text-anchor="middle" fill="#e8a04b" font-size="9" font-style="italic">picks</text>
    <text x="340" y="180" class="svg-text svg-text-faint" font-weight="700">STEP t+1</text>
    <g font-family="JetBrains Mono, monospace" font-size="10">
      <rect x="340" y="186" width="60" height="22" fill="#a89c84" opacity="0.4"/>
      <text x="370" y="201" text-anchor="middle" fill="#a89c84">… est</text>
      <rect x="404" y="186" width="48" height="22" fill="#ffc26b"/>
      <text x="428" y="201" text-anchor="middle" fill="#0f0d0a" font-weight="700">finie</text>
      <rect x="456" y="186" width="20" height="22" fill="none" stroke="#a89c84" stroke-dasharray="2,2"/>
      <text x="466" y="201" text-anchor="middle" fill="#a89c84">?</text>
    </g>
    <path d="M 480 197 Q 540 197 540 130 Q 540 80 280 80" fill="none" stroke="#c2553a" stroke-width="1.5" stroke-dasharray="3,3"/>
    <polygon points="285,76 275,80 285,84" fill="#c2553a"/>
    <text x="555" y="140" class="svg-text" text-anchor="middle" fill="#c2553a" font-size="9" font-style="italic" transform="rotate(90 555 140)">loop again</text>
    <text x="300" y="276" class="svg-label-big" text-anchor="middle">one token at a time, until done</text>
  </svg>
  <figcaption class="diagram-caption">At each step, the model re-reads the whole input to pick one token. (Example continues from Fig. 1.)</figcaption>
</figure>

This mechanic has a striking practical consequence. When the AI replies to you, it *doesn't know*, at the moment it writes the first word, how it will end its sentence. It writes, word after word, re-reading itself at every step to decide the next one. What looks like fluid thought is a string of probabilistic micro-decisions. That's why an AI can start a confident answer and end up with a false claim — it "drifted along" with its own generation.

<div class="section-num">§ 03 — The field of view</div>

## The <span class="accent">context window</span>, or why your AI "forgets."

If the model only predicts the next token from what came before, it needs a horizon — the amount of tokens it can "see" at once. That horizon is the **context window**.

This is *the* central notion to internalize if you use AI tools regularly. The window is everything at once: the model's zone of attention, its field of view, and its *only* support for information. Anything inside it can shape the response; anything outside doesn't exist for it.

This window has a **maximum size**, fixed when the model is built, measured in tokens. Depending on the model, you're talking a few thousand to several hundred thousand tokens. For current Claude models, for instance, the standard window is around **200,000 tokens** — equivalent to a 500-page book. Beyond that, you can't add anything: you have to remove existing content to make room.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 3</span><span>The window, at a glance</span></div>
  <svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The context window as a strip of tokens with a maximum limit">
    <text x="20" y="22" class="svg-text svg-text-faint">┌─ CONTEXT WINDOW</text>
    <text x="580" y="22" class="svg-text svg-text-faint" text-anchor="end">maximum capacity ─┐</text>
    <rect x="20" y="32" width="560" height="56" fill="none" stroke="#3d3525" stroke-width="1.5"/>
    <g font-family="JetBrains Mono, monospace" font-size="9">
      <rect x="22" y="34" width="280" height="52" fill="#e8a04b" opacity="0.15"/>
      <rect x="26" y="40" width="22" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="50" y="40" width="36" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="88" y="40" width="22" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="112" y="40" width="42" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="156" y="40" width="22" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="180" y="40" width="32" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="214" y="40" width="26" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="242" y="40" width="38" height="18" fill="#e8a04b" opacity="0.7"/>
      <text x="160" y="76" text-anchor="middle" fill="#a89c84" font-style="italic" font-size="9">tokens already present · what the model "sees"</text>
    </g>
    <rect x="302" y="34" width="276" height="52" fill="none" stroke-dasharray="2,3" stroke="#3d3525"/>
    <text x="440" y="64" class="svg-text svg-text-faint" text-anchor="middle" font-style="italic">available space</text>
    <text x="440" y="78" class="svg-text svg-text-faint" text-anchor="middle" font-size="9">for what comes next</text>
    <line x1="580" y1="28" x2="580" y2="92" stroke="#c2553a" stroke-width="2"/>
    <text x="578" y="104" class="svg-text" text-anchor="end" fill="#c2553a" font-size="9" font-weight="700">↑ ~200k tokens</text>
    <text x="578" y="116" class="svg-text" text-anchor="end" fill="#c2553a" font-size="9" font-style="italic">beyond: impossible</text>
    <path d="M 160 130 L 160 150 L 300 150 L 300 130" fill="none" stroke="#e8a04b" stroke-width="1"/>
    <line x1="230" y1="150" x2="230" y2="170" stroke="#e8a04b" stroke-width="1"/>
    <polygon points="226,164 230,174 234,164" fill="#e8a04b"/>
    <text x="230" y="186" class="svg-label-big" text-anchor="middle">the model predicts here, from all of this</text>
  </svg>
  <figcaption class="diagram-caption">A strip of tokens with a hard limit. No memory anywhere else.</figcaption>
</figure>

### Why your AI "forgets" after a while

You've probably had this experience: in a long conversation, the assistant seems to forget what you told it earlier. Not a bug. A direct consequence of what we just saw. When the history hits the window's limit, the application driving the model has to cut: either it prunes old messages, or it replaces them with a shorter summary. Either way, the original detail is lost to the model.

For the same reason, loading an 800-page PDF into a 200,000-token window may simply not fit. Past that, the tool has to get clever — chunk the document, load only relevant excerpts, or refuse. No magic.

<div class="section-num">§ 04 — The transformation</div>

## From a text predictor to an <span class="accent">assistant</span> that answers.

Here's the second counterintuitive idea in the field. A transformer, left to itself, doesn't "answer" questions. It **continues** text. Give it "The capital of France is" and it will most likely complete with "Paris." Give it "Hello, how are you?" and it could just as easily continue with "asked Mary as she opened the door." — because that, too, is a plausible continuation in the corpus of texts it was trained on.

For it to behave like an assistant — to *answer* instead of *continue* — you have to give it a frame. That frame is called the **system prompt**: a text placed before the conversation that defines who the model is, what it should do, how it should express itself, and what it should refuse. It's the first thing it "reads" on every request, and everything else is built on top of it.

The system prompt is the mechanism that powers the products you use — ChatGPT, Claude.ai, Gemini, Copilot, Le Chat. Behind every conversational interface, an instruction text turns a raw predictor into an assistant. Here's what that looks like in practice:

<div class="raw-prompt">You are an assistant for developers, specialized in SQL databases. You answer in English. For each question, you first provide the SQL query, then a brief explanation. If a question is outside your domain, you say so.

User: How do I find all customers who ordered in March?

Assistant: SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id WHERE EXTRACT(MONTH FROM o.date) = 3; — the join brings customers together with their orders, the EXTRACT filter isolates the ones from March.</div>

Change the system prompt, you change the assistant. Here's the same model, with another identity:

<div class="raw-prompt">You are Lea, a customer support agent for the ACME online store. You are polite, empathetic, and always formal. You only discuss orders, deliveries, and returns. For any other question, you redirect to the general contact form. You never reveal that you are an AI.

User: My order #4521 still hasn't arrived.

Assistant: Hello, I'm sorry for the delay. Could you confirm your postal code so I can check the status of order #4521?</div>

Same brain, two personalities. Everything is continuous text — the *User:* and *Assistant:* labels are nothing more than textual markers that help the model know when its turn to *continue* arrives.

<div class="section-num">§ 05 — The missing memory</div>

## The model remembers <span class="accent">nothing</span>.

Here's the third idea to internalize, and it has very concrete implications for you. The transformer is **stateless**. Between two requests, it has no memory of what was said. None. For a conversation to feel continuous, the application talking to the model has to **resend the entire conversation on every turn**.

When you type "And its population?" in a discussion that was about Canada, the application rebuilds the whole history behind the scenes and sends it to the model:

<div class="raw-prompt">You are a helpful, honest, and concise assistant. You answer in English.

User: What is the capital of Canada?

Assistant: The capital of Canada is Ottawa, in Ontario.

User: And its population?

Assistant: <span class="cursor">▮</span></div>

Everything is there, in one long string. The model receives that block, sees that it ends with *Assistant:* followed by a cursor, and continues the text. Without that full reconstruction, it would have no idea what "its" refers to in the last question.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 4</span><span>One conversation, two requests</span></div>
  <svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Diagram showing that on every turn, the entire conversation is sent back to the model">
    <text x="20" y="22" class="svg-text svg-text-faint" font-weight="700">TURN 1</text>
    <rect x="20" y="32" width="160" height="60" fill="#16130e" stroke="#3d3525" stroke-width="1"/>
    <g font-family="JetBrains Mono, monospace" font-size="9">
      <rect x="26" y="38" width="148" height="14" fill="#c2553a" opacity="0.5"/>
      <text x="30" y="48" fill="#f4ecdc">SYS · frame</text>
      <rect x="26" y="56" width="100" height="14" fill="#f4ecdc" opacity="0.5"/>
      <text x="30" y="66" fill="#0f0d0a">USR · capital?</text>
    </g>
    <text x="100" y="106" class="svg-text svg-text-faint" font-size="9" text-anchor="middle">→ model predicts</text>
    <line x1="180" y1="62" x2="220" y2="62" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="218,58 228,62 218,66" fill="#a89c84"/>
    <rect x="230" y="44" width="120" height="36" fill="#16130e" stroke="#e8a04b" stroke-width="1.5"/>
    <text x="290" y="66" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700">MODEL</text>
    <line x1="350" y1="62" x2="390" y2="62" stroke="#e8a04b" stroke-width="1.5"/>
    <polygon points="388,58 398,62 388,66" fill="#e8a04b"/>
    <rect x="400" y="50" width="180" height="22" fill="#7a8b5c" opacity="0.4" stroke="#7a8b5c" stroke-width="1"/>
    <text x="490" y="65" class="svg-text" text-anchor="middle" font-size="9">"Ottawa, in Ontario."</text>
    <line x1="20" y1="118" x2="580" y2="118" stroke="#3d3525" stroke-width="1" stroke-dasharray="2,4"/>
    <text x="20" y="138" class="svg-text svg-text-faint" font-weight="700">TURN 2 · contains everything before</text>
    <rect x="20" y="148" width="160" height="92" fill="#16130e" stroke="#3d3525" stroke-width="1"/>
    <g font-family="JetBrains Mono, monospace" font-size="9">
      <rect x="26" y="154" width="148" height="14" fill="#c2553a" opacity="0.5"/>
      <text x="30" y="164" fill="#f4ecdc">SYS · frame</text>
      <rect x="26" y="172" width="100" height="14" fill="#f4ecdc" opacity="0.5"/>
      <text x="30" y="182" fill="#0f0d0a">USR · capital?</text>
      <rect x="26" y="190" width="120" height="14" fill="#7a8b5c" opacity="0.6"/>
      <text x="30" y="200" fill="#0f0d0a">AST · Ottawa…</text>
      <rect x="26" y="208" width="100" height="14" fill="#f4ecdc" opacity="0.5"/>
      <text x="30" y="218" fill="#0f0d0a">USR · population?</text>
    </g>
    <line x1="180" y1="194" x2="220" y2="194" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="218,190 228,194 218,198" fill="#a89c84"/>
    <rect x="230" y="176" width="120" height="36" fill="#16130e" stroke="#e8a04b" stroke-width="1.5"/>
    <text x="290" y="198" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700">MODEL</text>
    <line x1="350" y1="194" x2="390" y2="194" stroke="#e8a04b" stroke-width="1.5"/>
    <polygon points="388,190 398,194 388,198" fill="#e8a04b"/>
    <rect x="400" y="182" width="180" height="22" fill="#7a8b5c" opacity="0.4" stroke="#7a8b5c" stroke-width="1"/>
    <text x="490" y="197" class="svg-text" text-anchor="middle" font-size="9">"~1.1 million."</text>
    <path d="M 100 92 Q 100 110 100 148" fill="none" stroke="#e8a04b" stroke-width="1" stroke-dasharray="2,3"/>
    <polygon points="96,144 100,154 104,144" fill="#e8a04b"/>
    <text x="160" y="128" class="svg-text" fill="#e8a04b" font-size="9" font-style="italic">keep it all</text>
    <text x="300" y="270" class="svg-label-big" text-anchor="middle">the history grows with every turn</text>
    <text x="300" y="294" class="svg-text svg-text-dim" text-anchor="middle" font-size="10" font-style="italic">the model itself has no memory between requests</text>
  </svg>
  <figcaption class="diagram-caption">The app rebuilds the history on every call. The app is what 'remembers,' not the model.</figcaption>
</figure>

This absence of internal memory has a very concrete consequence: each new exchange in a conversation **re-pays the cost of everything that came before**. The further the conversation goes, the more expensive each turn is in tokens, and the more the window fills up. That's why very long conversations end up dragging, slowing down, or restarting in a new thread.

It's also why modern products are starting to expose **persistent memory** features — a separate store from the conversation where the system records lasting facts about you (preferences, projects, professional context) to re-inject when relevant. It's not the model that remembers: it's the application that reminds it.

<div class="section-num">§ 06 — Action</div>

## How an AI can <span class="accent">act</span> on the world.

If an AI only predicts tokens, how can it "read a file," "search the web," or "send an email"? The answer is elegant: it still does nothing other than produce text — but that text can take the shape of an **action instruction** that the host program will recognize and execute on its behalf.

The trick comes down to two ingredients. First, you teach the model, in its system prompt, that it has access to **tools**: read a file, search the web, run code, etc. Second, the application watches what the model writes. When it produces a line that looks like a tool call — something like `read_file("/data/report.txt")` — the application intercepts it, actually runs the operation, and injects the result into the conversation. From the model's perspective, everything stays continuous text. From the application's perspective, it's the one doing the real work.

Here's what a full cycle looks like, in continuous text:

<div class="raw-prompt">User: Summarize the file /data/report.txt for me.

Action: read_file("/data/report.txt")
Observation: <span class="injected">The quarterly report shows a 12% revenue increase, an 8% drop in infrastructure costs, and three strategic recommendations [...4,200 tokens total...]</span>

Reply: The report shows a 12% revenue increase, an 8% drop in costs, and three strategic recommendations for next quarter.</div>

The model *requests* an action. The application *performs* it. The result comes back into context, the model sees it as if it had always known, and continues. This is the fundamental mechanic of modern assistants — Claude reading your Google Drive, ChatGPT searching the web, GitHub Copilot editing your code. Always the same loop: the AI asks, the app executes, the result returns to context.

### The cost on the window

All of this leaves a trace in the window, and every trace costs tokens. Reading a fifty-page file means dropping fifty pages into the window. Doing ten web searches means adding ten pages of results. That's why modern agents — the ones that chain actions on their own — can saturate their window surprisingly fast. And it's also the main subject of the next article, for anyone who wants to go deeper.

<aside class="pull-quote">
  <p>However sophisticated it gets, a large language model fundamentally does only one thing: predict the next token. Everything else is stagecraft — clever stagecraft, but stagecraft.</p>
</aside>

<div class="section-num">§ 07 — Takeaways</div>

## Three ideas that explain <span class="accent">everything</span>.

If you leave this page with three things in mind, let them be these. **First** — the AI reads tokens, not words, and everything it sees has to fit in a fixed-size window. **Second** — it does only one operation, predict the next token, in a loop that re-reads the input at every step. **Third** — it has no memory between requests: the application around it simulates continuity by resending the history on every turn, and actually executes the tools the model asks for.

With those three ideas, you can explain why your assistant forgets after a while, why a long document might "not fit," why the same model behaves differently from one product to another, and why an agent that consults a lot of sources can become slow or imprecise. Everything you read about the topic afterward — *RAG*, *MCP*, *compaction*, *sub-agents* — will be variations on these same constraints.

<div class="bridge">
  <div class="bridge-label">★ Further reading</div>
  <h3>If you build with agents, the story continues.</h3>
  <p>This article lays the foundations. If you use Claude Code, Cursor, custom agents, or you design your own tools on top of these models, the context window becomes a resource you have to manage actively: arbitrating between system prompt, tools, history, operation results, and persistent memory.</p>
  <p>The next article explores all of that in detail — the full toolkit of agent engineering, the phenomena that degrade quality, and the practical heuristics for staying below saturation.</p>
  <p><a class="bridge-cta" href="/en/articles/context-window/">Read the practitioner version</a></p>
</div>
