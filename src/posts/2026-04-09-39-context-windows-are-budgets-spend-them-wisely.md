---
title: "39. Context Windows Are Budgets — Spend Them Wisely"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "A context window isn't infinite space — it's a budget."
---

A context window isn't infinite space — it's a budget. Everything you put in costs tokens, competes for the model's attention, and potentially crowds out something more important. Treating the context window as a dumping ground for everything that might be relevant is one of the most common and costly mistakes in agentic system design.

The attention problem is subtler than the token limit problem. Models don't treat all parts of the context equally. Content near the beginning and end of the context tends to receive more attention than content in the middle — the so-called lost-in-the-middle effect. A context window that's technically within the token limit can still produce degraded performance if the most important information is buried in the middle of a long document, surrounded by less relevant material.

This means curation matters as much as capacity. The goal isn't to fit as much as possible into the context — it's to put the right things in, in the right order, at the right level of detail. A well-curated context of ten thousand tokens often produces better results than a bloated context of fifty thousand. The discipline is to ask, for every piece of information you're considering including: does the agent actually need this to do the task? If the answer isn't clearly yes, leave it out.

Retrieval systems make this worse before they make it better. The temptation is to retrieve generously — pull in the top twenty documents rather than the top five, just in case one of the less-likely candidates turns out to be relevant. The result is a context full of marginally relevant material that dilutes the signal. Better retrieval, not more retrieval, is the path to better context.

Conversation history is another common source of context bloat. Full history feels safe — you're not losing anything. But long histories push early context out of effective attention range and fill the window with content that's no longer relevant to the current task. Summarizing earlier turns, or dropping them selectively, often produces better results than preserving everything.

The context window is the most expensive real estate in your system. Treat it accordingly.
