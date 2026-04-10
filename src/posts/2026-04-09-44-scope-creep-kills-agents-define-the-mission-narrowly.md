---
title: "44. La dérive du périmètre tue les agents — définissez la mission étroitement"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Every successful agent faces the same pressure: it works, so people want it to do more."
---

Every successful agent faces the same pressure: it works, so people want it to do more. The customer service agent that handles returns gets asked to handle billing questions. The code review agent that checks for bugs gets asked to suggest architectural improvements. The research agent that summarizes documents gets asked to draft reports. Each extension seems incremental. Together they produce an agent that does too many things, does none of them as well as it should, and fails in ways that are hard to attribute to any single decision.

Scope creep in agents is more damaging than scope creep in conventional software because agents don't fail cleanly at their boundaries. A function called with the wrong arguments throws an exception. An agent asked to do something outside its design space will attempt it — and produce something that looks like a result, which is worse than an error. The user thinks the task was done. The agent has done something adjacent to what was asked, or confabulated a response, or applied a framework from its primary task to a secondary task where it doesn't fit. The failure is silent and the consequences arrive later.

The defense is a clear, written definition of what the agent is for — specific enough that you can answer, for any proposed extension, whether it's inside or outside scope. Not "helps with customer service" but "handles product return requests for orders placed in the last ninety days, escalates to a human for anything else." The specificity isn't bureaucracy — it's the thing that lets you say no coherently when the fifth team asks to add one more capability.

Saying no to scope extension is a product decision with real tradeoffs. Sometimes the extension is worth making — the capability is closely related, the agent handles it well, the user need is genuine. The point isn't to never extend, but to extend deliberately, with a full prompt review, a new round of testing, and explicit acknowledgment that you're changing what the agent is. Not an incremental tweak — a new version with a new scope.

The agent that does one thing exceptionally well is more valuable than the agent that does ten things adequately. Protect the mission.
