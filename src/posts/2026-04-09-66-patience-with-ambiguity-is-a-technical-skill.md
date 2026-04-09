---
title: "66. Patience with Ambiguity Is a Technical Skill"
date: 2026-04-09
tags:
  - agentic-programming
  - mindset
description: "Ambiguity is uncomfortable."
---

Ambiguity is uncomfortable. It creates the feeling that progress is blocked — that you can't move forward until you have a clearer picture of what you're building toward. The response to that discomfort is usually to resolve the ambiguity prematurely, picking a direction before you have enough information to pick the right one. In agentic programming, where so many of the important questions are genuinely unresolved, that premature resolution is a consistent source of wasted work.

The technical version of this is the system designed too early. The team is three days into a new agentic project, the requirements are still fuzzy, and someone starts designing the architecture. The architecture encodes assumptions that feel reasonable now but haven't been tested against real inputs. Two weeks later, the real inputs arrive and they're different from what the architecture assumed. The redesign costs more than the patience would have.

Patience with ambiguity doesn't mean doing nothing. It means doing the right things — the things that reduce uncertainty without locking in commitments. Running quick experiments on the prompts that are least clear. Prototyping the pieces that feel most uncertain. Talking to the people who will use the system before deciding what the system should do. Building the small version that will reveal the real constraints before building the large version that assumes you know them.

The skill is recognizing which decisions need to be made now and which can safely be deferred. Some decisions are genuinely blocking — you can't make progress without them. Others feel blocking because of the discomfort of uncertainty, but nothing actually depends on resolving them today. Distinguishing between these requires a kind of meta-judgment that experienced developers develop over time and less experienced ones often lack.

Agentic systems have a particular property that rewards patience: they're cheap to prototype. A prompt experiment takes minutes. A quick eval run takes an hour. The cost of staying in the exploratory phase longer than feels comfortable is low. The cost of locking in the wrong architecture because you needed to feel like you were making progress is high.

Sit with the unclear parts. They'll tell you something if you let them.
