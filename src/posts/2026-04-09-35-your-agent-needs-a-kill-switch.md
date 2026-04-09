---
title: "35. Your Agent Needs a Kill Switch"
date: 2026-04-09
tags:
  - agentic-programming
  - building-agentic-systems
description: "Every agentic system that operates with any degree of autonomy needs a way to stop it immediately — not gracefully, not after the current task completes, but now."
---

Every agentic system that operates with any degree of autonomy needs a way to stop it immediately — not gracefully, not after the current task completes, but now. This is not a feature you add after something goes wrong. It's infrastructure you build before deployment, because the scenarios that require it don't announce themselves in advance.

The kill switch is the physical embodiment of the principle that humans stay in control. An agent doing something wrong — sending bad outputs, making incorrect decisions, behaving unexpectedly at scale — needs to be stoppable by a person who isn't a developer, at any hour, without requiring a deployment or a database change. If stopping your agent requires a pull request, you've built something that's harder to control than it should be.

What the kill switch looks like depends on the system. At minimum, it's a feature flag that halts agent execution at the task level — checked at the start of each task, or at each step of a multi-step task, so that setting it takes effect within one cycle rather than after the current task completes. For higher-autonomy systems, it means the ability to pause mid-task, drain in-flight work cleanly, and prevent new work from starting — all from a single operation that non-technical stakeholders can perform.

Beyond the immediate stop, you want the ability to understand what was happening when you stopped. What tasks were in flight? What had the agent already done? What state was left behind that needs to be cleaned up? A kill switch without observability leaves you stopped but not informed — you know something was wrong, but not what or how bad.

There's a broader principle here that applies beyond the literal kill switch: design for reversibility wherever possible. Prefer operations the agent can undo over ones it can't. Prefer human confirmation for irreversible actions. Build in the assumption that you will sometimes need to stop, inspect, and reverse — and make sure the system supports it.

The agent that can't be stopped isn't trustworthy. Build the switch first.
