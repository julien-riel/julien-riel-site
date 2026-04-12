---
title: "40. Les meilleurs agents ont une personnalité étroite"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Un agent généraliste, ça sonne comme l'objectif."
---

Un agent généraliste, ça sonne comme l'objectif. Un seul agent, n'importe quelle tâche, flexibilité maximale. En pratique, les agents généralistes sont médiocres en tout et excellents en rien. Les agents qui fonctionnent le mieux en production ont une personnalité nettement définie — un sens cohérent de ce qu'ils sont, de ce qu'ils font et de la manière dont ils le font — et cette spécificité est une qualité, pas une limite.

La personnalité, ici, c'est plus que le ton. C'est un ensemble cohérent de valeurs qui gouvernent la façon dont l'agent fait ses arbitrages. Un agent de revue de code qui priorise l'exactitude sur la lisibilité se comportera différemment d'un agent qui priorise la lisibilité sur l'exactitude — pas juste dans ce qu'il dit, mais dans les problèmes qu'il signale, ceux qu'il laisse passer et la façon dont il explique son raisonnement. Ce sont des agents différents, adaptés à des contextes différents. Un agent qui essaie de concilier les deux sans priorité claire sera incohérent d'une manière qui frustre les gens qui comptent sur lui.

Une personnalité étroite rend aussi les agents plus prévisibles, ce qui les rend plus dignes de confiance. Les utilisateurs qui interagissent avec un agent de façon répétée développent un modèle mental de son comportement. Quand le comportement est cohérent — quand l'agent fait de manière fiable le même genre de choses de la même façon — ce modèle mental est juste et utile. Quand le comportement varie — quand la même question reçoit un traitement différent selon de subtiles différences de contexte — le modèle mental s'effondre et les utilisateurs cessent de faire confiance à leur intuition sur le système.

Le processus de conception d'une personnalité étroite est le même que celui d'un bon system prompt : détermine à quoi sert cet agent, à qui il s'adresse, ce qu'il valorise et ce qu'il refuse de faire — puis encode tout cela explicitement. La personnalité de l'agent est un artefact de conception, pas une propriété émergente. Non spécifiée, elle sera incohérente. Spécifiée précisément, elle devient une caractéristique fiable du système.

La tentation d'élargir les agents vient du désir d'éviter de construire plusieurs agents. C'est la mauvaise optimisation. Trois agents étroits qui font chacun bien leur travail valent mieux qu'un agent large qui fait tout passablement.

Sache ce qu'est l'agent. Fais-en exactement cela, complètement.

---

## Part 4 — Agents in the Real World
