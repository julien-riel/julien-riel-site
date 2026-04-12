---
title: "45. Les systèmes multi-agent multiplient les capacités et multiplient les failure modes"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Les arguments pour les systèmes multi-agent sont convaincants."
---

Les arguments pour les systèmes multi-agent sont convaincants. Les tâches complexes peuvent être décomposées en flux de travail parallèles. Des agents spécialisés surperforment les généralistes dans leurs domaines spécifiques. L'orchestration permet de combiner des capacités d'une façon qu'aucun agent unique ne pourrait atteindre. Le tout est supérieur à la somme des parties.

Les arguments contre la précipitation dans les systèmes multi-agent sont tout aussi convaincants, et moins souvent avancés. Chaque agent que tu ajoutes est une source de variance de plus, un failure mode de plus, un système de plus dont tu dois comprendre et tester le comportement. Dans un système à un seul agent, un échec a une origine. Dans un système multi-agent, un échec peut provenir de n'importe où, se propager à travers des handoffs de façons non évidentes, et arriver à la sortie en ressemblant à quelque chose de complètement différent de sa cause.

Le problème des handoffs est propre aux architectures multi-agent et particulièrement insidieux. L'agent A produit une sortie qui a l'air correcte. L'agent B la reçoit, l'interprète légèrement différemment de l'intention de l'agent A, et produit une sortie qui reflète cette mauvaise interprétation. L'agent C reçoit la sortie de B, en fait une inférence raisonnable, et le résultat final est confiant et faux d'une manière qui remonte à un glissement sémantique subtil trois étapes plus tôt. Chaque agent a fait son travail. Le système a échoué.

Ça ne plaide pas contre les systèmes multi-agent, mais pour les construire de façon incrémentale. Commence par la version à un seul agent, même si tu sais qu'elle ne passera pas à l'échelle pour le problème complet. Comprends ses failure modes. Puis décompose une pièce à la fois, en validant explicitement chaque handoff avant d'ajouter l'agent suivant. Les équipes qui conçoivent l'architecture multi-agent complète d'emblée et la construisent d'un coup accumulent une dette technique qu'elles ne voient pas jusqu'à ce que le tout tourne.

L'observabilité devient non négociable à l'échelle multi-agent. Tu dois pouvoir tracer une sortie finale à travers chaque agent qui y a contribué, avec le context complet et le raisonnement à chaque étape. Sans ça, déboguer un échec multi-agent, c'est deviner.

Plus d'agents signifie plus de capacité. Ça signifie aussi plus d'endroits où perdre le fil.
