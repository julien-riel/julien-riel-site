---
title: "26. Conçois pour l'observability avant de concevoir pour la capacité"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Le système agentique le plus capable que tu ne peux pas observer vaut moins qu'un système moins capable dans lequel tu peux voir."
---

Le système agentique le plus capable que tu ne peux pas observer vaut moins qu'un système moins capable dans lequel tu peux voir. Ce n'est pas une position philosophique — c'est une position pratique. Les systèmes que tu ne peux pas observer sont des systèmes que tu ne peux pas déboguer, que tu ne peux pas améliorer, et à qui tu ne peux pas faire confiance en production.

La tentation, quand on construit des systèmes agentiques, c'est de se concentrer sur la capacité en premier. Qu'est-ce que cet agent peut faire ? Jusqu'où peut-il aller ? Combien peut-il automatiser ? Ce sont des questions excitantes et elles alimentent les démos qui obtiennent l'adhésion des parties prenantes. L'observability est moins excitante. Elle ne rend pas l'agent plus intelligent. Elle n'ajoute pas de nouvelles fonctionnalités. C'est l'infrastructure qui rend tout le reste soutenable.

L'observability dans les systèmes agentiques, c'est être capable de répondre, à tout moment : qu'est-ce que l'agent a vu, qu'a-t-il décidé, qu'a-t-il fait, et pourquoi ? Le pourquoi est la partie difficile. L'observability logicielle traditionnelle — logs, métriques, traces — capture ce qui s'est passé. L'observability agentique doit capturer le reasoning derrière ce qui s'est passé, parce que le même input peut produire des outputs différents selon un reasoning qui n'est pas visible dans le log d'actions.

Le minimum pratique, c'est de logger le context window complet pour chaque invocation de l'agent, aux côtés des outputs et de tout tool call effectué. Ça a l'air cher — et à grande échelle, ça l'est — mais l'alternative, c'est de voler à l'aveugle. Tu ne peux pas déboguer un système que tu ne peux pas inspecter. Tu ne peux pas améliorer ce que tu ne peux pas mesurer. Le coût de stockage d'un logging exhaustif est presque toujours moins cher que le coût d'ingénierie pour diagnostiquer des échecs de production sans lui.

Au-delà du logging, l'observability veut dire construire des tools qui te permettent de rejouer les exécutions d'un agent. Étant donné un context loggé, peux-tu réexécuter l'agent avec un prompt modifié et comparer les outputs ? Peux-tu remonter un échec de production jusqu'aux inputs spécifiques qui l'ont causé ? Peux-tu échantillonner les outputs récents de l'agent et les réviser par rapport à ton seuil de qualité ? Ces capacités n'arrivent pas par accident — elles exigent un investissement avant que tu en aies besoin, pas après.

Construis les fenêtres avant d'emménager dans la maison. Tu vas avoir besoin de voir dehors éventuellement.
