---
title: "37. Les timeouts ne sont pas optionnels"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Chaque appel externe que fait ton agent — à une API de modèle, à un tool, à une base de données, à un service tiers — a besoin d'un timeout."
---

Chaque appel externe que fait ton agent — à une API de modèle, à un tool, à une base de données, à un service tiers — a besoin d'un timeout. C'est vrai dans le logiciel conventionnel et doublement vrai dans les systèmes agentic, où un appel qui pend ne bloque pas juste l'opération courante ; il peut geler une tâche entière, consommer le budget de la context window et laisser l'agent dans un state ambigu dont il est difficile de récupérer.

La raison pour laquelle les timeouts sont sautés, c'est l'optimisme. Le service est fiable. Le réseau est rapide. Le tool n'a jamais pendu auparavant. Ces observations sont exactes jusqu'au moment où elles ne le sont plus, et les systèmes sans timeouts sont des systèmes qui découvrent les modes de défaillance de leurs dépendances en production, sous charge, au pire moment possible.

Dans les systèmes agentic, le problème des timeouts a une dimension supplémentaire : l'agent lui-même peut être une source d'exécution non bornée. Un appel de modèle sans timeout peut pendre indéfiniment si l'API est sous charge. Un tool qui fait une requête réseau sans timeout peut bloquer toute la boucle de raisonnement de l'agent. Une stratégie de retry sans limite de temps totale peut étendre une tâche bien au-delà de toute attente raisonnable. Chacun de ces cas est un endroit où « ça marche d'habitude » devient « ça pend parfois pour toujours ».

Mettre de bons timeouts demande de savoir à quoi ressemble le normal. Si un appel de tool se complète typiquement en moins d'une seconde, un timeout de cinq secondes est conservateur et approprié. Si un appel de modèle prend typiquement trois secondes, un timeout de trente secondes laisse de la place pour les réponses lentes sans attendre pour toujours. Ces chiffres viennent de l'observation — instrumente tes appels, comprends la distribution des temps de réponse et règle des timeouts qui couvrent la queue légitime sans couvrir la queue infinie.

Ce qui se passe quand un timeout se déclenche est aussi important que le timeout lui-même. L'agent a besoin d'un comportement défini pour chaque cas de timeout — retry, échec de la tâche, escalade à un humain, ou saut de l'étape et poursuite avec une capacité dégradée. Un comportement de timeout indéfini produit un comportement d'agent indéfini, ce qui est précisément ce que tu essayais d'éviter.

Chaque appel a besoin d'une deadline. Le système qui finira par pendre, c'est celui qui n'en a pas.
