---
title: "38. Le coût est une contrainte architecturale"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Les coûts de tokens ont tendance à surprendre les équipes qui ne les ont pas planifiés."
---

Les coûts de tokens ont tendance à surprendre les équipes qui ne les ont pas planifiés. Le prototype tourne pour pas cher parce qu'il traite une poignée de requêtes par jour. La production en traite dix mille. Les context windows sont grosses parce que quelqu'un a décidé que plus de contexte était toujours mieux. La logique de retry fait trois tentatives par défaut. Personne n'a additionné ce que ça signifie à grande échelle, et le premier cycle de facturation est éducatif d'une façon que personne ne voulait.

Le coût n'est pas une préoccupation opérationnelle que tu traites après que l'architecture est posée. C'est une contrainte qui façonne l'architecture dès le départ — aussi réelle que la latence, la fiabilité ou la justesse. Un système qui produit de super sorties mais coûte dix dollars par interaction utilisateur n'est pas un système viable, peu importe à quel point la démo a l'air impressionnante.

Les leviers sont bien définis une fois que tu sais où regarder. Le choix du modèle est le plus gros : la différence de coût entre un gros modèle frontier et un modèle plus petit et plus rapide peut être d'un ordre de grandeur, et pour beaucoup de tâches le plus petit modèle est assez bon. La taille de la context window est le deuxième : chaque token dans la window coûte de l'argent, et les contextes gonflés — historiques de conversation complets, documents sur-récupérés, system prompts verbeux — s'additionnent vite. La décomposition de tâches est le troisième : un gros agent qui gère tout en un seul appel peut coûter plus qu'une pipeline de plus petits agents moins chers où seule l'étape finale utilise le modèle coûteux.

La discipline, c'est d'instrumenter le coût dès le premier jour. Sache ce que coûte chaque appel d'agent. Sache ce que coûte chaque appel de tool. Sache ce qu'est le coût total par tâche à travers la pipeline. Sans cette instrumentation, tu optimises à l'aveugle — tu ne peux pas faire de bons tradeoffs entre capacité et coût parce que tu ne sais pas ce que coûte quoi.

Il y a aussi une odeur de design à surveiller : le système où le coût est invisible pour les gens qui prennent les décisions de design. Quand les développeurs peuvent expérimenter librement sans voir le coût de leurs expériences, ils optimisent pour la capacité et la commodité. Quand le coût est visible — dans les dashboards, dans les breakdowns par requête, dans les résumés mensuels — les tradeoffs sont faits plus soigneusement.

Construis pour ce que ça coûte de le faire tourner, pas juste pour ce que ça coûte de le démontrer.
