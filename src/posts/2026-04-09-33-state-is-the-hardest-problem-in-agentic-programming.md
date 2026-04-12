---
title: "33. Le state est le problème le plus difficile en programmation agentic"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Tout problème difficile dans les systèmes distribués se réduit finalement au state."
---

Tout problème difficile dans les systèmes distribués se réduit finalement au state. Qui le possède, où il vit, comment il reste cohérent, ce qui se passe quand il diverge. Les systèmes agentic héritent de tous ces problèmes et en ajoutent de nouveaux, parce que l'agent lui-même est sans state — il n'a pas de mémoire entre les appels — tandis que les tâches qu'il effectue sont souvent profondément stateful. Combler cet écart est là où vit la plupart de la vraie complexité.

Prends une tâche multi-étapes : l'agent récupère de l'information, prend une décision, appelle un tool, attend un résultat, prend une autre décision. Chaque étape dépend des résultats des précédentes. Si la tâche échoue à mi-chemin — le tool fait timeout, la context window se remplit, l'utilisateur interrompt — tu dois savoir ce qui a été complété, ce qui ne l'a pas été, et s'il est sûr de reprendre ou s'il faut redémarrer. L'agent ne peut pas te le dire, parce que l'agent ne se souvient pas. Ton système doit le faire.

Les approches sont bien connues des systèmes distribués : faire du checkpointing du state à chaque étape, utiliser des event logs pour reconstruire ce qui s'est passé, concevoir les tâches pour être reprenables depuis n'importe quel checkpoint. Elles sont bien connues parce qu'elles sont nécessaires — le même problème fondamental a été résolu sous différentes formes de nombreuses fois. L'erreur, c'est de penser que les systèmes agentic sont en quelque sorte différents, que l'interface conversationnelle ou la colonne vertébrale IA change le défi sous-jacent de gestion du state. Non. L'agent n'est qu'un autre service sans state qui a besoin d'une gestion de state externe pour participer à des workflows stateful.

Ce qui est différent, c'est que le state dans les systèmes agentic inclut souvent des choses qui sont plus difficiles à sérialiser que des enregistrements de base de données. La compréhension actuelle qu'a l'agent d'un problème. Le contexte qu'on lui a donné. Les décisions implicites qu'il a prises au cours d'une longue conversation. Capturer tout ça d'une façon qui te permette de reprendre de manière significative — pas juste techniquement — demande de réfléchir à ce qui doit réellement persister et ce qui peut être reconstruit.

Les équipes qui gèrent bien ça conçoivent leur gestion de state avant de concevoir leur logique d'agent. Elles demandent : si cette tâche est interrompue à n'importe quel moment, de quoi avons-nous besoin pour la reprendre ? Elles répondent à cette question concrètement et construisent l'infrastructure pour la maintenir.

L'agent oublie tout. Conçois comme si c'était une contrainte, pas un oubli.
