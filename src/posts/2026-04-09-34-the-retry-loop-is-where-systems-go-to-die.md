---
title: "34. La boucle de retry est là où les systèmes vont mourir"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "La logique de retry est nécessaire."
---

La logique de retry est nécessaire. Tout système qui appelle des services externes en a besoin — les réseaux tombent, les services font timeout, des erreurs transitoires arrivent. Mais dans les systèmes agentic, la logique de retry a un mode de défaillance particulier qui vaut la peine d'être compris avant que tu ne la construises : l'agent qui fait du retry indéfiniment, convaincu qu'il progresse, consommant des tokens, du temps et de l'argent tout en ne produisant rien d'utile.

Le problème, c'est que les agents génèrent leurs propres raisons de faire un retry. Une boucle de retry conventionnelle a une condition fixe : l'opération a échoué, attends et réessaie. Un agent peut construire des raisons de continuer à partir du contenu de la conversation — le tool a retourné un résultat ambigu, donc réessaie avec une approche différente ; la sortie ne correspondait pas aux attentes, donc essaie une formulation différente ; la dernière tentative était presque bonne, donc itère encore une fois. Chacune de ces raisons est individuellement raisonnable. Ensemble, elles produisent une boucle qui peut tourner très longtemps avant que quelqu'un ne s'en aperçoive.

C'est particulièrement dangereux quand les retries ont des effets de bord. Un agent qui fait du retry sur une écriture en base de données, un envoi de message ou un appel d'API qui facture par requête peut causer de vrais dégâts avant que la boucle ne se termine. La logique de retry qui semblait être une fonctionnalité de sécurité devient elle-même le mode de défaillance.

Le correctif demande des limites explicites à plusieurs niveaux. Un nombre maximum de tentatives par opération. Un nombre maximum d'étapes par tâche. Un temps wall-clock maximum avant que la tâche ne soit abandonnée et signalée pour revue humaine. Ces limites devraient être réglées de manière conservatrice et ajustées en fonction du comportement observé — pas laissées ouvertes parce que la tâche pourrait vraiment avoir besoin de plus de tentatives.

Il y a aussi une question de design sur ce que l'agent fait quand il atteint une limite. Échouer silencieusement est le pire résultat — la tâche semble complétée alors qu'elle n'a rien fait. Échouer bruyamment, avec un état d'erreur clair et assez de contexte pour comprendre ce qui a été tenté, est le fondement de toute stratégie de retry significative au niveau humain.

Une logique de retry sans conditions de sortie, ce n'est pas de la fiabilité. C'est de l'optimisme sans plan.
