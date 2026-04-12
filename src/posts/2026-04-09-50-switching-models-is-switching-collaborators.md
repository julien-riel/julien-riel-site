---
title: "50. Changer de modèle c'est changer de collaborateur"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Quand un nouveau modèle sort avec de meilleurs scores aux benchmarks, la tentation est de le brancher à la place et de s'approprier l'amélioration."
---

Quand un nouveau modèle sort avec de meilleurs scores aux benchmarks, la tentation est de le brancher à la place et de s'approprier l'amélioration. Parfois ça marche. Plus souvent, ça introduit des changements comportementaux subtils qui cassent des choses dont tu ne savais pas que tu dépendais — et découvrir ces cassures après coup coûte beaucoup plus cher que de les tester avant.

Les modèles ont une personnalité, au sens plein du terme. Ils ont des façons caractéristiques de gérer l'ambiguïté, des niveaux caractéristiques de verbosité, des tendances caractéristiques à la prudence ou à la confiance. Un system prompt ajusté à la personnalité d'un modèle peut produire un comportement différent avec un autre modèle, même si le nouveau est objectivement plus capable. Plus capable sur les tâches du benchmark ne veut pas dire plus compatible avec les comportements spécifiques autour desquels ton système a été conçu.

Les changements de formatage sont les plus immédiatement visibles. Un modèle qui retournait fiablement du JSON peut retourner du JSON avec de la prose en plus une fois remplacé. Un modèle qui utilisait un certain délimiteur peut en utiliser un autre. Ce sont des corrections triviales individuellement et étonnamment coûteuses à trouver de façon exhaustive — il y a toujours plus de dépendances de format que tu ne le penses, éparpillées dans le code de parsing, les handlers en aval et la logique d'affichage.

Les changements de raisonnement sont plus difficiles à voir et plus lourds de conséquences. Un modèle qui était prudent dans l'expression de l'incertitude peut être remplacé par un modèle plus confiant — ce qui sonne comme une amélioration jusqu'à ce que tu sois dans un domaine où la prudence de l'ancien était appropriée et où la confiance du nouveau est déplacée. Un modèle avec une approche particulière des instructions ambiguës peut être remplacé par un modèle qui les interprète différemment, d'une façon raisonnable mais incompatible avec ton intention de design.

La discipline, c'est de traiter un changement de modèle comme un changement de version avec un risque de migration, pas comme un upgrade drop-in. Fais tourner ta suite d'évals complète contre le nouveau modèle avant de switcher. Compare les outputs sur un échantillon représentatif de tâches réelles. Cherche spécifiquement les changements de comportement dans tes cas limites, pas juste les améliorations de qualité moyenne. Donne au switch le même processus de revue qu'à un changement significatif de prompt.

Un meilleur modèle n'est pas automatiquement un meilleur choix. Mérite l'upgrade.
