---
title: "36. Enregistre tout ce que ton agent pensait, pas seulement ce qu'il a fait"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Les action logs sont nécessaires mais pas suffisants."
---

Les action logs sont nécessaires mais pas suffisants. Savoir que l'agent a appelé un tool, envoyé un message ou retourné une sortie te dit ce qui s'est passé. Ça ne te dit pas pourquoi, et dans les systèmes agentic, le pourquoi est souvent là où vit la défaillance.

La différence compte surtout pendant le debug. Un agent produit une mauvaise sortie. L'action log montre : document A récupéré, tool B appelé, sortie C retournée. Rien dans cette séquence n'a l'air mauvais — chaque étape était une action raisonnable. Mais le reasoning trace, si tu l'avais capturé, aurait montré l'agent mal interpréter une phrase du document A d'une façon qui faisait du tool B le choix logique, ce qui faisait de la sortie C le résultat inévitable. Sans le raisonnement, tu as un mystère. Avec, tu as un diagnostic.

Les reasoning traces révèlent aussi une classe de défaillances que les action logs ratent complètement : l'agent qui a fait la bonne chose pour la mauvaise raison. Il a récupéré le bon document, mais pas parce qu'il a compris la requête — parce que le document contenait par hasard des mots-clés qui correspondaient. Il a appelé le bon tool, mais avec des paramètres qui fonctionnaient par coïncidence. Ces défaillances sont invisibles dans les action logs et visibles dans les reasoning traces, et elles comptent parce que le prochain input légèrement différent va casser le pattern chanceux et tu ne sauras pas pourquoi.

L'objection pratique, c'est le coût. Les reasoning traces sont verbeuses. Les stocker à grande échelle coûte cher. C'est une vraie contrainte qui vaut la peine d'être gérée — tu peux échantillonner les traces plutôt que les capturer toutes, tu peux mettre en place des politiques de rétention qui gardent les traces récentes et archivent les plus anciennes, tu peux capturer des traces complètes seulement pour les tâches échouées ou signalées. Ce sont des tradeoffs raisonnables. Ce qui n'est pas raisonnable, c'est de ne rien capturer et d'espérer que l'action log suffit.

Il y a aussi un bénéfice cumulatif dans le temps. Un dépôt de reasoning traces de vraies tâches, c'est du matériel d'entraînement, des données d'évaluation et de la connaissance institutionnelle. C'est comme ça que tu comprends ce que ton agent fait réellement par rapport à ce que tu penses qu'il fait. Cette compréhension est le fondement de toute amélioration que tu feras au système.

Log la pensée. Les actions n'en sont que la surface visible.
