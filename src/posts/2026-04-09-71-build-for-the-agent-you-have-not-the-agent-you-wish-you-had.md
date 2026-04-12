---
title: "71. Construisez pour l'agent que vous avez, pas celui que vous aimeriez avoir"
date: 2026-04-09
tags:
  - mindset
description: "Chaque développeur qui travaille avec des agents a un écart entre les capacités actuelles des outils qu'il utilise et celles qu'il aimerait que ces outils aient."
---

Chaque développeur qui travaille avec des agents a un écart entre les capacités actuelles des outils qu'il utilise et celles qu'il aimerait que ces outils aient. Le modèle actuel est presque assez bon pour la tâche, mais pas tout à fait. La fenêtre de contexte est presque assez grande, mais se remplit au mauvais moment. Le raisonnement est presque assez fiable pour qu'on lui fasse confiance de façon autonome, mais il faut encore un checkpoint. Presque, presque, presque.

Construire pour l'agent que tu aimerais avoir, c'est concevoir des systèmes qui dépendent de capacités qui n'existent pas tout à fait encore — puis se demander pourquoi ça ne marche pas. Le système suppose un niveau de fiabilité dans le suivi des instructions que le modèle actuel n'atteint pas. Le workflow suppose une rétention du contexte sur une durée de session qui dépasse ce que le modèle gère bien. L'architecture suppose une précision d'utilisation d'outils que le modèle actuel atteint en test, mais pas de façon constante en production. Chaque hypothèse est individuellement raisonnable vu l'évolution du domaine. Ensemble, elles produisent un système qui marche en démo et échoue en déploiement.

Construire pour l'agent que tu as, c'est prendre au sérieux les capacités actuelles comme des contraintes, pas comme des obstacles temporaires à contourner. Si le modèle a du mal avec les tâches qui exigent de suivre plus de cinq variables simultanément, conçois la tâche pour en exiger moins. Si le modèle est peu fiable en planification à long horizon, ajoute des checkpoints humains plutôt que d'espérer que cette exécution-ci sera la bonne. Si la longueur du contexte cause de la dégradation, intègre de la synthèse dans le workflow plutôt que de supposer que le modèle gérera les longs contextes aussi bien que les courts.

Ce n'est pas du pessimisme sur le domaine. C'est le pragmatisme qui produit des systèmes qui fonctionnent réellement. Les capacités s'amélioreront — elles l'ont toujours fait. Quand elles le feront, tu retireras les contraintes que tu avais mises en place autour des anciennes limitations. Mais tu ne peux retirer que les contraintes que tu avais reconnues. Tu ne peux pas réparer un système bâti sur des hypothèses qui n'étaient jamais vraies.

Il y a aussi un bénéfice cumulatif à concevoir dans les contraintes actuelles : ça force à clarifier ce qui doit réellement se passer pour que le système fonctionne. Les contraintes révèlent la complexité essentielle. Les systèmes conçus sous contraintes serrées sont souvent meilleurs que ceux conçus en supposant des capacités illimitées, parce que les contraintes forcent la réflexion difficile que la conception sans contraintes reporte.

L'agent que tu as est celui avec lequel tu livres. Construis pour lui, honnêtement.
