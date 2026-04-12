---
title: "39. Les context windows sont des budgets — dépense-les judicieusement"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Une context window n'est pas un espace infini — c'est un budget."
---

Une context window n'est pas un espace infini — c'est un budget. Tout ce que tu y mets coûte des tokens, entre en compétition pour l'attention du modèle et évince potentiellement quelque chose de plus important. Traiter la context window comme un dépotoir pour tout ce qui pourrait être pertinent est une des erreurs les plus communes et les plus coûteuses dans le design de systèmes agentic.

Le problème d'attention est plus subtil que le problème de limite de tokens. Les modèles ne traitent pas toutes les parties du contexte également. Le contenu près du début et de la fin du contexte tend à recevoir plus d'attention que le contenu au milieu — ce qu'on appelle l'effet lost-in-the-middle. Une context window qui est techniquement dans la limite de tokens peut quand même produire une performance dégradée si l'information la plus importante est enterrée au milieu d'un long document, entourée de matériel moins pertinent.

Ça veut dire que la curation compte autant que la capacité. Le but n'est pas de faire tenir autant que possible dans le contexte — c'est de mettre les bonnes choses, dans le bon ordre, au bon niveau de détail. Un contexte bien curé de dix mille tokens produit souvent de meilleurs résultats qu'un contexte gonflé de cinquante mille. La discipline, c'est de demander, pour chaque morceau d'information que tu envisages d'inclure : est-ce que l'agent a réellement besoin de ça pour faire la tâche ? Si la réponse n'est pas clairement oui, laisse-le dehors.

Les systèmes de retrieval empirent ça avant de l'améliorer. La tentation est de récupérer généreusement — tirer les vingt meilleurs documents plutôt que les cinq meilleurs, juste au cas où un des candidats moins probables s'avère pertinent. Le résultat est un contexte rempli de matériel marginalement pertinent qui dilue le signal. Un meilleur retrieval, pas plus de retrieval, est le chemin vers un meilleur contexte.

L'historique de conversation est une autre source commune de gonflement du contexte. L'historique complet semble sûr — tu ne perds rien. Mais les longs historiques poussent le contexte initial hors de la portée d'attention effective et remplissent la window avec du contenu qui n'est plus pertinent pour la tâche courante. Résumer les tours antérieurs, ou les abandonner sélectivement, produit souvent de meilleurs résultats que tout préserver.

La context window est l'immobilier le plus cher de ton système. Traite-la en conséquence.
