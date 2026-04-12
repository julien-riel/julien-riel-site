---
title: "55. La programmation agentique récompense le penseur paresseux"
date: 2026-04-09
tags:
  - mindset
description: "Paresseux, ici, est un terme technique."
---

Paresseux, ici, est un terme technique. Le penseur paresseux est celui qui demande : quelle est la version la plus simple de ça qui pourrait marcher ? Qu'est-ce que je peux ne pas construire tout en résolvant le problème ? Où est-ce que j'ajoute de la complexité qui ne gagne pas sa place ? C'est la disposition qui produit des systèmes propres, et elle est particulièrement précieuse en programmation agentique parce que la tentation de complexité inutile y est particulièrement forte.

Les agents rendent la complexité peu coûteuse à ajouter. Tu peux brancher un autre tool, étendre le system prompt, ajouter un autre agent au pipeline — le tout sans écrire beaucoup de code. Le coût d'ajouter de la capacité semble bas. Le coût de la complexité que tu as ajoutée n'apparaît que quand tu débogues une panne en production et que tu n'arrives pas à dire lequel des sept composants y a contribué.

L'approche paresseuse commence par le plus petit système possible. Un agent, des tools minimaux, un prompt qui fait le moins qu'il lui faut. Lance-le. Regarde où ça casse. Ajoute exactement ce qui est nécessaire pour traiter la panne — rien de plus. Ce n'est pas du développement itératif comme méthodologie ; c'est du développement itératif comme discipline contre l'impulsion d'anticiper des problèmes qui ne se sont pas encore produits.

Le penseur industrieux construit le système qu'il a imaginé. Le penseur paresseux construit le système que le problème exige vraiment, qui est presque toujours plus petit. Les exigences imaginées sont généreuses. Les exigences réelles sont contraintes. L'écart entre les deux est du gaspillage — de la complexité qui consomme du temps de maintenance, introduit des modes de défaillance et rend le système plus dur à raisonner sans le rendre meilleur pour la tâche réelle.

Il y a aussi un argument d'économie cognitive. Les systèmes agentiques demandent un vrai effort mental à comprendre — le comportement probabiliste, les dynamiques de contexte, l'interaction entre composants. Chaque composant inutile est plus de surface que ton cerveau doit tenir. Le système paresseux est plus facile à déboguer, plus facile à expliquer, plus facile à transmettre et plus facile à améliorer parce que tu peux réellement le voir en entier.

Le penseur paresseux demande ce qui peut être enlevé. La réponse est généralement plus que prévu.
