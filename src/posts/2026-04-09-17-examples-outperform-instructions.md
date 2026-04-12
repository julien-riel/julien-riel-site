---
title: "17. Les exemples surpassent les instructions"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "Si tu veux qu'un agent produise un output dans un format, un style ou une structure particuliers, lui montrer un exemple est presque toujours plus efficace que de décrire ce que tu veux."
---

Si tu veux qu'un agent produise un output dans un format, un style ou une structure particuliers, lui montrer un exemple est presque toujours plus efficace que de décrire ce que tu veux. Ce n'est pas de l'intuition — c'est cohérent avec la façon dont ces modèles fonctionnent. Ils sont entraînés sur des exemples. Ils généralisent à partir d'exemples. Quand tu leur donnes un exemple, tu parles leur langue. C'est l'essence même du few-shot.

Le mode d'échec, c'est écrire des instructions élaborées là où un seul exemple ferait le travail en un tiers des mots. « Formate l'output comme un objet JSON avec une clé 'summary' contenant une chaîne de pas plus de deux phrases, une clé 'tags' contenant un tableau de chaînes, et une clé 'confidence' contenant un float entre 0 et 1 » fait douze mots de plus et est moins clair que de simplement montrer l'output que tu veux.

Les exemples sont aussi plus robustes face aux cas limites. Les instructions décrivent les cas auxquels tu as pensé. Les exemples, surtout plusieurs exemples, encodent la logique implicite qu'il faudrait des paragraphes pour spécifier pleinement. Un modèle qui voit trois exemples de gestion d'un input ambigu a appris quelque chose sur ton intention qui ne pourrait pas facilement être mis par écrit.

Le nombre d'exemples compte, mais pas linéairement. Passer de zéro exemple à un exemple est le plus grand saut de qualité. Passer de un à trois est significatif. Passer de cinq à dix est marginal pour la plupart des tâches. Le premier exemple pose le gabarit. Les exemples suivants affinent les bords. À un moment donné, tu ajoutes des exemples pour gérer des cas qui arrivent rarement et le retour diminue.

Il y a un effet de sélection à surveiller : les exemples que tu choisis encodent tes valeurs. Si tous tes exemples sont des inputs propres et bien formés, l'agent apprend à bien gérer les inputs propres et peut galérer avec les désordonnés. Inclure un exemple d'input difficile ou en cas limite — et montrer comment le gérer — vaut souvent plus que plusieurs exemples supplémentaires du chemin heureux.

Les instructions disent à l'agent ce que tu veux. Les exemples le lui montrent. Montre-lui.
