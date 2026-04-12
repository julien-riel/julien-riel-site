---
title: "23. Prompter, c'est penser à voix haute — alors pense soigneusement"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "Il y a une raison pour laquelle les mauvais prompts produisent de mauvais outputs : ils sont généralement le produit d'une pensée floue."
---

Il y a une raison pour laquelle les mauvais prompts produisent de mauvais outputs : ils sont généralement le produit d'une pensée floue. Le prompt, c'est là où ta compréhension du problème s'externalise. Si cette compréhension est incomplète, le prompt le sera aussi — et l'agent exécutera fidèlement ta confusion.

C'est inconfortable parce que ça supprime une excuse pratique. Quand l'agent produit quelque chose de faux, c'est tentant de l'attribuer au modèle — à ses limites, à ses bizarreries, à sa tendance à partir dans des directions inattendues. Parfois c'est vrai. Plus souvent, le prompt était en train de reporter une décision que tu n'avais pas encore prise. L'agent a heurté la question non résolue et y a répondu sans toi.

Écrire un bon prompt est un acte de pensée, pas de transcription. Ça exige de savoir ce que tu veux vraiment — au niveau de détail nécessaire pour agir dessus. C'est plus dur que ça en a l'air pour les tâches complexes, parce que l'écart entre « je saurai quand je le verrai » et « je peux le spécifier assez précisément pour qu'un agent le produise » est souvent plus large que prévu. Combler cet écart, c'est le travail. Le prompt est la trace qu'on l'a comblé.

Une habitude utile, c'est d'écrire le prompt, puis de le lire comme si tu n'avais jamais vu le problème avant. Contient-il tout ce qu'une personne compétente aurait besoin pour bien faire cette tâche ? Les contraintes sont-elles claires ? Les priorités sont-elles explicites quand les choses entrent en conflit ? Le succès est-il défini assez bien pour que tu le reconnaisses ? Si l'une de ces réponses est non, le prompt n'est pas fini — ta pensée n'est pas finie.

Une autre habitude utile, c'est d'écrire le prompt avant de construire le système. Essayer de spécifier exactement ce qu'un agent devrait faire te force à affronter les parties du problème que tu n'as pas encore complètement conçues. Les ambiguïtés dans le prompt sont des ambiguïtés dans la conception du système. Mieux vaut les trouver dans un éditeur de texte que dans les logs de production.

L'agent ne rend pas ta pensée plus claire. Il rend la qualité de ta pensée visible.
