---
title: "74. Donnez à l'assistant vos contraintes, pas juste vos exigences"
date: 2026-04-09
tags:
  - developer-as-user
description: "\"Écris une fonction qui parse ce fichier de config\" produit quelque chose."
---

"Écris une fonction qui parse ce fichier de config" produit quelque chose. "Écris une fonction qui parse ce fichier de config, gère les entrées malformées en retournant une erreur typée plutôt qu'en la lançant, utilise uniquement la librairie standard, et suit les conventions de gestion d'erreurs du reste de ce codebase" produit quelque chose d'utile. L'écart entre ces deux demandes, c'est l'écart entre une exigence et une spécification — et c'est le travail du développeur de le combler.

C'est plus difficile qu'il n'y paraît parce que les contraintes sont souvent tacites. Tu sais que ce projet n'utilise pas de dépendances externes sans avoir à y penser. Tu sais que les erreurs sont retournées, pas lancées, parce que tu as écrit la convention toi-même. Tu sais que cette fonction sera appelée dans un chemin critique et que la performance importe. Rien de tout cela n'est dans l'exigence. Tu dois le rendre explicite, ce qui exige d'abord de le rendre conscient.

L'exercice de lister tes contraintes avant de prompter est utile indépendamment de l'assistant. C'est une forme de design thinking — te forcer à articuler les exigences qui ne sont pas dans le spec parce que tout le monde dans l'équipe les connaît déjà. L'assistant a besoin qu'elles soient énoncées. Les écrire est souvent le moment où tu découvres que tu n'es pas d'accord dessus aussi clairement que tu le pensais.

Les contraintes qui comptent le plus sont celles qui disent ce que le code ne doit pas faire : ne doit pas lancer, ne doit pas muter l'entrée, ne doit pas faire d'appels réseau, ne doit pas casser l'interface existante. Les exigences positives décrivent la cible. Les contraintes négatives définissent les limites. Les deux sont nécessaires. La plupart des prompts n'en contiennent qu'une.

Dis à l'assistant ce que le code ne peut pas faire. C'est là que vit la vraie spécification.
