---
title: "89. Les grands projets ont besoin d'un document que l'assistant peut toujours lire"
date: 2026-04-09
tags:
  - developer-as-user
description: "Sur une petite tâche, le context dont tu as besoin tient dans un prompt."
---

Sur une petite tâche, le context dont tu as besoin tient dans un prompt. Sur un grand projet, non — et chaque nouvelle session commence sans la compréhension accumulée qui rend l'assistant utile. Les conventions que tu as établies, les décisions architecturales que tu as prises, les contraintes qui s'appliquent à travers le codebase — tout ça disparaît quand la fenêtre se ferme. Sans solution, tu passes les dix premières minutes de chaque session à ré-établir le context que tu avais déjà établi hier.

La solution est un document persistant — un `ARCHITECTURE.md`, un `AGENT.md`, un `CONTEXT.md` — que tu inclus au début de chaque session avec l'assistant. Pas une spécification complète, mais la version condensée de ce que l'assistant a besoin de savoir pour travailler efficacement dans ce projet : les patterns architecturaux que tu utilises, les conventions pour la gestion d'erreurs et le nommage, les décisions qui ont été prises et ne devraient pas être revisitées, les parties du codebase qui sont stables et les parties qui changent activement.

Ce document vaut la peine d'être maintenu soigneusement parce qu'il paie des dividendes à chaque session. Chaque fois que tu établis une nouvelle convention ou prends une décision architecturale significative, ajoute-la. Chaque fois que l'assistant produit quelque chose qui viole une contrainte de projet que tu as oublié de mentionner, ajoute cette contrainte. Le document grandit à mesure que le projet grandit, et la qualité de l'assistance s'améliore à mesure que le document s'améliore.

Il y a un bénéfice secondaire : le processus d'écrire le document force de la clarté sur ce que tu sais réellement de ton propre projet. Les contraintes qui vivent implicitement dans ta tête sont faciles à violer. Les contraintes écrites dans un document deviennent lisibles — pour l'assistant, pour les nouveaux membres de l'équipe, et pour toi-même quand tu reviens au projet après un mois d'absence.

Le document de context persistant est la mémoire que ton assistant n'a pas. Construis-le tôt et maintiens-le.
