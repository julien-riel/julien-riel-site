---
title: "93. Traitez votre CLAUDE.md comme un document d'embauche"
date: 2026-04-09
tags:
  - developer-as-user
description: "Claude Code lit un fichier `CLAUDE.md` au début de chaque session."
---

Claude Code lit un fichier `CLAUDE.md` au début de chaque session. La plupart des développeurs qui l'utilisent le traitent comme une liste de règles — n'utilise pas cette librairie, suis ces conventions, exécute ces commandes avant de commit. C'est sous-utiliser le fichier de façon significative. Le `CLAUDE.md` est le document qui fait l'onboarding de ton collaborateur IA sur ton projet, et il mérite le soin que tu mettrais à faire l'onboarding d'un nouveau membre d'équipe.

Un bon document d'onboarding pour un développeur humain lui dirait: ce qu'est ce projet et pourquoi il existe, les décisions architecturales qui définissent sa structure et le raisonnement derrière elles, les conventions non négociables et celles qui sont des préférences, les parties du codebase qui sont fragiles et demandent une manipulation soigneuse, les choses qui ont été essayées et qui n'ont pas marché, les outils et workflows que l'équipe utilise. Tout cela est pertinent pour l'assistant aussi.

Le raisonnement derrière les décisions est particulièrement précieux. « On utilise la librairie X » est une règle. « On utilise la librairie X parce que la librairie Y avait des problèmes de performance à notre échelle et que la librairie Z ne supportait pas le modèle d'authentification dont on a besoin » est une décision avec contexte — et un assistant qui comprend le contexte peut faire de meilleurs choix de jugement sur des décisions adjacentes que tu n'as pas spécifiées.

Le `CLAUDE.md` sert aussi de documentation pour les membres humains de l'équipe. Le processus de l'écrire — articuler les conventions et les décisions du projet assez clairement pour qu'une IA puisse agir dessus — produit exactement le genre de documentation dont les nouveaux membres d'équipe ont besoin et qui est rarement écrite parce qu'elle semble évidente aux gens qui savent déjà.

Écris le `CLAUDE.md` comme si tu expliquais le projet à une nouvelle recrue très capable mais qui ne connaît rien de ton contexte spécifique. C'est exactement ce que tu fais.
