---
title: "2. Le prompt est l'architecture"
date: 2026-04-09
tags:
  - working-with-agents
description: "La plupart des développeurs traitent le prompt comme une pensée après-coup — un truc que tu écris une fois, probablement mal, puis que tu bricoles quand quelque chose casse."
---

La plupart des développeurs traitent le prompt comme une pensée après-coup — un truc que tu écris une fois, probablement mal, puis que tu bricoles quand quelque chose casse. C'est le mauvais modèle mental. Le prompt est l'architecture. Change le prompt et tu changes le système. Rate-le et aucune infrastructure astucieuse ne te sauvera.

C'est contre-intuitif parce que les prompts ressemblent à du texte, et le texte donne l'impression d'être informel. On n'a pas l'impression de prendre une décision structurelle quand on en écrit un. Mais c'est ce que tu fais. Tu définis ce que l'agent sait de son rôle, ce à quoi il prête attention, ce qu'il ignore, comment il formate son output, et ce qu'il fait quand les choses deviennent ambiguës. Un prompt mal structuré ne produit pas juste de moins bons outputs — il produit des outputs imprévisibles, ce qui est pire.

L'analogie qui tient la route, c'est le design d'interface. Une API bien designée est explicite sur ses contrats : quels inputs sont valides, à quels outputs s'attendre, comment les erreurs sont communiquées. Un prompt bien designé fait le même travail. Il dit à l'agent dans quel context il opère, à quoi ressemble un bon output, et quoi faire aux marges. Un prompt vague est une interface qui fuit — ça marche quand les conditions sont idéales et ça échoue de façons imprévues quand elles ne le sont pas.

Considère ce qui se passe quand tu ajoutes un nouveau tool à un agent sans mettre à jour le system prompt. Le tool est disponible, mais l'agent n'a aucun modèle mental pour savoir quand l'utiliser, ou pourquoi, ou comment il s'intègre dans la tâche plus large. Tu as changé la capacité du système sans changer l'architecture qui le gouverne. C'est l'équivalent agentique d'ajouter une colonne à une base de données sans mettre à jour le schéma — techniquement possible, de façon fiable problématique.

Les meilleurs praticiens traitent l'écriture de prompts comme une activité d'ingénierie de premier ordre. Ils versionnent leurs prompts. Ils testent les changements systématiquement. Ils documentent le reasoning derrière les décisions de design — pas juste ce que le prompt dit, mais pourquoi il le dit de cette façon. Quand quelque chose casse, ils regardent le prompt en premier, pas l'infrastructure.

Il y a une discipline ici à laquelle la plupart des équipes arrivent tard : séparer ce que l'agent est censé faire (la définition de la tâche) de comment il devrait le faire (les contraintes comportementales) de ce qu'il devrait savoir (le context). Confondre ces trois éléments produit des prompts qui sont difficiles à maintenir et encore plus difficiles à déboguer. Les séparer produit des prompts sur lesquels on peut raisonner — et qu'on peut changer — avec confiance.

Le domaine bouge vite, et la tentation est de penser à quel modèle utiliser, quel framework d'orchestration adopter, quelle vector database brancher. Ces choix comptent. Mais un prompt médiocre sur une bonne infrastructure te donne quand même un agent médiocre. Un prompt précis et réfléchi fait travailler chaque autre partie du système plus fort.

Écris le prompt comme si tu écrivais la spec. Parce que c'est ce que tu fais.
