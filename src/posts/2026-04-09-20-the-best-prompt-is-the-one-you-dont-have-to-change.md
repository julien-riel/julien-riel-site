---
title: "20. Le meilleur prompt est celui que tu n'as pas à changer"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "Le prompt engineering a la réputation d'être itératif — tu écris quelque chose, tu vois ce qui casse, tu corriges, tu recommences."
---

Le prompt engineering a la réputation d'être itératif — tu écris quelque chose, tu vois ce qui casse, tu corriges, tu recommences. Cette boucle est réelle et nécessaire au début. Mais le but de la boucle, c'est d'en sortir. Un prompt que tu es encore en train d'ajuster après trois mois en production n'est pas un prompt raffiné. C'est un prompt instable.

La stabilité est une qualité sous-estimée d'un prompt. Un prompt qui produit des outputs légèrement moins bons mais de façon constante a souvent plus de valeur qu'un prompt qui produit d'excellents outputs la plupart du temps et des échecs mystérieux le reste. La cohérence, c'est ce qui rend un système prévisible. La prévisibilité, c'est ce qui le rend maintenable. La maintenabilité, c'est ce qui le fait survivre à son auteur original.

Le chemin vers un prompt stable passe par la compréhension de pourquoi il fonctionne, pas seulement du fait qu'il fonctionne. Les équipes qui ajustent leurs prompts de façon empirique — changer un mot, voir si c'est mieux, garder le changement si oui — finissent souvent avec des prompts fragiles d'une façon qu'elles ne peuvent pas expliquer. Le prompt marche, mais personne ne sait quelles parties sont porteuses. Quand quelque chose change — une nouvelle version du modèle, un changement dans la distribution des inputs, un nouveau cas limite — elles ne peuvent pas raisonner sur ce qu'il faut ajuster.

Comprendre pourquoi un prompt fonctionne exige la même discipline analytique que comprendre pourquoi du code fonctionne. Que fait chaque section ? Quel comportement changerait si cette contrainte était supprimée ? Qu'est-ce que cet exemple enseigne au modèle que les instructions n'enseignent pas ? Quand tu peux répondre à ces questions, tu peux maintenir le prompt. Quand tu ne peux pas, tu fais du cargo cult.

Il existe un test pratique de la stabilité d'un prompt : exécute-le contre un ensemble diversifié d'inputs et regarde la variance dans la qualité des outputs. Une variance élevée est le signal que le prompt fait quelque chose d'incohérent — que son comportement dépend de caractéristiques des inputs que tu n'as pas complètement cartographiées. Une variance faible signifie que le prompt fait quelque chose de cohérent qui généralise de façon fiable.

Le prompt que tu comprends complètement est le prompt qui t'appartient. Tout le reste te possède.
