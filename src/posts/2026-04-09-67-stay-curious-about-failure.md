---
title: "67. Reste curieux face à l'échec"
date: 2026-04-09
tags:
  - mindset
description: "L'échec dans les systèmes agentic, c'est de l'information."
---

L'échec dans les systèmes agentic, c'est de l'information. De l'information riche, précise, difficile à obtenir autrement, sur la façon dont ton système se comporte réellement par rapport à ce que tu pensais. Les développeurs qui traitent l'échec comme un embarras à corriger rapidement et à oublier jettent certaines des données les plus précieuses que leur système produit. Les développeurs qui restent curieux à son sujet — qui demandent non seulement ce qui a échoué mais pourquoi, et ce que l'échec révèle sur le comportement sous-jacent du système — s'améliorent plus vite.

La curiosité doit survivre à l'environnement émotionnel de l'échec, et c'est la partie difficile. Les échecs en production sont stressants. Ils créent de la pression pour agir — pour trouver le correctif immédiat, le déployer, passer à autre chose. Cette pression est légitime. Le système doit fonctionner. Mais le correctif et la compréhension sont des activités différentes, et faire le correctif sans faire la compréhension signifie que tu as résolu cette instance de l'échec sans rien apprendre qui prévienne la suivante.

Le type spécifique de curiosité qui paie, c'est celle qui demande : qu'est-ce que cet échec me dit sur ce qui est vrai? Pas ce que je devrais changer, mais ce que j'ai appris. Un échec qui révèle un trou dans ta suite d'évals, c'est de l'information sur l'endroit où tes tests étaient incomplets. Un échec qui révèle une hypothèse dans ton system prompt, c'est de l'information sur ce que l'agent inférait et que tu pensais avoir spécifié. Un échec qui révèle un cas limite que tu n'avais pas anticipé, c'est de l'information sur la vraie distribution des entrées, qui est toujours plus large que la distribution que tu imaginais.

Cette curiosité a aussi une qualité cumulative. Chaque échec que tu comprends en profondeur produit des insights qui préviennent plusieurs échecs futurs. Le développeur qui comprend cinq échecs en profondeur apprend plus que celui qui en rustine cinquante superficiellement. La compréhension se généralise. Les rustines, non.

Il y a une pratique qui vaut la peine d'être développée : avant de clore tout échec significatif, écris ce que tu as appris. Pas ce que tu as corrigé — ce que tu as appris. L'échec comme fenêtre sur le comportement du système. L'hypothèse qu'il a exposée. Le trou qu'il a révélé. Ce document vaut plus que le correctif.

Chaque échec est une question que le système te pose. Reste assez curieux pour y répondre.
