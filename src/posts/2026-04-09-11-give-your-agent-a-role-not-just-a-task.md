---
title: "11. Donnez à votre agent un rôle, pas juste une tâche"
date: 2026-04-09
tags:
  - working-with-agents
description: "Il y a une différence entre dire à un agent ce qu'il doit faire et lui dire ce qu'il est."
---

Il y a une différence entre dire à un agent ce qu'il doit faire et lui dire ce qu'il est. « Résume ce document » est une tâche. « Tu es un rédacteur technique senior qui résume de la documentation interne pour un public non technique » est un rôle. Le rôle produit de meilleurs résultats — pas grâce à des mots magiques, mais parce qu'il charge un ensemble cohérent de comportements, de contraintes et de priorités que le modèle peut appliquer de façon constante à tout ce que la tâche exige.

Les rôles fonctionnent parce que les LLM sont entraînés sur du texte produit par des humains, qui est rempli de comportements spécifiques à des rôles. La façon dont un avocat lit un contrat est différente de celle d'un ingénieur. La façon dont un correcteur aborde un paragraphe est différente de celle d'un développeur. Quand tu assignes un rôle, tu ne fixes pas juste un ton — tu actives un ensemble de comportements propres à un domaine que le modèle a appris à partir d'exemples de personnes occupant ce rôle et faisant ce type de travail.

La différence concrète est visible dans les cas limites. Donne à un agent la tâche « révise ce code pour y trouver des bugs » et il trouvera des bugs. Donne-lui le rôle « tu es un ingénieur senior qui fait une code review axée sécurité avant un déploiement en production » et il trouvera des bugs différents — il pondérera différemment, signalera différemment, et expliquera ses trouvailles d'une façon calibrée sur ce qui importerait à un senior soucieux de sécurité. La tâche est la même. L'angle est différent.

Les rôles fournissent aussi un repli cohérent pour les situations que la spécification de la tâche n'avait pas anticipées. Si tu as dit à l'agent qu'il est rédacteur technique pour un public non technique, et que le document contient du jargon que tu ne lui as pas explicitement dit de simplifier, il a un principe pour gérer ce cas. Sans le rôle, il doit deviner. C'est dans la devinette que vit l'inconsistance.

Le mode d'échec, ce sont les rôles trop vagues pour être utiles — « tu es un assistant serviable qui est bon à beaucoup de choses » ne donne rien au modèle sur quoi travailler. Les rôles utiles sont précis sur le domaine, le public et les valeurs qui doivent guider les arbitrages. Pas seulement ce qu'est l'agent, mais comment il pense.

Dis à l'agent ce qu'il est. La tâche découle du rôle.
