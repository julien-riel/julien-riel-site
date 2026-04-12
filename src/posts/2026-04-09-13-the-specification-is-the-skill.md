---
title: "13. La spécification est la compétence"
date: 2026-04-09
tags:
  - working-with-agents
description: "Les développeurs qui tirent le plus des agents ne sont pas ceux qui en savent le plus sur les modèles."
---

Les développeurs qui tirent le plus des agents ne sont pas ceux qui en savent le plus sur les modèles. Ce sont ceux qui savent écrire une spécification précise. Cette compétence — décomposer une tâche en exactement ce qui est nécessaire, ni plus ni moins — se révèle être le goulot d'étranglement, pas la technologie.

Ça surprend les gens parce que le discours autour des agents, c'est qu'ils réduisent le besoin de précision. Tu n'as plus à écrire du code exact — tu décris ce que tu veux et l'agent se débrouille. Et c'est vrai, jusqu'à un certain point. Pour les tâches simples, les descriptions lâches fonctionnent bien. Quand les tâches deviennent plus complexes, les descriptions lâches produisent des outputs approximativement corrects, ce qui en logiciel est une autre façon de dire faux.

Une bonne spécification fait plusieurs choses à la fois. Elle définit l'objectif assez clairement pour que le succès soit reconnaissable. Elle établit les contraintes — ce que l'output doit contenir, ce qu'il ne doit pas contenir, dans quel format il doit être, quels cas limites comptent. Elle anticipe les endroits où l'agent aura à faire un jugement et lui dit comment le faire. Elle définit à quoi ressemble « terminé » avant que le travail commence.

C'est sur cette dernière partie que la plupart des spécifications échouent. « Écris une fonction qui traite l'input utilisateur » n'est pas une spécification — c'est le début d'une conversation. Une spécification dit quels inputs sont valides, ce que la fonction doit retourner pour chacun, ce qu'elle doit faire quand l'input est invalide, et quelles caractéristiques de performance comptent. Mettre ça par écrit force une clarté que la version vague reporte à plus tard.

Le lien avec la qualité de l'agent est direct. Un agent qui travaille à partir d'une spécification vague comble les trous avec son propre jugement, qui peut être raisonnable mais ne sera pas constant et ne correspondra pas toujours au tien. Un agent qui travaille à partir d'une spécification précise a moins d'espace pour divaguer et plus de signal sur où aller quand il divague.

Le point plus profond, c'est qu'écrire de bonnes spécifications a de la valeur, qu'un agent soit impliqué ou non. C'est ce que les développeurs seniors font quand ils décomposent un problème avant d'écrire du code. Les agents rendent juste cette compétence plus visible — et son absence plus coûteuse.

Tu n'as pas besoin d'apprendre une nouvelle compétence. Tu dois prendre une ancienne plus au sérieux.
