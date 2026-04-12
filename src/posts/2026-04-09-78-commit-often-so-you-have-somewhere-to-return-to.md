---
title: "78. Committez souvent, pour avoir un endroit où revenir"
date: 2026-04-09
tags:
  - developer-as-user
description: "Travailler avec un AI coding assistant change le rythme du développement."
---

Travailler avec un AI coding assistant change le rythme du développement. Les changements arrivent en plus gros morceaux, plus vite. Une tâche qui prenait deux heures de travail incrémental arrive maintenant en vingt minutes de génération et de revue. L'accélération est réelle. Le risque aussi : quand quelque chose tourne mal — et ça arrivera — tu veux un état stable récent où revenir, pas un trou de deux heures duquel ressortir.

Les commits fréquents ne sont pas juste de l'hygiène de version control dans ce contexte. C'est le mécanisme qui rend sûr d'aller vite. Chaque commit est un checkpoint : le système était dans un bon état connu ici. Si le prochain lot de code généré casse quelque chose de subtil, tu peux bisecter, comparer, récupérer. Sans commits, tu as un long historique de changements rapides et aucune façon propre de comprendre quand le problème a été introduit.

Le message de commit importe aussi. "WIP" n'est pas utile quand tu débogues trois jours plus tard. "Add input validation per spec section 3.2", oui. L'assistant peut rédiger les messages de commit — et en produira de meilleurs si tu décris ce que le changement accomplit plutôt que ce qu'il fait. Prends trente secondes pour écrire un vrai message de commit. Ton futur toi l'utilisera.

Il y a aussi un bénéfice psychologique. Les commits fréquents créent un sens de sol stable — tu sais où tu as été et où tu peux revenir. Travailler sans eux, surtout au rythme qu'un AI assistant permet, crée une sorte de vertige. Tu bouges vite mais tu n'es pas sûr d'où tu es, et revenir en arrière semble plus difficile qu'il ne devrait l'être.

L'assistant rend facile d'aller vite. Les commits rendent sûr de le faire.
