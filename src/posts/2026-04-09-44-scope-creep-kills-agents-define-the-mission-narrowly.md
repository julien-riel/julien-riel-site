---
title: "44. Le scope creep tue les agents — définis la mission étroitement"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Chaque agent qui réussit fait face à la même pression : il fonctionne, donc les gens veulent qu'il en fasse plus."
---

Chaque agent qui réussit fait face à la même pression : il fonctionne, donc les gens veulent qu'il en fasse plus. L'agent de service client qui gère les retours se fait demander de gérer les questions de facturation. L'agent de revue de code qui cherche des bugs se fait demander de suggérer des améliorations architecturales. L'agent de recherche qui résume des documents se fait demander de rédiger des rapports. Chaque extension semble incrémentale. Ensemble, elles produisent un agent qui fait trop de choses, n'en fait aucune aussi bien qu'il le devrait, et échoue de façons difficiles à attribuer à une décision particulière.

Le scope creep dans les agents est plus dommageable que dans le logiciel conventionnel parce que les agents n'échouent pas proprement à leurs frontières. Une fonction appelée avec les mauvais arguments lève une exception. Un agent à qui on demande de faire quelque chose en dehors de son espace de conception tentera — et produira quelque chose qui ressemble à un résultat, ce qui est pire qu'une erreur. L'utilisateur pense que la tâche a été faite. L'agent a fait quelque chose d'adjacent à ce qui a été demandé, ou a fabulé une réponse, ou a appliqué un cadre de sa tâche principale à une tâche secondaire où ça ne colle pas. Le failure mode est silencieux et les conséquences arrivent plus tard.

La défense, c'est une définition claire et écrite de ce à quoi sert l'agent — assez spécifique pour que tu puisses répondre, pour toute extension proposée, si elle est dans le scope ou hors scope. Pas « aide au service client » mais « gère les demandes de retour de produit pour les commandes passées dans les quatre-vingt-dix derniers jours, escalade à un humain pour tout le reste ». La spécificité n'est pas de la bureaucratie — c'est ce qui te permet de dire non de façon cohérente quand la cinquième équipe demande d'ajouter une capacité de plus.

Dire non à une extension de scope est une décision produit avec de vrais arbitrages. Parfois l'extension vaut le coup — la capacité est étroitement liée, l'agent la gère bien, le besoin utilisateur est réel. Le but n'est pas de ne jamais étendre, mais d'étendre délibérément, avec une revue complète du prompt, un nouveau cycle de tests, et la reconnaissance explicite que tu changes ce qu'est l'agent. Pas un ajustement incrémental — une nouvelle version avec un nouveau scope.

L'agent qui fait une chose exceptionnellement bien est plus précieux que l'agent qui fait dix choses correctement. Protège la mission.
