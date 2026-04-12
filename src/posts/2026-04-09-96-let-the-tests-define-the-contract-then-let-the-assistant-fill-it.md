---
title: "96. Laissez les tests définir le contrat, puis laissez l'assistant le remplir"
date: 2026-04-09
tags:
  - developer-as-user
description: "Écrire les tests avant l'implémentation n'est pas juste une pratique de qualité dans un workflow assisté par IA — c'est un protocole de communication."
---

Écrire les tests avant l'implémentation n'est pas juste une pratique de qualité dans un workflow assisté par IA — c'est un protocole de communication. Une suite de tests bien écrite décrit précisément ce que le code doit faire: les inputs qu'il accepte, les outputs qu'il produit, les cas limites qu'il gère, les erreurs qu'il retourne. Étant donné cette spécification, l'assistant peut implémenter contre elle, et tu as un critère objectif pour savoir si l'implémentation est complète.

Ça change ce que « fini » veut dire d'une façon particulièrement utile quand tu travailles avec une IA. Sans tests, « fini » est un jugement — est-ce que ça a l'air correct, est-ce que ça semble gérer les cas qui m'importent, est-ce que ça suit les patterns que je voulais. Avec des tests, « fini » est vérifiable: les tests passent ou ne passent pas. L'assistant peut vérifier son propre travail plutôt que d'exiger que tu le fasses.

Les tests que tu écris avant l'implémentation sont aussi de meilleurs tests que ceux écrits après. Les tests post-implémentation tendent à refléter l'implémentation — ils testent ce que le code fait plutôt que ce qu'il devrait faire. Les tests pré-implémentation reflètent la spécification — ils testent le contract. Quand l'implémentation dérive du contract, les tests pré-implémentation l'attrapent. Les tests post-implémentation, souvent, non, parce qu'ils ont été écrits pour correspondre à l'implémentation qui a produit la dérive.

Il y a un workflow pratique: écris d'abord le fichier de tests, décris ce que chaque test vérifie dans un commentaire, et laisse l'assistant implémenter le code qui les fait passer. Révise l'implémentation par rapport aux tests. Les tests sont les critères d'acceptation; la revue, c'est de vérifier si l'implémentation les satisfait.

Définis « fini » avant de commencer. L'assistant sait quand il y est.
