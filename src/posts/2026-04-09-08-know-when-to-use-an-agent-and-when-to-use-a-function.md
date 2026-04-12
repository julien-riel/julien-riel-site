---
title: "8. Savoir quand utiliser un agent et quand utiliser une fonction"
date: 2026-04-09
tags:
  - working-with-agents
description: "Les agents sont assez impressionnants pour qu'on soit tenté de les utiliser pour tout."
---

Les agents sont assez impressionnants pour qu'on soit tenté de les utiliser pour tout. Ils gèrent l'ambiguïté, généralisent à travers les tâches, et peuvent faire des choses qu'aucun système déterministe ne pourrait. Mais ils sont aussi lents, chers, et non déterministes. Une fonction qui parse une chaîne de date n'a pas besoin d'un LLM. En utiliser un quand même n'est pas malin — c'est du gaspillage déguisé en sophistication.

La distinction est plus simple qu'elle en a l'air. Si la tâche a une réponse correcte qui peut être calculée de façon fiable avec du code, utilise du code. Si la tâche requiert du jugement, de la compréhension du langage, ou de la généralisation à travers des cas que tu ne peux pas énumérer, utilise un agent. La ligne n'est pas toujours nette, mais elle est plus nette que la plupart des équipes ne la tracent.

Là où les équipes se plantent, c'est dans les cas du milieu — des tâches qui ont l'air de requérir de l'intelligence mais qui en fait n'en ont pas besoin. Extraire des données structurées d'un format cohérent. Router des requêtes vers une des trois catégories connues. Valider qu'un output respecte une spec bien définie. Ces trucs ressemblent à des tâches d'agent parce qu'ils impliquent du texte et de l'interprétation. En fait c'est juste du pattern matching, et le pattern matching, c'est à ça que sert le code.

Le coût de mal classer dans la direction de l'agent est réel. Chaque appel d'agent a de la latence — typiquement des secondes, pas des millisecondes. Chaque appel coûte des tokens. Chaque appel introduit de la variance : le même input peut produire des outputs légèrement différents à différents runs. Pour une tâche qu'une regex ou un classificateur simple gérerait de façon déterministe en microsecondes, ce compromis ne vaut jamais le coup.

Il y a aussi un argument de fiabilité. Une fonction déterministe marche ou elle ne marche pas, et quand elle ne marche pas, tu le sais immédiatement. Un agent qui gère les cas faciles correctement mais dérive sur les cas limites te donne l'illusion de la fiabilité tout en échouant de façons difficiles à attraper. La complexité a le don de se cacher dans l'écart entre « marche la plupart du temps » et « marche de façon fiable ».

Le test pratique : pourrais-tu écrire un test unitaire qui couvre tous les cas que cette tâche rencontrera ? Si oui, écris la fonction. Si l'espace des cas est trop large, trop ambigu, ou trop dépendant du context pour être énuméré — c'est à ce moment-là que tu te tournes vers l'agent.

Utilise le bon outil. L'agent est un outil puissant. Ne t'en sers pas quand un tournevis suffit.
