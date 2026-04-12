---
title: "94. Découpez le projet en phases que l'assistant peut compléter"
date: 2026-04-09
tags:
  - developer-as-user
description: "Un projet décrit comme un seul flux continu est difficile à travailler avec un assistant IA."
---

Un projet décrit comme un seul flux continu est difficile à travailler avec un assistant IA. Le contexte change d'une session à l'autre, l'état du travail est dur à communiquer, et à n'importe quel moment donné, ce n'est pas clair ce que l'assistant devrait être en train de faire ou comment savoir quand un morceau est terminé.

Un projet découpé en phases — chacune avec un scope défini, des livrables clairs et des critères de complétion explicites — s'adapte naturellement à la façon dont l'assistant travaille. Chaque phase tient dans une session ou un petit nombre de sessions. Le livrable de chaque phase est testable avant que la phase suivante ne commence. Les critères de complétion définissent avec quel contexte la phase suivante devrait commencer.

Les phases devraient suivre les dépendances naturelles du projet: les fondations avant les features, les interfaces avant les implémentations, le happy path avant les cas limites. C'est le même séquençage que tu utiliserais pour n'importe quel projet bien planifié — l'assistant ne change pas la logique d'une bonne structure de projet, il rend juste la structure plus importante parce que chaque phase doit être vérifiable indépendamment.

Les frontières de phase sont aussi le bon endroit pour la revue humaine. À la fin de chaque phase, avant de commencer la suivante, révise ce qui a été produit. Est-ce que ça rencontre les critères de complétion? Est-ce que les interfaces sont comme conçues? Y a-t-il des décisions intégrées dans l'implémentation qui vont contraindre les phases futures d'une façon que tu n'avais pas l'intention? Attraper ces choses aux frontières de phase est bon marché. Les attraper après que trois phases ont été construites par-dessus, non.

Structure le projet pour des points de contrôle. L'assistant fait le travail entre eux. Toi, tu fais le travail à ces points.
