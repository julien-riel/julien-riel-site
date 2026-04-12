---
title: "60. Apprenez à lire l'échec comme un détective, pas comme un juge"
date: 2026-04-09
tags:
  - mindset
description: "Quand un agent échoue, l'instinct est de désigner un coupable."
---

Quand un agent échoue, l'instinct est de désigner un coupable. Le modèle a halluciné. Le prompt était mauvais. La récupération a raté. Tu choisis le coupable, tu corriges, tu passes à autre chose. Ça ressemble à du débogage. C'est en fait juste de la reconnaissance de patrons avec un verdict attaché.

Un détective ne commence pas par un verdict. Un détective commence par les preuves et remonte à rebours. Qu'est-ce qui s'est réellement passé? Que montre le log? Qu'y avait-il dans la fenêtre de contexte quand les choses ont mal tourné? Qu'a fait l'agent juste avant d'échouer? Les questions sont précises et les réponses sont descriptives avant d'être évaluatives.

Cette distinction compte parce que les échecs d'agents sont habituellement surdéterminés. Le modèle a halluciné, ET la récupération a raté, ET le prompt était ambigu, ET l'entrée de l'utilisateur était inhabituelle, et ces quatre choses ensemble ont produit l'échec. Si tu choisis un coupable et corriges, tu n'as peut-être rien corrigé du tout — tu as simplement changé quelle combinaison de facteurs causera le prochain échec.

Le mode de pensée du juge crée aussi un problème organisationnel subtil. Si le blâme tombe sur le modèle, la réponse est de changer de modèle. Si le blâme tombe sur le prompt, la réponse est de le réécrire. Ces interventions sont parfois justes, mais elles sont souvent prématurées, faites avant que tu comprennes vraiment ce qui s'est passé. Une équipe qui diagnostique régulièrement mal les échecs construit une base de code pleine de correctifs à des problèmes qu'elle n'avait pas.

Diagnostic avant intervention. Preuve avant conclusion. La discipline, c'est de rester curieux plus longtemps que ce qui est confortable — de résister à l'attraction vers le correctif jusqu'à ce que tu sois certain de comprendre ce que tu corriges.

Concrètement, ça veut dire logger plus que tu ne crois avoir besoin. Ça veut dire construire des outils qui te permettent de rejouer les exécutions d'agents avec différentes entrées. Ça veut dire rédiger des post-mortems d'échecs qui décrivent ce qui s'est passé, pas juste ce qui a été changé. Le but est une équipe qui accumule une vraie compréhension de comment ses systèmes échouent, pas juste une liste croissante de rustines.

Les systèmes d'agents échouent en combinaisons. Les développeurs qui s'améliorent sont ceux qui développent un goût pour la vue d'ensemble — qui peuvent regarder un échec et voir non pas une chose brisée, mais un ensemble de conditions mal alignées, et qui conçoivent ensuite contre les conditions plutôt que contre le symptôme.

Le coupable est rarement celui que tu pensais. L'enquête est toujours plus intéressante qu'elle n'en a l'air au départ.
