---
title: "30. Ne laissez pas votre agent toucher la production avant qu'il ne vous ait ennuyé en staging"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Dans chaque projet agentic, il arrive un moment où le système fonctionne assez bien en test pour que la tentation de déployer devienne presque irrésistible."
---

Dans chaque projet agentic, il arrive un moment où le système fonctionne assez bien en test pour que la tentation de déployer devienne presque irrésistible. Les démos sont propres. Les cas évidents passent tous. L'équipe le regarde depuis des semaines et personne ne trouve une nouvelle façon de le casser. On livre.

Non. Pas encore.

L'écart entre « ça marche en test » et « ça marche en production » est plus large pour les systèmes agentic que pour la plupart des logiciels, parce que les systèmes agentic rencontrent dans le monde réel une distribution d'inputs bien plus diverse qu'aucune suite de tests ne capture. Les utilisateurs font des choses inattendues. Ils fournissent du contexte dans des formats inhabituels. Ils posent des questions à la limite du périmètre. Ils combinent des capacités d'une façon que tu n'avais jamais anticipée. L'agent qui gère tes cas de test avec élégance peut quand même échouer lamentablement sur les inputs que tu n'as pas pensé à tester.

La discipline, c'est de faire tourner le système en staging — contre des inputs qui ressemblent au monde réel, avec la variabilité du monde réel — jusqu'à ce qu'il arrête de te surprendre. Pas jusqu'à ce qu'il gère tout parfaitement, mais jusqu'à ce que les modes de défaillance te soient familiers. Jusqu'à ce que tu aies vu les cas limites et décidé comment les traiter. Jusqu'à ce que le comportement te paraisse prévisible non pas parce qu'il n'échoue jamais, mais parce que quand il échoue, il échoue de façons que tu reconnais et que tu as anticipées.

Le standard « t'ennuyer » est délibérément subjectif. Ça veut dire que le système tourne depuis assez longtemps pour que tu ne découvres plus de nouveaux modes de défaillance. Tu as arrêté d'être surpris. Le dernier échec intéressant date d'un moment. C'est à ce moment-là que tu as assez confiance dans le comportement du système pour le confier à de vrais utilisateurs.

Ça demande une patience qui est vraiment difficile à maintenir quand les parties prenantes sont impatientes et que le système semble prêt. L'argument pour attendre est asymétrique : un déploiement prématuré qui échoue lamentablement coûte plus cher — en confiance des utilisateurs, en temps de debug, en réputation — qu'un déploiement prudent qui prend quelques semaines de plus.

Laisse-le t'ennuyer d'abord. La production n'est pas un environnement de test.
