---
title: "7. L'humain dans la boucle est une fonctionnalité, pas une faiblesse"
date: 2026-04-09
tags:
  - working-with-agents
description: "Il existe une version du futur agentique où l'automatisation est le but et l'intervention humaine est le mode d'échec — chaque étape qui requiert une personne pour réviser, approuver, ou corriger est une friction à éliminer."
---

Il existe une version du futur agentique où l'automatisation est le but et l'intervention humaine est le mode d'échec — chaque étape qui requiert une personne pour réviser, approuver, ou corriger est une friction à éliminer. C'est le mauvais cadre, et ça produit des systèmes fragiles.

La supervision humaine est un design pattern, pas une solution temporaire en attendant que les modèles s'améliorent. Certaines décisions devraient exiger un humain. Pas parce que l'agent ne peut pas les prendre — souvent il peut, avec une précision raisonnable — mais parce que les conséquences de se tromper sont assez élevées pour que le coût d'un checkpoint humain vaille la peine d'être payé. Envoyer un courriel à mille clients. Modifier une base de données de production. Exécuter une transaction financière. L'agent pourrait les faire correctement 98 % du temps. Les 2 %, c'est la raison pour laquelle tu gardes un humain dans la boucle.

La question plus intéressante, c'est où mettre l'humain. Tôt dans un workflow, tu peux attraper les mauvais inputs avant qu'ils se propagent. Tard dans un workflow, tu peux réviser les outputs avant qu'ils deviennent des effets dans le monde réel. Au milieu, tu peux intervenir sur des points de décision spécifiques — ceux où la confiance de l'agent est basse, ou les enjeux sont élevés, ou les deux. Chaque placement a des coûts différents et des caractéristiques d'échec différentes. La décision de design, c'est de choisir quelle combinaison correspond à ta tolérance au risque.

Les équipes qui résistent au design human-in-the-loop le justifient souvent par la vélocité — les étapes de revue ralentissent les choses, l'automatisation est le but, les utilisateurs veulent des résultats instantanés. Ce sont de vraies contraintes. Elles sont aussi souvent exagérées. Les utilisateurs ne veulent pas des résultats instantanés autant qu'ils veulent des résultats corrects. Un agent qui agit immédiatement et de travers est pire qu'un qui pause et demande. La pause ressemble à de la friction jusqu'à ce que l'alternative soit un courriel d'excuses.

Plus ton agent est autonome, plus tes checkpoints humains deviennent importants — pas moins. L'autonomie complète est appropriée pour des tâches étroites, bien comprises, à faibles enjeux, réversibles. Tout le reste mérite un checkpoint quelque part.

Construis les checkpoints en premier. Retire-les délibérément, un à la fois, à mesure que tu gagnes de la confiance. Les développeurs qui vont dans l'autre direction — qui automatisent d'abord et ajoutent la supervision après qu'un truc a mal tourné — l'ajoutent toujours sous pression, ce qui est le pire moment pour prendre de bonnes décisions de design.

L'autonomie se mérite. La supervision, c'est comment tu la mérites.
