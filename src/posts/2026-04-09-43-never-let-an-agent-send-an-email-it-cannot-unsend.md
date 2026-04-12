---
title: "43. Ne laisse jamais un agent envoyer un email qu'il ne peut pas annuler"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "L'irréversibilité des actions est la dimension la plus importante de la conception des systèmes d'agents, et c'est celle qui reçoit le moins d'attention avant que quelque chose tourne mal."
---

L'irréversibilité des actions est la dimension la plus importante de la conception des systèmes d'agents, et c'est celle qui reçoit le moins d'attention avant que quelque chose tourne mal. Lire des données est réversible — si l'agent lit la mauvaise chose, rien ne change. Écrire des données est généralement récupérable — les enregistrements peuvent être corrigés, l'état peut être restauré. Mais envoyer un email, publier un message, exécuter une transaction, publier du contenu — ce sont des actions qui existent dans le monde dès l'instant où elles sont prises, et les reprendre est soit impossible soit coûteux.

Le principe est simple : plus une action est irréversible, plus elle mérite de confirmation avant exécution. Un agent qui peut envoyer des emails en ton nom de façon autonome a besoin d'un seuil de confiance plus élevé avant d'agir qu'un agent qui rédige des emails que tu relis. Pas parce que l'action autonome est intrinsèquement mauvaise, mais parce que le coût d'une erreur est asymétrique — un email envoyé par erreur peut endommager une relation, violer une attente de confidentialité, ou créer une obligation légale qu'aucune excuse ne pourra annuler.

Les équipes sous-estiment ce risque aux premiers stades de la construction parce qu'elles testent avec leurs propres comptes, sur leurs propres données, avec des destinataires qui savent que le système est en développement. Les enjeux semblent bas. En production, avec de vrais utilisateurs, de vrais destinataires, de vraies conséquences — le calcul change.

Le pattern de conception, c'est une couche de confirmation entre la décision de l'agent et l'action dans le monde réel. Pour des actions à faibles enjeux et haute réversibilité, la confirmation peut être implicite — l'agent agit et logue ce qu'il a fait. Pour des actions à forts enjeux et faible réversibilité, la confirmation doit être explicite — l'agent présente ce qu'il est sur le point de faire, attend l'approbation, puis agit. La frontière entre ces catégories doit être tracée de façon conservatrice et révisée à mesure que tu apprends comment le système se comporte en production.

Il y a aussi une exigence de transparence. Quand un agent agit au nom de quelqu'un, le destinataire de cette action mérite souvent de le savoir. Un email d'un agent devrait probablement indiquer qu'il vient d'un agent, ou au minimum être relu par un humain qui prend la responsabilité de son contenu. L'alternative — des agents agissant sans couture comme des humains — crée un problème de confiance qui s'étend au-delà de ton système.

Relis avant d'envoyer. Certaines choses ne se reprennent pas.
