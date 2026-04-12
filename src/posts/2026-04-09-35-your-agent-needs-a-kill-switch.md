---
title: "35. Votre agent a besoin d'un kill switch"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Tout système agentic qui opère avec un quelconque degré d'autonomie a besoin d'un moyen de l'arrêter immédiatement — pas gracieusement, pas après la tâche courante, mais maintenant."
---

Tout système agentic qui opère avec un quelconque degré d'autonomie a besoin d'un moyen de l'arrêter immédiatement — pas gracieusement, pas après que la tâche courante soit complétée, mais maintenant. Ce n'est pas une fonctionnalité que tu ajoutes après que quelque chose a mal tourné. C'est de l'infrastructure que tu construis avant le déploiement, parce que les scénarios qui l'exigent ne s'annoncent pas à l'avance.

Le kill switch est l'incarnation physique du principe que les humains restent aux commandes. Un agent qui fait quelque chose de mal — qui envoie de mauvaises sorties, qui prend de mauvaises décisions, qui se comporte de manière inattendue à grande échelle — doit pouvoir être arrêté par une personne qui n'est pas développeuse, à n'importe quelle heure, sans nécessiter un déploiement ou un changement en base de données. Si arrêter ton agent exige une pull request, tu as construit quelque chose qui est plus difficile à contrôler que ça ne devrait l'être.

À quoi ressemble le kill switch dépend du système. Au minimum, c'est un feature flag qui arrête l'exécution de l'agent au niveau de la tâche — vérifié au début de chaque tâche, ou à chaque étape d'une tâche multi-étapes, pour que l'activation prenne effet en un cycle plutôt qu'après la fin de la tâche courante. Pour les systèmes à plus haute autonomie, ça veut dire la capacité de mettre en pause en cours de tâche, de drainer proprement le travail en cours et d'empêcher un nouveau travail de démarrer — le tout depuis une seule opération que des parties prenantes non techniques peuvent effectuer.

Au-delà de l'arrêt immédiat, tu veux la capacité de comprendre ce qui se passait quand tu as arrêté. Quelles tâches étaient en cours ? Qu'avait déjà fait l'agent ? Quel state a été laissé derrière et a besoin d'être nettoyé ? Un kill switch sans observabilité te laisse arrêté mais pas informé — tu sais que quelque chose n'allait pas, mais pas quoi ni à quel point.

Il y a un principe plus large ici qui s'applique au-delà du kill switch littéral : conçois pour la réversibilité partout où c'est possible. Préfère les opérations que l'agent peut annuler à celles qu'il ne peut pas. Préfère la confirmation humaine pour les actions irréversibles. Construis sur l'hypothèse que tu auras parfois besoin d'arrêter, d'inspecter et d'inverser — et assure-toi que le système le supporte.

L'agent qu'on ne peut pas arrêter n'est pas digne de confiance. Construis le switch en premier.
