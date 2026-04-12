---
title: "14. Révisez la sortie de l'agent comme vous révisez la PR d'un junior"
date: 2026-04-09
tags:
  - working-with-agents
description: "Le bon modèle mental pour réviser l'output d'un agent, ce n'est pas la relecture — c'est la code review."
---

Le bon modèle mental pour réviser l'output d'un agent, ce n'est pas la relecture — c'est la code review. Pas un survol rapide pour les fautes de frappe, mais une lecture attentive pour la justesse, les cas limites, les suppositions cachées, et les choses qui ont l'air correctes mais qui ne le sont pas.

La pull request d'un développeur junior mérite une vraie attention, non pas parce que les juniors sont mauvais dans leur travail, mais parce qu'ils travaillent avec moins de contexte que toi. Ils peuvent ne pas connaître le cas limite que tu as déjà vu. Ils peuvent avoir résolu le problème énoncé tout en manquant la contrainte implicite. Le code peut marcher aujourd'hui et échouer dans des conditions qu'ils n'ont pas pensé à tester. La review n'est pas une formalité — c'est là que le transfert de connaissance se fait et où les erreurs sont attrapées avant qu'elles n'aient de l'importance.

L'output d'un agent a le même profil. L'agent est capable, souvent de façon impressionnante, mais sa connaissance de ton contexte spécifique est limitée à ce que tu lui as donné. Il ne sait pas ce que tu as appris en trois ans sur ce codebase. Il ne connaît pas le client qui fait la chose inhabituelle qui casse l'implémentation évidente. Il ne sait pas que la dernière personne qui a pris cette approche l'a regretté. Il sait ce que tu lui as dit, et il a généralisé à partir de là.

Réviser avec ce cadre change ce que tu cherches. Tu arrêtes de demander « est-ce que c'est grammaticalement correct » ou « est-ce que ça a du sens en général » et tu commences à demander « est-ce que c'est vraiment juste pour notre situation ». Tu cherches les endroits où l'agent a fait une supposition raisonnable qui se trouve être fausse dans ton contexte. Tu vérifies les bords — ce qui se passe avec un input vide, avec un input inhabituellement gros, avec l'utilisateur qui fait la chose que personne n'avait anticipée.

Le mode d'échec consistant à traiter l'output d'un agent comme du travail terminé est subtil parce que l'output a souvent l'air terminé. Il est fluide, bien structuré, cohérent en interne. Ce sont des propriétés de surface. La justesse dans ta situation spécifique est une propriété plus profonde, et elle ne vient pas du modèle — elle vient de toi.

L'agent a écrit le brouillon. Tu en es responsable. Révise en conséquence.
