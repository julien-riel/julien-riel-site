---
title: "41. Les utilisateurs casseront votre agent de manières imprévisibles"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Tu peux passer des semaines à tester un agent contre chaque scénario imaginable, et un utilisateur le cassera dès le premier jour avec une entrée que tu n'as jamais envisagée."
---

Tu peux passer des semaines à tester un agent contre chaque scénario imaginable, et un utilisateur le cassera dès le premier jour avec une entrée que tu n'as jamais envisagée. Ce n'est pas un échec d'imagination — c'est une propriété de l'écart entre ceux qui construisent les systèmes et ceux qui les utilisent. Les utilisateurs apportent leurs propres modèles mentaux, leur propre vocabulaire, leurs propres suppositions sur ce que le système peut faire. Ces modèles ne correspondent pas aux tiens, et ce décalage produit des entrées que ton agent n'a jamais vues.

Les entrées qui cassent les agents ne sont généralement pas malveillantes, ni même inhabituelles du point de vue de l'utilisateur. C'est l'expression naturelle de la manière dont cette personne pense au problème. Un utilisateur qui colle tout un fil d'emails dans un champ conçu pour une seule question. Un utilisateur qui demande à l'agent de faire quelque chose d'adjacent à sa mission, en supposant qu'il devinera ce qu'il veut dire. Un utilisateur qui tape dans sa langue maternelle alors que l'agent a été conçu pour l'anglais. Un utilisateur qui pose la même question de cinq façons différentes, convaincu que la bonne formulation débloquera la réponse qu'il cherche. Chacun de ces cas est un comportement humain normal. Aucun n'est dans ta suite de tests.

La réponse à ça, ce n'est pas de tester plus exhaustivement — tu ne peux pas énumérer l'espace du comportement humain. La réponse, c'est de concevoir pour une gestion gracieuse des entrées inattendues. Que fait l'agent quand il reçoit quelque chose qu'il ne comprend pas ? Échoue-t-il de manière utile, en expliquant ce qu'il peut et ne peut pas faire ? Fait-il une tentative raisonnable en signalant son incertitude ? Produit-il silencieusement quelque chose de plausible mais faux ? La dernière option est celle à écarter, parce qu'elle crée la pire UX : l'utilisateur pense avoir obtenu une réponse, agit dessus, et découvre le problème plus tard.

Le premier mois de production est ta période de test la plus précieuse. Les entrées qui arrivent pendant ce mois représentent la distribution réelle de la façon dont tes utilisateurs pensent au problème — qui est toujours différente de la tienne. Collecte-les. Analyse les échecs. Sers-t'en pour construire le jeu d'évaluation que tes tests pré-lancement n'auraient jamais pu produire.

Conçois pour les utilisateurs que tu as, pas ceux que tu as imaginés.
