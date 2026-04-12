---
title: "16. Les prompts dérivent — versionnez-les comme du code"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "Un prompt qui marche aujourd'hui ne marchera pas nécessairement demain."
---

Un prompt qui marche aujourd'hui ne marchera pas nécessairement demain. Les modèles sont mis à jour. Ton application évolue. Les cas limites auxquels ton prompt était ajusté cèdent la place à de nouveaux. Quelqu'un ajuste la formulation pour corriger un comportement et casse un autre par inadvertance. Six mois plus tard, personne ne peut expliquer pourquoi le prompt dit ce qu'il dit, et le changer semble risqué parce que personne ne sait ce qu'il tient ensemble.

C'est de la décomposition logicielle, et ça arrive aux prompts pour les mêmes raisons que ça arrive au code : ils accumulent des changements sans documentation, ils deviennent porteurs sans que personne ne le déclare, et le contexte qui les rendait sensés à l'époque s'évapore en même temps que les gens qui les ont écrits.

Le version control est la solution évidente. Les prompts appartiennent à des repositories, avec des messages de commit qui expliquent non seulement ce qui a changé mais pourquoi. « Ton rendu plus formel » est un mauvais message de commit. « Ton rendu plus formel après que les retours clients ont indiqué que le registre précédent semblait trop décontracté pour les utilisateurs entreprise » est un document de design. Toi du futur — ou le collègue qui hérite de ce système — a besoin du pourquoi, pas juste du quoi.

Au-delà du version control, les prompts bénéficient de la même culture de review que le code. Les changements aux system prompts devraient passer par une review, surtout pour les systèmes en production. Le reviewer ne vérifie pas la grammaire — il demande si ce changement pourrait affecter le comportement d'une façon que l'auteur n'a pas anticipée. Un changement de prompt d'une ligne peut avoir des effets étendus qui ne sont pas évidents avant qu'ils ne surgissent en production.

Le problème plus invisible, c'est le prompt qui dérive sans que personne ne le remarque. Personne n'a changé le fichier. Le modèle a changé. Un system prompt qui était calibré contre une version d'un modèle peut se comporter différemment contre la suivante — subtilement, d'une façon qui ne déclenche pas d'erreurs évidentes mais qui décale la distribution des outputs dans des directions que personne n'avait prévues. Attraper ça exige de l'évaluation : faire tourner tes prompts contre un jeu de test et comparer les outputs entre les versions de modèles.

Traite le prompt comme du code source. Il a la même fragilité, le même besoin de documentation, et la même capacité à devenir immaintable si tu n'en prends pas soin dès le début.

Ce qui n'est pas versionné est déjà perdu.
