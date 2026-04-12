---
title: "76. L'assistant ne connaît pas votre codebase sauf si vous la lui montrez"
date: 2026-04-09
tags:
  - developer-as-user
description: "Chaque session repart à zéro."
---

Chaque session repart à zéro. L'assistant n'a aucun souvenir du refactoring que tu as fait la semaine dernière, de la convention que tu as établie le mois dernier, de la décision architecturale que tu as prise l'année dernière et des raisons derrière. Il sait ce que tu as mis dans la fenêtre de contexte, et rien d'autre. C'est la même contrainte qui s'applique à tout agent — mais elle surprend les développeurs qui travaillent productivement avec un assistant depuis des mois et commencent à sentir qu'il connaît le projet.

Ce sentiment est compréhensible. Quand tu as eu des centaines de bonnes interactions, quand l'assistant produit constamment du code qui s'intègre à tes patterns, on commence à avoir l'impression qu'un contexte partagé s'est accumulé. Ce n'est pas le cas. Ce qui s'est passé, c'est que tu es devenu meilleur à fournir le contexte implicitement — tu as appris à formuler les demandes de façons qui encodent tes conventions, à coller le bon code de référence, à décrire des contraintes que tu laissais autrefois non dites. L'assistant n'a pas appris ton codebase. Tu as appris à le porter avec toi.

Cette distinction importe quand quelque chose tourne mal. Si l'assistant produit du code qui viole une convention du projet, l'échec n'est pas l'assistant qui oublie — c'est toi qui ne fournis pas. Le modèle mental d'un collègue oublieux te mène à te sentir frustré contre l'assistant. Le bon modèle mental d'un système sans état te mène à corriger le contexte.

La réponse pratique est de développer des habitudes de fourniture de contexte : coller un exemple représentatif du pattern à suivre, inclure l'interface à laquelle le nouveau code doit se conformer, ajouter un commentaire sur la contrainte qui n'est pas évidente. Ces habitudes ne font pas qu'aider l'assistant — elles documentent les choses qui sont actuellement implicites dans ta tête, ce qui rend le codebase plus facile à maintenir peu importe qui l'écrit.

L'assistant est aussi bon que le contexte que tu fournis. C'est entièrement sous ton contrôle.
