---
title: "81. Le contexte est une compétence que tu peux améliorer"
date: 2026-04-09
tags:
  - developer-as-user
description: "Savoir quel context fournir — et comment le fournir — est la compétence la plus à effet de levier quand tu travailles avec un AI coding assistant."
---

Savoir quel context fournir — et comment le fournir — est la compétence la plus à effet de levier quand tu travailles avec un AI coding assistant. Deux développeurs qui donnent au même assistant la même tâche vont obtenir des résultats différents, et la différence se joue souvent sur la qualité du context. L'un colle l'interface pertinente et un exemple représentatif. L'autre écrit une requête d'une ligne. Les outputs ne sont pas comparables.

La qualité du context a plusieurs dimensions. Pertinence : l'assistant travaille mieux avec le fichier spécifique qu'il doit comprendre qu'avec le repository en entier. Précision : un exemple concret du pattern que tu veux suivre est plus utile qu'une description abstraite de celui-ci. Complétude : les contraintes qui te semblent évidentes — le style de gestion d'erreurs, les conventions de nommage, les dépendances que tu veux éviter — doivent être énoncées explicitement. Format : un context structuré est plus facile à utiliser pour le modèle qu'un mur de texte collé.

La compétence se développe par une attention délibérée à l'échec. Quand l'assistant produit quelque chose de faux, demande-toi : qu'est-ce qui manquait dans le context qui aurait pu empêcher ça ? Habituellement, quelque chose manquait — une contrainte que tu as oublié de mentionner, un exemple que tu n'as pas collé, une convention que tu supposais évidente. Ajoute-la à ta liste mentale pour le prochain prompt.

Avec le temps, tu développes un sens de quel context un type de tâche donné nécessite. Le refactoring de code a besoin du code existant et de l'interface cible. L'écriture de tests a besoin de la signature de la fonction et d'un exemple de comment les tests du module sont structurés. Le bug fixing a besoin du message d'erreur, de la stack trace, et du chemin de code qui les a produits. Ces patterns deviennent intuitifs avec la pratique.

La capacité de l'assistant est fixe. Ta capacité à l'utiliser ne l'est pas. Le context est là où vit l'amélioration.
