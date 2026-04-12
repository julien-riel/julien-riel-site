---
title: "86. Découpe les grandes tâches en prompts, pas juste en étapes"
date: 2026-04-09
tags:
  - developer-as-user
description: "Un prompt qui demande cinq cents lignes de code demande à l'assistant de prendre des dizaines de décisions de design sans savoir lesquelles tu as déjà prises, lesquelles sont contraintes par le reste du..."
---

Un prompt qui demande cinq cents lignes de code demande à l'assistant de prendre des dizaines de décisions de design sans savoir lesquelles tu as déjà prises, lesquelles sont contraintes par le reste du codebase, et lesquelles t'importent. L'output sera techniquement cohérent mais architecturalement déconnecté de tes intentions. Tu passeras plus de temps à l'éditer pour le mettre en forme que si tu avais découpé la tâche en plus petits morceaux.

Les plus petits prompts produisent de meilleurs outputs pour la même raison que les petites fonctions sont meilleures que les grandes : chaque unité a une responsabilité unique et claire. Un prompt qui demande une chose — « implement the validation logic for this form, returning a typed result for each field » — peut être précis sur les contraintes et peut produire quelque chose que tu peux évaluer complètement. Un prompt qui demande toute la couche de gestion du formulaire produit quelque chose que tu ne peux évaluer que partiellement jusqu'à ce que tout soit là, moment auquel changer les fondations coûte cher.

La décomposition devrait suivre les coutures naturelles du problème : les couches de l'architecture, la séparation entre transformation de données et effets de bord, la frontière entre logique métier et infrastructure. Ce sont les mêmes coutures que tu utiliserais pour décomposer la tâche pour un développeur junior. L'assistant répond bien à la même structure.

Il y a aussi un argument d'économie de l'attention. Un grand prompt demande au modèle de garder beaucoup de contraintes en attention simultanément, et aux bords de cette fenêtre les choses dérivent — le code plus tardif ne respecte pas pleinement les contraintes établies tôt dans l'output. Les plus petits prompts réinitialisent cette fenêtre proprement et te laissent valider à chaque frontière avant d'avancer.

Décompose avant de prompter. Le prompt est la spécification d'une unité de travail.
