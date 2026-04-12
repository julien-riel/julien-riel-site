---
title: "46. L'agent qui fait tout ne fait rien de bien"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Il existe une version fantasmée d'un système d'agents où un seul agent gère tout — n'importe quelle question, n'importe quelle tâche, n'importe quel domaine, avec une compétence égale partout."
---

Il existe une version fantasmée d'un système d'agents où un seul agent gère tout — n'importe quelle question, n'importe quelle tâche, n'importe quel domaine, avec une compétence égale partout. C'est un fantasme séduisant parce qu'il est simple. Un système à construire, un système à maintenir, un système à expliquer aux parties prenantes. La réalité, c'est un agent médiocre partout et excellent nulle part, parce que l'excellence exige de la spécificité et la spécificité exige des limites.

Le mécanisme est direct. Un agent généraliste a besoin d'un system prompt assez large pour couvrir chaque cas qu'il pourrait rencontrer. Les prompts larges produisent des comportements larges — l'agent n'a pas de prior fort sur ce qui est bon dans un contexte donné, donc il produit la moyenne de tout ce qu'il a vu. Cette moyenne est cohérente, fluide et constamment décevante. Elle manque de la netteté qui vient d'un système qui sait exactement pour quoi il optimise.

Les preuves apparaissent dans le comportement des utilisateurs. Les utilisateurs d'agents généralistes développent des contournements — des rituels de prompt élaborés conçus pour pousser l'agent vers le comportement spécifique qu'ils veulent vraiment. Ils apprennent à spécifier le rôle, le format, les contraintes, le ton — toutes les choses qu'un agent dédié aurait encodées dès le départ. L'utilisateur fait le travail de spécialisation au moment de l'interaction, à chaque fois, parce que le système ne l'a pas fait au moment de la conception.

Il y a aussi une dimension organisationnelle. Un agent généraliste n'a pas de propriétaire clair. Quand il échoue à une tâche juridique, est-ce un problème juridique ou un problème d'agent ? Quand il sous-performe en revue de code, est-ce le prompt qui est mauvais, le modèle qui est mauvais, ou le cas d'usage qui est mauvais ? Sans scope défini, pas de responsabilité claire, et sans responsabilité, la qualité ne s'améliore pas — elle dérive.

L'alternative, ce n'est pas nécessairement une prolifération d'agents mono-tâche sans recoupement. C'est un portefeuille d'agents avec des scopes clairs et distincts, chacun optimisé pour son domaine, orchestrés par quelque chose qui route les tâches vers le bon spécialiste. Plus complexe à construire, beaucoup mieux à utiliser.

La généralité est une qualité dans un modèle. Dans un agent, c'est généralement un trou de conception.
