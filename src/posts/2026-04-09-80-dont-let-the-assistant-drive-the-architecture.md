---
title: "80. Ne laissez pas l'assistant piloter l'architecture"
date: 2026-04-09
tags:
  - developer-as-user
description: "L'assistant est excellent pour implémenter des décisions."
---

L'assistant est excellent pour implémenter des décisions. Ce n'est pas la bonne entité pour les prendre. Cette distinction s'effondre rapidement en pratique — tu demandes une implémentation, l'implémentation implique une architecture, tu acceptes l'implémentation, et l'architecture est maintenant dans ton codebase sans avoir été choisie délibérément.

Le mode d'échec est subtil parce que les choix architecturaux de l'assistant sont généralement raisonnables. Il prend des patterns bien connus, utilise des abstractions standards, fait des décisions conventionnelles. Le problème n'est pas que les choix sont mauvais dans l'absolu — c'est qu'ils ne sont peut-être pas les bons pour ton contexte spécifique, les conventions de ton équipe, les contraintes de ton système, ta direction à long terme. L'assistant ne sait rien de tout ça à moins que tu le lui aies dit, et les décisions architecturales sont exactement le genre de chose qu'il est difficile de spécifier complètement dans un prompt.

La règle pratique est de prendre la décision architecturale avant de prompter, pas après. Avant de demander à l'assistant d'implémenter une nouvelle fonctionnalité, décide comment elle devrait s'insérer dans la structure existante. Avant de lui demander d'ajouter un nouveau module, décide où ce module appartient et comment il communique avec ses voisins. Le prompt devrait spécifier l'architecture. L'assistant devrait l'implémenter.

Quand tu n'es pas sûr de quelle est la bonne architecture — ce qui est souvent le cas — c'est un signal qu'il faut faire le travail de design en premier, pas de le déléguer à l'assistant. Demande à l'assistant de t'aider à réfléchir aux options, décrire les compromis, identifier les contraintes que tu pourrais manquer. Utilise-le comme partenaire de réflexion dans le processus de design. Mais prends la décision toi-même, explicitement, avant que le code n'existe.

L'assistant construit ce que tu lui donnes à construire. Assure-toi que c'est toi qui décides quoi construire.
