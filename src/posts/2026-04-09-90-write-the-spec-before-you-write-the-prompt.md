---
title: "90. Écrivez la spec avant d'écrire le prompt"
date: 2026-04-09
tags:
  - developer-as-user
description: "Pour une petite tâche — corriger ce bug, ajouter ce champ — le prompt peut être la spec."
---

Pour une petite tâche — corriger ce bug, ajouter ce champ — le prompt peut être la spec. Pour n'importe quoi de significatif — un nouveau module, une nouvelle API, un refactoring substantiel — prompter sans spec produit du code qui implémente ce que tu as demandé plutôt que ce dont tu avais besoin. Ce sont les mêmes choses seulement quand tu as réfléchi soigneusement à ce dont tu as besoin, et c'est précisément le travail qu'écrire une spec exige.

Une spec n'a pas à être un document formel. Ça peut être une courte description en prose: ce que la chose fait, ce qu'elle ne fait pas, comment elle s'intègre au système existant, quels sont les cas limites, à quoi ressemble le succès. La discipline de l'écrire — avant que le code n'existe — force les décisions que le prompting essaie de contourner. D'où viennent les données? Comment les erreurs sont-elles remontées? Qu'arrive-t-il quand la dépendance n'est pas disponible? Écrire la spec fait émerger ces questions. Prompter les enterre.

La spec sert aussi de point de référence pour la revue. Quand l'assistant produit une implémentation, la question n'est pas « est-ce que ça a l'air raisonnable? » — c'est « est-ce que ça implémente la spec? ». Du code qui a l'air raisonnable mais qui n'implémente pas la spec, c'est un échec. La revue pilotée par la spec est plus rapide et plus fiable que la revue pilotée par l'intuition, parce que la cible est explicite.

Dans une équipe collaborative, la spec est aussi de la communication — c'est comme ça que tu établis un alignement sur ce qui sera construit avant que le code n'existe, au moment où changer de direction est bon marché. Un prompt envoyé directement à un assistant avant que l'équipe se soit alignée sur la spec, c'est une façon de générer du code que tu pourrais avoir à jeter.

Écris la spec. Le prompt, c'est comme ça que tu la passes à l'assistant.
