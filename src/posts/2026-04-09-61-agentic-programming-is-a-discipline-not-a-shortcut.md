---
title: "61. La programmation agentic est une discipline, pas un raccourci"
date: 2026-04-09
tags:
  - mindset
description: "Le pitch de la programmation agentic sonne souvent comme une promesse de moins de travail."
---

Le pitch de la programmation agentic sonne souvent comme une promesse de moins de travail. Tu décris ce que tu veux, l'agent le fait, tu révises et tu livres. Moins de code, moins de débogage, moins de tâches fastidieuses qui ralentissent tout. Il y a du vrai là-dedans — les agents réduisent certains types de travail de façon significative. L'erreur est de conclure que moins d'un type de travail signifie moins de travail au total.

Le travail que les agents réduisent est en grande partie du travail d'exécution — la traduction d'une spécification bien comprise en code ou contenu fonctionnel. C'est du vrai labeur et les agents le gèrent bien. Le travail que les agents ne réduisent pas — et qu'ils augmentent à certains égards — c'est le travail de réflexion : comprendre le problème assez clairement pour le spécifier, concevoir le système de manière assez réfléchie pour qu'il soit maintenable, réviser les sorties assez soigneusement pour détecter ce qui a mal tourné, construire l'infrastructure d'évaluation pour savoir si les choses s'améliorent ou se dégradent.

En fait, la programmation agentic relève la barre du travail de réflexion. Quand l'exécution est bon marché, le goulot d'étranglement se déplace vers la spécification. Le développeur qui pouvait s'en tirer avec un modèle mental flou du problème — parce que l'implémentation révélerait les lacunes rapidement et à bas coût — a maintenant besoin d'un modèle plus net en amont, parce que l'agent exécutera fidèlement la spécification floue et produira quelque chose qui a l'air complet mais qui n'est pas juste. La taxe sur la pensée imprécise est plus élevée, pas plus basse.

La discipline se manifeste dans les pratiques qui distinguent les équipes qui livrent des systèmes agentic fiables de celles qui livrent des démos impressionnantes. Des evals. Des prompts versionnés. De l'infrastructure d'observabilité. Une définition soignée du scope. Des points de contrôle humains pour les actions conséquentes. Aucun de ces éléments n'est un raccourci — c'est la rigueur d'ingénierie qui rend le système digne de confiance plutôt que simplement fonctionnel.

Les développeurs qui abordent la programmation agentic comme un raccourci tendent à accumuler de la dette technique précisément aux endroits où les systèmes d'agents sont le plus fragiles : gestion des prompts, validation des sorties, gestion des échecs. La démo a fonctionné. Le système en production ne fonctionne pas, pas de façon fiable, et maintenant ils font le travail d'ingénierie qu'ils avaient reporté, sous pression, avec un système vivant qui se comporte déjà mal.

Le raccourci est un prêt. La discipline est ce qui le rend valable.
