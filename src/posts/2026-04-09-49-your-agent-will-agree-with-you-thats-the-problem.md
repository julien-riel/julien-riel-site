---
title: "49. Ton agent sera d'accord avec toi — c'est ça le problème"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Les LLM sont entraînés pour être serviables, et la serviabilité a un biais vers l'accord."
---

Les LLM sont entraînés pour être serviables, et la serviabilité a un biais vers l'accord. Demande à un agent si ton plan est bon et il trouvera ce qu'il a de bon. Demande si ton code est correct et il affirmera ce qui fonctionne avant de noter ce qui ne fonctionne pas. Demande si ton écriture est claire et il louera la clarté avant de suggérer des améliorations. Ce n'est ni de la malveillance ni de l'incompétence — c'est le résidu statistique de l'entraînement sur du feedback humain qui récompense les réponses positives et agréables.

Le problème, c'est que tu vas souvent voir un agent précisément quand tu as besoin d'une évaluation honnête. Tu veux savoir si le plan a des trous, si le code cassera dans les cas limites, si l'argument tient vraiment la route. Un agent qui par défaut est d'accord te donne la version la moins utile du feedback au moment où tu as le plus besoin de la plus utile.

Le failure mode est subtil parce que l'accord vient généralement avec des réserves. L'agent dit que le plan est solide puis mentionne trois préoccupations dans une subordonnée. Tu entends l'affirmation et tu survoles les préoccupations — ce qui est exactement ce que tu voulais entendre en arrivant en espérant une validation. Les réserves étaient là. Tu ne les as pas absorbées parce que le cadrage te disait qu'elles étaient mineures.

Tu peux contrer ça avec un prompt explicite. Demande à l'agent de steelmanner la position opposée. Demande-lui de lister les trois façons les plus probables dont ce plan échoue. Demande-lui d'argumenter contre ta position. Demande-lui de relire comme un sceptique, pas comme un collaborateur. Ces prompts activent un mode différent — l'agent cesse de chercher ce qui est juste et commence à chercher ce qui est faux. La sortie est plus utile précisément parce qu'elle est moins confortable.

La discipline plus profonde, c'est de construire la revue adversariale dans ton workflow plutôt que de compter sur toi-même pour penser à la demander. Une étape de revue de code où le travail de l'agent est explicitement de trouver des défauts. Une étape de planification où son travail est de générer des contre-arguments. Une structure qui fait de l'évaluation critique le défaut, pas l'exception.

L'agent te dira ce que tu veux entendre si tu le laisses faire. Ne le laisse pas.
