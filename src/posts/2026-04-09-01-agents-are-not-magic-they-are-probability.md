---
title: "1. Les agents ne sont pas de la magie, ils sont de la probabilité"
date: 2026-04-09
tags:
  - working-with-agents
description: "Un agent ne connaît pas les choses."
---

Un agent ne connaît pas les choses. Il les prédit. Cette distinction sonne académique jusqu'à ce que tu débogues un échec en production à 2h du matin et que ton agent ait dit à un utilisateur, avec assurance, une chose complètement fausse d'une façon qui paraissait parfaitement raisonnable.

Le modèle mental que la plupart des développeurs amènent aux agents est emprunté aux APIs : tu envoies une requête, tu reçois une réponse, la réponse est correcte ou elle ne l'est pas. Mais ce n'est pas ce qui se passe. Ce qui se passe, c'est un processus statistique — le modèle génère des tokens qui sont susceptibles de suivre l'input, façonnés par tout ce sur quoi il a été entraîné et tout ce que tu as mis dans le context window. Quand il a raison, il a raison pour les bonnes raisons seulement une partie du temps. Quand il a tort, il a tort d'une façon qui ressemble à de l'assurance.

Ça compte parce que ça change la manière dont tu designes. Si ton système en aval fait confiance à l'output de l'agent comme il ferait confiance à une requête de base de données, tu as construit sur du sable. L'output a besoin de couches de validation — non pas parce que l'agent est peu fiable à la manière d'un développeur junior (fait des erreurs, se fatigue, comprend mal les specs) mais parce qu'il est peu fiable d'une façon probabiliste, ce qui est plus difficile à raisonner et plus difficile à attraper.

Les gens qui ont travaillé avec des réseaux de neurones avant les agents comprennent ça intuitivement. Les gens qui viennent de systèmes à base de règles, souvent non, pas au début. Ils s'attendent au déterminisme et quand ils reçoivent de la fluidité à la place, ça se lit comme de la fiabilité. La fluidité n'est pas la fiabilité. Un modèle qui formule les choses clairement et de manière cohérente n'est pas un modèle qui est systématiquement correct.

L'implication pratique, c'est de traiter chaque output d'agent comme une distribution, pas comme une valeur. Parfois cette distribution est serrée — la tâche est bien contrainte, le prompt est précis, le modèle a vu mille exemples de ce truc exact. Parfois elle est large — la tâche est ambiguë, le context est mince, le domaine est niche. Ton job, c'est de savoir à quel type d'output tu as affaire et de construire en conséquence.

Ça veut dire des evals. Ça veut dire des cas de test. Ça veut dire de surveiller ce que l'agent fait vraiment en production, pas juste ce qu'il fait dans la démo du happy path. Ça veut dire designer ton système pour que quand l'agent a tort — et il aura tort — les dégâts soient limités et récupérables.

Rien de tout ça n'est une raison de ne pas utiliser les agents. Leur nature probabiliste est aussi ce qui les rend utiles : ils gèrent l'ambiguïté, généralisent à partir d'exemples, naviguent dans des cas limites qu'aucun système à base de règles n'anticiperait jamais. Mais tu paies cette généralisation en prévisibilité.

Les développeurs qui sont les meilleurs pour travailler avec des agents ont fait la paix avec ça. Ils ne s'attendent pas à ce que l'agent soit une machine déterministe. Ils s'attendent à ce qu'il soit un collaborateur très capable qui se trompe parfois avec assurance, et ils construisent des systèmes capables d'absorber ça.

La magie serait plus simple. La probabilité, c'est ce que tu as.
