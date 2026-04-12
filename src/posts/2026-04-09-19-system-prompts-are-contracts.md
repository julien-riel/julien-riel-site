---
title: "19. Les system prompts sont des contrats"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "Un system prompt n'est pas une instruction — c'est un contrat."
---

Un system prompt n'est pas une instruction — c'est un contrat. Il définit ce qu'est l'agent, ce qu'il fait, et ce qu'il refuse de faire. À l'instant où tu le traites comme une suggestion, tu as perdu le contrôle du système.

Les contrats ont des propriétés spécifiques. Ils sont explicites, pas implicites. Ils sont stables — on ne change pas un contrat en pleine transaction sans l'accord des deux parties. Les cas limites y sont énoncés, pas laissés à l'interprétation. Et surtout, ils créent des attentes : les systèmes en aval, les utilisateurs, et les autres agents se comportent tous en fonction de ce que le contrat promet. Casse le contrat silencieusement et tout ce qui est en aval casse d'une façon difficile à tracer.

La plupart des system prompts sont écrits comme des brouillons. Vagues sur le périmètre, silencieux sur les modes d'échec, inconsistants sur le format. Ils fonctionnent bien dans le chemin heureux et s'effondrent au moment où quelque chose d'inattendu arrive. Ce n'est pas un problème de prompt — c'est un problème de contrat. Le contrat ne couvrait pas le cas.

Écrire un system prompt comme un contrat veut dire être explicite sur les choses auxquelles tu préférerais ne pas penser. Que fait l'agent quand l'utilisateur demande quelque chose en dehors de son périmètre ? Que fait-il quand les tool calls échouent ? Quand le contexte est ambigu, est-ce qu'il demande des clarifications ou fait sa meilleure supposition ? Ce ne sont pas des cas limites que tu peux différer — ce sont les cas qui définissent le comportement réel du système en production.

Il y a aussi l'exigence de stabilité. Les équipes qui itèrent rapidement sur les system prompts créent souvent un problème plus subtil : le contrat change, mais rien en aval n'est notifié. Un agent qui renvoyait du JSON structuré renvoie maintenant du texte parce que quelqu'un a « amélioré » le system prompt. La pipeline qui parsait ce JSON casse. C'est pourquoi le versioning des prompts n'est pas juste une bonne hygiène — c'est de la gestion de contrat.

La partie la plus difficile d'écrire un bon system prompt, c'est l'espace négatif : ce que l'agent ne fera pas. C'est tentant de ne spécifier que le comportement positif. Mais un agent sans contraintes explicites comblera l'ambiguïté avec quelque chose, et ce quelque chose pourrait ne pas être ce que tu voulais. C'est souvent dans les contraintes négatives que vit le vrai contrat.

Traite un system prompt modifié comme tu traiterais un contrat d'API modifié — avec des tests, avec du versioning, et avec l'hypothèse que quelque chose en aval dépend de l'ancien comportement.

L'agent honorera le contrat que tu lui as donné. Écris-en un qui mérite d'être honoré.
