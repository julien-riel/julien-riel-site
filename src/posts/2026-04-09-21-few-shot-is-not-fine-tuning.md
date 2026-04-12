---
title: "21. Le few-shot n'est pas du fine-tuning"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "Le prompting few-shot — fournir des exemples dans le context window pour façonner le comportement du modèle — est puissant et largement utilisé."
---

Le prompting few-shot — fournir des exemples dans le context window pour façonner le comportement du modèle — est puissant et largement utilisé. Il est aussi largement mal compris. Les développeurs qui obtiennent de bons résultats avec des exemples few-shot concluent parfois qu'ils ont effectivement personnalisé le modèle. Ce n'est pas le cas. Ils ont influencé une seule inférence. La différence compte énormément quand tu conçois un système qui doit être fiable à grande échelle.

Le fine-tuning change les poids du modèle. Le comportement appris est cuit dedans — il généralise à travers les inputs, persiste à travers les sessions, et ne consomme pas d'espace dans le context window. Le prompting few-shot ne change rien au modèle. Il fournit des exemples qui influencent la génération courante, et quand le context window se ferme, l'influence se ferme avec. Chaque nouvel appel repart du modèle de base.

Ça veut dire que les exemples few-shot doivent voyager avec chaque requête. Dans un système à fort volume, c'est un vrai coût — les tokens dépensés sur les exemples sont des tokens non dépensés sur le contenu pertinent pour la tâche. Ça veut aussi dire que les exemples sont soumis à la dynamique du context window : dans une longue conversation, les premiers exemples peuvent perdre leur influence à mesure que le contenu plus récent les éloigne du point de génération.

Le malentendu le plus lourd de conséquences concerne la généralisation. Les exemples few-shot enseignent au modèle un pattern pour les cas que tu lui as montrés. Le fine-tuning lui enseigne quelque chose de plus durable — un comportement qui généralise à travers la distribution des inputs qu'il rencontrera. Si ton cas d'usage exige un comportement cohérent sur une grande variété d'inputs, le prompting few-shot peut te donner une fausse confiance : il fonctionne sur les exemples que tu as testés et se dégrade sur les inputs qui ne leur ressemblent pas.

Rien de tout ça ne veut dire que le prompting few-shot n'a pas de valeur — c'est souvent le bon outil, surtout pour le contrôle du format, l'imitation de style, et les tâches où tu as quelques exemples représentatifs. Mais c'est une technique de prompting, pas une technique d'entraînement. S'attendre à ce qu'elle se comporte comme une technique d'entraînement te fera investir dans des exemples alors que tu devrais investir dans l'évaluation, ou sauter le fine-tuning quand la tâche le justifie vraiment.

Sache ce que fait le tool. Utilise-le pour ce à quoi il sert.
