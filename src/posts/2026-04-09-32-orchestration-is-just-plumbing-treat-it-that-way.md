---
title: "32. L'orchestration n'est que de la plomberie — traitez-la ainsi"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Les frameworks d'orchestration ont tendance à devenir le centre de l'attention dans les systèmes agentic."
---

Les frameworks d'orchestration ont tendance à devenir le centre de l'attention dans les systèmes agentic. Le framework est nouveau, il a des opinions, il introduit des abstractions, et très vite tu écris du code qui sert plus à satisfaire le framework qu'à résoudre le problème. C'est un piège familier en logiciel — ça arrive avec les ORMs, avec les meshes microservices, avec les frameworks frontend — et ça arrive aussi avec l'orchestration d'agents.

Le but de l'orchestration, c'est de déplacer les données entre agents, de gérer le state à travers les étapes, de gérer les retries et de brancher les tools. Ce sont de vrais besoins. Ce sont aussi des préoccupations d'infrastructure fondamentalement ennuyeuses. La valeur de ton système vit dans les agents eux-mêmes — dans les prompts, les tools, les évals, la connaissance du domaine que tu as encodée. L'orchestration, ce sont les tuyaux. Personne ne se soucie des tuyaux tant qu'ils fonctionnent.

Traiter l'orchestration comme de la plomberie a des implications pratiques. Ça veut dire choisir l'approche d'orchestration la plus simple qui répond à tes besoins, pas la plus sophistiquée disponible. Ça veut dire garder ta logique d'orchestration mince — routage, séquençage, gestion d'erreurs — et ta logique d'agent grasse. Ça veut dire être prêt à remplacer les frameworks d'orchestration sans réécrire tes agents, ce qui demande de les garder découplés.

Les équipes qui surinvestissent dans l'orchestration le font souvent parce que ça a l'air d'être du progrès. Tu construis de l'infrastructure, tu conçois des systèmes, tu fais des choix techniques. Ça a la texture du vrai travail d'ingénierie. Mais une orchestration qui ne sert pas la capacité des agents, c'est du surcoût. La question à poser pour chaque décision d'orchestration est : est-ce que ça rend mes agents meilleurs, ou est-ce que ça rend mon orchestration plus élaborée ?

Le churn des frameworks est aussi réel dans cet espace. Le framework d'orchestration populaire aujourd'hui peut être dépassé dans un an. Les agents qui sont étroitement couplés à leur framework d'orchestration sont difficiles à migrer. Les agents qui traitent l'orchestration comme une infrastructure interchangeable bougent beaucoup plus librement.

Sache où est la valeur. Elle n'est pas dans les tuyaux.
