---
title: "10. Votre agent n'a pas de mémoire sauf si vous lui en donnez une"
date: 2026-04-09
tags:
  - working-with-agents
description: "Chaque fois que tu appelles un LLM, il repart de zéro."
---

Chaque fois que tu appelles un LLM, il repart de zéro. Il n'a aucun souvenir de la conversation précédente, de la dernière tâche, de la dernière erreur qu'il a faite ou de la correction que tu lui as donnée. Le context window est l'intégralité de ce qu'il sait. Quand la fenêtre se referme, tout ce qu'elle contenait disparaît.

C'est la partie de l'architecture des agents qui surprend le plus les gens, et qui continue de les surprendre même après qu'ils l'ont comprise intellectuellement. L'agent semblait comprendre le projet. Il semblait avoir une intuition du codebase. Puis tu démarres une nouvelle session et c'est un inconnu, qui te pose des questions auxquelles tu as déjà répondu la semaine dernière.

La memory dans les systèmes agentiques est un problème d'ingénierie que tu dois résoudre explicitement. Il y a quelques approches, chacune avec ses compromis. Tu peux étendre le context window — continuer à ajouter l'historique de conversation jusqu'à ce que ça rentre. Ça marche jusqu'à ce que ça ne marche plus : les context windows ont des limites, les longs contextes ralentissent le reasoning, et les modèles ont tendance à perdre de vue les informations des premières parties d'un long contexte. Tu peux utiliser le retrieval — stocker les interactions passées dans une base vectorielle et ramener les morceaux pertinents au début de chaque nouvelle session. Ça passe mieux à l'échelle mais ça exige que tu fasses bien le retrieval, ce qui est un problème en soi. Tu peux maintenir un state structuré — un document ou une base de données qui capture les faits clés que tu veux que l'agent conserve, mis à jour explicitement après chaque session.

La bonne approche dépend du type de memory dont tu as besoin. Il y a une différence entre la memory épisodique — ce qui s'est passé dans les sessions précédentes — et la memory sémantique — les faits que l'agent devrait connaître sur le domaine. Il y a une différence entre une memory qui doit être exacte et une memory qui doit juste être approximativement correcte. Concevoir pour la memory, c'est être précis sur ce qui doit persister, pourquoi, et à quelle fidélité.

L'erreur des équipes, c'est de supposer que la memory va émerger du modèle. Non. Le modèle est stateless par conception. Si ton agent a besoin de continuité entre les sessions, tu dois la construire, la maintenir, et la passer explicitement à chaque fois.

L'agent ne se souvient de rien. C'est toi qui décides ce qu'il garde.
