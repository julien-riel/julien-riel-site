---
title: "15. La conversation est un environnement de développement"
date: 2026-04-09
tags:
  - working-with-agents
description: "L'interface conversationnelle d'un LLM n'est pas juste un moyen d'obtenir des réponses — c'est un endroit où penser."
---

L'interface conversationnelle d'un LLM n'est pas juste un moyen d'obtenir des réponses — c'est un endroit où penser. Les développeurs qui la traitent comme un moteur de recherche posent une question et évaluent la réponse. Les développeurs qui la traitent comme un environnement de développement itèrent, poussent en retour, explorent des alternatives, et utilisent l'agent comme partenaire de pensée sur tout un problème.

La différence en qualité d'output est significative. Une interaction à un seul tour avec un agent produit ce que le modèle estime être la meilleure réponse probable étant donné le prompt initial. Une conversation multi-tours produit quelque chose façonné par ton feedback, tes corrections, ta connaissance du domaine injectée aux bons moments. Le premier, c'est la meilleure supposition de l'agent. Le second, c'est une collaboration.

Ça recadre ce que signifie être compétent dans le travail avec les agents. Ce n'est pas juste une question d'écrire de meilleurs prompts initiaux — c'est savoir comment piloter une conversation de manière productive. Ça veut dire reconnaître quand l'agent est parti dans la mauvaise direction tôt, avant que tu aies construit par-dessus une fondation défectueuse. Ça veut dire savoir quand demander des alternatives plutôt que d'accepter la première réponse. Ça veut dire comprendre quand injecter du contexte en cours de conversation — « en fait, il y a une contrainte que je n'ai pas mentionnée » — plutôt que de recommencer.

La conversation sert aussi de trace de ta pensée. Les questions que tu as posées, les directions explorées, les impasses identifiées — c'est un log d'un processus de conception. Les équipes qui traitent le développement conversationnel comme du travail jetable perdent cette trace. Les équipes qui la préservent, même de façon informelle, se construisent une image de la façon dont les décisions ont été prises.

Il y a une limite pratique : les longues conversations accumulent du contexte qui peut dériver. La compréhension précoce du problème par l'agent façonne tout ce qui suit, et si cette compréhension précoce était fausse, la correction devient plus difficile à mesure que la conversation grandit. La compétence, c'est de savoir quand repartir à neuf avec de meilleurs inputs plutôt que de continuer à construire sur ce qui est là.

La boîte de prompt vide n'est pas un champ de requête. C'est là que le travail commence.

---

## Partie 2 — Le prompting comme ingénierie
