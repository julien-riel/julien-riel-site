---
title: "3. Votre agent n'est aussi bon que son contexte"
date: 2026-04-09
tags:
  - working-with-agents
description: "Garbage in, garbage out est un des plus vieux principes en informatique."
---

Garbage in, garbage out est un des plus vieux principes en informatique. Avec les agents, c'est plus insidieux : du sophistiqué en entrée, de la merde au son sophistiqué en sortie. L'agent va travailler avec n'importe quel context que tu lui donnes, et il va le faire avec fluidité, ce qui veut dire qu'un mauvais context ne produit pas d'erreurs évidentes — il produit des réponses fausses qui ont l'air plausibles.

Le context, c'est tout ce que l'agent sait quand il commence à travailler : le system prompt, l'historique de conversation, les documents que tu as récupérés, les outputs de tools que tu lui as repassés. L'agent n'a accès à rien en dehors de cette fenêtre. Il ne peut pas vérifier. Il ne peut pas demander (sauf si tu l'as construit comme ça). Il raisonne à partir de ce qu'il a, et si ce qu'il a est incomplet, périmé, ou subtilement faux, le reasoning le sera aussi.

Le mode d'échec que les développeurs rencontrent le plus souvent n'est pas le context manquant — c'est le context supposé. Tu connais la codebase, les règles métier, les cas limites qui comptent. L'agent, non, à moins que tu le lui aies dit. Quand tu lances une tâche et que tu reçois un résultat techniquement correct mais manifestement faux pour ta situation, c'est généralement pour ça. L'agent a résolu le problème qu'on lui a donné. Tu lui as donné le mauvais problème.

Les systèmes retrieval-augmented rendent ça concret. Tu construis un pipeline qui tire les documents pertinents dans le context avant que l'agent tourne. Ça marche magnifiquement en test, là où ton retrieval tombe sur les bons documents. En production, le retrieval rate. L'agent reçoit des documents adjacents — assez liés pour paraître bons, assez faux pour que ça compte. Et parce que l'agent ne sait pas ce qu'il ne sait pas, il avance avec assurance avec ce qu'il a.

La discipline, c'est d'auditer ton context avant d'auditer ton prompt. Quand un agent échoue, la première question, ce n'est pas « le modèle s'est-il embrouillé ? » — c'est « qu'est-ce que le modèle a vraiment vu ? ». Logge le context complet. Lis-le comme l'agent le ferait. Souvent l'échec est évident au moment où tu fais ça : un élément d'information clé n'était pas là, ou quelque chose de contradictoire l'était.

Designer un bon context est une compétence sous-estimée. Ça veut dire savoir quoi inclure, quoi exclure, et comment structurer l'information pour que l'agent puisse l'utiliser. Trop de context est un problème en soi — l'agent enterre le signal dans le bruit, ou atteint la limite du context window et perd les premières parties de la conversation. Trop peu et tu attends de l'inférence là où tu as besoin de faits.

L'agent fait de son mieux avec ce que tu lui as donné. Donne-lui de meilleures choses.
