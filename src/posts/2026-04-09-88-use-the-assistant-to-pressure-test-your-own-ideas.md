---
title: "88. Utilise l'assistant pour mettre tes propres idées à l'épreuve"
date: 2026-04-09
tags:
  - developer-as-user
description: "Avant de t'engager dans une approche d'implémentation, décris-la à l'assistant et demande ce qui pourrait mal tourner."
---

Avant de t'engager dans une approche d'implémentation, décris-la à l'assistant et demande ce qui pourrait mal tourner. Pas « implement this » — « here's what I'm thinking, what are the failure modes? ». L'assistant n'a aucun attachement à ton idée, aucune incitation sociale à protéger tes sentiments, et une large connaissance de comment des approches similaires ont échoué dans des contexts similaires. Il trouvera des choses que tu as manquées.

C'est du prompting adversarial appliqué à ton propre travail, et c'est l'une des utilisations à plus haute valeur d'un coding assistant. La dynamique sans ego qui rend l'assistant réticent à critiquer — si tu ne demandes pas de critique — devient un atout puissant quand tu l'invites explicitement. Demande-lui de défendre la meilleure version de l'alternative que tu as rejetée. Demande-lui les trois façons les plus probables dont ce design échoue sous charge. Demande-lui ce qu'un reviewer de code sceptique dirait de l'approche.

Le feedback est plus utile avant que le code n'existe. Une fois que tu as écrit l'implémentation, la dynamique du coût irrécupérable entre en jeu et le feedback critique devient plus difficile à mettre en action même quand il est juste. Avant que le code n'existe, le feedback est de l'information pure — ça ne coûte rien de mettre à jour ton design en réponse à une critique que tu ne peux pas immédiatement écarter.

Il y a une version spécifique de ça qui est particulièrement précieuse : demande à l'assistant de proposer une approche alternative et d'expliquer les compromis. Pas parce que l'alternative est nécessairement meilleure, mais parce que comprendre pourquoi tu ne la prends pas affine ta compréhension de pourquoi tu prends l'autre. La meilleure justification pour un choix architectural est celle que tu as articulée explicitement, pas celle qui vit dans ta tête comme « ça semblait juste ».

Ton attachement à tes propres idées est le plus grand obstacle à leur amélioration. L'assistant n'en a aucun.

---

### Building at Scale with an Assistant
