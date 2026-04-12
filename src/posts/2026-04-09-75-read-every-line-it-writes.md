---
title: "75. Lisez chaque ligne qu'il écrit"
date: 2026-04-09
tags:
  - developer-as-user
description: "La vitesse de génération, c'est le piège."
---

La vitesse de génération, c'est le piège. L'assistant produit cinquante lignes de code en trois secondes, ça a l'air plausible, les tests passent, tu commit. Deux jours plus tard, tu débogues une défaillance qui remonte à une erreur de logique subtile dans ces cinquante lignes — une erreur qui aurait été évidente si tu avais lu attentivement, ce que tu n'as pas fait parce qu'elle est arrivée vite et avait l'air correcte.

La vitesse de production et la justesse de production sont des variables indépendantes. L'assistant génère du code à un rythme qui crée une pression psychologique pour l'accepter au même rythme. Résiste. Le code mérite la même lecture que tu accorderais à une pull request d'un collègue compétent — ni méfiante, ni une analyse mot à mot, mais une vraie compréhension. Comprends-tu ce que chaque partie fait ? Fait-elle ce que tu voulais ? Y a-t-il des cas limites que l'implémentation ne gère pas ?

Les cas où la lecture attentive importe le plus sont exactement ceux où c'est le plus difficile à maintenir : quand tu es fatigué, quand tu as une échéance, quand la tâche semble routinière, quand tu as demandé quelque chose de similaire plusieurs fois auparavant et que c'était toujours correct. L'assistant ne se fatigue pas. Il ne devient pas négligent sous pression. Mais il ne sait pas non plus ce dont tu as réellement besoin — il sait ce que tu as demandé. Ce sont parfois des choses différentes, et toi seul peux attraper l'écart.

Il y a un mode d'échec spécifique qui vaut la peine d'être nommé : le code qui est techniquement correct mais faux pour ta situation. Il compile, les tests passent, la logique est saine — mais il résout un problème subtilement différent de celui que tu as. Cet échec est invisible si tu vérifies seulement que le code tourne. Il est visible si tu lis pour comprendre.

L'assistant écrit la première version. Tu es responsable de chaque ligne qui est livrée.
