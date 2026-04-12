---
title: "85. Quand l'output est mauvais, corrige le prompt avant de corriger le code"
date: 2026-04-09
tags:
  - developer-as-user
description: "Quand l'assistant produit du code qui n'est pas tout à fait correct, l'instinct est d'éditer le code directement — c'est plus rapide, c'est familier, ça produit le résultat dont tu as besoin immédiatement."
---

Quand l'assistant produit du code qui n'est pas tout à fait correct, l'instinct est d'éditer le code directement — c'est plus rapide, c'est familier, ça produit le résultat dont tu as besoin immédiatement. Mais éditer le code manuellement est une solution à une instance. Corriger le prompt est une solution à la classe de cas que le prompt représente, et l'assistant rencontrera cette classe à nouveau.

L'habitude du debugging orienté prompt paie plus à mesure que le codebase grandit et que les mêmes tâches reviennent. Si l'assistant produit systématiquement des fonctions avec une gestion d'erreurs insuffisante, et que tu les corriges systématiquement à la main, tu as établi un workflow où l'assistant fait les parties faciles et tu nettoies les parties difficiles à chaque passe. Si au lieu de ça tu identifies le pattern, ajoutes une contrainte claire à ton prompt — « always handle the case where the input is null and return a typed error » — tu changes l'output pour la suite.

Ça demande de faire une pause avant d'éditer, ce qui est la partie difficile. Quand tu es fatigué et que le code doit être correct, saisir le clavier est plus rapide que de penser à pourquoi le prompt a échoué. Mais l'accumulation de corrections manuelles sans amélioration du prompt est une dette technique d'un autre genre — tu compenses un écart connu dans ton workflow sans t'y attaquer.

La question à se poser avant d'éditer : pourquoi le prompt a-t-il produit ça ? Habituellement la réponse est l'une d'un petit ensemble : la contrainte n'a pas été énoncée, l'exemple montrait quelque chose de différent de ce que tu voulais, le résultat n'a pas été spécifié assez clairement. Identifier laquelle prend trente secondes et produit un meilleur prompt pour la prochaine fois.

Édite le code quand tu dois livrer maintenant. Corrige le prompt pour ne pas avoir à éditer la prochaine fois.
