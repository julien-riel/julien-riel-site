---
title: "5. Faites confiance au résultat, pas au raisonnement"
date: 2026-04-09
tags:
  - working-with-agents
description: "Le chain-of-thought reasoning est vraiment utile — il améliore la qualité de l'output, rend le processus de l'agent plus lisible, et te donne quelque chose à déboguer quand ça tourne mal."
---

Le chain-of-thought reasoning est vraiment utile — il améliore la qualité de l'output, rend le processus de l'agent plus lisible, et te donne quelque chose à déboguer quand ça tourne mal. Mais ça crée un piège : le reasoning a l'air si cohérent que tu commences à lui faire confiance. Tu lis la logique étape par étape de l'agent, ça fait sens, et tu conclus que l'output doit être correct. C'est à l'envers.

Le reasoning n'est pas une fenêtre sur ce que le modèle fait. C'est un autre output. Le modèle génère le reasoning de la même manière qu'il génère tout le reste — en prédisant les tokens probables étant donné le context. Ce reasoning peut être cohérent en interne, logiquement structuré, et complètement déconnecté du calcul réel qui a produit la réponse finale. On a montré que des modèles produisent des justifications assurées et cohérentes pour des réponses carrément fausses. L'explication sonne bien. La réponse est quand même fausse.

Ça compte parce que le reasoning crée une fausse confiance d'une manière spécifique. Quand un agent produit une réponse nue et qu'elle est fausse, tu vois une réponse fausse. Quand il produit une réponse fausse magnifiquement raisonnée, tu vois un argument convaincant. Le second est plus difficile à attraper et plus facile à suivre. Surtout quand tu bouges vite, quand le domaine est inconnu, quand le reasoning couvre un terrain que tu devrais réfléchir dur pour vérifier indépendamment.

La discipline, c'est d'évaluer les outputs selon leurs propres termes — est-ce que l'output correspond à la réalité, respecte la spec, passe les tests — pas selon la qualité du reasoning qui les a précédés. Traite le reasoning comme un artefact de debugging utile, pas comme une preuve de justesse. Si l'output est faux, le reasoning te dit où regarder. Si l'output est juste, le reasoning est intéressant mais n'est pas le point.

Il y a une erreur liée dans l'autre sens : ne pas faire confiance à un output parce que le reasoning a l'air faux. Parfois les modèles arrivent à des réponses correctes via des chaînes de reasoning qui paraissent bizarres ou prennent des détours inutiles. Le reasoning est un croquis, pas une preuve. Ce qui compte, c'est si la réponse tient quand tu la vérifies contre la ground truth.

Vérifie les outputs. Utilise le reasoning pour comprendre les échecs. Ne laisse pas un bon argument remplacer une bonne réponse.

Le modèle ne montre pas son travail. Il génère l'apparence de montrer son travail. Garde cette distinction proche.
