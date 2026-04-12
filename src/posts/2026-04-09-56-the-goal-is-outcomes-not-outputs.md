---
title: "56. L'objectif est les résultats, pas les outputs"
date: 2026-04-09
tags:
  - mindset
description: "Un agent qui produit un beau résumé d'un document n'a pas réussi."
---

Un agent qui produit un beau résumé d'un document n'a pas réussi. Il a réussi si la personne qui lit le résumé comprend quelque chose qu'elle avait besoin de comprendre, prend une meilleure décision, sauve du temps qu'elle aurait passé à lire le document complet. L'output est le moyen. Le résultat est le but. Les confondre, c'est comme ça qu'on construit des systèmes techniquement impressionnants qui n'aident personne en pratique.

Cette distinction compte surtout en évaluation. Les équipes qui évaluent la qualité d'un agent par la qualité de l'output — le résumé est-il bien écrit, le code est-il syntaxiquement correct, la réponse est-elle grammaticalement fluide — mesurent la mauvaise chose. Ces propriétés sont corrélées à la qualité mais ne la définissent pas. Un résumé bien écrit du mauvais contenu rate l'utilisateur. Du code syntaxiquement correct qui ne résout pas le vrai problème rate le développeur. Des réponses fluides à la mauvaise question ratent tout le monde.

L'évaluation orientée résultats demande de savoir ce que l'utilisateur essayait réellement d'accomplir et si l'agent l'a aidé à l'accomplir. C'est plus difficile à mesurer que la qualité d'output, ce qui est probablement pourquoi les équipes mesurent la qualité d'output à la place. Mais difficile à mesurer ne veut pas dire optionnel. Tu peux mesurer les résultats via le comportement utilisateur — a-t-il pris l'action que l'information devait permettre ? Via les taux de suivi — est-il revenu avec des questions de clarification qui suggèrent que la première réponse a manqué sa cible ? Via le feedback direct — l'output a-t-il aidé ?

Le focus output distord aussi ce qui se construit. Les équipes qui optimisent la qualité d'output investissent à rendre les outputs plus beaux — prose plus polie, meilleur formatage, couverture plus exhaustive. Les équipes qui optimisent les résultats investissent à comprendre le but réel de l'utilisateur, ce qui veut parfois dire des outputs plus courts, une couverture moins exhaustive et des réponses plus directes qui ne mettent pas en valeur la capacité de l'agent mais qui répondent au besoin.

Il y a une question de design sous-jacente : sais-tu ce que tes utilisateurs essaient d'accomplir ? Pas ce qu'ils demandent — ce qu'ils essaient d'accomplir. Ce sont souvent des choses différentes. L'utilisateur qui demande un résumé d'un document juridique essaie de prendre une décision, pas de collecter de l'information. L'agent qui l'aide à prendre la décision a réussi. L'agent qui résume le document magnifiquement tout en laissant la décision aussi dure qu'avant a produit un bon output et un mauvais résultat.

Mesure ce qui compte. L'output est une preuve. Le résultat est le verdict.
