---
title: "47. La sécurité commence par ce que tu mets dans le context window"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Le context window est la surface la plus sensible d'un système d'agents."
---

Le context window est la surface la plus sensible d'un système d'agents. Tout ce que l'agent sait, tout ce sur quoi il peut agir, tout ce qui façonne son comportement — tout passe par le context. Ça en fait la surface d'attaque principale, le risque principal de fuite de données, et l'endroit principal où les décisions de sécurité sont soit prises correctement, soit reportées jusqu'à ce que quelque chose tourne mal.

Le risque de fuite de données est le plus immédiat. Les développeurs qui construisent des systèmes de récupération tirent des documents dans le context pour donner à l'agent des informations pertinentes. Si ces documents contiennent des données sensibles — informations personnelles, identifiants, données internes — et que la sortie de l'agent expose ces données à des utilisateurs qui ne devraient pas les voir, le système de récupération est devenu un mécanisme d'exposition de données. L'agent ne sait pas ce qui est sensible. Il sait ce qu'on lui a donné et ce qu'on lui a demandé. Si on lui a donné des données sensibles et posé une question dont la réponse implique ces données, il les utilisera.

La solution exige de réfléchir soigneusement à ce qui entre dans la récupération. Pas seulement ce qui est pertinent, mais ce qui est approprié pour que l'agent le voie compte tenu de l'identité et des permissions de l'utilisateur qui fait la requête. Le contrôle d'accès à la couche de récupération — s'assurer que l'agent ne voit que les documents que l'utilisateur est autorisé à voir — n'est pas optionnel dans un système qui gère des données avec des différences de sensibilité significatives entre utilisateurs.

Les identifiants méritent une attention particulière. Les system prompts qui contiennent des clés d'API, des mots de passe de base de données ou des tokens d'authentification sont courants en développement précoce et catastrophiquement faux en production. Le context window est logué. Il passe par des APIs. Il finit à des endroits que tu n'avais pas prévus. Les identifiants appartiennent aux variables d'environnement et aux gestionnaires de secrets, accédés à l'exécution, jamais intégrés dans les prompts.

Il y a un principe plus large ici sur l'exposition minimale. L'agent devrait voir le minimum d'informations nécessaire pour faire son travail. Pas tout ce qui pourrait être utile — le minimum réellement nécessaire. Chaque morceau de context supplémentaire est un morceau d'information supplémentaire qui peut être détourné, divulgué ou manipulé.

Ce que tu mets dans le context, c'est ce que tu confies à l'agent. Choisis soigneusement.
