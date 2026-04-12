---
title: "28. Le tool est l'interface"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Quand tu donnes un tool à un agent, tu ne fais pas qu'étendre ses capacités — tu définis la frontière entre ce que l'agent décide et ce que le monde fait."
---

Quand tu donnes un tool à un agent, tu ne fais pas qu'étendre ses capacités — tu définis la frontière entre ce que l'agent décide et ce que le monde fait. Cette frontière est la décision de conception la plus lourde de conséquences dans un système agentique, et la plupart des équipes la prennent sans réaliser qu'elles sont en train de la prendre.

Un tool est une interface au sens plein. Il a un contrat : des inputs qu'il accepte, des outputs qu'il retourne, des erreurs qu'il peut produire. Il a une sémantique : ce que ça veut dire de l'appeler, quel état il change, ce qu'il suppose sur le monde avant l'appel et ce qu'il garantit sur le monde après. Un tool bien conçu rend le travail de l'agent plus clair. Un tool mal conçu introduit une ambiguïté que l'agent résoudra de façon imprévisible.

L'erreur de conception de tool la plus courante, c'est de rendre les tools trop larges. Un tool appelé `execute_action` qui prend une chaîne libre et fait ce qu'il en extrait n'est pas un tool — c'est une délégation de la conception d'interface au modèle. Le modèle l'utilisera de façon incohérente parce qu'il n'y a pas de contrat avec lequel être cohérent. Un tool appelé `send_email` avec des paramètres explicites pour destinataire, sujet et corps est une vraie interface. Le modèle sait quoi fournir et quoi attendre en retour.

Les tools étroits se composent mieux que les larges. Un agent avec dix tools spécifiques — chacun faisant une chose bien — est plus fiable et plus débogable qu'un agent avec deux tools omnibus. Quand quelque chose tourne mal, tu peux demander quel tool a été appelé et avec quels paramètres. L'échec est localisé. Avec des tools larges, l'échec est quelque part à l'intérieur de l'interprétation par le tool d'un input en forme libre, ce qui est beaucoup plus dur à trouver.

La conception du tool détermine aussi le rayon d'impact. Un tool en lecture seule qui récupère des données peut être appelé librement — s'il échoue ou retourne des données fausses, les dégâts sont limités à la tâche courante. Un tool qui modifie l'état — écrit dans une base de données, envoie un message, exécute un paiement — porte des conséquences sur le monde réel qui ne peuvent pas être annulées. Ces tools méritent un soin supplémentaire dans leur conception : des paramètres de confirmation explicites, des garanties d'idempotency, des états d'erreur clairs sur lesquels l'agent peut raisonner.

L'agent est aussi bon que les tools que tu lui as donnés. Conçois-les comme les interfaces qu'ils sont.
