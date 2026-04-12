---
title: "6. Les agents échouent gracieusement ou pas — il n'y a pas de milieu"
date: 2026-04-09
tags:
  - working-with-agents
description: "La plupart des systèmes échouent sur un spectre."
---

La plupart des systèmes échouent sur un spectre. Un serveur web sous charge commence à laisser tomber des requêtes lentement, te donnant le temps de remarquer et de réagir. Une base de données à court d'espace disque se dégrade gracieusement, t'avertissant avant de s'arrêter. L'échec est visible, incrémental, et récupérable. Tu construis du monitoring pour exactement ce genre de décrépitude.

Les agents échouent différemment. Ils ne se dégradent pas — ils dérivent. Les outputs deviennent subtilement pires avec le temps, de façons difficiles à détecter à moins que tu les cherches spécifiquement. L'agent commence à faire des hypothèses légèrement différentes. Son ton change. Il commence à gérer les cas limites de nouvelles façons. Rien ne casse bruyamment. Le système tourne toujours. Les outputs arrivent toujours. Ils ne sont juste plus justes.

Ça fait de la dégradation gracieuse dans les systèmes agentiques un problème de design que tu dois résoudre exprès, pas une propriété que tu reçois gratuitement. Tu dois décider, à l'avance, à quoi ressemble l'échec et comment tu veux que le système se comporte quand il y arrive. Un agent qui se heurte à un tool failure — est-ce qu'il retente silencieusement, remonte l'erreur à l'utilisateur, ou tente un contournement ? Un agent qui reçoit des informations contradictoires — est-ce qu'il signale la contradiction, choisit la source la plus récente, ou demande des clarifications ? Chacune de ces choses est une décision de design. Laisse-les non spécifiées et l'agent les prendra pour toi, de façon incohérente.

L'échec ingracieux est plus facile à designer, paradoxalement. Si ton agent va échouer salement, qu'il échoue bruyamment. Remonte l'erreur. Arrête le processus. Fais du bruit. Un échec bruyant est déboguable. Une dérive silencieuse qui corrompt tes données ou induit tes utilisateurs en erreur pendant trois semaines avant que quelqu'un remarque — c'est le mode d'échec que tu ne peux vraiment pas te permettre.

La question pratique à te poser pour chaque tool que ton agent utilise, chaque dépendance externe qu'il touche, chaque cas limite dans sa tâche : qu'est-ce que je veux qu'il se passe quand ça tourne mal ? Écris-le. Construis-le. Teste-le. Ne le laisse pas au hasard et ne suppose pas que le modèle le gérera sensément, parce que sensé et cohérent ne sont pas la même chose.

L'échec viendra. La seule variable, c'est si tu l'as designé.
