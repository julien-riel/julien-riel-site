---
title: "31. Les petits agents battent les gros agents"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "L'instinct quand on construit des systèmes agentic, c'est de rendre l'agent capable de tout."
---

L'instinct quand on construit des systèmes agentic, c'est de rendre l'agent capable de tout. Un seul agent, un seul prompt, tous les tools, toutes les tâches. Ça semble efficace. C'est en fait un piège.

Les gros agents sont difficiles à raisonner. Quand un seul agent est responsable de comprendre l'intention de l'utilisateur, de récupérer l'information pertinente, d'appeler des APIs externes, de formater la sortie et de gérer les erreurs, tu as créé un système où n'importe quelle défaillance peut être causée par n'importe quoi. Le debug devient de l'archéologie. Tu fouilles dans les logs en essayant de comprendre quelle partie du raisonnement de l'agent a déraillé, et souvent tu ne peux pas le dire, parce que la défaillance est quelque part au milieu d'une longue chaîne de décisions que l'agent a prises sans s'expliquer.

Les petits agents ont un job plus étroit. Un classificateur qui détermine le type de tâche. Un retriever qui tire le contexte pertinent. Un générateur qui rédige la sortie. Un validateur qui la vérifie. Chacun fait une chose et est testable isolément. Quand quelque chose casse, tu sais où regarder. Quand tu veux améliorer la performance, tu sais quoi changer sans te soucier de casser autre chose.

Ça reflète tout ce qu'on sait déjà sur le design logiciel. Les petites fonctions ciblées sont plus faciles à tester que les grosses fonctions tentaculaires. Le même principe s'applique ici — l'unité de composition dans un système agentic est l'agent, et les petites unités se composent mieux que les grosses.

L'objection pratique, c'est la latence : plusieurs agents en séquence veut dire plusieurs appels au modèle, et les appels au modèle sont lents. C'est réel. Mais c'est souvent surpondéré. Une pipeline de trois petits agents qui produit de manière fiable une sortie correcte vaut généralement mieux qu'un gros agent qui est rapide mais qui se trompe quinze pour cent du temps et qui est opaque quand il échoue. La fiabilité se compose d'une façon que la latence ne fait pas.

Il y a aussi un argument context window en faveur des petits agents. Un agent focalisé a besoin d'un contexte focalisé — une tranche d'information plus petite et plus précise. Un gros agent accumule du contexte à travers plusieurs sous-tâches, brûle la window et commence à perdre des informations importantes du début de la conversation. Les petits agents repartent proprement entre les tâches.

Commence avec le plus petit agent qui pourrait fonctionner. Rends-le plus gros seulement quand les coutures commencent à se voir.
