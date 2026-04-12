---
title: "4. Arrêtez d'anthropomorphiser, commencez à déboguer"
date: 2026-04-09
tags:
  - working-with-agents
description: "Quand un agent fait quelque chose d'inattendu, les développeurs se rabattent sur des explications humaines."
---

Quand un agent fait quelque chose d'inattendu, les développeurs se rabattent sur des explications humaines. « Il s'est embrouillé. » « Il a mal compris l'intention. » « Il était paresseux. » Ces phrases ont l'air descriptives. Elles ne le sont pas — elles sont une façon d'éviter la vraie question, qui est : qu'est-ce qui s'est vraiment passé dans le système ?

Anthropomorphiser est confortable parce que ça mappe le comportement de l'agent sur un domaine qu'on comprend déjà — la cognition humaine. On sait comment gérer un collègue embrouillé. On ne sait pas toujours comment gérer un modèle transformer qui produit des séquences de tokens inattendues. Alors on traduit l'inconnu en connu, et ce faisant, on perd en précision.

Le coût apparaît au debugging. Si l'agent « a mal compris », le fix, c'est d'expliquer plus clairement — réécrire le prompt, ajouter plus de context, être plus explicite. Parfois c'est la bonne chose à faire. Mais parfois l'agent n'a rien mal compris. Il a parfaitement compris et les instructions étaient contradictoires, ou le document récupéré contenait des données périmées, ou le tool a retourné une réponse mal formée que l'agent a gérée gracieusement d'une façon qui a produit le mauvais résultat. Rien de tout ça n'est de l'incompréhension. Ce sont des échecs systèmes avec des causes spécifiques. Et tu ne les trouveras jamais si tu t'es arrêté à « il s'est embrouillé ».

La meilleure habitude, c'est de narrer mécaniquement. Pas « l'agent a mal compris la tâche » mais « l'agent a reçu ces inputs, a produit ce reasoning intermédiaire, a appelé ce tool, a reçu cette réponse, et a généré cet output ». Cette chaîne d'événements est déboguable. Tu peux pointer chaque étape et te demander si elle était correcte. Tu peux la reproduire. Tu peux changer une variable et observer l'effet.

Ça ne veut pas dire traiter les agents comme de simples systèmes déterministes — ils ne le sont pas. Ça veut dire tenir la complexité probabiliste d'une main tout en exigeant de l'autre une précision mécanique dans ton debugging. Le modèle est une boîte noire, mais tout ce qui l'entoure — le context qu'il a reçu, les tools qu'il a appelés, les outputs qu'il a produits — est observable. Débogue ce que tu peux observer.

Le piège de l'anthropomorphisation distord aussi les attentes. Les développeurs qui pensent aux agents comme des collègues embrouillés essaient de les réparer comme ils répareraient un collègue embrouillé — avec une meilleure communication. Les développeurs qui pensent aux agents comme des systèmes probabilistes construisent des harnais d'évaluation, loggent les états intermédiaires, et mesurent les distributions d'output. Le deuxième groupe livre des systèmes plus fiables.

L'agent n'a pas eu une mauvaise journée. Quelque chose dans le système a produit un mauvais output. Trouve-le.
