---
title: "24. L'agent qui a l'air confiant n'est pas nécessairement correct"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "Les LLM sont fluides par défaut."
---

Les LLM sont fluides par défaut. Ils produisent du texte qui se lit comme assuré, cohérent et autoritaire, peu importe si le contenu sous-jacent est exact. Ce n'est pas un bug que les développeurs ont oublié de corriger — c'est une conséquence de la façon dont ces modèles sont entraînés. La fluidité et la justesse sont des propriétés différentes, et le processus d'entraînement optimise fortement la première.

Le problème, c'est que les humains lisent la confiance comme un signal de fiabilité. On a évolué pour faire ça — dans la plupart des communications humaines, quelqu'un qui parle avec conviction a généralement vérifié ses faits, ou du moins croit l'avoir fait. Cette heuristique casse violemment avec les LLM, qui produisent de la prose confiante sur des choses qu'ils n'ont aucune base fiable pour affirmer.

L'effet pratique, c'est que les outputs d'un agent exigent un scepticisme proportionnel aux enjeux, pas proportionnel à la façon dont l'output se lit. Un agent qui résume un document avec une autorité tranquille pourrait avoir manqué une nuance clé. Un agent qui fournit une procédure technique étape par étape pourrait avoir fabriqué une étape qui sonne plausible. Le texte ne te donne aucun signal fiable sur lequel des deux est en train de se produire.

La calibration, c'est la compétence que tu développes ici — la capacité d'évaluer la probabilité qu'un output d'agent soit correct étant donné le type de tâche, la qualité du context, et ta connaissance des endroits où ce modèle a tendance à échouer. Dans les domaines que tu connais bien, la calibration vient naturellement : tu peux repérer la mauvaise réponse parce que tu sais à quoi ressemble la bonne. Dans les domaines où tu t'appuies sur l'agent précisément parce que tu ne connais pas le domaine — ce qui est courant, et légitime — la calibration exige une vérification externe. Vérifie les affirmations. Suis les citations. Teste le code.

Certaines techniques de prompting peuvent réduire la confiance injustifiée — demander à l'agent d'exprimer son incertitude explicitement, lui demander d'identifier les parties de sa réponse dont il est le moins sûr, lui demander de distinguer entre ce qu'il sait et ce qu'il infère. Ça aide. Ça ne règle pas le problème.

Lis l'output de l'agent comme tu lirais le premier brouillon d'un stagiaire brillant : avec de l'appréciation pour l'effort et un jugement indépendant sur le contenu.
