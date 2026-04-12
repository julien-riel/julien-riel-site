---
title: "18. L'espace négatif compte — dites à votre agent ce qu'il ne doit pas faire"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "La plupart des prompts décrivent ce que l'agent doit faire."
---

La plupart des prompts décrivent ce que l'agent doit faire. Peu décrivent ce qu'il ne doit pas faire. Cette asymétrie est là où vit un nombre surprenant d'échecs en production.

La raison est simple : un LLM comble les trous avec de la probabilité. Quand tu ne spécifies pas un comportement, le modèle se replie sur la réponse la plus probable étant donné son entraînement. D'habitude, c'est correct. Parfois, c'est exactement ce que tu ne voulais pas — l'agent qui ajoute des précautions non sollicitées à chaque réponse, celui qui reformate l'output d'une façon qui casse le parsing en aval, celui qui s'excuse longuement avant d'annoncer de mauvaises nouvelles alors que tu voulais qu'il annonce juste la nouvelle. Aucun de ces comportements n'est déraisonnable dans l'abstrait. Ils sont juste mauvais pour ton système, et tu ne l'as jamais dit à l'agent.

Les contraintes négatives sont plus difficiles à écrire que les positives parce qu'elles t'obligent à anticiper les modes d'échec avant qu'ils n'arrivent. Tu dois demander : qu'est-ce qu'un agent raisonnable ferait ici que je ne voudrais pas ? Cette question est inconfortable parce qu'elle te force à imaginer le système qui part en vrille, ce qui semble pessimiste quand tu es dans la phase optimiste de construire quelque chose de nouveau. Fais-le quand même.

Certaines contraintes négatives sont assez universelles pour appartenir à chaque system prompt. Ne fabrique pas de citations. Ne suppose pas d'information qui n'a pas été fournie. Ne continue pas au-delà du périmètre de la tâche. D'autres sont spécifiques à ton cas d'usage et tes utilisateurs. Un agent de service client ne devrait probablement pas spéculer sur les produits des concurrents. Un agent de code review ne devrait probablement pas réécrire du code qu'on ne lui a pas demandé de réécrire. Un agent de résumé ne devrait probablement pas éditorialiser.

La discipline d'écrire des contraintes négatives force une clarté utile sur ce à quoi sert réellement l'agent. Quand tu t'assois pour énumérer ce que l'agent ne devrait pas faire, tu découvres souvent que tu n'avais pas pleinement articulé ce qu'il devrait faire non plus. L'espace négatif éclaire le positif.

Il y a un équilibre. Un prompt qui est principalement des interdictions est fragile et déroutant — l'agent dépense son budget cognitif à naviguer des restrictions plutôt qu'à faire le travail. Les contraintes négatives devraient être ciblées : les comportements spécifiques qui seraient plausibles sans elles et problématiques s'ils survenaient.

Définis la forme en décrivant les bords. Le milieu se gère tout seul.
