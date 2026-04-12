---
title: "64. L'expertise compte encore — elle se manifeste juste différemment maintenant"
date: 2026-04-09
tags:
  - mindset
description: "Il existe une version du futur agentic où l'expertise est dévaluée — où l'écart entre l'expert et le novice se referme parce que les deux peuvent prompt un agent pour faire le travail."
---

Il existe une version du futur agentic où l'expertise est dévaluée — où l'écart entre l'expert et le novice se referme parce que les deux peuvent prompt un agent pour faire le travail. Cette version est fausse, mais elle est fausse d'une façon qui demande une explication, parce que l'évidence de surface est réelle. Un développeur avec deux ans d'expérience en utilisant des agents peut produire des résultats qui auraient exigé dix ans d'expérience sans eux. L'écart se referme pour l'exécution. Il ne se referme pas pour le jugement.

L'expertise dans un contexte agentic se manifeste dans la qualité de la spécification, la précision de la révision et l'exactitude du diagnostic d'échec. L'expert sait à quoi ressemble un bon résultat assez bien pour reconnaître quand l'agent a produit quelque chose qui a l'air bien mais qui ne l'est pas. Le novice, n'ayant pas ce point de référence, accepte la sortie plausible. L'agent les amplifie tous les deux, ce qui veut dire qu'il amplifie la différence entre eux — l'expert obtient plus de levier de son expertise, le novice devient plus confiant dans ses erreurs.

Ça se joue concrètement dans la revue de code. Un développeur senior qui révise du code généré par un agent apporte la même connaissance qu'il apporterait à la revue de code écrit par un humain : les patrons architecturaux qui causent des problèmes à l'échelle, les cas limites que l'implémentation évidente manque, les caractéristiques de performance qui ne comptent que sous charge. L'agent peut écrire le code. Il ne peut pas le réviser avec la connaissance qui vient d'avoir vu ce patron échouer en production trois fois.

L'expertise de domaine détermine aussi la qualité du contexte que l'agent reçoit. Un expert sait quels détails comptent et lesquels ne comptent pas — quelles contraintes sont essentielles à spécifier et lesquelles l'agent peut être digne de confiance pour les gérer raisonnablement. Il écrit des prompts qui sont précis là où la précision compte et flexibles là où la flexibilité est appropriée. Un novice soit sur-spécifie — enfouissant les contraintes importantes dans le bruit — soit sous-spécifie — laissant des trous que l'agent comble avec des hypothèses raisonnables mais fausses.

Les experts qui s'épanouissent sont ceux qui redirigent leur expertise vers les choses que les agents ne peuvent pas faire : jugement, évaluation, spécification, diagnostic. Les experts qui souffrent sont ceux qui concurrencent les agents sur l'exécution, essayant de rester pertinents en faisant le travail plus vite plutôt qu'en réfléchissant mieux.

L'expertise n'est pas devenue moins précieuse. Son expression a changé.
