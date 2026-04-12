---
title: "9. Le déterminisme est un choix que vous devez faire délibérément"
date: 2026-04-09
tags:
  - working-with-agents
description: "Par défaut, les LLM sont non déterministes."
---

Par défaut, les LLM sont non déterministes. Lance le même prompt deux fois et tu obtiendras des outputs similaires, pas identiques. Pour certaines tâches, c'est correct — même souhaitable. Pour d'autres, c'est un bug caché qui attend de faire surface en production.

Le problème, ce n'est pas le non-déterminisme lui-même. Le problème, c'est le non-déterminisme que tu n'as pas choisi. Quand tu construis un système sans réfléchir à savoir s'il doit être déterministe, tu obtiens un système dont tu ne peux pas pleinement raisonner le comportement, que tu ne peux pas pleinement tester, et que tu ne peux pas pleinement expliquer à tes utilisateurs quand ils demandent pourquoi ils ont eu un résultat différent aujourd'hui que hier.

La plupart des APIs exposent un paramètre de temperature pour exactement cette raison. Temperature zéro — ou proche — fait que le modèle choisit le token le plus probable à chaque étape, ce qui produit des outputs quasi déterministes pour la plupart des inputs. Des températures plus élevées introduisent plus d'aléatoire, ce qui produit des outputs plus variés. C'est une molette que tu peux tourner. La tourner intentionnellement fait partie de l'architecture ; la laisser à la valeur par défaut est une décision par omission.

Les cas où le déterminisme compte le plus sont ceux où l'output de ton système alimente quelque chose d'autre. Si l'output de l'agent est parsé par du code en aval, la variabilité de format casse le parser. Si l'agent prend une décision qui est loggée et auditée, tu dois pouvoir la reproduire. Si l'output de l'agent est montré à un utilisateur et qu'il revient le lendemain en s'attendant à de la cohérence, le non-déterminisme est un problème d'UX.

Les cas où le non-déterminisme est un atout sont les tâches créatives, le brainstorming, et toute situation où tu veux de la variété à travers plusieurs runs. Générer cinq titres alternatifs bénéficie de la variabilité. Extraire une adresse structurée d'un formulaire soumis, non.

C'est une décision qui mérite d'être prise explicitement, par tâche, par système. Pas une seule fois au niveau supérieur — différents agents dans le même pipeline peuvent avoir des exigences de déterminisme différentes. Le classificateur qui route les tâches veut probablement une temperature proche de zéro. L'agent qui rédige des réponses veut peut-être un peu plus de marge.

Sache ce dont tu as besoin. Règle-le exprès. Le défaut n'est pas une décision de design — c'est une décision reportée.
