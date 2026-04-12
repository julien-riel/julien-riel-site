---
title: "92. Utilisez le Markdown, pas la prose, pour les spécifications"
date: 2026-04-09
tags:
  - developer-as-user
description: "Une spécification écrite en prose continue est difficile à référencer, difficile à mettre à jour et difficile à fournir comme contexte."
---

Une spécification écrite en prose continue est difficile à référencer, difficile à mettre à jour et difficile à fournir comme contexte. Une spécification écrite en markdown structuré — avec des en-têtes, des listes, des exemples de code, et des sections explicites pour les exigences, les contraintes et les cas limites — est facile à naviguer, facile à maintenir, et facile à injecter dans un contexte de prompt.

La structure fait un travail que la prose ne peut pas faire. Une liste de critères d'acceptation est non ambiguë d'une manière qu'un paragraphe décrivant la feature ne l'est pas. Un exemple de code montrant l'interface attendue est plus clair que trois phrases qui l'expliquent. Une section explicite intitulée « Out of Scope » empêche l'assistant d'ajouter serviablement des features que tu n'as pas demandées. Le format impose une discipline de spécificité que la prose tend à saper.

Les spécifications en markdown se composent aussi bien. Tu peux inclure la section pertinente de la spec dans un prompt plutôt que le document entier. Tu peux mettre à jour une seule section sans réécrire toute la spec. Tu peux créer des liens entre sections quand une contrainte dépend d'une autre. Ces propriétés comptent davantage à mesure que le projet grandit et que la spec devient un document vivant plutôt qu'un artefact ponctuel.

Il y a un template qui vaut la peine d'être développé pour ton propre usage: une structure standard pour les specs de features qui marche bien pour toi et pour l'assistant. Quelque chose comme: Énoncé du problème, Solution proposée, Critères d'acceptation, Cas limites, Hors-scope, Questions ouvertes. Les sections spécifiques comptent moins que l'habitude de les utiliser de manière cohérente — la cohérence signifie que tu sais toujours où chercher la contrainte dont tu as besoin.

Le format de la spec détermine à quel point tu peux l'utiliser. Structure-la pour le travail que tu vas faire avec.
