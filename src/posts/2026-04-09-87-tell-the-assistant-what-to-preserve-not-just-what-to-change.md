---
title: "87. Dis à l'assistant ce qu'il faut préserver, pas juste ce qu'il faut changer"
date: 2026-04-09
tags:
  - developer-as-user
description: "Chaque prompt demande implicitement à l'assistant d'optimiser pour l'objectif que tu as énoncé."
---

Chaque prompt demande implicitement à l'assistant d'optimiser pour l'objectif que tu as énoncé. Si tu lui demandes d'améliorer la performance, il améliorera la performance — et il pourrait changer la signature de la fonction, retirer une étape de validation, ou restructurer la gestion d'erreurs pour le faire, parce qu'aucune de ces choses n'a été mentionnée comme contrainte. L'assistant ne sait pas ce qui t'importe au-delà de ce que tu as dit. Il remplit le reste avec un jugement raisonnable qui peut ne pas correspondre au tien.

La moitié manquante de la plupart des prompts est la contrainte de préservation : ce qui doit rester pareil. L'interface publique. Les tests existants. Le contrat de gestion d'erreurs. Le comportement pour les cas limites qui sont déjà gérés correctement. Ce sont les parties porteuses du code existant qu'une nouvelle optimisation pourrait casser par inadvertance. Les énoncer explicitement fait que l'assistant les traite comme des points fixes plutôt que comme des variables.

C'est particulièrement important pour les tâches de refactoring, où tout l'intérêt est de changer l'implémentation tout en préservant le comportement. « Refactor this function to reduce cyclomatic complexity » sans spécifier que tous les tests existants doivent continuer à passer est une invitation ouverte à changer ce que la fonction fait. L'assistant pourrait produire quelque chose de plus simple et de faux.

La discipline est de penser à ce que tu n'essaies pas de changer avant de décrire ce que tu essaies de changer. Fais une liste, même mentale : l'interface est fixe, la couverture de tests ne doit pas régresser, le comportement de logging doit rester pareil. Puis inclus ces contraintes dans le prompt. L'output sera meilleur et la review sera plus rapide, parce que tu sauras exactement quoi vérifier.

Énonce ce qui ne peut pas bouger. L'assistant travaillera autour.
