---
title: "83. Commence ton prompt par le résultat, pas la méthode"
date: 2026-04-09
tags:
  - developer-as-user
description: "« Refactor this function » est une instruction de méthode."
---

« Refactor this function » est une instruction de méthode. « Make this function testable in isolation without changing its public interface » est une instruction de résultat. La différence est significative : l'instruction de méthode délègue toute la décision de design à l'assistant, tandis que l'instruction de résultat spécifie à quoi ressemble le succès et laisse le chemin d'implémentation ouvert.

Les prompts orientés résultat produisent de meilleurs outputs parce qu'ils donnent à l'assistant une cible à optimiser plutôt qu'une procédure à exécuter. Quand tu spécifies le résultat, l'assistant peut évaluer si son approche l'atteint et ajuster. Quand tu spécifies la méthode, l'assistant exécute la méthode qu'elle atteigne ou non ce dont tu avais vraiment besoin.

La discipline du prompting orienté résultat te force aussi à clarifier tes propres objectifs. « Refactor this function » peut signifier plusieurs choses différentes — améliorer la lisibilité, réduire la complexité, améliorer la performance, rendre testable — et tu n'as peut-être pas décidé laquelle tu veux vraiment. Écrire le résultat force la décision. Quoi, spécifiquement, devrait être vrai à propos du code quand tu as fini qui ne l'est pas maintenant ?

Ça ne veut pas dire que tu ne peux jamais spécifier la méthode. Parfois tu sais que la méthode est correcte et tu veux que l'assistant l'implémente. Mais même alors, ajouter le résultat comme vérification — « implement X approach so that Y is achieved » — donne à l'assistant un moyen de signaler quand la méthode ne sert pas le résultat. Ce feedback vaut souvent plus que l'implémentation elle-même.

Sache à quoi ressemble « fait » avant de décrire comment y arriver.
