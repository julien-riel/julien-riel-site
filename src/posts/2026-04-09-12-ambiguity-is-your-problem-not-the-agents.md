---
title: "12. L'ambiguïté est votre problème, pas celui de l'agent"
date: 2026-04-09
tags:
  - working-with-agents
description: "Quand un agent produit un output qui n'est pas ce que tu voulais, la tentation est de dire que le prompt était ambigu."
---

Quand un agent produit un output qui n'est pas ce que tu voulais, la tentation est de dire que le prompt était ambigu. C'est généralement vrai. C'est aussi une manière de détourner la responsabilité. L'ambiguïté était là avant que l'agent la voie. Tu l'as mise là, ou tu n'as pas réussi à l'enlever. L'agent n'a pas créé le problème — il l'a juste rendu visible.

L'ambiguïté dans les instructions est normale. Le langage naturel est imprécis par conception ; il repose sur un contexte partagé, le bon sens et la réparation conversationnelle pour combler les trous. Quand tu parles à un collègue, il peut demander ce que tu voulais dire. Il peut inférer à partir de ton ton. Il peut puiser dans des semaines d'historique de projet partagé pour interpréter une demande sous-spécifiée. Les agents n'ont rien de tout ça à moins que tu le fournisses explicitement. Ce qui te semble clair — parce que tu combles tous les trous avec tes propres connaissances — est réellement ambigu pour le modèle, qui n'a que le context window.

La discipline, c'est de lire tes prompts comme si tu ne savais rien de plus que ce qui est écrit. Pas en tant qu'auteur qui sait ce qu'il a voulu dire, mais en tant que lecteur qui rencontre le texte à froid. Mieux encore : donne le prompt à un collègue et demande-lui ce qu'il pense que ça demande. S'il hésite, ou donne une réponse différente de celle que tu attendais, tu as trouvé ton ambiguïté avant l'agent.

Il y a un type spécifique d'ambiguïté qui coûte particulièrement cher : les contraintes contradictoires. « Sois concis mais exhaustif. » « Sois direct mais diplomate. » « Résume pour un public général mais préserve la précision technique. » Chacune de ces paires contient une vraie tension, et l'agent va la résoudre d'une manière ou d'une autre — mais pas nécessairement comme tu le voudrais. Quand tu as des contraintes contradictoires, priorise-les explicitement. Dis à l'agent laquelle gagne quand les deux ne peuvent pas être satisfaites.

Lever l'ambiguïté est plus difficile qu'il n'y paraît parce que ça exige que tu saches ce que tu veux vraiment — précisément, au niveau de détail sur lequel l'agent doit agir. C'est souvent là qu'est le vrai travail. Des instructions vagues sont fréquemment le signe d'une pensée vague.

Clarifie ta pensée d'abord. Le prompt n'est que la transcription.
