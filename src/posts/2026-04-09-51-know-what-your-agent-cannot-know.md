---
title: "51. Sachez ce que votre agent ne peut pas savoir"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Chaque agent a une frontière épistémique — une ligne entre ce qu'il peut savoir et ce qu'il ne peut pas."
---

Chaque agent a une frontière épistémique — une ligne entre ce qu'il peut savoir et ce qu'il ne peut pas. D'un côté : tout ce qui se trouve dans ses données d'entraînement, tout ce qui se trouve dans sa context window, tout ce que ses tools retournent. De l'autre : tout le reste. Les développeurs qui travaillent le plus fiablement avec les agents ont cartographié cette frontière avec soin. Ceux qui se brûlent ont supposé qu'elle était plus loin qu'elle ne l'est.

Le training cutoff est la limitation la plus discutée et la moins subtile. L'agent ne connaît pas les événements postérieurs à la fin de ses données d'entraînement. C'est bien compris, fréquemment oublié en pratique, et facile à vérifier — demande à l'agent quelque chose de récent et regarde ce qu'il dit. Les trous épistémiques les plus dangereux sont ceux qui ne sont pas évidents.

L'agent ne connaît pas ton organisation. Il ne connaît pas ton codebase, tes clients, tes processus internes, tes décisions historiques et pourquoi elles ont été prises. Il peut raisonner sur ces choses si tu les mets dans le contexte, mais il n'y a aucun accès autrement. Les équipes qui travaillent avec un agent depuis assez longtemps l'oublient parfois — l'agent a été utile si longtemps qu'il commence à avoir l'air de connaître l'entreprise. Il ne la connaît pas. Il connaît ce qui était dans la context window des sessions auxquelles il a participé, ce qui est une tranche petite et curatée de la connaissance institutionnelle.

L'agent ne sait pas ce qu'il ne sait pas. C'est le trou le plus important opérationnellement. Un expert humain qui arrive au bord de ses connaissances sait généralement qu'il est au bord — il y a un sentiment d'incertitude qui déclenche la prudence. Les agents n'ont pas ça. Ils génèrent la réponse la plus probable étant donné leurs inputs, et si leurs inputs ne contiennent pas l'information nécessaire pour répondre correctement, ils génèrent à la place la réponse la plus probable qui sonne plausible. L'output a le même air, que l'agent connaisse la réponse ou qu'il la confabule.

Concevoir en tenant compte des limites épistémiques, c'est intégrer de la vérification pour les affirmations qui comptent, restreindre le scope de l'agent aux domaines où ses connaissances sont fiables, et être explicite avec les utilisateurs sur ce que l'agent peut et ne peut pas être digne de confiance pour savoir.

L'agent ne sait pas ce qu'il ne sait pas. Tu dois le savoir pour vous deux.
