---
title: "29. L'idempotency compte plus dans les systèmes agentiques que partout ailleurs"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "L'idempotency — la propriété selon laquelle appeler quelque chose plusieurs fois produit le même résultat que l'appeler une seule fois — est une bonne pratique dans tout système distribué."
---

L'idempotency — la propriété selon laquelle appeler quelque chose plusieurs fois produit le même résultat que l'appeler une seule fois — est une bonne pratique dans tout système distribué. Dans les systèmes agentiques, c'est proche d'une exigence. Les agents réessaient. Ils bouclent. Ils perdent le fil de ce qu'ils ont déjà fait. Sans opérations idempotentes, ces comportements se transforment en actions dupliquées, en états incohérents, et en échecs très difficiles à démêler.

La raison pour laquelle les agents créent plus de pression sur l'idempotency que le logiciel traditionnel, c'est que leur flux de contrôle est probabiliste. Un programme conventionnel réessaie une opération échouée parce qu'une boucle de retry explicite le lui a dit. Un agent réessaie parce qu'il a généré une séquence de tokens qui incluait de tenter l'opération à nouveau — peut-être parce qu'il n'a pas enregistré la première tentative, peut-être parce que la première tentative a retourné un résultat ambigu, peut-être parce que quelque chose dans le context de conversation a fait croire que l'action n'avait pas encore été prise. Tu ne peux souvent pas prédire quand un retry va arriver ni pourquoi.

La conséquence pratique, c'est que tout tool que ton agent peut appeler et qui a des effets de bord devrait être conçu pour être appelé plusieurs fois en toute sécurité. Créer un enregistrement : vérifier d'abord s'il existe, ou utiliser une clé d'idempotency fournie par le client. Envoyer une notification : suivre ce qui a été envoyé et dédupliquer. Facturer un paiement : exiger une clé d'idempotency qui empêche les doubles facturations. Ce ne sont pas des patterns d'ingénierie exotiques — c'est la pratique standard dans les systèmes distribués. La raison de les appliquer plus agressivement dans les contextes agentiques, c'est que le comportement de retry est moins prévisible et moins contrôlable que dans les systèmes que tu as écrits toi-même.

Le mode d'échec est mémorable quand il survient. Un utilisateur reçoit le même courriel cinq fois. Un enregistrement de base de données est créé en double. Une transaction financière est traitée deux fois. Ces échecs sont embarrassants au mieux et coûteux au pire, et ils surviennent souvent en production bien après que ta suite de tests t'a donné un faux sentiment de sécurité, parce que les conditions qui déclenchent des retries inattendus sont difficiles à reproduire en test.

Conçois chaque tool qui change l'état comme s'il allait être appelé deux fois. Parce qu'éventuellement, il le sera.
