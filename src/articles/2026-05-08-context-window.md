---
title: "Le contexte est fini. Programmons en conséquence."
date: 2026-05-08
tags:
  - context-engineering
  - agents
  - architecture
description: "L'inventaire des techniques qui peuplent la fenêtre, les phénomènes qui la dégradent, les heuristiques pour la maîtriser. Et au passage, l'anti-pattern le plus coûteux qu'on rencontre dans les agents en production."
---

<p class="deck">
L'inventaire des techniques qui peuplent la fenêtre, les phénomènes qui la dégradent, les heuristiques pour la maîtriser. Et au passage, l'anti-pattern le plus coûteux qu'on rencontre dans les agents en production.
</p>

<div class="notice-prereq">
  <strong>Prérequis.</strong> Cet article suppose que vous savez ce qu'est un <em>token</em>, comment fonctionne grossièrement un transformateur, et pourquoi le modèle reçoit l'historique entier à chaque tour. Si ces notions ne sont pas déjà en place, l'article compagnon <a href="/articles/understanding-llms/">« Que se passe-t-il, vraiment, quand vous parlez à une IA ? »</a> pose le décor en quinze minutes.
</div>

<div class="section-num">§ 01 — Inventaire</div>

## Tout ce qu'on a inventé pour <span class="accent">domestiquer</span> un prédicteur de jetons.

Un transformateur seul ne fait qu'une chose : prédire le prochain jeton à partir de ce qu'il a sous les yeux. Pour le rendre utile en production — qu'il *réponde*, qu'il *se souvienne*, qu'il *agisse*, qu'il *tienne dans le temps* — on a inventé une douzaine de techniques. Chacune répond à un manque précis. Chacune *habite* la fenêtre de contexte d'une manière ou d'une autre. Voici l'inventaire, du point de vue de ce que ça **coûte** et de ce que ça **débloque**.

### Cadrer le comportement · le system prompt

Le texte d'instructions placé en tête de chaque requête. Définit rôle, ton, règles, garde-fous, format de sortie, parfois exemples. C'est ce qui transforme un prédicteur de texte en assistant. **Coût :** permanent et payé à chaque tour. Souvent 5 000 à 25 000 jetons pour un produit grand public, plus pour un agent avec beaucoup d'outils.

### Personnaliser sans dupliquer · les préférences utilisateur

Un petit bloc supplémentaire propre à l'utilisateur, injecté avant la conversation — langue, ton, expertise, projets en cours. **Coût :** faible en jetons mais à haute priorité, ces lignes pèsent lourd dans la prédiction.

### Donner des capacités · les outils et le MCP

Un modèle ne peut ni lire un fichier, ni interroger une base, ni envoyer un courriel — il ne fait que produire du texte. La solution : déclarer des outils qu'il invoque en écrivant un appel structuré (function calling, tool use), que l'application exécute pour lui. Le **Model Context Protocol** (MCP) standardise la déclaration et l'exposition d'outils, permettant de brancher des serveurs tiers (Asana, Gmail, GitLab, bases internes…) sans réécrire le pipeline. **Coût :** chaque outil déclaré occupe la fenêtre — schéma JSON, description, paramètres — *même quand il n'est jamais appelé*. Brancher dix serveurs MCP, c'est dix fois la facture.

### Enseigner des procédures · les skills

Des fichiers `SKILL.md` contenant des recettes procédurales injectées seulement quand un déclencheur correspond. Au lieu de gonfler le system prompt avec toutes les recettes possibles, on les stocke à part et on charge à la demande. **Coût :** nul tant qu'ils ne sont pas activés ; modéré quand ils le sont. Le piège majeur — un skill mal conçu peut faire entrer dans la fenêtre des données qu'il aurait dû traiter à part. C'est l'objet du § 04.

### Garder le fil · l'historique de conversation

Le modèle est sans état. Pour qu'une conversation paraisse continue, l'application reconstitue l'historique entier à chaque tour. **Coût :** linéaire dans le nombre d'échanges. Au tour 40, on repaie 40 fois le même prix.

### Compresser ce qui est ancien · la summarization automatique

Quand on s'approche de la limite, l'application remplace les tours anciens par un résumé condensé produit par le modèle lui-même. **Coût :** la compression est *irréversible* — un détail effacé ne revient pas.

### Persister entre les conversations · la mémoire

Un magasin distinct de l'historique, qui contient des faits durables (préférences, projets, contexte professionnel) ré-injectés en fenêtre quand pertinent. **Coût :** faible en jetons, mais demande une discipline — quoi mémoriser, quoi oublier, quoi proposer.

### Récupérer plutôt que tout charger · le RAG

Un corpus documentaire (centaines de docs, milliers de pages) ne tient pas dans la fenêtre. Le *Retrieval-Augmented Generation* indexe le corpus à part, et au moment d'une requête, ne récupère que les passages pertinents pour injection. L'évolution récente — le RAG *agentique* — laisse l'agent décider *quand* et *quoi* récupérer plutôt que d'imposer une étape pré-LLM figée. **Coût :** infrastructure d'indexation à part, et la qualité de la réponse dépend de la qualité de la récupération.

### Réduire le coût des préfixes stables · le prompt caching

Chaque requête recalcule le system prompt et les définitions d'outils — même quand rien n'a changé. Les fournisseurs mettent désormais en cache le calcul d'attention (*KV cache*) pour les portions stables. Lors des requêtes suivantes, ces jetons coûtent une fraction de leur prix normal et la latence chute. **Coût :** aucun en jetons — c'est une optimisation pure — mais demande de garder le préfixe identique d'une requête à l'autre, à l'octet près.

### Isoler le bruit · les sous-agents

Certaines tâches exigent de lire de gros volumes (web, fichiers, recherches multiples) qui satureraient la fenêtre du parent. Déléguer à un sous-agent qui a sa propre fenêtre, traite le bruit chez lui, et ne renvoie qu'une synthèse compacte. Permet aussi de paralléliser. **Coût :** chaque sous-agent paie son propre system prompt et ses propres outils ; la compression du résumé reste irréversible. Voir § 06.

### Compacter le contexte · l'opération de fond

Au fil d'une longue session agentique — outils appelés, fichiers lus, sous-agents invoqués — la fenêtre se remplit de matériel qui n'est plus pertinent. La **compaction** élague ou résume les portions périphériques pour libérer de l'espace. C'est l'idée plus générale dont la summarization n'est qu'une instance. **Coût :** comme toute compression, on perd quelque chose. Le challenge est de perdre *la bonne chose*.

### L'allocation typique

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 1</span><span>Allocation typique d'un agent en production</span></div>
  <svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Répartition de la fenêtre entre les différents artefacts">
    <rect x="20" y="40" width="560" height="220" fill="none" stroke="#3d3525" stroke-width="1.5"/>
    <text x="20" y="30" class="svg-text svg-text-faint">┌─ FENÊTRE</text>
    <text x="580" y="30" class="svg-text svg-text-faint" text-anchor="end">~200 000 tokens ─┐</text>
    <rect x="20" y="40" width="140" height="220" fill="#c2553a" opacity="0.85"/>
    <text x="90" y="155" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">SYSTEM</text>
    <text x="90" y="170" class="svg-text" text-anchor="middle" fill="#0f0d0a">~15-25k</text>
    <rect x="160" y="40" width="68" height="220" fill="#e8a04b" opacity="0.85"/>
    <text x="194" y="148" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">TOOLS</text>
    <text x="194" y="162" class="svg-text" text-anchor="middle" fill="#0f0d0a">defs</text>
    <rect x="228" y="40" width="46" height="220" fill="#ffc26b" opacity="0.85"/>
    <text x="251" y="148" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700" font-size="9">SKILLS</text>
    <rect x="274" y="40" width="18" height="220" fill="#7a8b5c" opacity="0.85"/>
    <rect x="292" y="40" width="170" height="220" fill="#4d8a8a" opacity="0.85"/>
    <text x="377" y="148" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">RÉSULTATS D'OUTILS</text>
    <text x="377" y="164" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-style="italic">le grand vecteur de saturation</text>
    <rect x="462" y="40" width="80" height="220" fill="#a89c84" opacity="0.85"/>
    <text x="502" y="148" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">HISTOIRE</text>
    <rect x="542" y="40" width="22" height="220" fill="#5a5040" opacity="0.85"/>
    <rect x="564" y="40" width="16" height="220" fill="#f4ecdc" opacity="0.95"/>
    <text x="572" y="280" class="svg-text" text-anchor="middle" font-size="8">↑ user</text>
    <text x="20" y="290" class="svg-text svg-text-dim">└─ chaque solution active = un bloc de jetons à payer</text>
    <text x="300" y="312" class="svg-label-big" text-anchor="middle">les solutions cohabitent dans le même réservoir</text>
  </svg>
  <figcaption class="diagram-caption">Toutes les techniques laissent une empreinte ici. Aucune n'est gratuite.</figcaption>
</figure>

<div class="section-num">§ 02 — Phénomènes</div>

## Six choses qui se passent <span class="accent">dans</span> la fenêtre, et qu'on ne contrôle pas vraiment.

Les solutions précédentes sont des leviers qu'on actionne. Il existe aussi des phénomènes qu'on subit — propriétés du modèle, propriétés de l'attention, propriétés des données — et qu'il faut intégrer comme contraintes. Ces six-là reviennent dans presque tous les agents en production. Avoir un nom pour les désigner est la première étape pour les traiter.

<div class="glossary">

  <div class="gl-item">
    <div class="gl-term">Lost in the middle <span class="alt">— l'oubli au milieu</span></div>
    <div class="gl-def">L'attention du modèle <strong>n'est pas uniforme</strong> sur la fenêtre. Le début et la fin sont privilégiés ; le milieu est sous-exploité. C'est un effet d'architecture documenté empiriquement (papier <em>Lost in the Middle</em>, Liu et al., 2023), atténué sur les modèles récents mais pas disparu.</div>
    <div class="gl-signal">→ SIGNAL · l'agent ignore une instruction que vous savez présente, mais enfouie au milieu d'un long contexte. Remontez-la en début ou en fin.</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Context rot <span class="alt">— la pourriture progressive</span></div>
    <div class="gl-def">Plus la fenêtre se remplit, plus la qualité de raisonnement <strong>tend à baisser</strong>, même bien en-deçà de la limite théorique. Un agent à 150 000 jetons n'est pas équivalent au même agent à 30 000. La compaction n'est donc pas qu'une question d'espace — c'est aussi une question de performance.</div>
    <div class="gl-signal">→ SIGNAL · les premières actions de votre agent sont précises, les dernières dérivent. Compactez à 50-60 % de remplissage, pas à 95 %.</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Attention dilution <span class="alt">— la dilution sous bruit</span></div>
    <div class="gl-def">Cas particulier du <em>context rot</em> : même si le modèle a la capacité théorique de tout regarder, ajouter du contenu non pertinent <strong>réduit l'importance relative</strong> du contenu pertinent. Le bruit ne fait pas que coûter des jetons — il dilue les signaux.</div>
    <div class="gl-signal">→ SIGNAL · ajouter de la documentation « au cas où » dégrade les performances au lieu de les améliorer. Coupez l'inutile, ne le chargez jamais « par précaution ».</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Tool soup <span class="alt">— la soupe d'outils</span></div>
    <div class="gl-def">Au-delà d'une certaine quantité d'outils déclarés (en pratique, autour de quinze à vingt selon les modèles), l'agent <strong>commence à choisir mal</strong> — outils proches confondus, outils manquants ignorés, outils complexes mal paramétrés. Plus c'est gros, plus c'est lent et plus c'est faux.</div>
    <div class="gl-signal">→ SIGNAL · l'agent invoque le mauvais outil, ou en oublie un dont vous savez qu'il était disponible. Activez les outils par phase, pas tous en permanence.</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Runaway agent <span class="alt">— l'agent en fuite</span></div>
    <div class="gl-def">Sans plafond explicite, un agent peut entrer dans une boucle où chaque appel d'outil produit un résultat qui justifie un autre appel. La fenêtre enfle, la qualité chute, et la facture grimpe en silence. Particulièrement fréquent quand l'agent cherche, ne trouve pas, et reformule.</div>
    <div class="gl-signal">→ SIGNAL · une session « simple » consomme dix fois plus de jetons que prévu. Imposez un plafond d'appels d'outils, des points de contrôle, et des seuils de remplissage qui déclenchent une compaction ou un arrêt.</div>
  </div>

  <div class="gl-item">
    <div class="gl-term">Prompt injection <span class="alt">— l'injection d'instructions</span></div>
    <div class="gl-def">Tout contenu externe — page web, e-mail, fichier, résultat d'outil — peut contenir des <strong>instructions cachées</strong> qui détournent l'agent. Le modèle ne distingue pas naturellement <em>données</em> et <em>ordres</em>. Plus l'agent a d'outils puissants, plus le risque est sérieux. Hygiène mentale obligatoire : traiter les contenus tiers comme potentiellement hostiles.</div>
    <div class="gl-signal">→ SIGNAL · l'agent fait quelque chose que vous n'avez pas demandé après avoir lu un contenu externe. Marquez les contenus tiers, restreignez les outils utilisables après lecture, validez humainement les actions irréversibles.</div>
  </div>

</div>

<div class="section-num">§ 03 — Heuristiques</div>

## Onze principes pour <span class="accent">arbitrer</span> les appétits concurrents.

Connaître les solutions et les phénomènes ne suffit pas : il faut savoir les composer. Voici les heuristiques que j'utilise et que je vois utilisées dans les agents en production. Aucune n'est révolutionnaire prise isolément ; leur valeur tient à la *discipline* de les appliquer ensemble. Pour chacune, un signal d'alarme qui déclenche son application, et un cas où elle ne s'applique pas.

<div class="heuristics-list">

  <div class="heuristic">
    <div class="heur-name">Mesurer avant d'optimiser</div>
    <p class="heur-body">Avant de chercher à compresser ou reformuler, savoir <strong>combien chaque artefact pèse réellement</strong>. Toutes les API modernes exposent un compte de jetons par message. Compter d'abord, cibler le plus gros poste, puis seulement optimiser.</p>
    <div class="heur-signal"><strong>Signal</strong> Vous « sentez » que l'agent rame mais vous ne savez pas où. Ouvrez les logs, comptez les jetons par catégorie (system, tools, history, results).</div>
    <div class="heur-counter"><strong>Sauf si</strong> Prototype rapide pour valider une idée. Ne pas optimiser ce qui n'est pas encore stable.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Précision plutôt qu'exhaustivité dans le system prompt</div>
    <p class="heur-body">Le réflexe est de bourrer le system prompt d'exemples « au cas où ». Un system prompt long fatigue le modèle (cf. <em>context rot</em>) et augmente le coût de chaque requête. Mieux vaut un cadre <strong>resserré</strong> et déléguer les détails à des skills chargés à la demande.</p>
    <div class="heur-signal"><strong>Signal</strong> System prompt &gt; 30k jetons, ou contenant des sections jamais déclenchées, ou réécrites tous les sprints.</div>
    <div class="heur-counter"><strong>Sauf si</strong> Le contexte métier est tellement spécialisé qu'aucun skill ne peut le remplacer (réglementation stricte, ton de marque non-négociable).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Ne brancher que les outils nécessaires</div>
    <p class="heur-body">Chaque outil déclaré occupe la fenêtre <em>même quand il n'est jamais utilisé</em>. Brancher dix serveurs MCP « pour le futur », c'est dépenser des milliers de jetons en permanence et nourrir la <em>tool soup</em>. Activer les outils par <strong>profil de tâche ou par phase</strong> produit des agents nettement plus performants.</p>
    <div class="heur-signal"><strong>Signal</strong> Plus de quinze outils déclarés, ou agent qui hésite entre deux outils proches.</div>
    <div class="heur-counter"><strong>Sauf si</strong> Vous mesurez et vous savez qu'aucun outil n'est superflu. Dans ce cas, documentez la raison de chacun.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Ne jamais charger un fichier brut quand on peut le traiter par code</div>
    <p class="heur-body">C'est le principe le plus important — il fait l'objet du § 04. Demander au modèle de « regarder » un CSV de 100 000 lignes ou un PDF de cinquante pages, c'est la cause la plus fréquente de saturation. Donner au modèle le moyen d'<strong>écrire du code qui opère sur les données</strong> et de ne ramener que le résultat est le pivot fondamental.</p>
    <div class="heur-signal"><strong>Signal</strong> Un seul appel d'outil ramène plus de 5 000 jetons en contexte.</div>
    <div class="heur-counter"><strong>Sauf si</strong> Le fichier est petit (&lt; 2k jetons) et le modèle doit en saisir la totalité (relecture nuancée d'un texte court, par exemple).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Placer l'essentiel aux extrémités</div>
    <p class="heur-body">Compte tenu de l'effet <em>lost in the middle</em>, les instructions critiques vont en début ou en fin de la fenêtre. La règle métier qu'on ne veut pas voir ignorée ? En fin de system prompt. La consigne immédiate la plus importante ? Dans le dernier message utilisateur.</p>
    <div class="heur-signal"><strong>Signal</strong> Une instruction documentée n'est pas suivie. Avant d'en déduire que « le modèle est nul », vérifier sa position dans la fenêtre.</div>
    <div class="heur-counter"><strong>Sauf si</strong> Vous avez peu de contenu et tout tient dans un horizon court. La règle n'apparaît qu'au-delà de quelques milliers de jetons.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Stabiliser le préfixe pour activer le KV cache</div>
    <p class="heur-body">Le <em>prompt caching</em> ne fonctionne que si la portion en tête est <strong>identique d'une requête à l'autre, à l'octet près</strong>. Mettre la date du jour ou un identifiant de session au tout début, c'est invalider le cache à chaque tour. Garder le préfixe immuable et placer les éléments variables plus loin est une optimisation gratuite — souvent 80-90 % de réduction sur le coût des prefixes stables, et latence divisée par deux ou trois.</p>
    <div class="heur-signal"><strong>Signal</strong> Vos appels Anthropic / OpenAI ne montrent pas de <em>cache hit</em> alors que le system prompt est « identique ».</div>
    <div class="heur-counter"><strong>Sauf si</strong> Vos requêtes sont rares ou irrégulières — le cache a une durée de vie limitée (5 min chez Anthropic par défaut).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Compacter tôt, pas en panique</div>
    <p class="heur-body">Attendre que la fenêtre soit pleine pour compacter, c'est compacter dans l'urgence — donc mal. Les agents bien construits déclenchent la compaction <strong>par seuil</strong> (60 % de remplissage est un bon point de départ), avec une stratégie réfléchie : quoi résumer, quoi élaguer, quoi garder verbatim.</p>
    <div class="heur-signal"><strong>Signal</strong> La compaction s'enclenche à 95 %, ou pire, n'existe pas et les sessions longues plantent.</div>
    <div class="heur-counter"><strong>Sauf si</strong> Vous êtes dans une session courte par construction (single-turn, ou plafond d'appels imposé).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Déléguer aux sous-agents le travail bruyant</div>
    <p class="heur-body">Toute tâche qui implique de <strong>lire beaucoup pour produire peu</strong> — exploration web, lecture de fichiers volumineux, recherches multi-sources — est candidate naturelle pour un sous-agent. Le parent garde sa fenêtre légère ; le sous-agent absorbe le bruit dans la sienne et ne renvoie qu'une synthèse.</p>
    <div class="heur-signal"><strong>Signal</strong> Le contexte de l'agent principal est rempli à 70 % par des résultats de recherche ou des contenus bruts.</div>
    <div class="heur-counter"><strong>Sauf si</strong> La tâche exige que le parent voie le détail (audit, traçabilité, raisonnement multi-étapes sur des éléments précis).</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Traiter tout contenu externe comme hostile</div>
    <p class="heur-body">Une page web, un courriel, un résultat d'outil sont des données — et peuvent contenir des instructions cachées (cf. <em>prompt injection</em>). Pour les agents avec outils sensibles (envoi d'emails, accès systèmes internes, exécution de code), c'est non-négociable. Marquer les contenus tiers, restreindre les outils utilisables après lecture, valider humainement les actions irréversibles — disciplines, pas options.</p>
    <div class="heur-signal"><strong>Signal</strong> Votre agent a accès à du courriel, à un browser, ou à des données externes ET peut exécuter des actions à effet de bord.</div>
    <div class="heur-counter"><strong>Sauf si</strong> L'agent est purement read-only et n'a aucun outil à effet de bord. Le risque devient théorique.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Mémoriser ce qui dure, pas ce qui passe</div>
    <p class="heur-body">La mémoire persistante est précieuse mais piégeuse. On y met des faits durables (préférences, projets en cours, contexte professionnel), pas des micro-détails d'une conversation. Règle utile : <strong>si l'information n'est pas pertinente dans au moins trois conversations futures, elle n'a rien à faire en mémoire</strong>.</p>
    <div class="heur-signal"><strong>Signal</strong> La mémoire contient « l'utilisateur a dit X mardi » pour des X qui ne reviendront jamais. Ou pire, des contradictions accumulées.</div>
    <div class="heur-counter"><strong>Sauf si</strong> C'est explicitement un agent de prise de notes ou de journal personnel — la rétention granulaire est alors la fonctionnalité.</div>
  </div>

  <div class="heuristic">
    <div class="heur-name">Itérer avec des évals, pas à l'œil</div>
    <p class="heur-body">L'optimisation de contexte ressemble au tuning de performance : on a souvent tort en jugeant à l'intuition. Construire <strong>quelques tests reproductibles</strong> — voici une question, voici la réponse attendue — et mesurer l'impact de chaque changement empêche les régressions silencieuses. Ajouter un outil ou un skill sans mesurer dégrade étonnamment vite.</p>
    <div class="heur-signal"><strong>Signal</strong> Vous ajoutez une fonctionnalité et un autre comportement, sans lien apparent, devient instable.</div>
    <div class="heur-counter"><strong>Sauf si</strong> Vous êtes en exploration pure et la performance n'est pas encore un critère. Une fois en production, plus d'excuses.</div>
  </div>

</div>

<div class="section-num">§ 04 — L'anti-pattern</div>

## Skills qui <span class="accent">lisent</span> vs skills qui <span class="accent">exécutent</span>.

C'est la distinction la plus mal comprise de l'ingénierie d'agent. Un skill, ce n'est pas un endroit où on dépose des données pour que le modèle les contemple : c'est un mode d'emploi pour les *opérer hors contexte*. C'est aussi l'optimisation qui produit les gains les plus spectaculaires — souvent **deux ordres de grandeur sur la consommation de jetons**.

<div class="compare">
  <div class="compare-card bad">
    <span class="compare-tag">↯ Anti-pattern</span>
    <h4>Le skill qui lit</h4>
    <p>Charge le fichier brut dans la fenêtre, demande au modèle de tout regarder puis de tout résumer. Coûteux, lent, fragile, plafonné par la taille du fichier, et soumis au <em>context rot</em>.</p>
  </div>
  <div class="compare-card good">
    <span class="compare-tag">✓ Bon pattern</span>
    <h4>Le skill qui exécute</h4>
    <p>Apprend au modèle à écrire du code qui opère sur les données — analyse, filtre, agrège, valide. Seul le <em>résultat compact</em> revient en contexte. Le code voit les octets, le modèle voit l'agrégat.</p>
  </div>
</div>

### Le coût réel, en chiffres

Cas concret : « Combien de transactions de plus de 1 000 $ y a-t-il dans ce CSV de 100 000 lignes ? » Le fichier fait environ 8 Mo de texte, soit grossièrement **2 millions de jetons**. Comparons les deux trajectoires :

<div class="showcase">
  <div class="lbl">A · Le skill qui lit (anti-pattern)</div>
  <div class="row head"><span class="lhs">poste</span><span class="rhs">jetons</span></div>
  <div class="row bad"><span class="lhs">→ Tentative de chargement intégral</span><span class="rhs">2 000 000</span></div>
  <div class="row bad"><span class="lhs">→ Limite de fenêtre dépassée (200k)</span><span class="rhs">échec</span></div>
  <div class="row bad"><span class="lhs">→ Stratégie de repli : chunking + résumés</span><span class="rhs">~180 000</span></div>
  <div class="row bad"><span class="lhs">→ Résultat : approximation, pas de comptage exact</span><span class="rhs">imprécis</span></div>
  <div class="row total"><span class="lhs">TOTAL · 1 réponse approximative</span><span class="rhs">~180 000 tk</span></div>
</div>

<div class="showcase">
  <div class="lbl">B · Le skill qui exécute (bon pattern)</div>
  <div class="row head"><span class="lhs">poste</span><span class="rhs">jetons</span></div>
  <div class="row"><span class="lhs">→ Skill chargé en contexte</span><span class="rhs">~400</span></div>
  <div class="row"><span class="lhs">→ Modèle écrit un script Python</span><span class="rhs">~200</span></div>
  <div class="row good"><span class="lhs">→ Script lit le CSV hors contexte (pandas)</span><span class="rhs">0</span></div>
  <div class="row good"><span class="lhs">→ Sortie du script en contexte : "47 322"</span><span class="rhs">~5</span></div>
  <div class="row total"><span class="lhs">TOTAL · 1 réponse exacte</span><span class="rhs">~605 tk</span></div>
</div>

Rapport **~300×**. Et au passage : la réponse B est *exacte* alors que la A est nécessairement approximative. Le bon pattern est plus rapide, moins cher, et plus précis. Ce n'est pas un compromis — c'est juste une meilleure architecture.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 2</span><span>Deux trajectoires de données</span></div>
  <svg viewBox="0 0 600 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Comparaison des deux approches : lecture vs exécution">
    <text x="20" y="20" class="svg-text" font-weight="700" fill="#c2553a">A · LECTURE DIRECTE</text>
    <rect x="20" y="34" width="80" height="50" fill="none" stroke="#c2553a" stroke-width="1.5"/>
    <text x="60" y="58" class="svg-text" text-anchor="middle">fichier</text>
    <text x="60" y="72" class="svg-text svg-text-dim" font-size="9">8 Mo</text>
    <line x1="100" y1="59" x2="180" y2="59" stroke="#c2553a" stroke-width="6"/>
    <polygon points="180,55 192,59 180,63" fill="#c2553a"/>
    <text x="140" y="48" class="svg-text" text-anchor="middle" fill="#c2553a" font-size="9">tout passe</text>
    <rect x="195" y="34" width="180" height="50" fill="#c2553a" opacity="0.4" stroke="#c2553a" stroke-width="1.5"/>
    <text x="285" y="58" class="svg-text" text-anchor="middle" fill="#f4ecdc" font-weight="700">FENÊTRE SATURÉE</text>
    <text x="285" y="72" class="svg-text" text-anchor="middle" fill="#f4ecdc" font-size="9">le modèle « regarde » tout</text>
    <line x1="375" y1="59" x2="455" y2="59" stroke="#c2553a" stroke-width="2"/>
    <polygon points="455,55 467,59 455,63" fill="#c2553a"/>
    <rect x="470" y="34" width="80" height="50" fill="none" stroke="#c2553a" stroke-width="1.5"/>
    <text x="510" y="58" class="svg-text" text-anchor="middle">résumé</text>
    <text x="510" y="72" class="svg-text svg-text-dim" font-size="9">approximatif</text>
    <line x1="20" y1="120" x2="580" y2="120" stroke="#3d3525" stroke-width="1" stroke-dasharray="2,4"/>
    <text x="20" y="150" class="svg-text" font-weight="700" fill="#7a8b5c">B · EXÉCUTION DE CODE</text>
    <rect x="20" y="164" width="80" height="50" fill="none" stroke="#7a8b5c" stroke-width="1.5"/>
    <text x="60" y="188" class="svg-text" text-anchor="middle">fichier</text>
    <text x="60" y="202" class="svg-text svg-text-dim" font-size="9">8 Mo</text>
    <line x1="100" y1="189" x2="180" y2="189" stroke="#7a8b5c" stroke-width="2"/>
    <polygon points="180,185 192,189 180,193" fill="#7a8b5c"/>
    <text x="140" y="178" class="svg-text" text-anchor="middle" fill="#7a8b5c" font-size="9">reste sur disque</text>
    <rect x="195" y="164" width="100" height="50" fill="none" stroke="#7a8b5c" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="245" y="184" class="svg-text" text-anchor="middle" fill="#7a8b5c">SKILL.md</text>
    <text x="245" y="200" class="svg-text" text-anchor="middle" font-size="9" fill="#7a8b5c" font-style="italic">→ écrit du code</text>
    <line x1="295" y1="189" x2="345" y2="189" stroke="#7a8b5c" stroke-width="2"/>
    <polygon points="345,185 357,189 345,193" fill="#7a8b5c"/>
    <rect x="360" y="164" width="80" height="50" fill="#7a8b5c" opacity="0.2" stroke="#7a8b5c" stroke-width="1.5"/>
    <text x="400" y="184" class="svg-text" text-anchor="middle" font-weight="700">exec</text>
    <text x="400" y="200" class="svg-text" text-anchor="middle" font-size="9" fill="#7a8b5c">hors fenêtre</text>
    <line x1="440" y1="189" x2="490" y2="189" stroke="#7a8b5c" stroke-width="2"/>
    <polygon points="490,185 502,189 490,193" fill="#7a8b5c"/>
    <rect x="505" y="164" width="60" height="50" fill="#7a8b5c" opacity="0.7"/>
    <text x="535" y="188" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-weight="700">data</text>
    <text x="535" y="202" class="svg-text" text-anchor="middle" fill="#0f0d0a" font-size="9">précise</text>
    <text x="300" y="260" class="svg-label-big" text-anchor="middle">le code voit les octets ; le modèle voit le résultat</text>
    <line x1="60" y1="290" x2="540" y2="290" stroke="#3d3525"/>
    <text x="160" y="312" class="svg-text" text-anchor="middle" fill="#c2553a">A · ~180 000 tk · imprécis · plafonné</text>
    <text x="440" y="312" class="svg-text" text-anchor="middle" fill="#7a8b5c">B · ~600 tk · exact · scalable</text>
  </svg>
  <figcaption class="diagram-caption">Le skill bien conçu garde les données sur disque et ne ramène que le calcul.</figcaption>
</figure>

Cette idée — *code execution as context compression* — est le pattern le plus rentable de l'ingénierie d'agent contemporaine. Quand vous concevez un skill, demandez-vous toujours : **est-ce que le modèle a besoin de voir les données, ou seulement le résultat de leur traitement ?** La réponse est presque toujours « le résultat ».

<div class="section-num">§ 05 — Audit</div>

## Comment <span class="accent">mesurer</span> ce qui se passe vraiment dans votre fenêtre.

Tout le reste de cet article suppose que vous savez ce que votre agent consomme. La plupart des équipes que je rencontre n'en ont qu'une intuition. L'audit n'est pas compliqué ; il demande juste qu'on s'y mette une fois et qu'on instrumente proprement.

### Les quatre métriques de base

Pour chaque appel au modèle, journalisez quatre nombres. **Jetons d'entrée totaux** — la taille complète envoyée au modèle. **Jetons de sortie** — ce que le modèle a généré. **Jetons mis en cache** (cache hit) — ce qui a coûté la fraction. **Jetons facturés au plein tarif** — la différence. Toutes les API sérieuses (Anthropic, OpenAI, Google) exposent ces compteurs dans la réponse ; il faut les capturer et les agréger.

### La répartition par catégorie

Une fois les totaux connus, ventilez l'entrée. Combien pour le **system prompt** ? Combien pour les **définitions d'outils** ? Combien pour l'**historique** ? Combien pour les **résultats d'outils** de la session courante ? Combien pour les **skills** chargés ? La majorité des agents en production découvre à ce stade que *les résultats d'outils dévorent 40-60 % de la fenêtre* et que personne ne le savait. C'est typiquement là qu'il faut tirer.

### Les indicateurs de santé

Trois indicateurs valent la peine d'être suivis dans le temps. Le **taux de cache hit** — sous 70 %, votre préfixe n'est pas stable. Le **remplissage moyen de la fenêtre en fin de session** — au-dessus de 70 %, vous êtes en zone de *context rot*. Le **nombre moyen d'appels d'outils par session** — s'il dérive vers le haut sans gain de qualité, vous avez un *runaway agent* en formation.

### Outils pratiques

Au minimum, un middleware qui capture les compteurs API et les écrit dans une base ou un fichier de logs structuré. Pour aller plus loin : les fournisseurs offrent des dashboards (Anthropic Console, OpenAI Usage), qui donnent une vue globale mais sans la ventilation par catégorie. Pour Claude Code spécifiquement, la commande `/context` affiche en temps réel la répartition de la fenêtre courante — c'est la lecture la plus précieuse à apprendre. On y revient au § 07.

<div class="section-num">§ 06 — Architecture</div>

## Sous-agents : des fenêtres <span class="accent">isolées</span>.

Quand un parent délègue à un sous-agent, il lui ouvre une fenêtre propre. Le sous-agent absorbe le bruit — lecture brute, recherches, exploration — puis ne renvoie qu'une **synthèse compacte**. Le parent reçoit un télégramme, pas un fleuve. C'est le pattern qui permet à un agent d'orchestration de traiter des problèmes qui dépassent largement sa propre fenêtre.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 3</span><span>Délégation parallèle</span></div>
  <svg viewBox="0 0 600 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Architecture parent et sous-agents avec fenêtres isolées">
    <rect x="180" y="20" width="240" height="80" fill="#16130e" stroke="#e8a04b" stroke-width="2"/>
    <text x="300" y="45" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700">PARENT</text>
    <text x="300" y="62" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="9">fenêtre principale</text>
    <text x="300" y="80" class="svg-text" text-anchor="middle" font-size="9" font-style="italic">légère, orchestre</text>
    <line x1="240" y1="100" x2="100" y2="160" stroke="#a89c84" stroke-width="1.5"/>
    <line x1="300" y1="100" x2="300" y2="160" stroke="#a89c84" stroke-width="1.5"/>
    <line x1="360" y1="100" x2="500" y2="160" stroke="#a89c84" stroke-width="1.5"/>
    <text x="170" y="130" class="svg-text svg-text-faint" text-anchor="middle" font-size="9">délègue</text>
    <text x="430" y="130" class="svg-text svg-text-faint" text-anchor="middle" font-size="9">en parallèle</text>
    <rect x="30" y="160" width="140" height="120" fill="#0f0d0a" stroke="#4d8a8a" stroke-width="1.5"/>
    <text x="100" y="180" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-weight="700" font-size="9">SOUS-AGENT 1</text>
    <rect x="40" y="190" width="120" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="40" y="200" width="100" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="40" y="210" width="115" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="40" y="220" width="90" height="6" fill="#4d8a8a" opacity="0.25"/>
    <rect x="40" y="230" width="120" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="40" y="240" width="105" height="6" fill="#4d8a8a" opacity="0.3"/>
    <text x="100" y="262" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-size="9" font-style="italic">absorbe le bruit</text>
    <text x="100" y="274" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="8">fichier · web · recherche</text>
    <rect x="230" y="160" width="140" height="120" fill="#0f0d0a" stroke="#4d8a8a" stroke-width="1.5"/>
    <text x="300" y="180" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-weight="700" font-size="9">SOUS-AGENT 2</text>
    <rect x="240" y="190" width="120" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="240" y="200" width="115" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="240" y="210" width="105" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="240" y="220" width="120" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="240" y="230" width="95" height="6" fill="#4d8a8a" opacity="0.4"/>
    <text x="300" y="262" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-size="9" font-style="italic">fenêtre isolée</text>
    <text x="300" y="274" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="8">contexte propre</text>
    <rect x="430" y="160" width="140" height="120" fill="#0f0d0a" stroke="#4d8a8a" stroke-width="1.5"/>
    <text x="500" y="180" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-weight="700" font-size="9">SOUS-AGENT 3</text>
    <rect x="440" y="190" width="120" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="440" y="200" width="100" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="440" y="210" width="115" height="6" fill="#4d8a8a" opacity="0.4"/>
    <rect x="440" y="220" width="105" height="6" fill="#4d8a8a" opacity="0.3"/>
    <rect x="440" y="230" width="118" height="6" fill="#4d8a8a" opacity="0.4"/>
    <text x="500" y="262" class="svg-text" text-anchor="middle" fill="#4d8a8a" font-size="9" font-style="italic">parallélisable</text>
    <text x="500" y="274" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="8">indépendant</text>
    <line x1="100" y1="280" x2="240" y2="328" stroke="#e8a04b" stroke-width="1"/>
    <polygon points="238,323 246,330 234,332" fill="#e8a04b"/>
    <line x1="300" y1="280" x2="300" y2="328" stroke="#e8a04b" stroke-width="1"/>
    <polygon points="296,322 300,332 304,322" fill="#e8a04b"/>
    <line x1="500" y1="280" x2="360" y2="328" stroke="#e8a04b" stroke-width="1"/>
    <polygon points="362,323 354,330 366,332" fill="#e8a04b"/>
    <rect x="220" y="330" width="160" height="32" fill="#e8a04b" opacity="0.2" stroke="#e8a04b" stroke-width="1.5"/>
    <text x="300" y="350" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700" font-size="10">SYNTHÈSES COMPACTES</text>
    <text x="300" y="376" class="svg-text" text-anchor="middle" fill="#c2553a" font-size="9" font-style="italic">⚠ la compression est irréversible</text>
  </svg>
  <figcaption class="diagram-caption">Chaque sous-agent ouvre sa propre fenêtre, traite le bruit, renvoie un télégramme.</figcaption>
</figure>

### Avantages

**Isolation** : un sous-agent qui sature sa propre fenêtre n'affecte pas le parent. **Parallélisation** : plusieurs sous-agents peuvent travailler simultanément, ce que la fenêtre unique d'un agent monolithique interdit. **Spécialisation** : chaque sous-agent peut avoir son propre system prompt et ses propres outils, finement adaptés à sa tâche.

### Limite

La compression est *irréversible*. Si le sous-agent omet un détail dans son résumé, le parent n'a aucun moyen de le récupérer — sauf à relancer une délégation, ce qui coûte un nouveau cycle complet. C'est pour cette raison que les sous-agents demandent un soin particulier dans la définition de leur *contrat de retour* : que doit-il *impérativement* remonter, même si ça allonge la synthèse ?

<div class="section-num">§ 07 — Focus pratique</div>

## Comment ça se traduit dans <span class="accent">Claude Code</span> et compagnie.

Vous utilisez probablement Claude Code, Cursor, Cline, ou un agent maison basé sur l'API Anthropic ou OpenAI. Voici comment les principes précédents se manifestent dans ces outils — et où regarder pour les diagnostiquer.

### Lire la fenêtre en temps réel

Dans Claude Code, la commande `/context` affiche la répartition exacte de votre fenêtre courante : system prompt, outils MCP, skills chargés, historique, résultats d'outils. C'est la lecture la plus utile à apprendre. Lancez-la régulièrement pendant une session longue ; vous identifierez très vite quel poste dévore l'espace. La majorité du temps, c'est les résultats d'outils — typiquement les `Read` de gros fichiers ou les `Bash` qui ramènent du JSON volumineux.

### La compaction automatique

Claude Code déclenche une compaction automatique quand la fenêtre approche de sa limite. Les anciens tours sont remplacés par un résumé. Vous pouvez aussi la déclencher manuellement avec `/compact`, en ajoutant des instructions sur ce que la compaction doit préserver (« garde la liste des fichiers que j'ai modifiés, les commandes Bash exécutées et leur résultat »). Compacter tôt et avec des instructions explicites donne presque toujours de meilleurs résultats que laisser l'auto-compaction décider seule au bord du gouffre.

### L'arbitrage MCP

Quand vous branchez plusieurs serveurs MCP (GitHub, Linear, base de données, Sentry, etc.), chacun ajoute son lot de définitions d'outils en permanence. Mesurez le coût : `/context` vous le donne. Si vous voyez 20-30k jetons en outils MCP qui ne servent qu'occasionnellement, envisagez d'activer les serveurs *par projet* via la configuration plutôt que globalement. C'est un des leviers les plus rentables sur Claude Code.

### Les skills, en pratique

Les `SKILL.md` ne sont pas chargés par défaut : ils sont décrits dans le system prompt sous forme d'index, et l'agent les ouvre via leur outil `view` quand un déclencheur correspond. Ce design est *l'application directe du § 04* : la procédure n'occupe la fenêtre qu'à la demande, et seulement quand elle sert. Quand vous écrivez vos propres skills, suivez le même principe : instructions courtes, références à du code, jamais de données brutes empaquetées dans le markdown.

### Le sous-agent Task

Claude Code expose un outil `Task` qui lance un sous-agent avec son propre contexte. Excellente application du § 06 : déléguez les recherches multi-fichiers, les explorations de gros répertoires, les audits de code à un sous-agent. Vous récupérerez une synthèse au lieu d'inonder votre contexte principal.

### Cursor, Cline, Copilot, et les autres

Les principes sont les mêmes, l'instrumentation diffère. Cursor expose moins de visibilité sur la composition de la fenêtre ; il faut souvent passer par les logs API. Cline et les agents open-source basés sur le Model Context Protocol exposent généralement plus de détails. Quel que soit l'outil, la question à se poser reste la même : *qu'est-ce qui occupe ma fenêtre, et pourquoi ?*

<div class="section-num">§ 08 — État des lieux</div>

## Où on en est, en <span class="accent">mai 2026</span>.

Le terrain bouge vite. Cette section est datée pour cette raison : ce qui est vrai au moment de la publication ne le sera peut-être plus dans six mois. Quelques tendances notables que vous pouvez intégrer dans votre raisonnement d'ingénieur.

**Les fenêtres standard ont stagné autour de 200k**, mais des offres expérimentales à 1M de jetons existent (Claude Sonnet en bêta, Gemini depuis longtemps). Le coût par jeton en mode « long contexte » reste sensiblement plus élevé, et la dégradation à grande fenêtre y est plus marquée — autrement dit, l'option « 1M » est utile pour les cas singuliers (un gros document à traiter d'un coup) mais reste un mauvais réflexe par défaut.

**Le KV cache est devenu un acquis universel**. Anthropic, OpenAI et Google exposent tous des mécanismes de prompt caching avec des tarifications explicites. Si vous ne les utilisez pas, vous laissez de l'argent sur la table. La discipline du préfixe stable n'est plus une optimisation avancée : c'est l'attendu de base.

**Le MCP est devenu le standard de fait** pour la déclaration d'outils tiers. L'écosystème compte désormais des centaines de serveurs publics, ce qui est à la fois une chance (capacités énormes accessibles rapidement) et un piège (tentation de la *tool soup*). Le défi 2026 est moins de *brancher* que de *choisir judicieusement quoi brancher*.

**Les skills ont quitté la marge**. Anthropic les a popularisés en 2025 avec Claude Code ; le pattern s'est diffusé. Les agents qui n'ont pas de système de skills explicite ont tendance à accumuler tout dans le system prompt — c'est-à-dire à payer en permanence ce qu'ils pourraient charger à la demande.

**Le pattern « code execution as context compression »** — l'idée du § 04 — est devenu un sujet de discussion dans la communauté d'ingénierie d'agent et fait l'objet d'articles techniques d'Anthropic et d'autres. Si vous ne l'avez pas encore appliqué dans votre architecture, c'est probablement la plus haute priorité de votre prochaine itération.

**L'évaluation systématique reste sous-pratiquée**. C'est la mesure que je vois le moins souvent en place dans les équipes qui construisent des agents ; et c'est paradoxalement celle qui permet d'appliquer toutes les autres avec confiance. Cela bouge — des outils comme Promptfoo, Inspect, et les évals d'Anthropic se diffusent — mais l'écart entre les équipes qui évaluent et celles qui n'évaluent pas reste considérable.

<aside class="pull-quote">
  <p>Chaque jeton a un coût, chaque artefact a un mode de défaillance, et l'ingénierie d'agent consiste largement à arbitrer ces appétits concurrents.</p>
</aside>

<div class="further">
  <div class="further-label">★ Pour aller plus loin</div>
  <ul>
    <li>
      <a href="https://www.anthropic.com/news/context-engineering" target="_blank" rel="noopener">Anthropic · Effective context engineering for AI agents</a>
      <span class="desc">L'article fondateur sur la discipline, par l'équipe applied AI d'Anthropic.</span>
    </li>
    <li>
      <a href="https://arxiv.org/abs/2307.03172" target="_blank" rel="noopener">Liu et al. · Lost in the Middle (2023)</a>
      <span class="desc">Le papier qui a documenté empiriquement la non-uniformité de l'attention sur la fenêtre.</span>
    </li>
    <li>
      <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener">Model Context Protocol · spécification</a>
      <span class="desc">Le standard ouvert pour la déclaration et l'exposition d'outils aux agents.</span>
    </li>
    <li>
      <a href="https://docs.claude.com/en/docs/claude-code/overview" target="_blank" rel="noopener">Anthropic · Claude Code documentation</a>
      <span class="desc">La référence pour les commandes <code>/context</code>, <code>/compact</code>, et le système de skills.</span>
    </li>
    <li>
      <a href="https://docs.claude.com/en/docs/build-with-claude/prompt-caching" target="_blank" rel="noopener">Anthropic · Prompt caching</a>
      <span class="desc">Comment activer le KV cache et structurer son préfixe pour en tirer le maximum.</span>
    </li>
    <li>
      <a href="/articles/understanding-llms/">Article compagnon · Que se passe-t-il, vraiment, quand vous parlez à une IA ?</a>
      <span class="desc">Les fondations conceptuelles, pour qui veut partager le sujet à un public moins technique.</span>
    </li>
  </ul>
</div>
