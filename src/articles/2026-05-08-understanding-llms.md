---
title: "Que se passe-t-il, vraiment, quand vous parlez à une IA ?"
date: 2026-05-08
tags:
  - fondamentaux
  - vulgarisation
description: "Tokens, transformateurs, fenêtre de contexte, system prompt, outils : les fondations conceptuelles pour comprendre vraiment comment fonctionne ChatGPT, Claude ou Gemini. Sans équations."
---

<p class="deck">
Trois idées suffisent pour comprendre — et démystifier — ChatGPT, Claude, Gemini et tous leurs cousins : le jeton, le transformateur, et la fenêtre de contexte. Aucune formule, juste des analogies qui tiennent.
</p>

<div class="section-num">§ 01 — Le matériau de base</div>

## L'IA ne lit pas de mots. Elle lit des <span class="accent">jetons</span>.

Première surprise : quand vous écrivez « Bonjour, comment vas-tu ? » à une intelligence artificielle, elle ne voit ni votre phrase, ni vos mots, ni même vos lettres. Elle voit une suite de **jetons** — un mot anglais qu'on garde en français, *tokens* — produits par un découpage automatique de votre texte.

Un jeton, ce n'est pas un mot complet ni une lettre seule : c'est un fragment, quelque part entre les deux. En français, un jeton vaut typiquement **3 à 4 caractères**, soit environ trois quarts de mot. Le mot `fenêtre` peut tenir en un seul jeton parce qu'il est fréquent. `fenêtres` peut se découper en deux (`fenêtre` + `s`). Un nom propre rare ou un mot technique peut éclater en quatre ou cinq morceaux.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 1</span><span>Une phrase, vue par l'IA</span></div>
  <svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Décomposition d'une phrase en jetons">
    <text x="20" y="30" class="svg-text svg-text-faint">┌─ CE QUE VOUS ÉCRIVEZ</text>
    <text x="20" y="58" class="svg-label-big" font-size="18">"La fenêtre de contexte est finie."</text>
    <line x1="300" y1="72" x2="300" y2="98" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="296,92 300,102 304,92" fill="#a89c84"/>
    <text x="312" y="89" class="svg-text svg-text-dim" font-size="9" font-style="italic">découpage automatique</text>
    <text x="20" y="120" class="svg-text svg-text-faint">└─ CE QUE L'IA VOIT</text>
    <g font-family="JetBrains Mono, monospace" font-size="11">
      <rect x="20" y="130" width="38" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="39" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">La</text>
      <rect x="62" y="130" width="58" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="91" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">fenêtre</text>
      <rect x="124" y="130" width="34" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="141" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">de</text>
      <rect x="162" y="130" width="62" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="193" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">contexte</text>
      <rect x="228" y="130" width="36" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="246" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">est</text>
      <rect x="268" y="130" width="38" height="28" fill="#e8a04b" opacity="0.85"/>
      <text x="287" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">fin</text>
      <rect x="310" y="130" width="22" height="28" fill="#ffc26b" opacity="0.85"/>
      <text x="321" y="148" text-anchor="middle" fill="#0f0d0a" font-weight="700">ie</text>
      <rect x="336" y="130" width="20" height="28" fill="#5a5040" opacity="0.85"/>
      <text x="346" y="148" text-anchor="middle" fill="#f4ecdc" font-weight="700">.</text>
    </g>
    <g font-family="JetBrains Mono, monospace" font-size="8" fill="#5a5040">
      <text x="39" y="174" text-anchor="middle">1</text>
      <text x="91" y="174" text-anchor="middle">2</text>
      <text x="141" y="174" text-anchor="middle">3</text>
      <text x="193" y="174" text-anchor="middle">4</text>
      <text x="246" y="174" text-anchor="middle">5</text>
      <text x="287" y="174" text-anchor="middle">6</text>
      <text x="321" y="174" text-anchor="middle">7</text>
      <text x="346" y="174" text-anchor="middle">8</text>
    </g>
    <text x="380" y="148" class="svg-text svg-text-dim" font-style="italic">8 jetons · "finie" se scinde en deux</text>
  </svg>
  <figcaption class="diagram-caption">Le découpage privilégie les fragments fréquents. « finie » = « fin » + « ie ».</figcaption>
</figure>

Pourquoi est-ce important pour vous ? Parce que tout, dans les outils d'IA, se mesure en jetons : la facture si vous payez à l'usage, la longueur maximale d'une conversation, la taille des documents que vous pouvez analyser. Quand un fournisseur annonce « 200 000 jetons de contexte », c'est l'équivalent d'environ **500 pages de livre**. Quand vous lui collez un document, il le découpe en jetons avant de le regarder.

<div class="section-num">§ 02 — La mécanique</div>

## Une seule opération, répétée des milliers de fois : prédire le <span class="accent">prochain jeton</span>.

Voici l'idée la plus contre-intuitive du domaine, et celle qui change tout : aussi sophistiqué soit-il, un grand modèle de langage ne fait fondamentalement qu'une seule chose. **Étant donnée une suite de jetons, prédire celui qui vient ensuite.**

Pas de planification globale. Pas de réflexion préalable sur l'ensemble de la réponse. Pas de plan caché. Un jeton à la fois, dans une boucle qui ne s'arrête que quand le modèle décide qu'il a terminé.

Comment fait-il ? L'architecture qui réalise cette prédiction s'appelle un **transformateur**. Ce qu'il faut retenir, sans entrer dans la machinerie, c'est son principe central — *l'attention*. Pour chaque jeton à produire, le modèle pèse l'importance relative de tous les jetons déjà présents. Chaque mot regarde tous les autres et décide lesquels comptent. Une sorte de relecture intégrale, à chaque pas.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 2</span><span>La boucle, un pas à la fois</span></div>
  <svg viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Boucle de prédiction du prochain jeton">
    <text x="20" y="24" class="svg-text svg-text-faint" font-weight="700">ÉTAPE t</text>
    <g font-family="JetBrains Mono, monospace" font-size="10">
      <rect x="20" y="38" width="42" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="41" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">La</text>
      <rect x="66" y="38" width="62" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="97" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">fenêtre</text>
      <rect x="132" y="38" width="36" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="150" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">de</text>
      <rect x="172" y="38" width="62" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="203" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">contexte</text>
      <rect x="238" y="38" width="36" height="26" fill="#e8a04b" opacity="0.85"/>
      <text x="256" y="55" text-anchor="middle" fill="#0f0d0a" font-weight="700">est</text>
      <rect x="278" y="38" width="20" height="26" fill="none" stroke="#a89c84" stroke-dasharray="2,2"/>
      <text x="288" y="55" text-anchor="middle" fill="#a89c84">?</text>
    </g>
    <line x1="160" y1="74" x2="160" y2="96" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="156,90 160,100 164,90" fill="#a89c84"/>
    <rect x="60" y="100" width="200" height="40" fill="#16130e" stroke="#e8a04b" stroke-width="1.5"/>
    <text x="160" y="118" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700">TRANSFORMATEUR</text>
    <text x="160" y="132" class="svg-text" text-anchor="middle" fill="#a89c84" font-size="9" font-style="italic">attention sur tous les jetons</text>
    <line x1="160" y1="142" x2="160" y2="164" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="156,158 160,168 164,158" fill="#a89c84"/>
    <text x="20" y="180" class="svg-text svg-text-faint" font-size="9">PROBABILITÉS · top 4 candidats</text>
    <g font-family="JetBrains Mono, monospace" font-size="9">
      <rect x="20" y="186" width="100" height="14" fill="#7a8b5c" opacity="0.85"/>
      <text x="124" y="197" fill="#a89c84">finie · 0.62</text>
      <rect x="20" y="204" width="48" height="14" fill="#7a8b5c" opacity="0.55"/>
      <text x="124" y="215" fill="#a89c84">limitée · 0.18</text>
      <rect x="20" y="222" width="22" height="14" fill="#7a8b5c" opacity="0.4"/>
      <text x="124" y="233" fill="#a89c84">large · 0.07</text>
      <rect x="20" y="240" width="14" height="14" fill="#7a8b5c" opacity="0.3"/>
      <text x="124" y="251" fill="#a89c84">vaste · 0.04</text>
    </g>
    <line x1="240" y1="193" x2="320" y2="193" stroke="#e8a04b" stroke-width="1.5"/>
    <polygon points="318,189 328,193 318,197" fill="#e8a04b"/>
    <text x="280" y="184" class="svg-text" text-anchor="middle" fill="#e8a04b" font-size="9" font-style="italic">choisit</text>
    <text x="340" y="180" class="svg-text svg-text-faint" font-weight="700">ÉTAPE t+1</text>
    <g font-family="JetBrains Mono, monospace" font-size="10">
      <rect x="340" y="186" width="60" height="22" fill="#a89c84" opacity="0.4"/>
      <text x="370" y="201" text-anchor="middle" fill="#a89c84">… est</text>
      <rect x="404" y="186" width="48" height="22" fill="#ffc26b"/>
      <text x="428" y="201" text-anchor="middle" fill="#0f0d0a" font-weight="700">finie</text>
      <rect x="456" y="186" width="20" height="22" fill="none" stroke="#a89c84" stroke-dasharray="2,2"/>
      <text x="466" y="201" text-anchor="middle" fill="#a89c84">?</text>
    </g>
    <path d="M 480 197 Q 540 197 540 130 Q 540 80 280 80" fill="none" stroke="#c2553a" stroke-width="1.5" stroke-dasharray="3,3"/>
    <polygon points="285,76 275,80 285,84" fill="#c2553a"/>
    <text x="555" y="140" class="svg-text" text-anchor="middle" fill="#c2553a" font-size="9" font-style="italic" transform="rotate(90 555 140)">on recommence</text>
    <text x="300" y="276" class="svg-label-big" text-anchor="middle">un jeton à la fois, jusqu'à la fin</text>
  </svg>
  <figcaption class="diagram-caption">À chaque pas, le modèle relit toute l'entrée pour choisir un seul jeton.</figcaption>
</figure>

Cette mécanique a une conséquence pratique étonnante. Quand l'IA vous répond, elle *ne sait pas*, au moment où elle écrit le premier mot, comment elle finira sa phrase. Elle écrit, mot après mot, en se relisant à chaque pas pour décider du suivant. Ce qui ressemble à de la pensée fluide est une succession de micro-décisions probabilistes. Cela explique pourquoi une IA peut commencer une réponse confiante et finir par une affirmation fausse — elle s'est « laissée porter » par sa propre génération.

<div class="section-num">§ 03 — Le champ de vision</div>

## La <span class="accent">fenêtre de contexte</span>, ou pourquoi votre IA « oublie ».

Si le modèle ne fait que prédire le prochain jeton à partir de ce qui précède, il faut lui définir un horizon — la quantité de jetons qu'il peut « voir » en même temps. Cet horizon s'appelle la **fenêtre de contexte**.

C'est *la* notion centrale à intégrer si vous utilisez régulièrement des outils d'IA. La fenêtre est tout à la fois : la zone d'attention du modèle, son champ de vision, et son *seul* support d'information. Tout ce qui s'y trouve peut influencer sa réponse ; tout ce qui n'y est pas n'existe pas pour lui.

Cette fenêtre a une **taille maximale**, fixée à la fabrication du modèle, mesurée en jetons. Selon le modèle, on parle de quelques milliers à plusieurs centaines de milliers de jetons. Pour les modèles Claude actuels par exemple, la fenêtre standard est d'environ **200 000 jetons**, soit l'équivalent d'un livre de 500 pages. Au-delà, on ne peut plus rien ajouter : il faut retirer du contenu existant pour faire de la place.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 3</span><span>La fenêtre, vue d'ensemble</span></div>
  <svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="La fenêtre de contexte comme une bande de jetons avec une limite maximale">
    <text x="20" y="22" class="svg-text svg-text-faint">┌─ FENÊTRE DE CONTEXTE</text>
    <text x="580" y="22" class="svg-text svg-text-faint" text-anchor="end">capacité maximale ─┐</text>
    <rect x="20" y="32" width="560" height="56" fill="none" stroke="#3d3525" stroke-width="1.5"/>
    <g font-family="JetBrains Mono, monospace" font-size="9">
      <rect x="22" y="34" width="280" height="52" fill="#e8a04b" opacity="0.15"/>
      <rect x="26" y="40" width="22" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="50" y="40" width="36" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="88" y="40" width="22" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="112" y="40" width="42" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="156" y="40" width="22" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="180" y="40" width="32" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="214" y="40" width="26" height="18" fill="#e8a04b" opacity="0.7"/>
      <rect x="242" y="40" width="38" height="18" fill="#e8a04b" opacity="0.7"/>
      <text x="160" y="76" text-anchor="middle" fill="#a89c84" font-style="italic" font-size="9">jetons déjà présents · ce que le modèle "voit"</text>
    </g>
    <rect x="302" y="34" width="276" height="52" fill="none" stroke-dasharray="2,3" stroke="#3d3525"/>
    <text x="440" y="64" class="svg-text svg-text-faint" text-anchor="middle" font-style="italic">espace disponible</text>
    <text x="440" y="78" class="svg-text svg-text-faint" text-anchor="middle" font-size="9">pour la suite</text>
    <line x1="580" y1="28" x2="580" y2="92" stroke="#c2553a" stroke-width="2"/>
    <text x="578" y="104" class="svg-text" text-anchor="end" fill="#c2553a" font-size="9" font-weight="700">↑ ~200k jetons</text>
    <text x="578" y="116" class="svg-text" text-anchor="end" fill="#c2553a" font-size="9" font-style="italic">au-delà : impossible</text>
    <path d="M 160 130 L 160 150 L 300 150 L 300 130" fill="none" stroke="#e8a04b" stroke-width="1"/>
    <line x1="230" y1="150" x2="230" y2="170" stroke="#e8a04b" stroke-width="1"/>
    <polygon points="226,164 230,174 234,164" fill="#e8a04b"/>
    <text x="230" y="186" class="svg-label-big" text-anchor="middle">le modèle prédit ici, à partir de tout ça</text>
  </svg>
  <figcaption class="diagram-caption">Une bande de jetons avec une limite stricte. Pas de mémoire ailleurs.</figcaption>
</figure>

### Pourquoi votre IA « oublie » au bout d'un moment

Vous avez peut-être déjà eu cette expérience : dans une longue conversation, l'assistant semble oublier ce que vous lui avez dit en début d'échange. Pas un bug. Une conséquence directe de ce qu'on vient de voir. Quand l'historique atteint la limite de la fenêtre, l'application qui pilote le modèle est obligée de couper : soit elle élague les anciens messages, soit elle les remplace par un résumé plus court. Dans les deux cas, le détail original est perdu pour le modèle.

Et pour la même raison, charger un PDF de 800 pages dans une fenêtre de 200 000 jetons peut tout simplement ne pas tenir. Au-delà, l'outil doit ruser — découper le document, n'en charger que des extraits pertinents, ou refuser. Aucune magie.

<div class="section-num">§ 04 — La transformation</div>

## D'un prédicteur de texte à un <span class="accent">assistant</span> qui répond.

Voici la deuxième idée contre-intuitive du domaine. Un transformateur, livré à lui-même, ne « répond » pas aux questions. Il **continue** du texte. Donnez-lui « La capitale de la France est », il complétera vraisemblablement par « Paris. ». Donnez-lui « Bonjour, comment vas-tu ? », il pourrait tout aussi bien continuer par « demanda Marie en ouvrant la porte. » — parce que c'est aussi une suite plausible dans le corpus de textes qui l'a entraîné.

Pour qu'il se comporte comme un assistant — qu'il *réponde* au lieu de *continuer* — il faut lui donner un cadre. Ce cadre s'appelle le **system prompt** : un texte placé avant la conversation qui définit qui le modèle est, ce qu'il doit faire, comment il doit s'exprimer, et ce qu'il doit refuser. C'est la première chose qu'il « lit » à chaque requête, et tout le reste se construit par-dessus.

Le system prompt est le mécanisme qui anime les produits que vous utilisez — ChatGPT, Claude.ai, Gemini, Copilot, Le Chat. Derrière chaque interface conversationnelle, un texte d'instructions transforme un prédicteur brut en assistant. Voici à quoi ça ressemble en pratique :

<div class="raw-prompt">Tu es un assistant pour développeurs spécialisé en bases de données SQL. Tu réponds en français. Pour chaque question, tu fournis d'abord la requête SQL, puis une brève explication. Si la question sort de ton domaine, tu le signales.

Utilisateur : Comment trouver tous les clients qui ont commandé en mars ?

Assistant : SELECT * FROM clients c JOIN commandes o ON c.id = o.client_id WHERE EXTRACT(MONTH FROM o.date) = 3; — la jointure ramène les clients liés à leurs commandes, le filtre EXTRACT isole celles de mars.</div>

Changez le system prompt, vous changez l'assistant. Voici le même modèle, avec une autre identité :

<div class="raw-prompt">Tu es Léa, agente de soutien client pour la boutique en ligne ACME. Tu es polie, empathique et tu vouvoies toujours. Tu ne discutes que de commandes, livraisons et retours. Pour toute autre question, tu rediriges vers le formulaire général. Tu ne révèles jamais que tu es une IA.

Utilisateur : Ma commande #4521 n'est toujours pas arrivée.

Assistant : Bonjour, je suis désolée pour ce délai. Pourriez-vous me confirmer votre code postal afin que je vérifie le statut de la commande #4521 ?</div>

Même cerveau, deux personnalités. Tout est du texte continu — les étiquettes *Utilisateur :* et *Assistant :* ne sont rien d'autre que des marqueurs textuels qui aident le modèle à savoir quand son tour de *continuer* arrive.

<div class="section-num">§ 05 — La mémoire absente</div>

## Le modèle ne se souvient de <span class="accent">rien</span>.

Voici la troisième idée à intégrer, et elle a des implications très concrètes pour vous. Le transformateur est **sans état**. Entre deux requêtes, il n'a aucun souvenir de ce qui a été dit. Aucun. Pour qu'une conversation paraisse continue, l'application qui parle au modèle doit lui **renvoyer la conversation entière à chaque tour**.

Quand vous tapez « Et sa population ? » dans une discussion qui parlait du Canada, l'application reconstruit en coulisse tout l'historique et l'envoie au modèle :

<div class="raw-prompt">Tu es un assistant utile, honnête et concis. Tu réponds en français.

Utilisateur : Quelle est la capitale du Canada ?

Assistant : La capitale du Canada est Ottawa, en Ontario.

Utilisateur : Et sa population ?

Assistant : <span class="cursor">▮</span></div>

Tout est là, dans une seule longue chaîne. Le modèle reçoit ce bloc, voit qu'il se termine par *Assistant :* avec un curseur, et continue le texte. Sans cette reconstitution intégrale, il n'aurait aucune idée de ce que désigne « sa » dans la dernière question.

<figure class="diagram">
  <div class="diagram-label"><span class="fig">Fig. 4</span><span>Une conversation, deux requêtes</span></div>
  <svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Schéma montrant qu'à chaque tour, la conversation entière est renvoyée au modèle">
    <text x="20" y="22" class="svg-text svg-text-faint" font-weight="700">TOUR 1</text>
    <rect x="20" y="32" width="160" height="60" fill="#16130e" stroke="#3d3525" stroke-width="1"/>
    <g font-family="JetBrains Mono, monospace" font-size="9">
      <rect x="26" y="38" width="148" height="14" fill="#c2553a" opacity="0.5"/>
      <text x="30" y="48" fill="#f4ecdc">SYS · cadre</text>
      <rect x="26" y="56" width="100" height="14" fill="#f4ecdc" opacity="0.5"/>
      <text x="30" y="66" fill="#0f0d0a">USR · capitale ?</text>
    </g>
    <text x="100" y="106" class="svg-text svg-text-faint" font-size="9" text-anchor="middle">→ modèle prédit</text>
    <line x1="180" y1="62" x2="220" y2="62" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="218,58 228,62 218,66" fill="#a89c84"/>
    <rect x="230" y="44" width="120" height="36" fill="#16130e" stroke="#e8a04b" stroke-width="1.5"/>
    <text x="290" y="66" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700">MODÈLE</text>
    <line x1="350" y1="62" x2="390" y2="62" stroke="#e8a04b" stroke-width="1.5"/>
    <polygon points="388,58 398,62 388,66" fill="#e8a04b"/>
    <rect x="400" y="50" width="180" height="22" fill="#7a8b5c" opacity="0.4" stroke="#7a8b5c" stroke-width="1"/>
    <text x="490" y="65" class="svg-text" text-anchor="middle" font-size="9">"Ottawa, en Ontario."</text>
    <line x1="20" y1="118" x2="580" y2="118" stroke="#3d3525" stroke-width="1" stroke-dasharray="2,4"/>
    <text x="20" y="138" class="svg-text svg-text-faint" font-weight="700">TOUR 2 · contient tout ce qui précède</text>
    <rect x="20" y="148" width="160" height="92" fill="#16130e" stroke="#3d3525" stroke-width="1"/>
    <g font-family="JetBrains Mono, monospace" font-size="9">
      <rect x="26" y="154" width="148" height="14" fill="#c2553a" opacity="0.5"/>
      <text x="30" y="164" fill="#f4ecdc">SYS · cadre</text>
      <rect x="26" y="172" width="100" height="14" fill="#f4ecdc" opacity="0.5"/>
      <text x="30" y="182" fill="#0f0d0a">USR · capitale ?</text>
      <rect x="26" y="190" width="120" height="14" fill="#7a8b5c" opacity="0.6"/>
      <text x="30" y="200" fill="#0f0d0a">AST · Ottawa…</text>
      <rect x="26" y="208" width="100" height="14" fill="#f4ecdc" opacity="0.5"/>
      <text x="30" y="218" fill="#0f0d0a">USR · population ?</text>
    </g>
    <line x1="180" y1="194" x2="220" y2="194" stroke="#a89c84" stroke-width="1.5"/>
    <polygon points="218,190 228,194 218,198" fill="#a89c84"/>
    <rect x="230" y="176" width="120" height="36" fill="#16130e" stroke="#e8a04b" stroke-width="1.5"/>
    <text x="290" y="198" class="svg-text" text-anchor="middle" fill="#e8a04b" font-weight="700">MODÈLE</text>
    <line x1="350" y1="194" x2="390" y2="194" stroke="#e8a04b" stroke-width="1.5"/>
    <polygon points="388,190 398,194 388,198" fill="#e8a04b"/>
    <rect x="400" y="182" width="180" height="22" fill="#7a8b5c" opacity="0.4" stroke="#7a8b5c" stroke-width="1"/>
    <text x="490" y="197" class="svg-text" text-anchor="middle" font-size="9">"~1,1 million."</text>
    <path d="M 100 92 Q 100 110 100 148" fill="none" stroke="#e8a04b" stroke-width="1" stroke-dasharray="2,3"/>
    <polygon points="96,144 100,154 104,144" fill="#e8a04b"/>
    <text x="160" y="128" class="svg-text" fill="#e8a04b" font-size="9" font-style="italic">on garde tout</text>
    <text x="300" y="270" class="svg-label-big" text-anchor="middle">l'historique grossit à chaque tour</text>
    <text x="300" y="294" class="svg-text svg-text-dim" text-anchor="middle" font-size="10" font-style="italic">le modèle, lui, n'a aucune mémoire entre les requêtes</text>
  </svg>
  <figcaption class="diagram-caption">L'application reconstitue l'historique à chaque appel. C'est elle qui « se souvient », pas le modèle.</figcaption>
</figure>

Cette absence de mémoire interne a une conséquence très concrète : chaque nouvel échange dans une conversation **repaie le coût de tout ce qui précède**. Plus la conversation avance, plus chaque tour est cher en jetons et plus la fenêtre se remplit. C'est pour ça que les conversations très longues finissent par se tasser, ralentir, ou démarrer dans un nouveau fil.

Et c'est aussi pour ça qu'on voit apparaître, dans les produits modernes, des fonctions de **mémoire persistante** — un magasin distinct de la conversation où le système enregistre les faits durables sur vous (préférences, projets, contexte professionnel) pour les ré-injecter quand c'est pertinent. Ce n'est pas le modèle qui se souvient : c'est l'application qui lui rappelle.

<div class="section-num">§ 06 — L'action</div>

## Comment une IA peut <span class="accent">agir</span> sur le monde.

Si une IA ne fait que prédire des jetons, comment peut-elle « lire un fichier », « chercher sur le web » ou « envoyer un courriel » ? La réponse est élégante : elle ne fait toujours rien d'autre que produire du texte — mais ce texte peut prendre la forme d'une **instruction d'action** que le programme hôte va reconnaître et exécuter pour elle.

L'astuce tient en deux ingrédients. D'abord, on apprend au modèle, dans son system prompt, qu'il a accès à des **outils** : lire un fichier, chercher sur le web, exécuter du code, etc. Ensuite, l'application surveille ce que le modèle écrit. Quand il produit une ligne qui ressemble à un appel d'outil — quelque chose comme `read_file("/data/rapport.txt")` — l'application l'intercepte, exécute réellement l'opération, et injecte le résultat dans la conversation. Du point de vue du modèle, tout reste du texte continu. Du point de vue de l'application, c'est elle qui fait le vrai travail.

Voici à quoi ressemble un cycle complet, en texte continu :

<div class="raw-prompt">Utilisateur : Résume-moi le fichier /data/rapport.txt.

Action: read_file("/data/rapport.txt")
Observation: <span class="injected">Le rapport trimestriel indique une hausse de 12% des revenus, une baisse des coûts d'infrastructure de 8%, et trois recommandations stratégiques [...4 200 jetons au total...]</span>

Réponse : Le rapport présente une hausse de 12 % des revenus, une baisse des coûts de 8 %, et trois recommandations stratégiques pour le prochain trimestre.</div>

Le modèle *demande* une action. L'application *la fait*. Le résultat revient en contexte, le modèle le voit comme s'il l'avait toujours su, et il continue. C'est la mécanique fondamentale des assistants modernes — Claude qui lit votre Google Drive, ChatGPT qui cherche sur le web, GitHub Copilot qui édite votre code. Toujours la même boucle : l'IA demande, l'application exécute, le résultat retourne en contexte.

### La conséquence sur la fenêtre

Tout cela laisse une trace dans la fenêtre, et chaque trace coûte des jetons. Lire un fichier de cinquante pages, c'est aussi déposer cinquante pages dans la fenêtre. Faire dix recherches web, c'est ajouter dix pages de résultats. C'est pour ça que les agents modernes — ceux qui enchaînent des actions de leur propre chef — peuvent saturer leur fenêtre étonnamment vite. Et c'est aussi le sujet principal de l'article suivant, pour qui veut aller plus loin.

<aside class="pull-quote">
  <p>Aussi sophistiqué soit-il, un grand modèle de langage ne fait fondamentalement qu'une seule chose : prédire le prochain jeton. Tout le reste est de la mise en scène — bien pensée, mais de la mise en scène.</p>
</aside>

<div class="section-num">§ 07 — À retenir</div>

## Trois idées qui suffisent à <span class="accent">tout</span> expliquer.

Si vous quittez cette page avec trois choses en tête, qu'elles soient celles-ci. **Premièrement** — l'IA lit des jetons, pas des mots, et tout ce qu'elle voit doit tenir dans une fenêtre de taille fixe. **Deuxièmement** — elle ne fait qu'une opération, prédire le prochain jeton, dans une boucle qui relit l'entrée à chaque pas. **Troisièmement** — elle n'a aucune mémoire entre deux requêtes : c'est l'application autour d'elle qui simule la continuité, en lui renvoyant l'historique à chaque tour, et qui exécute réellement les outils qu'elle réclame.

Avec ces trois idées, vous pouvez expliquer pourquoi votre assistant oublie au bout d'un moment, pourquoi un long document peut « ne pas tenir », pourquoi un même modèle se comporte différemment d'un produit à l'autre, et pourquoi un agent qui consulte beaucoup de sources peut devenir lent ou imprécis. Tout ce que vous lirez ensuite sur le sujet — *RAG*, *MCP*, *compaction*, *sous-agents* — sera de la variation autour de ces mêmes contraintes.

<div class="bridge">
  <div class="bridge-label">★ Pour aller plus loin</div>
  <h3>Si vous construisez avec des agents, l'histoire continue.</h3>
  <p>Cet article pose les fondations. Si vous utilisez Claude Code, Cursor, des agents personnalisés ou que vous concevez vous-même des outils basés sur ces modèles, la fenêtre de contexte devient une ressource qu'il faut gérer activement : arbitrer entre system prompt, outils, historique, résultats d'opérations et mémoire persistante.</p>
  <p>L'article suivant explore tout ça en détail — la boîte à outils complète de l'ingénierie d'agent, les phénomènes qui dégradent la qualité, et les heuristiques pratiques pour rester en-deçà de la saturation.</p>
  <p><a class="bridge-cta" href="/articles/context-window/">Lire la version praticien</a></p>
</div>
