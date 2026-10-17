# Livrer à tout prix — architecture et plan d'implémentation

Compagnon de [CONCEPTION.md](CONCEPTION.md). Ce document fixe les **contrats** entre les modules
pour que plusieurs sous-agents puissent développer en parallèle sans se marcher dessus.

## 1. Choix techniques

| Choix | Décision | Raison |
|---|---|---|
| Livrable | Une seule page HTML, sans dépendance, comme les autres projets | Cohérence du site, hors ligne, aucune chaîne de build côté client |
| Sources | Modules séparés dans `jeux/livrer-a-tout-prix/`, assemblés par un script | Développement parallèle sans conflits ; le premier jeu était déjà découpé (`00-core.js`, `20-engine.js`…) |
| Langage | JavaScript classique (pas de modules ES), un espace de noms global `LTP` | Concaténation triviale ; chargement facile dans Node pour les tests |
| Moteur | Pur, déterministe, sans DOM, graine aléatoire (mulberry32) | Tests sans navigateur, équilibrage par simulation, parties rejouables |
| Rendu | DOM et SVG à la main, rendu ciblé par région | Pas de framework ; l'open space SVG est mis à jour par différences |
| Textes | Interface dans `05-textes.fr.js` ; contenu narratif dans les fichiers de données, indexé par `id` | L'édition anglaise sera une surcouche de fichiers `*.en.js` |
| Persistance | `localStorage` sous try/catch, versionné | Même approche que le premier jeu |
| Nunjucks | Le script enveloppe le JS dans `{% raw %}…{% endraw %}` | `htmlTemplateEngine` vaut `njk` : un `{{` dans le JS casserait le build |

## 2. Arborescence

```
jeux/livrer-a-tout-prix/
├── CONCEPTION.md
├── ARCHITECTURE.md
├── page.fr.html                  # coquille : front matter, <head>, marqueurs @styles et @scripts
├── styles/
│   ├── 00-jetons.css             # couleurs, espacements, typo (clair et sombre)
│   ├── 10-base.css
│   ├── 20-hud.css
│   ├── 30-open-space.css
│   ├── 40-panneaux.css           # leviers, tableau de bord, registre
│   ├── 50-modales.css            # fiche, événement, comité, bilan
│   └── 60-mobile.css
├── js/
│   ├── 00-noyau.js               # espace de noms, rng, outils, t(), bus
│   ├── 05-textes.fr.js           # chaînes de l'interface
│   ├── 10-donnees-ilots.js       # 7 lignes d'affaires et leurs paramètres
│   ├── 11-donnees-personnages.js # 21 développeurs + personnages secondaires
│   ├── 12-donnees-leviers.js     # définition de tous les leviers (contrat moteur ↔ UI)
│   ├── 13-donnees-capacites.js   # skills et MCP
│   ├── 14-donnees-evenements.js  # cartes d'événement
│   ├── 15-donnees-comites.js     # questions des comités trimestriels
│   ├── 16-donnees-dialogues.js   # tutoriel, indices de Ginette, bulles génériques
│   ├── 17-donnees-fins.js        # profils de fin et textes du bilan
│   ├── 18-donnees-metriques.js   # définition des métriques (contrat moteur ↔ tableau de bord)
│   ├── 19-effets.js              # effets déclaratifs et effets différés
│   ├── 20-etat.js                # schéma, creerEtat(), lecture des leviers avec surcharge
│   ├── 21-production.js
│   ├── 22-qualite.js             # revue, cascade de défauts, couverture, agent qui triche
│   ├── 23-dette.js               # compréhension, MTTR
│   ├── 24-couts.js               # jetons, budget
│   ├── 25-donnees-secu.js        # données, triade létale, risques de sécurité
│   ├── 26-registre.js            # capacités : conflits, saturation, péremption, développement maison
│   ├── 27-personnes.js           # fatigue, moral, compétence, départs, relève
│   ├── 28-demandes.js            # backlog, portée, satisfaction, non-TI, Goodhart
│   ├── 29-evenements.js          # tirage et résolution des cartes d'événement
│   ├── 30-metriques.js           # agrégats, visibilité, historique
│   ├── 31-boucle.js              # semaine(), action(), trimestres, fin
│   ├── 32-fin.js                 # score et profil
│   ├── 35-demo.js                # états de démonstration pour développer l'UI
│   ├── 40-ui-dom.js              # h(), svg(), app (store), dispatch, planification du rendu
│   ├── 41-ui-avatars.js
│   ├── 42-ui-open-space.js
│   ├── 43-ui-leviers.js
│   ├── 44-ui-tableau-bord.js
│   ├── 45-ui-registre.js
│   ├── 46-ui-modales.js          # fiche de développeur, carte d'événement
│   ├── 47-ui-comite.js           # comité trimestriel et bilan annuel
│   ├── 48-ui-ecrans.js           # accueil, tutoriel, HUD, journal
│   ├── 49-ui-sauvegarde.js
│   └── 90-main.js                # démarrage, paramètres d'URL, horloge de lecture
└── tests/
    ├── charger.js                # charge les fichiers 00–39 dans un contexte vm
    ├── invariants.test.js        # bornes, NaN, durée d'une partie
    ├── contrats.test.js          # références croisées des données
    ├── <systeme>.test.js         # un fichier par système
    └── equilibre.test.js         # stratégies scriptées × graines

scripts/build-livrer-a-tout-prix.js   # assemble → src/projets/livrer-a-tout-prix.html
tests/livrer-a-tout-prix.spec.js      # Playwright (bout en bout)
```

Règle de numérotation : `00–39` ne touchent jamais au DOM et sont chargés dans Node ; `40–99` sont l'interface.
Les tests Node sont dans `jeux/…/tests/` et non dans `tests/`, sinon Playwright les ramasserait
(son motif par défaut inclut `*.test.js`).

## 3. Assemblage

`scripts/build-livrer-a-tout-prix.js` :

1. lit `page.fr.html` ;
2. concatène `styles/*.css` et `js/*.js` en ordre lexical, avec un commentaire `/* nom-du-fichier */` avant chaque fichier ;
3. remplace `<!-- @styles -->` et `<!-- @scripts -->` ; enveloppe le script dans `{% raw %}` ;
4. écrit `src/projets/livrer-a-tout-prix.html` avec un en-tête « fichier généré, ne pas éditer ».

Options :

- `--check` : échoue si le fichier généré n'est pas à jour (utilisé par les tests).
- `--autonome <chemin>` : écrit une version sans front matter ni `{% raw %}`, ouvrable en `file://`.
  Chaque sous-agent d'interface produit la sienne dans son propre chemin, ce qui évite que deux agents
  lancent Eleventy sur le même port ou écrivent `_site` en même temps.

Scripts npm ajoutés :

```json
"build:livrer": "node scripts/build-livrer-a-tout-prix.js",
"test:livrer": "node --test 'jeux/livrer-a-tout-prix/tests/*.test.js' && node scripts/build-livrer-a-tout-prix.js --check"
```

et `build` appelle `build:livrer` avant `eleventy`. Le fichier généré est versionné, comme les autres projets.

## 4. Contrats

### 4.1 Forme d'un module

```js
/* 21-production.js */
(function (LTP) {
  'use strict';
  LTP.K.production = { base: { stagiaire: 2, junior: 3, inter: 5, senior: 6 }, exposantSessions: 0.6 /* … */ };

  LTP.systemes.production = {
    initialiser(etat) { /* champs propres au système, si besoin */ },
    semaine(etat, ctx) { /* mute etat (une copie), écrit dans ctx.journal */ },
  };
})(globalThis.LTP = globalThis.LTP || {});
```

Chaque système possède **ses constantes** (`LTP.K.<systeme>`), seul endroit que l'agent d'équilibrage modifiera.

### 4.2 API du moteur (31-boucle.js)

```js
LTP.moteur.creer(graine, options)        // → etat   (options : { serveurAchete, trimestreDepart })
LTP.moteur.semaine(etat)                 // → { etat, journal }   copie profonde, puis systèmes en ordre
LTP.moteur.action(etat, action)          // → { etat, journal, erreur? }
LTP.moteur.simuler(etat, n, strategie)   // → etat   utilitaire pour tests et mode animateur
```

Actions :

```js
{ type: 'levier', id, valeur, ilot? }          // ilot absent : global ; présent : surcharge (leviers ◆)
{ type: 'retirerSurcharge', id, ilot }
{ type: 'equiper', carte, ilot } | { type: 'desequiper', carte, ilot }
{ type: 'developper', carte, dev }             // lance une capacité maison (effet différé)
{ type: 'proprietaire', carte, dev }
{ type: 'choix', evenement, choix }            // résout la carte en attente
{ type: 'comite', question, reponse }
{ type: 'phase', phase }                       // planification → execution, comite → planification…
```

Ordre des systèmes dans `semaine()` :

0. remise à zéro de `derniere` (îlots, devs) et leviers à délai arrivés à échéance (`31`)
1. différés échus (`19-effets`)
2. disponibilité des personnes : congés, formation (`27-personnes.disponibilite`)
3. registre : fraîcheur, développement maison, conflits, poids du contexte ; **calcule `ilot.mods` et `ilot.triade`** (`26`)
4. production (`21`)
5. revue et qualité (`22`)
6. dette cognitive (`23`)
7. coûts (`24`)
8. données et sécurité : risques → incidents (`25`)
9. bogues → incidents, MTTR (`23.incidents`)
10. demandes, satisfaction, non-TI, Goodhart (`28`)
11. personnes : fatigue, moral, compétence, départs (`27`)
12. tirage d'un événement : au plus un par semaine, met `etat.evenementEnAttente` (`29`)
13. métriques et historique (`30`)
14. fin de trimestre → `phase = 'comite'` ; semaine 52 ou confiance < 15 → `phase = 'fin'` (`31`, `32`)

Tant que `etat.evenementEnAttente` existe, `semaine()` refuse d'avancer et renvoie l'état inchangé.

### 4.3 Schéma de l'état (20-etat.js)

```js
{
  version: 1, graine, semaine: 0, trimestre: 1,
  phase: 'accueil' | 'tutoriel' | 'planification' | 'execution' | 'comite' | 'fin',
  rngEtat,                                   // état sérialisable du générateur
  leviers: { global: { [id]: valeur }, surcharges: { [ilotId]: { [id]: valeur } } },
  debloques: { panneaux: ['pratiques', 'ia'], leviers: [...] },
  org: {
    budget, budgetAnnuel, jetonsSemaine, jetonsCumul, confiance,
    donnees: { qualite, duplication, fraicheur, classification },
    menace, observabilite, releve, objectif,
    drapeaux: { [nom]: true },               // ex. 'kevinAppli', 'ginetteRetraite'
  },
  ilots: { [id]: {
    id, criticite, sensibilite, volatilite, adequationPile,
    backlog, satisfaction, applisFantomes,
    codebase: { volume, defautsLatents, couvertureApparente, couvertureReelle, conventions },
    fileRevue, capacites: [carteId], triade: { prive, nonFiable, sortie },
    mods: { debit, tauxDefaut, jetons, mttr, comprehension, adequationPile, couverture, verbosite, poidsContexte, incoherence,
            taillePR, attention },
                                             // calculés par 26-registre à l'étape 3, lus par 21–25 (voir 20-etat.js)
    derniere: { points, pointsLivres, code, pr, prFusionnees, deploiements, defautsInjectes, defautsEchappes,
                bogues, incidents, jetons, cout, delai, attention, lgtm, demandes, demandesRefusees },
                                             // statistiques de la semaine écoulée, remises à zéro par la boucle
  } },
  devs: { [id]: {
    id, ilot, niveau, traits: [...],
    competence, competenceIA, adhesion, rigueur, comprehension, fatigue, moral,
    sessions, etat, indisponibleJusqua, present: true,
    occupation: { registre, formation },     // temps réservé par 26 et 27, soustrait par 21 ; remis à zéro chaque semaine
                                             // etat : 'conge' | 'formation' | 'absent' posés par 27 (étape 2) ;
                                             //        'code' | 'revue' | 'attente' par 21 (étape 4) ; 'incident' par 23 (étape 9)
    derniere: { points, code, pr, revues, lgtm, jetons, ecritSoiMeme, luEnRevue },
  } },
  capacites: { [carteId]: { equipeeSur: [ilotId], proprietaire, fraicheur, disponibleSemaine, decouverte } },
  incidents: [{ id, ilot, gravite, debut, mttrPrevu, cause }],
  differes: [{ semaine, effets, source }],
  evenementEnAttente: null | { id, ref: { ilot?, dev? }, vars },
  evenementsForces: [{ id, semaine, ref }],   // suites planifiées par l'effet { evenement }
  evenementsVus: { [id]: semaine },
  derniereDecision: null | { evenement, choix, semaine, ref },   // posé par 29.resoudre, affiché par l'interface
  comite: null | { questions, reponses },
  historique: { [metrique]: [valeurParSemaine] },
  journal: [{ semaine, type, cle, vars, ilot?, dev? }],  // 200 dernières entrées
}
```

Champs ajoutés par les systèmes dans leur `initialiser` (internes, documentés en tête de chaque fichier) :
`etat.travail` (21–24), `etat.registre` (26 : orphelines, périmées, connaissance capturée), `etat.securite`
(25 : surface d'attaque, exposition Loi 25, cumul par cause), `etat.personnes` (27 : départs), `etat.evenementsCompte` (29),
`etat.metriquesDetail` et `etat.cumuls` (30), `etat.fin` (32), `ilot.retard` et `ilot.applisEncadrees` (28).
`etat.comite.questions[]` contient `{ id, personnage, reponses: [{ id }] }`, et `etat.comite` a aussi `scores` et `reactions`.
`etat.resultat` contient aussi `faits` (chiffres du bilan) et `affinites` (proximité de chaque profil).

Fonctions utilitaires exposées (`LTP.q`) pour le moteur, les conditions d'événements et l'UI :
`levier`, `rangLevier`, `levierDebloque`, `devsDe`, `devsPresents`, `disponible`, `ilotDe`, `aTrait`, `modTrait`,
`triadeComplete`, `metriqueVisible`, `creerIncident`, `incidentsActifs`, `estIncidentSecurite`, `def*` (définitions de contenu).
La liste à jour est dans `20-etat.js`.

**Textes du journal** : un système note `ctx.noter(type, cle, vars, ref)` et enregistre lui-même le gabarit
de ses clés avec `LTP.enregistrerTextes({ '<systeme>.<cle>': '…' })` dans son propre fichier.
Les identifiants (`levier`, `ilot`, `dev`) passés dans `vars` sont convertis en libellés par l'interface.

### 4.4 Effets déclaratifs (19-effets.js)

Les fichiers de contenu n'appellent jamais le moteur directement ; ils décrivent des effets :

```js
{ cible: 'org' | 'ilot' | 'dev' | 'ilots' | 'devs', chemin: 'satisfaction', op: 'add' | 'mul' | 'set', valeur,
  pendant?: 4,      // effet temporaire, annulé après n semaines
  dans?: 2,         // effet différé
  filtre?: { trait: 'junior' } }
{ drapeau: 'kevinAppli' }
{ carte: 'githbu-mcp', op: 'decouvrir' | 'retirer', ilot? }
{ incident: { gravite: 0.7, cause: 'injection' } }
{ journal: 'cle-de-texte', vars: {} }
```

`cible: 'ilot'` et `'dev'` désignent la cible choisie au tirage (`etat.evenementEnAttente.cible`).
Les chemins sont validés par `contrats.test.js` contre le schéma.

Une carte d'événement (forme définie dans CONCEPTION.md §6) peut avoir une `condition(etat, q)` et une
`choisirCible(etat, q, rng)` ; ce sont les seules fonctions permises dans les données.

### 4.5 Leviers (12-donnees-leviers.js)

```js
{ id: 'autonomie', panneau: 'ia', trimestre: 1, surchargeable: true, delai: 0,
  valeurs: [{ v: 0, cle: 'autonomie.0' }, /* … */], defaut: 1, aideCle: 'autonomie.aide' }
```

L'UI génère les panneaux à partir de cette table ; le moteur lit les valeurs avec `LTP.q.levier()`.
Ajouter un levier = une entrée ici + sa lecture dans un système. Personne d'autre n'a à changer.

### 4.6 Interface (40-ui-dom.js)

```js
LTP.app = { etat, ui: { vue, selection, vitesse, panneau, modale } }
LTP.app.dispatch(action)   // moteur.action → sauvegarde → notifie
LTP.app.avancer()          // moteur.semaine → animation → notifie (appelé par l'horloge de 90-main)
// régions : 'hud', 'openspace', 'panneau' (leviers et registre), 'tableau', 'journal', 'modale' ; écrans : 'accueil', 'fin'
LTP.ui.h(tag, attrs, ...enfants), LTP.ui.svg(...)
LTP.ui.regions[region] = { monter(racine), rendre(etat, ui, precedent) }
// Le comité et la visite guidée sont des couches de la région 'modale' (47, 48).
```

Chaque module d'interface enregistre sa région et ne touche qu'à sa racine DOM.
`35-demo.js` fournit `LTP.demo.etat(nom)` (« debut », « t2-tendu », « triade-appro », « fin-usine »)
pour développer l'interface avant que le moteur soit complet.

### 4.7 Paramètres d'URL et crochets de test

- `?graine=123` fixe la graine ; `?animateur=1` active le mode animateur ; `?demo=t2-tendu` charge un état de démonstration.
- `window.LTP` reste exposé ; les tests Playwright s'en servent pour sauter des semaines (`LTP.app.simuler(n)`).
- Identifiants stables pour les tests : `#btn-lecture`, `#btn-semaine`, `[data-dev="<id>"]`, `[data-ilot="<id>"]`,
  `[data-levier="<id>"]`, `.carte-evenement`, `[data-choix]`, `.comite`, `.bilan`, `.profil-fin`.

## 5. Tests

| Niveau | Fichier | Ce qui est vérifié |
|---|---|---|
| Unitaire | `jeux/…/tests/<systeme>.test.js` | Chaque formule de CONCEPTION.md §4 réagit dans le bon sens (ex. plus de réviseurs → moins de défauts échappés, plus de délai) |
| Invariants | `invariants.test.js` | 100 parties aléatoires : aucune valeur hors bornes ni NaN, 52 semaines en moins de 200 ms |
| Contrats | `contrats.test.js` | Toute référence dans les données existe (ids de devs, d'îlots, de cartes, de leviers, de clés de texte, chemins d'effets) |
| Équilibrage | `equilibre.test.js` | Stratégies scriptées × 30 graines (voir ci-dessous) |
| Bout en bout | `tests/livrer-a-tout-prix.spec.js` | Accueil sans erreur ; tutoriel ; un trimestre jusqu'au comité ; un événement résolu ; fiche de dev ; équiper un MCP allume un voyant ; bilan via le mode animateur ; 375 px sans défilement horizontal ; `--check` à jour |

Stratégies d'équilibrage, qui servent de **critères d'acceptation** du modèle :

| Stratégie | Réglages | Attendu |
|---|---|---|
| `tout-ia` | Autonomie 4 partout, 6 sessions, aucun garde-fou, libre-service | « Usine à PR » ou « Far West » dans ≥ 80 % des graines ; meilleur débit au T1 |
| `musee` | Autonomie 0–1, 2 réviseurs, interdire le non-TI | « Le musée » dans ≥ 70 % ; satisfaction la plus basse |
| `equilibree` | Autonomie selon la criticité, CI complète, tests + mutation, 1 réviseur, PR de 400 lignes, formation, catalogue, bac à sable, liste blanche, encadrer le non-TI | « Transition réussie » dans ≥ 70 % ; meilleur score moyen |
| `economique` | Comme `equilibree` mais modèle économique, sans formation | Score inférieur à `equilibree`, plus d'incidents de sécurité |
| `aleatoire` | Leviers au hasard chaque trimestre | Aucun plantage, distribution de profils variée |

## 6. Plan d'implémentation

### Phase 0 — Fondations (moi, séquentiel)

Rien n'est parallélisé avant que les contrats soient en place et testés.

1. Arborescence, `page.fr.html`, script d'assemblage avec `--check` et `--autonome`, scripts npm.
2. `00-noyau.js` (rng, outils, `t()`, bus), `20-etat.js` (schéma complet, `creerEtat`, `LTP.q`).
3. `31-boucle.js` avec l'ordre des systèmes et des **systèmes bouchons** (un fichier par système, `semaine()` vide).
4. Noyau de `29-evenements.js` : application des effets déclaratifs et différés (le tirage viendra en phase 1).
5. `10-donnees-ilots.js` complet ; `12-donnees-leviers.js` complet ; squelette de `11-donnees-personnages.js`
   (21 ids, îlot, niveau, traits, sans les textes).
6. `35-demo.js`, `40-ui-dom.js`, `90-main.js` : la page démarre, affiche un open space provisoire
   et fait défiler les semaines sur les bouchons.
7. `tests/charger.js`, `invariants.test.js`, `contrats.test.js`, test Playwright de fumée.
8. Commit « squelette ».

### Phase 1 — Développement parallèle (7 sous-agents)

Tous dans le même arbre de travail, avec **propriété exclusive des fichiers** : un agent ne modifie que ses fichiers
et les tests qui portent son nom. Aucun agent ne commite ; je fais l'intégration. Chaque agent reçoit
CONCEPTION.md, ARCHITECTURE.md et la liste de ses fichiers, et doit terminer avec ses tests au vert.
Un besoin de changer un contrat (schéma, API, leviers) est **signalé dans le rapport**, pas appliqué.

| Agent | Fichiers | Livrable | Dépend de |
|---|---|---|---|
| **A · Moteur du travail** | `21-production`, `22-qualite`, `23-dette`, `24-couts` + tests | §4.4 à 4.6 de la conception | Phase 0 |
| **B · Moteur des risques** | `25-donnees-secu`, `26-registre`, `28-demandes` + tests | §4.7 à 4.9, 4.11, 4.12 | Phase 0 |
| **C · Moteur des personnes et du pilotage** | `27-personnes`, `29-evenements` (tirage et résolution), `30-metriques`, `32-fin` + tests | §4.10, §5, §2.3, tirage des événements | Phase 0 |
| **D1 · Contenu des personnages** | `11-donnees-personnages` (textes, bulles, réactions), `15-donnees-comites`, `16-donnees-dialogues`, `17-donnees-fins` | Les voix des personnages, le tutoriel, les comités, le bilan | Phase 0 |
| **D2 · Contenu des événements et des capacités** | `13-donnees-capacites`, `14-donnees-evenements` | ~33 cartes de capacités, ~45 événements, en effets déclaratifs | Phase 0 (DSL) |
| **E · Interface de l'open space** | `41-ui-avatars`, `42-ui-open-space`, `30-open-space.css` | Vue de haut, avatars paramétriques, signaux, animations, version téléphone | `35-demo` |
| **F · Interface de pilotage** | `05-textes.fr`, `43` à `49`, `00-jetons.css`, `10-base`, `20-hud`, `40-panneaux`, `50-modales`, `60-mobile` | HUD, leviers, tableau de bord, registre, modales, comités, accueil, tutoriel, bilan, sauvegarde | `35-demo` |

Les agents d'interface travaillent sur les états de démonstration et produisent leur version
`--autonome` dans leur propre dossier temporaire pour leurs captures Playwright.
Les agents E et F chargent la compétence de design avant d'écrire du CSS ; F charge celle de visualisation
de données pour le tableau de bord.

### Phase 2 — Intégration (moi)

1. Lire les rapports, arbitrer les demandes de changement de contrat.
2. Assembler, lancer `test:livrer`, `contrats.test.js` et Playwright ; corriger les écarts.
3. Première partie complète à la main avec la vraie simulation. Commit.

### Phase 3 — Équilibrage et revue (3 sous-agents en parallèle)

| Agent | Mission | Modifie |
|---|---|---|
| **G · Équilibrage** | Fait passer `equilibre.test.js`, rapporte la distribution des profils par stratégie | Uniquement les blocs `LTP.K.*` |
| **H · Essai de jeu et ergonomie** | Joue des campagnes avec Playwright au bureau et au téléphone, captures d'écran, liste des problèmes classés | Rien (rapport) |
| **I · Relecture** | Français (accents, typographie, espaces insécables avant « : ; ! ? » et dans les guillemets), ton, cohérence des personnages, exactitude technique (MCP, skills, triade létale, DORA, Loi 25) | Les fichiers de textes et de données (textes seulement) |

### Phase 4 — Finition et publication (moi)

1. Corrections issues de H (ou nouvel envoi à E ou F si le lot est gros).
2. Intégration au site : entrée dans `src/_data/projets.json` (emoji, modèle, date), lien de retour, `hreflang` réservé à l'édition anglaise.
3. Tests complets, build de production, commit. Pas de push sans ton accord.

### Phase 5 — Plus tard

- Édition anglaise : `page.en.html` et des fichiers `*.en.js` qui remplacent les textes par `id` ;
  un sous-agent de traduction, comme pour *Server at Any Price*.
- Missions courtes construites sur des états de départ préréglés.

## 7. Budget et risques

**Budget** : environ 7 000 lignes au total, page générée sous 450 Ko (le premier jeu fait 375 Ko).
Partie complète simulée en moins de 200 ms.

| Risque | Mitigation |
|---|---|
| Le modèle est trop riche pour être lisible | Divulgation progressive par trimestre ; chaque levier a une ligne d'aide et un indice de Ginette |
| L'équilibrage déraille | Stratégies scriptées comme critères d'acceptation, constantes isolées, mode animateur pour l'essai |
| Les agents divergent sur les contrats | Phase 0 complète et testée avant le parallélisme ; propriété exclusive des fichiers ; `contrats.test.js` |
| Le contenu écrit des effets impossibles | Effets déclaratifs validés contre le schéma |
| L'open space est illisible sur téléphone | Vue en liste d'îlots dédiée, pas un simple rétrécissement ; essai au téléphone en phase 3 |
| Nunjucks interprète le JS | Enveloppe `{% raw %}` dans le script d'assemblage, vérifiée par le test `--check` |
| Deux agents qui lancent un serveur ou un build en même temps | Builds `--autonome` dans des dossiers temporaires distincts, pas d'Eleventy en phase 1 |
