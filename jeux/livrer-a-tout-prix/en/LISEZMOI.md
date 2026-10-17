# Édition anglaise : *Ship at Any Price*

Ce dossier contient les surcouches qui transforment *Livrer à tout prix* en *Ship at Any Price*.
Le français reste la source : le moteur, les conditions et les effets ne sont écrits qu'une fois.
Les fichiers de ce dossier ne remplacent que des textes.

- Page : `/en/projets/ship-at-any-price/`, générée dans `src/projets/ship-at-any-price.html` par `npm run build:livrer`
  (coquille : `page.en.html`).
- Sauvegarde distincte de l'édition française (`ship-at-any-price:v2`).

## Mécanique

Chaque fichier `en/NN-nom.js` porte le nom du fichier qu'il traduit et se charge juste après lui
(ordre fixé par `sources.js`). `en/00-langue.js` se charge en premier et pose `LTP.langue = 'en'`.

**Textes de l'interface et du journal** (`t('cle')`) : on enregistre les mêmes clés en anglais.

```js
LTP.enregistrerTextes({ 'hud.semaine': 'Week {semaine}', 'levier.change': function (v) { … } }, 'en');
```

Les clés manquantes retombent sur le français. Ce repli est relevé dans `LTP.textesManquants` et fait
échouer les tests.

**Contenu** (`LTP.donnees`) : `LTP.traduire(chemin, patch)` fusionne un patch dans les données, en place.

```js
LTP.traduire('ilots', { appro: { nom: 'Procurement', court: 'Procurement', mission: '…' } });
LTP.traduire('evenements', { 'kevin-appli': { titre: '…', texte: function (etat, q, ref) { … }, choix: { accepter: { libelle: '…', consequence: '…' } } } });
LTP.traduire('reflexions.leviers.reviseurs', { 2: ['Two reviewers? My PR is going to wait.', '…'] });
```

Règles de fusion :

- **Chaîne ou fonction** : remplace la valeur. Posée sur un objet qui a un champ `texte` (entrée de réflexion
  `{ texte, effet, ton… }`), elle remplace seulement ce `texte`.
- **Tableau** : fusionné élément par élément, avec la même longueur que le français. `null` garde l'élément.
- **Objet posé sur un tableau** : désigne les éléments par `id`, ou par `v` pour les valeurs d'un levier.
- **Objet posé sur un objet** : fusion clé par clé.
- Toute clé ou tout `id` absent du français est une erreur, gardée dans `LTP.erreursTraduction`.

Ne traduire que les champs de texte. Ne jamais toucher aux `id`, `v`, conditions, effets, poids ou drapeaux.
Une fonction de texte (`texte: function (etat, q, ref) {…}`) se traduit par une fonction anglaise qui garde la
même logique (mêmes lectures de l'état, mêmes branches) et ne change que les phrases.

## Vérifier

```
node --test jeux/livrer-a-tout-prix/tests/traduction.test.js               # tout
PORTEE=evenements node --test jeux/livrer-a-tout-prix/tests/traduction.test.js   # seulement donnees.evenements et les clés « evenements. »
```

`PORTEE` accepte une liste séparée par des virgules (`PORTEE=comites,personnages,devs`). Le test signale :

- les textes restés identiques au français ;
- le français résiduel : mots outils, mots accentués en minuscules, « Mme », « Me », guillemets « » ;
- les erreurs de fusion.

Les noms propres accentués (Régis, Bérubé, Marie-Ève) sont permis.

## Style

- **Canadian English** :
  - licence (noun), license (verb) ; catalogue, centre, colour, behaviour, favour, cheque ;
  - program (pas programme) ; analyze ; travelled.
- **Ton** : la même comédie de bureau sèche et bienveillante que le français. Des phrases courtes, du concret
  (des noms, des chiffres, des outils). On traduit l'intention et la chute de la blague, pas les mots ;
  on reste naturel pour un lecteur anglophone.
- **Références québécoises** :
  - on garde le chalet (« the cottage »), la cabane à sucre (« the sugar shack »), la Loi 25
    (« Law 25 », ou « Quebec's Law 25 » à la première mention) ;
  - on adapte une expression idiomatique intraduisible.
- **Typographie** :
  - apostrophe droite `'` (échappée `\'` dans les chaînes à guillemets simples) ;
  - guillemets doubles typographiques “ ” ;
  - tiret cadratin — sans espaces ;
  - aucune espace insécable.
- **Nombres** dans les phrases :
  - $1.2M, $45k, $1,200, 87%, 3.5 ;
  - 4:55 p.m. ; 1,000 lines ; 12k tokens.
- **Jours** : Monday … Friday, avec la majuscule.
- **Semaines et trimestres** :
  - W14 (« S14 ») et Q2 (« T2 ») ;
  - Week 14, Quarter 2 (en toutes lettres).
- **Gabarits** : garder `{dev}`, `{ilot}`, `{n}` et tous les `{…}` tels quels.
- **Personnes** :
  - les personnages gardent leur prénom français, accents compris (Benoît, Jérémy, Marie-Ève, Chloé) ;
  - « il/elle » suit le personnage.

## Glossaire

| Français | Anglais |
|---|---|
| Groupe Polyvalent Universel inc. (GPU) | General Purpose Unlimited Inc. (GPU) |
| Mme Lavigueur, présidente-directrice générale | Ms. Lavigueur, President and CEO |
| Me Bérubé, contentieux et conformité | Counsel Bérubé, Legal and Compliance |
| Régis, VP Finances | Régis, VP Finance |
| Ginette, technicienne senior, TI | Ginette, Senior Technician, IT |
| Kevin, marketing, développeur citoyen | Kevin, Marketing, citizen developer |
| open space | open-plan office (ou « the floor ») |
| îlot | pod |
| ligne d'affaires | business line |
| direction (d'une ligne d'affaires) | business owner / the business |
| comité de direction | executive committee |
| note de service | memo |
| levier | lever |
| tableau de bord | dashboard |
| rapport du vendredi | Friday report |
| « Ce qu'on entend » | “Overheard” |
| « À surveiller » | “Watch this” |
| semaine, trimestre | week, quarter |
| jeton(s) | token(s) |
| PR, revue de code, réviseur | PR, code review, reviewer |
| fusionner, fusion | merge |
| file de revue | review queue |
| carnet de demandes, demandes en attente | request backlog |
| dette cognitive, dette technique | cognitive debt, technical debt |
| triade létale | lethal trifecta |
| injection de prompt | prompt injection |
| paquet halluciné | hallucinated package |
| typosquattage | typosquatting |
| registre (des skills et MCP) | registry |
| carte (skill ou MCP du registre) | card |
| serveur MCP, passerelle MCP | MCP server, MCP gateway |
| skill | skill |
| groupe pilote, volontaire | pilot group, volunteer |
| autocomplétion, clavardage | autocomplete, chat |
| session d'agent | agent session |
| bac à sable | sandbox |
| sorties externes | external outputs |
| catalogue de données, source officielle | data catalogue, source of truth |
| développeur citoyen, hors TI | citizen developer, outside IT |
| application fantôme | shadow app |
| relève | succession (programme de relève : succession program) |
| formation, atelier, mentorat | training, workshop, mentoring |
| de garde | on call |
| traitement de nuit | nightly batch |
| couverture de tests | test coverage |
| temps moyen de rétablissement | mean time to recovery (MTTR) |
| confiance de la direction | executive confidence |
| moral, fatigue | morale, fatigue |

### Îlots

| id | Français | Anglais (nom · court) |
|---|---|---|
| appro | Approvisionnement · Appro | Procurement · Procurement |
| operations | Opérations | Operations · Ops |
| bureautique | Bureautique | Office Automation · Office |
| inspections | Inspections | Inspections · Inspections |
| iot | Capteurs et IoT · IoT | Sensors and IoT · IoT |
| web | Sites web · Web | Websites · Web |
| portail | Portail client · Portail | Customer Portal · Portal |

### Panneaux et leviers (libellés exacts, à réutiliser tels quels dans la prose)

| id | Libellé anglais | Valeurs |
|---|---|---|
| *panneau* ia | AI use | |
| *panneau* pratiques | Practices | |
| *panneau* produit | Product | |
| *panneau* registre | Registry | |
| *panneau* gouvernance | Data and security | |
| *panneau* organisation | Organization | |
| licences | Copilot licences | None · Pilot group (7) · Whole team (21) |
| autonomie | Agent autonomy level | 0 · Autocomplete · 1 · Chat · 2 · Agent with approval at every step · 3 · Agent that opens PRs · 4 · Agent that merges on its own |
| sessionsMax | Agent sessions per person | 1 session … 6 sessions |
| modele | Default model | Economy · Mid-range · High-end · Local (on the GPU server) |
| modeleParTache | Model chosen per task | No · Yes |
| pair | Pair programming | None · Human with human · Human with AI · Rotation (every other week) |
| reviseurs | Reviewers per PR | None · One · Two |
| approbation | PR approval | Automatic (AI review) · A human · The code owner |
| taillePR | Maximum PR size | Unlimited · 400 lines · 200 lines |
| ci | Continuous integration and deployment | None · Build · Build and tests · Full pipeline and continuous deployment |
| lint | Lint and formatting | None · Auto-formatting · Strict lint · Strict, locked lint |
| tests | Tests | None · Unit · Unit and integration · With mutation testing |
| revuesExplication | Explanation reviews | No · Monthly · Every major PR |
| formation | Training program | None · One workshop · Full program · Program and mentoring |
| plafondJetons | Token cap | None · Soft (alert) · Hard (cut-off) |
| disciplineNon | “Saying no” discipline | Lax · Normal · Firm |
| specs | Specs before code | No · Light · Mandatory |
| catalogue | Data catalogue | None · Inventory · Catalogue with sources of truth |
| classification | Sensitive data classification | No · Manual · Automatic |
| bacASable | Agent sandbox | None · Container · Container and filtered network |
| accesProd | Agent access to production | Read and write · Read-only · None |
| approbationSorties | Human approval of external outputs | No · Yes |
| politiqueRegistre | Registry policy | Self-service · Allowlist · Allowlist and mandatory owner |
| passerelleMCP | MCP gateway | No · Yes |
| politiqueNonTI | Development outside IT | Ignore · Ban · Support (approved registry and coaching) |
| observabilite | Observability | None · Basic · Full |
| embauche | Junior hiring | Freeze · Normal · Succession program |
| objectif | Executive target | None · Number of PRs · Lead time · Satisfaction · Number of incidents |

### Métriques

| Français | Anglais |
|---|---|
| groupes : Livraison · Qualité · Coûts · Risques · Personnes | Delivery · Quality · Costs · Risks · People |
| PR fusionnées par semaine | PRs merged per week |
| Valeur livrée (points) | Value delivered (points) |
| Délai de livraison | Lead time |
| Déploiements par semaine | Deployments per week |
| Demandes en attente (points) | Request backlog (points) |
| Satisfaction des directions | Business satisfaction |
| Bogues en production | Production bugs |
| Incidents | Incidents |
| Taux d'échec des changements | Change failure rate |
| Temps moyen de rétablissement | Mean time to recovery |
| Volume de code | Code volume |
| Couverture de tests (affichée) | Test coverage (reported) |
| Couverture réelle | Actual coverage |
| Compréhension du code | Code comprehension |
| Jetons consommés par semaine | Tokens used per week |
| Coût de l'IA et des outils par semaine | AI and tooling cost per week |
| Budget restant | Remaining budget |
| Projets en retard | Late projects |
| Surface d'attaque | Attack surface |
| Exposition Loi 25 | Law 25 exposure |
| Applications hors TI connues | Known apps built outside IT |
| Skills et MCP sans propriétaire | Ownerless skills and MCPs |
| Moral moyen · Fatigue moyenne | Average morale · Average fatigue |
| Relève | Succession |
| Connaissance capturée | Captured knowledge |
| Confiance de la direction | Executive confidence |

### Profils de fin

| id | Français | Anglais |
|---|---|---|
| reussie | La transition réussie | The Successful Transition |
| usine | L'usine à PR | The PR Factory |
| musee | Le musée | The Museum |
| farwest | Le Far West | The Wild West |
| facture | La facture salée | The Eye-Watering Bill |
| chateau | Le château de cartes | The House of Cards |
| congedie | Merci pour votre contribution | Thank You for Your Contribution |

### Cartes du registre

Les noms de cartes sont des identifiants de dépôt en kebab-case.
On les traduit en anglais, en kebab-case, en gardant les fautes voulues :

- `githbu-mcp` reste tel quel : c'est le typosquat ;
- `github-mcp (officiel)` devient `github-mcp (official)` ;
- exemples : `conventions-gpu` devient `gpu-conventions`, `deploiement-nuit` devient `night-deploy`,
  `explique-moi` devient `explain-it-to-me`.

Noms anglais fixés (en/13-donnees-capacites.js), à citer tels quels dans la prose :

| Français | Anglais |
|---|---|
| conventions-gpu | gpu-conventions |
| deploiement-nuit | night-deploy |
| migration-as400 | as400-migration |
| revue-securite | security-review |
| redaction-specs | spec-writing |
| tests-mutation | mutation-testing |
| documentation-vivante | living-docs |
| explique-moi | explain-it-to-me |
| macros-paie | payroll-macros |
| firmware-prudent | careful-firmware |
| wordpress-sauveteur | wordpress-rescuer |
| orchestre-sessions | session-orchestrator |
| commits-soignes | tidy-commits |
| refactor-total | total-refactor |
| mcp-courriel-fournisseurs | mcp-supplier-email |
| mcp-telemetrie-iot | mcp-iot-telemetry |
| mcp-suite-bureautique | mcp-office-suite |
| mcp-billetterie | mcp-ticketing |
| mcp-gestion-projet | mcp-project-management |
| mcp-signature-electronique | mcp-e-signature |
| mcp-observabilite | mcp-observability |
| mcp-scraper-web | mcp-web-scraper |
| mcp-docs-bibliotheques | mcp-library-docs |
| github-mcp (officiel) | github-mcp (official) |

Noms inchangés : deploy, tdd-strict, ship-it, snapshot-tests-pro, super-productivity-pack, mcp-catalogue, mcp-as400,
mcp-sql-prod, mcp-crm, githbu-mcp, mcp-power-automate-pro.
