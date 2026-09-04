/* ======================================================================
   Du cep à la bouteille — parcours interactif de la fabrication du vin.
   Aucune dépendance. Trois styles (rouge, blanc, rosé) partagent les
   mêmes étapes de vigne, puis divergent au chai : la place du pressurage
   par rapport à la fermentation fait toute la différence.
   ====================================================================== */

/* ---------- Utilitaires ---------- */

const $ = (sel, racine = document) => racine.querySelector(sel);
const $$ = (sel, racine = document) => Array.from(racine.querySelectorAll(sel));
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const lerp = (a, b, t) => a + (b - a) * t;
const fmt = (v, d = 0) => v.toLocaleString('fr-CA', { minimumFractionDigits: d, maximumFractionDigits: d });
const CLE = 'vin-style';

function h(tag, attrs = {}, ...enfants) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  for (const e of enfants.flat()) {
    if (e === null || e === undefined) continue;
    el.append(e.nodeType ? e : document.createTextNode(e));
  }
  return el;
}

/* Prépare un canvas net sur écran haute densité et renvoie son contexte. */
function contexte(canvas, largeur, hauteur) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(largeur * dpr);
  canvas.height = Math.round(hauteur * dpr);
  canvas.style.aspectRatio = `${largeur} / ${hauteur}`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/* ---------- État global ---------- */

const NOMS = { rouge: 'Rouge', blanc: 'Blanc', rose: 'Rosé' };
const TITRES = { rouge: 'un vin rouge', blanc: 'un vin blanc', rose: 'un vin rosé' };

const ETAT = {
  style: 'rouge',
  sucre: 220,           // g/L au moment de la vendange, fixé par l'atelier « maturité »
  piedsHa: 5000,        // fixé par l'atelier « plantation »
};
try { if (NOMS[localStorage.getItem(CLE)]) ETAT.style = localStorage.getItem(CLE); } catch (e) { /* stockage indisponible */ }

/* Texte selon le style : une chaîne, ou un objet { rouge, blanc, rose, _ }. */
const t = (v) => (v && typeof v === 'object' && !Array.isArray(v)) ? (v[ETAT.style] ?? v._ ?? '') : v;

/* ---------- Les étapes ---------- */

const STADES = [
  {
    id: 'plantation',
    titre: 'Planter la vigne',
    duree: 'Année 0',
    intro: "Tout commence par un choix de terrain : le sol (drainant, plutôt pauvre), l'exposition, le climat. Puis un choix de plant. Depuis la crise du phylloxéra, à la fin du XIX<sup>e</sup> siècle, presque toute vigne européenne (<i>Vitis vinifera</i>) est <b>greffée</b> sur un porte-greffe américain dont les racines résistent au puceron. On plante au printemps de jeunes plants greffés-soudés, en rangs, puis on installe le palissage : piquets et fils qui guideront la végétation.",
    reperes: [
      "<b>Densité</b> : de 2 000 à 10 000 pieds à l'hectare. Serré à Bordeaux ou en Bourgogne, plus large dans les vignobles mécanisés.",
      "<b>Porte-greffe</b> : choisi pour le sol (calcaire, humide, sec) et la vigueur voulue. Le cépage ne pousse que sur le greffon.",
      "<b>Au Québec</b>, on plante surtout des hybrides rustiques (Frontenac, Marquette, Vidal, Seyval) qui tolèrent −30 °C, et on butte les autres.",
      "Une vigne bien conduite produit <b>30 à 50 ans</b>, parfois bien plus.",
    ],
    atelier: 'densite',
  },
  {
    id: 'jeunesse',
    titre: 'Les premières années',
    duree: 'Années 1 à 3',
    intro: "Une jeune vigne ne donne rien d'utile avant sa troisième feuille. Les deux premières années servent à <b>former le tronc</b> et le système racinaire : on taille court, on tuteure, on retire les grappes pour ne pas épuiser le plant. La première vraie vendange arrive la troisième année, modeste ; la pleine production vers cinq à sept ans. Les appellations imposent d'ailleurs souvent un âge minimal avant que le raisin ait droit au nom.",
    reperes: [
      "<b>Année 1</b> : le plant s'enracine, un seul rameau conservé, tuteuré.",
      "<b>Année 2</b> : formation du tronc et des bras (la « charpente ») selon le mode de taille choisi : Guyot, cordon de Royat, gobelet…",
      "<b>Année 3</b> : première récolte, souvent la moitié d'un rendement normal.",
      "<b>Vieilles vignes</b> : au-delà de 30 à 40 ans, le rendement décline mais les baies sont plus concentrées ; les racines profondes tamponnent la sécheresse.",
    ],
    atelier: 'age',
  },
  {
    id: 'cycle',
    titre: 'Une année dans la vigne',
    duree: 'Chaque année',
    intro: "La vigne est une liane à feuilles caduques : chaque année, elle refait tout à partir de rien. Elle dort l'hiver, pleure au dégel, débourre en avril, fleurit en juin, change de couleur en août et mûrit jusqu'aux vendanges. Le vigneron accompagne chaque phase : la <b>taille d'hiver</b> décide du nombre de grappes, les travaux en vert (épamprage, relevage, rognage, effeuillage) disciplinent la végétation, et les traitements protègent des champignons, mildiou et oïdium en tête. Déplacez le curseur pour suivre l'année.",
    reperes: [
      "<b>Règle des 100 jours</b> : de la mi-floraison à la vendange, il s'écoule en moyenne une centaine de jours.",
      "<b>Gel de printemps</b> : un bourgeon débourré meurt à −2 °C. D'où les bougies, les éoliennes et l'aspersion dans les vignes en avril.",
      "<b>Aoûtement</b> : en fin d'été, les rameaux verts se lignifient et deviennent le bois qui sera taillé l'hiver suivant.",
    ],
    atelier: 'cycle',
    large: true,
  },
  {
    id: 'maturite',
    titre: 'Décider de vendanger',
    duree: 'Août → octobre',
    intro: "Après la véraison, la baie se remplit de sucre et perd son acidité : le sucre passe d'une cinquantaine à plus de 220 g/L, l'acidité de près de 30 g/L à moins de 7. La date de vendange est la <b>décision la plus importante de l'année</b>. Trop tôt, le vin est vert et maigre ; trop tard, il est lourd, alcoolisé, mou. On suit la maturité en prélevant des baies et en mesurant sucre, acidité, pH, et pour les rouges la <b>maturité phénolique</b> : pépins bruns et croquants, peaux qui lâchent leur couleur.",
    reperes: [
      "<b>17 g/L de sucre</b> donnent environ 1 % d'alcool. Un moût à 220 g/L donne un vin à 13 %.",
      { rouge: "Pour un <b>rouge</b>, on attend que les tanins soient mûrs, souvent au-delà de la maturité « sucre ».", blanc: "Pour un <b>blanc</b>, on vendange plus tôt pour garder la fraîcheur et les arômes ; souvent de nuit, au frais.", rose: "Pour un <b>rosé</b>, on cherche l'acidité et le fruit : on vendange tôt, et souvent de nuit." },
      "Le <b>réchauffement climatique</b> avance les vendanges de deux à trois semaines par rapport aux années 1980, et le sucre monte avant que les arômes soient là.",
    ],
    atelier: 'maturite',
  },
  {
    id: 'vendanges',
    titre: 'Les vendanges',
    duree: 'Septembre → octobre',
    intro: "À la main, au sécateur, grappe par grappe dans des cagettes de 10 à 20 kg pour ne pas écraser le raisin ; ou à la machine, qui secoue les rangs et fait tomber les baies, de nuit s'il le faut. La vendange manuelle permet un premier tri sur pied ; la machine récolte un hectare en une heure. Dans les deux cas, il faut aller vite : un raisin cueilli s'oxyde et commence à fermenter tout seul. Au chai, une <b>table de tri</b> écarte feuilles, grappes pourries ou pas mûres.",
    reperes: [
      "<b>Rendement</b> : de 25 hL/ha pour un grand cru à plus de 100 hL/ha pour un vin de table. Les appellations fixent un plafond.",
      "Il faut environ <b>1,1 à 1,3 kg de raisin</b> pour une bouteille de 75 cL : un pied, une bouteille, en gros.",
      "Vendanger de <b>nuit</b> ou tôt le matin conserve les arômes et limite l'oxydation, surtout pour les blancs et rosés.",
    ],
    atelier: 'rendement',
  },
  {
    id: 'eraflage',
    styles: ['rouge'],
    titre: 'Éraflage, foulage, encuvage',
    duree: 'Jour de la vendange',
    intro: "Le raisin rouge n'est pas pressé tout de suite : c'est la peau qui contient la couleur et les tanins, et il faut la laisser macérer dans le jus. On <b>érafle</b> (on retire les rafles, ces tiges vertes qui donneraient de l'amertume), on <b>foule</b> légèrement pour faire éclater les baies, puis on met le tout, jus, pulpe, peaux et pépins, en cuve. Une pincée de soufre (SO<sub>2</sub>) protège le moût des bactéries et des oxydations, et on laisse les levures faire, indigènes ou ajoutées.",
    reperes: [
      "<b>Vendange entière</b> : certains gardent une part de rafles (Bourgogne, Rhône, Beaujolais) pour la fraîcheur et la structure.",
      "<b>Macération à froid</b> : quelques jours à 8 à 10 °C avant la fermentation, pour extraire le fruit et la couleur sans les tanins.",
      "<b>Levures</b> : indigènes (celles du raisin et du chai, moins prévisibles) ou sélectionnées (souche connue, départ rapide, degré alcoolique toléré plus élevé).",
    ],
  },
  {
    id: 'pressurage-blanc',
    styles: ['blanc'],
    titre: 'Pressurage direct',
    duree: 'Jour de la vendange',
    intro: "Pour un blanc, on presse tout de suite, avant toute fermentation : le jus seul fermentera, sans les peaux. Les grappes vont, entières ou éraflées, dans un <b>pressoir pneumatique</b> : une membrane se gonfle doucement contre les grappes, et le jus s'écoule par les grilles. Les premiers litres, le <b>jus de goutte</b>, sont les plus fins ; les dernières pressées sont plus tanniques et souvent écartées. Cent kilos de raisin donnent 60 à 70 litres de moût.",
    reperes: [
      "<b>Macération pelliculaire</b> : quelques heures de contact avec les peaux, à froid, avant de presser, pour les cépages aromatiques (sauvignon, gewurztraminer).",
      "<b>Blanc de noirs</b> : pressé sans macération, un raisin noir donne un jus presque incolore. C'est ainsi que le pinot noir devient du champagne blanc.",
      "Tout se fait <b>au frais et sous gaz inerte</b> pour éviter l'oxydation, qui brunit le jus et efface les arômes.",
    ],
  },
  {
    id: 'maceration-rose',
    styles: ['rose'],
    titre: 'Macération courte ou pressurage direct',
    duree: 'Quelques heures',
    intro: "Un rosé est un vin de raisins noirs qui a très peu vu ses peaux. Deux méthodes. Le <b>pressurage direct</b>, comme un blanc, mais la pression fait sortir un peu de couleur des peaux : c'est le rosé pâle de Provence. La <b>saignée</b> : on laisse macérer quelques heures à un jour en cuve, puis on tire (« on saigne ») une partie du jus, déjà rosé, qui fermente à part. Le reste, plus concentré, devient un rouge. Le rosé n'est jamais, en Europe, un mélange de rouge et de blanc, sauf en Champagne.",
    reperes: [
      "La <b>couleur</b> se règle à l'heure près : 2 heures de contact donnent un pelure d'oignon, 24 heures un rosé soutenu.",
      "Après la saignée ou le pressurage, le jus est traité <b>comme un blanc</b> : débourbage, fermentation au frais, pas ou peu de bois.",
    ],
  },
  {
    id: 'debourbage',
    styles: ['blanc', 'rose'],
    titre: 'Débourbage',
    duree: '12 à 24 heures',
    intro: "Le jus qui sort du pressoir est trouble : il charrie des débris de pulpe, de peau, de la terre. On le laisse reposer une nuit à 8 à 12 °C, assez froid pour que la fermentation ne démarre pas, et les <b>bourbes</b> tombent au fond. On soutire le jus clair par-dessus. Un jus trop débourbé fermente mal (les levures manquent de nutriments) ; pas assez, le vin prend des goûts herbacés et de réduction. Puis on remonte doucement la température et on ensemence les levures.",
    reperes: [
      "<b>Turbidité visée</b> : quelques dizaines à quelques centaines de NTU, mesurée au néphélomètre, selon le style recherché.",
      "Les bourbes ne sont pas perdues : filtrées, elles donnent un jus qui rejoint souvent des cuvées plus simples.",
    ],
  },
  {
    id: 'fermentation',
    titre: { rouge: 'Fermentation alcoolique et macération', _: 'Fermentation alcoolique' },
    duree: { rouge: '5 à 20 jours', blanc: '2 à 4 semaines', rose: '2 à 3 semaines' },
    intro: {
      rouge: "C'est ici que le jus devient vin. Les levures transforment le sucre en alcool et en gaz carbonique, en dégageant beaucoup de chaleur. Le gaz remonte les peaux en surface, où elles forment le <b>chapeau de marc</b>, épais et sec, qu'il faut sans cesse remouiller : par <b>remontage</b> (on pompe le jus du bas vers le haut), par <b>pigeage</b> (on enfonce le chapeau), ou par <b>délestage</b> (on vide la cuve et on la remplit). C'est ce contact qui extrait couleur, tanins et arômes. Tout se joue sur la température : 25 à 30 °C pour extraire, jamais au-delà de 35 °C, où les levures meurent.",
      blanc: "C'est ici que le jus devient vin. Les levures transforment le sucre en alcool et en gaz carbonique, en dégageant beaucoup de chaleur. Pour un blanc, on fermente <b>au frais</b>, entre 12 et 18 °C, en cuve inox thermorégulée ou en barrique : les arômes de fruit et de fleur, volatils, s'évaporent avec le gaz si la cuve chauffe. La fermentation est donc lente, deux à quatre semaines, et le chai sent la brioche et la pomme. Le simulateur montre ce qui arrive si l'on coupe le refroidissement.",
      rose: "C'est ici que le jus devient vin. Les levures transforment le sucre en alcool et en gaz carbonique, en dégageant beaucoup de chaleur. Un rosé se fermente comme un blanc, <b>au frais</b>, entre 14 et 18 °C, pour garder ses arômes de petits fruits et sa couleur fragile. Deux à trois semaines en cuve inox thermorégulée. Le simulateur montre ce qui arrive si l'on coupe le refroidissement.",
    },
    reperes: [
      "<b>C<sub>6</sub>H<sub>12</sub>O<sub>6</sub> → 2 C<sub>2</sub>H<sub>5</sub>OH + 2 CO<sub>2</sub></b>, plus de la chaleur : environ 100 kJ par mole de glucose. Chaque litre de moût dégage près de 50 L de gaz carbonique.",
      "La <b>densité</b> est l'instrument du vigneron : le moût sucré pèse 1,090 à 1,100 ; le vin sec, 0,990 à 0,996. On la mesure au mustimètre chaque matin.",
      { rouge: "La <b>couleur</b> s'extrait dans les trois à cinq premiers jours ; les <b>tanins</b> continuent tant que le vin macère, aidés par l'alcool. La durée de cuvaison règle la structure.", _: "Le <b>gaz carbonique</b> qui s'échappe protège le vin de l'air pendant la fermentation ; c'est après, quand il s'arrête, que le vin devient vulnérable." },
    ],
    atelier: 'fermentation',
    large: true,
  },
  {
    id: 'pressurage-rouge',
    styles: ['rouge'],
    titre: 'Décuvage et pressurage',
    duree: "À la fin de la macération",
    intro: "Quand le vigneron juge l'extraction suffisante, on <b>décuve</b> : le vin s'écoule par gravité, c'est le <b>vin de goutte</b>, le plus fin. Le marc (peaux et pépins gorgés de vin) est sorti à la pelle ou à la vis et passé au pressoir : c'est le <b>vin de presse</b>, plus tannique, plus coloré, qu'on gardera à part pour en ajouter, ou pas, à l'assemblage. Le marc épuisé part à la distillerie (marc, grappa) ou au compost.",
    reperes: [
      "Le vin de presse représente <b>10 à 15 %</b> du volume. Les premières pressées sont bonnes, les dernières dures et amères.",
      "Le vin qui sort est <b>trouble, chaud, gazeux</b> et encore plein de levures : il n'est pas fini.",
    ],
  },
  {
    id: 'malo',
    titre: 'Fermentation malolactique',
    duree: '2 à 8 semaines',
    intro: {
      rouge: "Une seconde fermentation, discrète, sans alcool cette fois. Des bactéries lactiques (<i>Oenococcus oeni</i>) transforment l'<b>acide malique</b>, dur et vert comme une pomme, en <b>acide lactique</b>, plus doux, en dégageant un peu de gaz. Le vin perd du mordant, gagne en rondeur et se stabilise : un rouge qui ne l'aurait pas faite pourrait la refaire en bouteille, avec du gaz et du trouble. Elle est <b>systématique</b> pour les rouges. Elle démarre spontanément vers 18 à 22 °C, ou avec un ensemencement.",
      blanc: "Une seconde fermentation, discrète, sans alcool cette fois. Des bactéries lactiques (<i>Oenococcus oeni</i>) transforment l'<b>acide malique</b>, dur et vert comme une pomme, en <b>acide lactique</b>, plus doux, en dégageant un peu de gaz. Pour un blanc, c'est un <b>choix de style</b> : oui pour un chardonnay bourguignon, qui gagne des notes de beurre et de noisette (le diacétyle) ; non pour un sauvignon ou un riesling, dont on veut garder la tension. On la bloque alors par le froid et le soufre.",
      rose: "Une seconde fermentation, discrète, sans alcool cette fois. Des bactéries lactiques transforment l'<b>acide malique</b>, dur et vert comme une pomme, en <b>acide lactique</b>, plus doux. Pour un rosé, on l'évite presque toujours : c'est l'acidité qui fait sa fraîcheur. On la bloque par le froid et une dose de soufre, puis on surveille.",
    },
    reperes: [
      "1 g de malique donne <b>0,67 g de lactique</b> : l'acidité totale baisse de 1 à 3 g/L et le pH monte de 0,1 à 0,3.",
      "Elle marche mal sous 15 °C, sous pH 3,1 ou au-delà de 14 % d'alcool : c'est pourquoi on chauffe les chais en novembre.",
    ],
    atelier: 'malo',
  },
  {
    id: 'elevage',
    titre: 'Élevage',
    duree: { rouge: '6 à 24 mois', blanc: '3 à 18 mois', rose: '2 à 6 mois' },
    intro: {
      rouge: "Le vin nouveau est brut, gazeux, anguleux. L'élevage, c'est le temps qu'on lui laisse pour se faire, et le contenant dans lequel on le laisse. En <b>barrique</b> de chêne (225 L à Bordeaux, 228 en Bourgogne), le bois cède ses arômes (vanille, épices, toasté) et laisse passer un filet d'oxygène qui assouplit les tanins et fixe la couleur. En <b>cuve inox</b>, rien n'entre et rien ne sort : le fruit reste intact. Entre les deux, le béton, les foudres, les amphores. On <b>soutire</b> régulièrement pour séparer le vin de ses lies, et on <b>ouille</b> : on complète ce que les anges ont bu.",
      blanc: "Le vin nouveau est brut, gazeux, un peu trouble. L'élevage, c'est le temps qu'on lui laisse pour se faire, et le contenant dans lequel on le laisse. Beaucoup de blancs restent en <b>cuve inox</b>, au frais, pour garder le fruit ; les grands chardonnays vont en <b>barrique</b>, où le bois cède ses arômes et un filet d'oxygène. Souvent on les laisse <b>sur lies</b> : les levures mortes, qu'on remet en suspension au bâton (le bâtonnage), donnent du gras et protègent de l'oxydation.",
      rose: "Un rosé s'élève peu : quelques mois en <b>cuve inox</b>, au frais, parfois sur lies fines pour le gras, et on le met en bouteille avant le printemps pour le vendre jeune. Le bois est rare, et discret quand il est là. L'atelier ci-dessous montre pourquoi une longue barrique ne lui irait pas.",
    },
    reperes: [
      "<b>Part des anges</b> : 2 à 5 % du volume s'évapore chaque année à travers le bois. On ouille toutes les semaines.",
      "Une barrique neuve coûte <b>900 à 1 200 $</b> et marque fortement le vin ; dès le troisième vin, elle n'apporte plus guère que l'oxygène.",
      "<b>Chauffe</b> : le tonnelier cintre les douelles au feu ; légère, moyenne ou forte, elle change les arômes du bois, de la vanille au café.",
    ],
    atelier: 'elevage',
    large: true,
  },
  {
    id: 'clarification',
    titre: 'Clarifier et stabiliser',
    duree: 'Avant la mise',
    intro: "Un vin doit arriver en bouteille limpide et sans surprise. Le <b>soutirage</b> a déjà retiré l'essentiel des dépôts. Le <b>collage</b> précipite ce qui reste en suspension : on ajoute une protéine (blanc d'œuf, pois, colle de poisson) ou une argile (bentonite) qui agglomère les particules et tombe au fond. La <b>filtration</b>, sur plaques ou membranes, finit le travail et retire les levures et bactéries. Enfin, la <b>stabilisation tartrique</b> : quelques jours au froid, vers −4 °C, font cristalliser le tartre, faute de quoi il le ferait plus tard dans la bouteille.",
    reperes: [
      "Les cristaux de tartre dans une bouteille sont <b>inoffensifs</b> : c'est le même acide tartrique que dans le raisin.",
      "Beaucoup de vignerons filtrent peu ou pas : un vin « non filtré » peut déposer, mais garde plus de matière.",
      "Le <b>soufre</b> (SO<sub>2</sub>) est ajusté à chaque étape : il bloque les oxydations et les microbes. Un vin « sans soufre ajouté » en contient quand même un peu, produit par les levures.",
    ],
  },
  {
    id: 'assemblage',
    titre: "L'assemblage",
    duree: 'Quelques jours de dégustation',
    intro: {
      rouge: "Un domaine n'a jamais une seule cuve : il en a une par parcelle, par cépage, par date de vendange, puis des barriques neuves et d'autres usagées, du vin de goutte et du vin de presse. L'<b>assemblage</b> est le moment où l'on goûte tout et où l'on compose le vin final. À Bordeaux, c'est le mariage du merlot et du cabernet ; en Bourgogne, un seul cépage, mais on décide quelle barrique entre dans la cuvée et laquelle part dans le second vin.",
      blanc: "Un domaine n'a jamais une seule cuve : il en a une par parcelle, par cépage, par pressée, en inox et en barrique. L'<b>assemblage</b> est le moment où l'on goûte tout et où l'on compose le vin final : la cuve inox pour la fraîcheur, la barrique pour le gras, le jus de presse ou pas. C'est un exercice de dégustation à l'aveugle, avec des éprouvettes et des proportions.",
      rose: "Même pour un rosé, on assemble : la cuvée de saignée avec celle de pressurage direct, le grenache avec le cinsault, une cuve plus colorée avec une plus pâle, pour arriver à la teinte et à l'équilibre voulus. L'<b>assemblage</b> se fait à la dégustation, avec des éprouvettes et des proportions.",
    },
    reperes: [
      "Les proportions se testent en <b>éprouvette</b> avant de toucher aux cuves ; une cuvée se compose à 1 % près.",
      "Le <b>second vin</b> d'un château, c'est ce qui n'est pas entré dans le premier : jeunes vignes, barriques moins réussies.",
    ],
  },
  {
    id: 'mise',
    titre: 'La mise en bouteille',
    duree: 'Une journée',
    intro: "Le vin assemblé, stabilisé, dosé en soufre, passe sur la chaîne de mise : rinçage des bouteilles, souvent inertage à l'azote, remplissage à niveau, <b>bouchage</b>, capsule, étiquette. Le choix du bouchon décide de la quantité d'oxygène qui entrera dans les années suivantes, donc du vieillissement. Le liège naturel respire un peu et varie d'un bouchon à l'autre ; la capsule à vis est régulière et étanche ; le liège aggloméré traité se situe entre les deux. Une bouteille mal bouchée s'oxyde ; un bouchon contaminé par le TCA donne le fameux goût de bouchon.",
    reperes: [
      "<b>SO<sub>2</sub> libre</b> à la mise : 25 à 40 mg/L pour un blanc, 20 à 30 pour un rouge, ajusté au pH. C'est la fraction moléculaire, quelques dixièmes de mg/L, qui protège vraiment.",
      "Le <b>goût de bouchon</b> (TCA) touchait 3 à 5 % des bouteilles il y a vingt ans ; le tri des lièges l'a ramené sous 1 %.",
      "Après la mise, le vin est <b>choqué</b> : on le laisse reposer quelques semaines avant de le vendre.",
    ],
    atelier: 'bouchage',
  },
  {
    id: 'bouteille',
    titre: 'Dans la bouteille',
    duree: 'Des mois aux décennies',
    intro: {
      rouge: "Le vin n'est pas mort en bouteille : il continue d'évoluer, lentement, à l'abri de l'air. Les tanins s'assemblent en longues chaînes et deviennent soyeux, la couleur passe du violet au rubis puis au grenat et à l'orangé, un dépôt se forme. Les arômes changent de registre : le fruit frais (<b>arômes primaires</b>) et les notes d'élevage (<b>secondaires</b>) laissent la place aux <b>tertiaires</b> : sous-bois, cuir, tabac, truffe. Tout dépend du vin de départ : la plupart des rouges sont faits pour cinq ans, quelques-uns pour cinquante.",
      blanc: "Le vin n'est pas mort en bouteille : il continue d'évoluer, lentement, à l'abri de l'air. La couleur, presque incolore au départ, tourne à l'or puis à l'ambre. Les arômes changent de registre : les fleurs et les agrumes (<b>arômes primaires</b>) laissent la place aux notes de miel, de noix, de cire, de pétrole pour un riesling (<b>tertiaires</b>). L'acidité et le sucre sont les garants de la garde : la plupart des blancs secs se boivent dans les trois ans, les grands rieslings, chenins et chardonnays dans les vingt.",
      rose: "Un rosé est un vin de l'année : sa couleur pâlit et vire à l'orangé, ses arômes de petits fruits s'éteignent en deux ou trois ans. Il n'y a rien à gagner à l'attendre. On le garde au frais et à l'ombre, on l'ouvre l'été qui suit, et on recommence l'année d'après.",
    },
    reperes: [
      "<b>Conditions</b> : 12 à 14 °C constants, obscurité, couchée si liège, un peu d'humidité. Les chocs de température vieillissent plus vite que les années.",
      "L'<b>apogée</b> n'est pas un point mais une fenêtre, parfois de plusieurs années ; après, le vin décline sans mourir d'un coup.",
    ],
    atelier: 'garde',
  },
];

/* ---------- Rendu du parcours ---------- */

const ATELIERS = {};   // rempli plus bas : nom → fonction(conteneur) → api
const API = { etat: ETAT, ateliers: {} };

function stadesVisibles() {
  return STADES.filter((s) => !s.styles || s.styles.includes(ETAT.style));
}

function rendre() {
  document.documentElement.dataset.style = ETAT.style;
  $('#etiquetteStyle').textContent = NOMS[ETAT.style];
  $('#titreStyle').textContent = TITRES[ETAT.style];
  $('#resumeDuree').textContent = ETAT.style === 'rouge' ? '4 à 6 ans' : ETAT.style === 'blanc' ? '4 à 5 ans' : '3 ans et demi';
  $$('#choixStyle button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.style === ETAT.style)));

  const liste = $('#etapes');
  const conteneur = $('#stades');
  liste.replaceChildren();
  conteneur.replaceChildren();
  API.ateliers = {};

  stadesVisibles().forEach((s, i) => {
    const num = String(i + 1).padStart(2, '0');
    liste.append(h('li', {}, h('a', { href: `#s-${s.id}`, 'data-etape': s.id }, h('span', { class: 'n' }, num), t(s.titre))));

    const section = h('section', { class: 'stade', id: `s-${s.id}`, 'data-stade': s.id },
      h('div', { class: 'stade-tete' },
        h('span', { class: 'num' }, `Étape ${num}`),
        h('h2', {}, t(s.titre)),
        h('span', { class: 'duree' }, t(s.duree))));
    const texte = h('div', { class: 'texte' },
      h('p', { class: 'intro', html: t(s.intro) }),
      h('ul', { class: 'reperes' }, s.reperes.map((r) => h('li', { html: t(r) }))));
    const corps = h('div', { class: 'stade-corps' + (s.large ? ' large' : '') }, texte);
    if (s.atelier && ATELIERS[s.atelier]) {
      const boite = h('div', { class: 'atelier', 'data-atelier': s.atelier });
      corps.append(boite);
      API.ateliers[s.atelier] = ATELIERS[s.atelier](boite);
    }
    section.append(corps);
    conteneur.append(section);
  });

  observerEtapes();
}

let observateur = null;
function observerEtapes() {
  if (observateur) observateur.disconnect();
  const liens = new Map($$('#etapes a').map((a) => [a.dataset.etape, a]));
  const visibles = new Set();
  observateur = new IntersectionObserver((entrees) => {
    for (const e of entrees) {
      if (e.isIntersecting) visibles.add(e.target.dataset.stade);
      else visibles.delete(e.target.dataset.stade);
    }
    const ordre = stadesVisibles().map((s) => s.id);
    const courant = ordre.find((id) => visibles.has(id));
    liens.forEach((a, id) => a.classList.toggle('actif', id === courant));
    if (courant) {
      const a = liens.get(courant);
      const nav = a.closest('ol');
      const r = a.getBoundingClientRect(), rn = nav.getBoundingClientRect();
      if (r.left < rn.left || r.right > rn.right) nav.scrollTo({ left: a.offsetLeft - 40, behavior: 'smooth' });
    }
  }, { rootMargin: '-120px 0px -55% 0px' });
  $$('.stade').forEach((s) => observateur.observe(s));
}

function choisirStyle(style) {
  if (!NOMS[style] || style === ETAT.style) return;
  ETAT.style = style;
  try { localStorage.setItem(CLE, style); } catch (e) { /* stockage indisponible */ }
  rendre();
}

function atelierEntete(boite, marque, titre) {
  boite.append(h('div', { class: 'atelier-tete' }, h('span', { class: 'marque' }, marque), h('h3', {}, titre)));
}

/* Un curseur étiqueté ; renvoie l'input. */
function curseur(parent, { id, label, min, max, step = 1, valeur, unite = '', affiche }) {
  const sortie = h('output', { for: id });
  const input = h('input', { type: 'range', id, min, max, step, value: valeur });
  const maj = () => { sortie.textContent = affiche ? affiche(+input.value) : `${fmt(+input.value)}${unite}`; };
  input.addEventListener('input', maj);
  maj();
  parent.append(h('div', { class: 'reglage' }, h('label', { for: id }, h('span', {}, h('span', {}, label), sortie)), input));
  input.majSortie = maj;
  return input;
}

function selection(parent, { id, label, options, valeur }) {
  const sel = h('select', { id }, options.map(([v, l]) => h('option', { value: v, selected: v === valeur ? '' : null }, l)));
  parent.append(h('div', { class: 'reglage' }, h('label', { for: id }, h('span', {}, h('span', {}, label))), sel));
  return sel;
}

function jauge(parent, label, classe = '') {
  const barre = h('i', { class: classe });
  const sortie = h('output');
  parent.append(h('div', { class: 'jauge' }, h('span', {}, label), h('div', { class: 'piste' }, barre), sortie));
  return (valeur, texte) => { barre.style.width = `${clamp(valeur, 0, 100)}%`; sortie.textContent = texte ?? `${fmt(valeur)} %`; };
}

/* ======================================================================
   Ateliers
   ====================================================================== */

/* ---------- Plantation : densité ---------- */

ATELIERS.densite = (boite) => {
  atelierEntete(boite, 'Atelier', 'Choisir la densité de plantation');
  const canvas = h('canvas', { 'aria-label': 'Vue du dessus d’une parcelle : chaque point est un cep' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const rang = curseur(reglages, { id: 'dRang', label: 'Entre les rangs', min: 1, max: 3, step: 0.1, valeur: 2, affiche: (v) => `${fmt(v, 1)} m` });
  const cep = curseur(reglages, { id: 'dCep', label: 'Entre les ceps', min: 0.8, max: 2, step: 0.1, valeur: 1, affiche: (v) => `${fmt(v, 1)} m` });
  const mesures = h('dl', { class: 'mesures' });
  const dDens = h('dd', { class: 'grand', 'data-testid': 'densite' });
  const dSurf = h('dd');
  const dTracteur = h('dd');
  mesures.append(h('dt', {}, "Pieds à l'hectare"), dDens, h('dt', {}, 'Surface par cep'), dSurf, h('dt', {}, 'Machine adaptée'), dTracteur);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture' });
  boite.append(lecture);

  function dessiner() {
    const L = 520, H = 200;
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#2b1e14';
    ctx.fillRect(0, 0, L, H);
    // 20 m × 8 m de parcelle, vue du dessus, 26 px par mètre
    const px = 26;
    const er = +rang.value, ec = +cep.value;
    ctx.strokeStyle = '#3d2b1c';
    ctx.lineWidth = px * 0.5;
    for (let x = er / 2; x < 20; x += er) {
      ctx.beginPath(); ctx.moveTo(x * px, 0); ctx.lineTo(x * px, H); ctx.stroke();
    }
    ctx.strokeStyle = '#6a5a3a';
    ctx.lineWidth = 1;
    for (let x = er / 2; x < 20; x += er) {
      ctx.beginPath(); ctx.moveTo(x * px, 0); ctx.lineTo(x * px, H); ctx.stroke();
      for (let y = ec / 2; y < H / px; y += ec) {
        ctx.fillStyle = '#7fae55';
        ctx.beginPath(); ctx.arc(x * px, y * px, Math.min(6, ec * px * 0.32), 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#4b3a25';
        ctx.beginPath(); ctx.arc(x * px, y * px, 1.5, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.fillStyle = 'rgba(243,235,228,.7)';
    ctx.font = '500 10px JetBrains Mono, monospace';
    ctx.fillText('20 m ×  8 m', 8, H - 8);
  }
  function maj() {
    const er = +rang.value, ec = +cep.value;
    const dens = 10000 / (er * ec);
    ETAT.piedsHa = Math.round(dens);
    dDens.textContent = fmt(dens);
    dSurf.textContent = `${fmt(er * ec, 2)} m²`;
    dTracteur.textContent = er < 1.5 ? 'Enjambeur ou cheval' : er < 2.2 ? 'Tracteur vigneron étroit' : 'Tracteur standard';
    let l;
    if (dens >= 8000) l = "<b>Vigne serrée</b>, à la bourguignonne ou bordelaise : chaque cep porte peu de grappes, les racines se concurrencent et plongent, les baies sont plus concentrées. Mais rien ne passe entre les rangs : il faut un tracteur enjambeur et beaucoup de main-d'œuvre. Une plantation dense coûte deux fois plus de plants.";
    else if (dens >= 4500) l = "<b>Densité moyenne</b>, la plus courante en Europe : bon compromis entre concentration, mécanisation et coût. Un tracteur vigneron passe entre les rangs.";
    else if (dens >= 2500) l = "<b>Vigne large</b>, typique des vignobles du Nouveau Monde et des hybrides québécois : chaque cep est vigoureux et porte beaucoup, la mécanisation est facile, la vendangeuse passe partout. Le rendement par cep est élevé, la concentration moindre.";
    else l = "<b>Très large</b> : c'est presque un verger. On ne trouve ces densités qu'en climat très sec, où chaque cep doit disposer de beaucoup de sol pour trouver son eau.";
    lecture.innerHTML = `${l} Pour rappel, un hectare fait 100 m × 100 m ; ici, ${fmt(dens)} pieds.`;
    dessiner();
    API.ateliers.rendement?.maj?.();
  }
  rang.addEventListener('input', maj);
  cep.addEventListener('input', maj);
  maj();
  return { maj };
};

/* ---------- Jeunesse : rendement selon l'âge ---------- */

function rendementAge(a) {
  // Fraction du rendement de pleine production selon l'âge du cep.
  if (a < 3) return 0;
  if (a < 7) return lerp(0.45, 1, (a - 3) / 4);
  if (a < 30) return 1;
  return Math.max(0.35, 1 - (a - 30) * 0.02);
}

ATELIERS.age = (boite) => {
  atelierEntete(boite, 'Atelier', "Ce qu'un cep donne selon son âge");
  const canvas = h('canvas', { 'aria-label': 'Courbe du rendement selon l’âge de la vigne' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages une' });
  boite.append(reglages);
  const age = curseur(reglages, { id: 'aAge', label: 'Âge de la vigne', min: 0, max: 60, valeur: 3, affiche: (v) => v === 0 ? 'plantation' : `${v} an${v > 1 ? 's' : ''}` });
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-age' });
  boite.append(lecture);

  function dessiner() {
    const L = 520, H = 190, m = { g: 36, d: 12, h: 12, b: 26 };
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#120c0e'; ctx.fillRect(0, 0, L, H);
    const X = (a) => m.g + (a / 60) * (L - m.g - m.d);
    const Y = (r) => H - m.b - r * (H - m.h - m.b);
    ctx.strokeStyle = '#2f2226'; ctx.lineWidth = 1;
    for (let r = 0; r <= 1; r += 0.25) { ctx.beginPath(); ctx.moveTo(m.g, Y(r)); ctx.lineTo(L - m.d, Y(r)); ctx.stroke(); }
    ctx.fillStyle = '#a89a95'; ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.textAlign = 'right';
    for (let r = 0; r <= 1; r += 0.5) ctx.fillText(`${r * 100} %`, m.g - 5, Y(r) + 3);
    ctx.textAlign = 'center';
    for (let a = 0; a <= 60; a += 10) ctx.fillText(`${a} ans`, X(a), H - 8);
    // concentration (qualitatif) en pointillé
    ctx.setLineDash([3, 4]); ctx.strokeStyle = '#d9b25f'; ctx.beginPath();
    for (let a = 0; a <= 60; a += 0.5) {
      const c = a < 3 ? 0 : clamp(0.55 + (a - 3) * 0.012, 0, 1);
      a === 0 ? ctx.moveTo(X(a), Y(c)) : ctx.lineTo(X(a), Y(c));
    }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#d4577a';
    ctx.lineWidth = 2.5; ctx.beginPath();
    for (let a = 0; a <= 60; a += 0.5) { const p = [X(a), Y(rendementAge(a))]; a === 0 ? ctx.moveTo(...p) : ctx.lineTo(...p); }
    ctx.stroke();
    const a = +age.value;
    ctx.fillStyle = '#f3ebe4'; ctx.beginPath(); ctx.arc(X(a), Y(rendementAge(a)), 5, 0, Math.PI * 2); ctx.fill();
    ctx.textAlign = 'left'; ctx.fillStyle = '#d9b25f'; ctx.fillText('- - concentration des baies', m.g + 6, m.h + 10);
    ctx.fillStyle = '#f3ebe4'; ctx.fillText('— rendement', m.g + 6, m.h + 22);
  }
  function maj() {
    const a = +age.value;
    let l;
    if (a === 0) l = "<b>Plantation.</b> Un plant greffé-soudé de 30 cm, planté au printemps, arrosé s'il le faut. Aucune récolte.";
    else if (a === 1) l = "<b>Première feuille.</b> Le plant s'enracine ; on garde un seul rameau, tuteuré, et on retire toute grappe. Aucune récolte.";
    else if (a === 2) l = "<b>Deuxième feuille.</b> On forme le tronc et la charpente par la taille. Les quelques grappes sont supprimées pour ne pas épuiser le cep.";
    else if (a === 3) l = `<b>Troisième feuille : première vendange</b>, à peu près ${fmt(rendementAge(a) * 100)} % d'une récolte normale. Beaucoup d'appellations n'acceptent le raisin qu'à partir de cet âge.`;
    else if (a < 7) l = `<b>Jeune vigne</b> en montée de production : ${fmt(rendementAge(a) * 100)} % du plein rendement. Les vins sont fruités, simples, un peu dilués.`;
    else if (a < 30) l = "<b>Pleine production.</b> Le cep est formé, ses racines descendent de plusieurs mètres, le rendement est régulier d'une année à l'autre, aux caprices du millésime près.";
    else if (a < 45) l = `<b>Vigne mûre</b> : le rendement fléchit (${fmt(rendementAge(a) * 100)} %) mais les baies, moins nombreuses, sont plus concentrées. C'est l'âge où la mention « vieilles vignes » apparaît sur l'étiquette, sans aucune règle légale.`;
    else l = `<b>Vieille vigne</b> : ${fmt(rendementAge(a) * 100)} % du rendement, des ceps manquants qu'on remplace un à un, des racines profondes qui ignorent la sécheresse. Un vigneron arrache en général entre 40 et 60 ans, quand le rendement ne paie plus le travail ; certains ceps centenaires produisent encore.`;
    lecture.innerHTML = l;
    dessiner();
  }
  age.addEventListener('input', maj);
  maj();
  return { maj };
};

/* ---------- Cycle annuel ---------- */

const MOIS = ['janv.', 'févr.', 'mars', 'avril', 'mai', 'juin', 'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.'];
const JOURS_MOIS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
function dateDuJour(j) {
  let m = 0;
  while (m < 11 && j > JOURS_MOIS[m]) { j -= JOURS_MOIS[m]; m++; }
  return `${j} ${MOIS[m]}`;
}
const D = (m, j) => JOURS_MOIS.slice(0, m - 1).reduce((a, b) => a + b, 0) + j;   // jour de l'année

/* Phases du cycle, hémisphère Nord tempéré ; décalage pour le Québec. */
function phases(quebec) {
  const q = quebec;
  return [
    { fin: q ? D(3, 31) : D(3, 10), nom: 'Dormance', vigne: "La vigne dort. Ses réserves sont dans le tronc et les racines ; les bourgeons de l'an prochain, formés l'été dernier, attendent sous leurs écailles. Le bois résiste à −15 °C environ" + (q ? ", jusqu'à −30 °C pour les hybrides rustiques ; les vinifera buttés dorment sous la terre et la neige." : "."), vigneron: q ? "Taille d'hiver dès que le froid le permet, souvent en mars sur la neige. Entretien du palissage, réparation des fils." : "Taille d'hiver : on retire 80 à 90 % du bois de l'année et on choisit les bourgeons qui donneront les rameaux. C'est la décision qui fixe le rendement. Entretien du palissage." },
    { fin: q ? D(5, 5) : D(4, 5), nom: 'Pleurs', vigne: "Le sol se réchauffe, les racines repompent l'eau et la sève monte : elle coule aux plaies de taille, goutte à goutte. C'est le signal du réveil.", vigneron: q ? "Débuttage : on retire la terre qui protégeait le point de greffe. Fin de taille, pliage et attachage des baguettes sur le fil." : "Fin de taille, pliage et attachage des baguettes sur le fil de palissage. Premiers labours." },
    { fin: q ? D(5, 25) : D(4, 28), nom: 'Débourrement', vigne: "Les bourgeons gonflent, s'ouvrent en bourre cotonneuse, puis en petites feuilles. Le cep est alors très vulnérable : un gel à −2 °C tue les jeunes pousses.", vigneron: "Veille de gel : bougies allumées la nuit, éoliennes qui brassent l'air, aspersion d'eau qui gaine les bourgeons de glace protectrice. Ébourgeonnage des pousses en trop." },
    { fin: q ? D(6, 25) : D(6, 5), nom: 'Croissance', vigne: "Les rameaux s'allongent de plusieurs centimètres par jour, les feuilles se déploient, les vrilles s'accrochent, les grappes en boutons apparaissent. La vigne fabrique son usine à sucre.", vigneron: "Épamprage (retrait des gourmands sur le tronc), relevage des rameaux entre les fils, premiers traitements contre le mildiou et l'oïdium dès que l'humidité menace." },
    { fin: q ? D(7, 8) : D(6, 20), nom: 'Floraison', vigne: "Des fleurs minuscules, sans pétales, qui s'autofécondent. Dix jours de beau temps sont décisifs : sous la pluie, les fleurs coulent (coulure) ou donnent des baies sans pépin (millerandage). Le compte à rebours des 100 jours commence.", vigneron: "On regarde le ciel. Relevage et palissage continuent ; on évite de traiter en pleine fleur." },
    { fin: q ? D(8, 20) : D(7, 28), nom: 'Nouaison et fermeture', vigne: "Les fleurs fécondées deviennent des baies vertes, dures, acides, qui grossissent par division cellulaire puis par grossissement. La grappe se referme sur elle-même. Les rameaux atteignent leur longueur maximale.", vigneron: "Rognage (on coupe le haut des rameaux), effeuillage côté soleil levant pour aérer les grappes, vendange verte si la charge est trop lourde. Derniers traitements." },
    { fin: q ? D(9, 5) : D(8, 18), nom: 'Véraison', vigne: "Les baies se ramollissent et changent de couleur : rouges pour les cépages noirs, translucides et dorées pour les blancs. Le sucre entre, l'acidité sort. En même temps, les rameaux se lignifient : c'est l'aoûtement.", vigneron: "Filets contre les oiseaux et la grêle, contrôles de maturité au réfractomètre chaque semaine. On prépare le chai." },
    { fin: q ? D(10, 15) : D(10, 10), nom: 'Maturation et vendanges', vigne: "La baie se remplit de sucre, perd son acide malique, développe ses arômes ; les pépins brunissent, les peaux s'assouplissent. Puis la vigne met ses réserves dans le bois et les racines pour l'hiver.", vigneron: "Vendanges, à la main ou à la machine, cépage par cépage, parcelle par parcelle, selon la maturité. Les journées sont longues." },
    { fin: q ? D(11, 20) : D(11, 25), nom: 'Chute des feuilles', vigne: "Les feuilles jaunissent (blancs) ou rougissent (rouges), et tombent aux premières gelées. Le cep entre en dormance ; ses bourgeons de l'année prochaine sont prêts.", vigneron: q ? "Buttage : on relève la terre sur le pied des vinifera et des hybrides fragiles, pour couvrir le point de greffe avant les grands froids. Fumure, labour, prétaillage." : "Fumure, labour ou semis d'engrais vert entre les rangs. Prétaillage mécanique des rameaux longs." },
    { fin: 366, nom: 'Dormance', vigne: "La vigne dort. Ses réserves sont dans le tronc et les racines ; les bourgeons attendent." + (q ? " Sous la neige, la température au pied reste proche de 0 °C, quoi qu'il fasse dehors." : ""), vigneron: q ? "Repos, entretien du matériel. La taille attendra le cœur de l'hiver ou mars." : "Début de la taille d'hiver, en décembre pour les grands domaines, sinon de janvier à mars." },
  ];
}

ATELIERS.cycle = (boite) => {
  atelierEntete(boite, 'Atelier', "Suivre la vigne au fil de l'année");
  const canvas = h('canvas', { 'aria-label': 'Dessin d’un cep de vigne selon la saison' });
  const colG = h('div', {}, canvas);
  const colD = h('div');
  boite.append(h('div', { class: 'double' }, colG, colD));
  const reglages = h('div', { class: 'reglages une' });
  colG.append(reglages);
  const jour = curseur(reglages, { id: 'cJour', label: 'Date', min: 1, max: 365, valeur: D(8, 10), affiche: dateDuJour });
  const quebec = h('input', { type: 'checkbox', id: 'cQuebec' });
  colG.append(h('label', { class: 'case' }, quebec, 'Vignoble québécois (saison courte, buttage, hybrides)'));
  const lecture = h('p', { class: 'boutons' });
  const btnLecture = h('button', { class: 'bouton', type: 'button' }, '▶ Dérouler l’année');
  lecture.append(btnLecture);
  colG.append(lecture);

  const phaseNom = h('h4', { style: 'font:400 22px/1.1 var(--serif);margin:0 0 6px', 'data-testid': 'phase' });
  const phaseBarre = h('div', { class: 'phases' });
  const pVigne = h('p', { class: 'lecture', style: 'margin-top:10px' });
  const pVigneron = h('p', { class: 'lecture' });
  colD.append(phaseNom, phaseBarre, pVigne, pVigneron);

  let anim = null;
  function phaseCourante(j, q) {
    const ph = phases(q);
    return ph.find((p) => j <= p.fin) || ph[ph.length - 1];
  }

  function dessiner() {
    const j = +jour.value, q = quebec.checked;
    const L = 480, H = 300;
    const ctx = contexte(canvas, L, H);
    // repères phénologiques (jour de l'année)
    const deb = q ? D(5, 10) : D(4, 10), flo = q ? D(6, 28) : D(6, 12), ver = q ? D(8, 25) : D(8, 5);
    const ven = q ? D(10, 5) : D(9, 25), chute = q ? D(11, 5) : D(11, 15);
    const hiver = j < deb - 20 || j > chute;
    const froid = q && (j < D(4, 15) || j > D(11, 20));
    // ciel
    const ciel = ctx.createLinearGradient(0, 0, 0, H);
    if (hiver) { ciel.addColorStop(0, '#2a2f3d'); ciel.addColorStop(1, '#4b5262'); }
    else if (j > ver) { ciel.addColorStop(0, '#3a2f3d'); ciel.addColorStop(1, '#b98e5a'); }
    else { ciel.addColorStop(0, '#2e4a6b'); ciel.addColorStop(1, '#8fb2cf'); }
    ctx.fillStyle = ciel; ctx.fillRect(0, 0, L, H);
    // soleil
    const hauteurSoleil = 0.5 - 0.5 * Math.cos((j / 365) * Math.PI * 2 - Math.PI);   // 0 en hiver, 1 en été
    ctx.fillStyle = hiver ? 'rgba(255,235,200,.35)' : 'rgba(255,225,150,.85)';
    ctx.beginPath(); ctx.arc(L - 70, 90 - hauteurSoleil * 60, 22, 0, Math.PI * 2); ctx.fill();
    // sol
    const ySol = 236;
    ctx.fillStyle = froid ? '#e9edf2' : hiver ? '#4a3a2c' : '#5a4632';
    ctx.fillRect(0, ySol, L, H - ySol);
    ctx.fillStyle = froid ? '#d5dbe3' : '#3a2c20';
    ctx.fillRect(0, ySol + 26, L, H - ySol - 26);
    // piquets et fils
    ctx.strokeStyle = '#8b8378'; ctx.lineWidth = 2;
    for (const x of [40, L - 40]) { ctx.beginPath(); ctx.moveTo(x, ySol); ctx.lineTo(x, 70); ctx.stroke(); }
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(200,190,170,.6)';
    for (const y of [95, 130, 170]) { ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(L - 40, y); ctx.stroke(); }
    // butte (Québec, de novembre à début mai)
    const butte = q && (j > D(11, 1) || j < D(5, 1));
    if (butte) {
      ctx.fillStyle = '#5e4632';
      ctx.beginPath(); ctx.moveTo(L / 2 - 60, ySol); ctx.quadraticCurveTo(L / 2, ySol - 46, L / 2 + 60, ySol); ctx.fill();
      if (froid) { ctx.fillStyle = '#e9edf2'; ctx.beginPath(); ctx.moveTo(L / 2 - 58, ySol - 2); ctx.quadraticCurveTo(L / 2, ySol - 52, L / 2 + 58, ySol - 2); ctx.quadraticCurveTo(L / 2, ySol - 40, L / 2 - 58, ySol - 2); ctx.fill(); }
    }
    // tronc et bras (cordon de Royat)
    const x0 = L / 2, yBras = 172;
    ctx.strokeStyle = '#5a4030'; ctx.lineCap = 'round'; ctx.lineWidth = 9;
    ctx.beginPath(); ctx.moveTo(x0, ySol + 4); ctx.quadraticCurveTo(x0 - 4, 210, x0, yBras); ctx.stroke();
    ctx.lineWidth = 6;
    ctx.beginPath(); ctx.moveTo(x0, yBras); ctx.lineTo(x0 - 110, yBras - 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0, yBras); ctx.lineTo(x0 + 110, yBras - 2); ctx.stroke();
    // rameaux : longueur selon la saison
    let pousse = 0;      // 0..1
    if (j >= deb && j < chute) pousse = clamp((j - deb) / (ver - deb), 0.04, 1);
    else if (j >= chute) pousse = 1;
    let feuillage = 0;   // densité de feuilles
    if (j >= deb && j < chute) feuillage = clamp((j - deb) / 50, 0.05, 1);
    const automne = j > ven ? clamp((j - ven) / (chute - ven), 0, 1) : 0;
    const boisNouveau = j >= chute || j < deb;   // sarments lignifiés, avant taille
    const taille = q ? j >= D(3, 1) && j < deb : j >= D(1, 5) && j < deb;   // sarments coupés
    const positions = [-95, -70, -45, -20, 20, 45, 70, 95];
    positions.forEach((dx, i) => {
      if (boisNouveau && taille) {      // coursons taillés
        ctx.strokeStyle = '#6d4d38'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(x0 + dx, yBras); ctx.lineTo(x0 + dx + 3, yBras - 12); ctx.stroke();
        return;
      }
      const long = boisNouveau ? 1 : pousse;
      if (long <= 0) { ctx.fillStyle = '#7a5a45'; ctx.beginPath(); ctx.arc(x0 + dx, yBras - 3, 2.5, 0, Math.PI * 2); ctx.fill(); return; }
      const haut = yBras - long * (105 + (i % 3) * 8);
      const courbe = (i % 2 ? 1 : -1) * 10 * long;
      ctx.strokeStyle = boisNouveau || j > D(9, 1) ? '#8a6b4a' : '#6e9a4a'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x0 + dx, yBras); ctx.quadraticCurveTo(x0 + dx + courbe, (yBras + haut) / 2, x0 + dx + courbe / 2, haut); ctx.stroke();
      // feuilles
      if (feuillage > 0) {
        const n = Math.round(feuillage * 5);
        for (let k = 1; k <= n; k++) {
          const tt = k / (n + 1);
          const lx = x0 + dx + courbe * tt * 0.9 + (k % 2 ? 9 : -9);
          const ly = yBras + (haut - yBras) * tt;
          const r = 7 + feuillage * 4;
          const vert = automne > 0 ? (ETAT.style === 'rouge' ? `rgba(${Math.round(lerp(90, 190, automne))},${Math.round(lerp(150, 70, automne))},40,.95)` : `rgba(${Math.round(lerp(90, 210, automne))},${Math.round(lerp(150, 170, automne))},40,.95)`) : 'rgba(95,155,55,.95)';
          ctx.fillStyle = vert;
          ctx.beginPath(); ctx.ellipse(lx, ly, r, r * 0.75, (k % 2 ? 0.5 : -0.5), 0, Math.PI * 2); ctx.fill();
        }
      }
      // grappe
      if (j >= flo - 5 && j < ven + 2 && (i % 2 === 0)) {
        const gy = yBras - 8;
        const taille = j < flo ? 0.35 : clamp(0.45 + (j - flo) / (ver - flo) * 0.55, 0.45, 1.05);
        const mur = j > ver ? clamp((j - ver) / (ven - ver), 0, 1) : 0;
        const couleur = ETAT.style === 'blanc'
          ? `rgb(${Math.round(lerp(140, 205, mur))},${Math.round(lerp(190, 185, mur))},${Math.round(lerp(80, 90, mur))})`
          : `rgb(${Math.round(lerp(140, 70, mur))},${Math.round(lerp(190, 30, mur))},${Math.round(lerp(80, 80, mur))})`;
        ctx.fillStyle = j < flo ? 'rgba(200,220,150,.9)' : couleur;
        const rangs = [3, 3, 2, 2, 1];
        rangs.forEach((nb, r) => {
          for (let b = 0; b < nb; b++) {
            const bx = x0 + dx + (b - (nb - 1) / 2) * 7 * taille;
            const by = gy + 6 + r * 7 * taille;
            ctx.beginPath(); ctx.arc(bx, by, 3.6 * taille, 0, Math.PI * 2); ctx.fill();
          }
        });
      }
    });
    // pleurs
    if ((q ? j >= D(4, 1) : j >= D(3, 10)) && j < deb) {
      ctx.fillStyle = 'rgba(200,230,255,.8)';
      for (const dx of [-70, 20, 95]) { ctx.beginPath(); ctx.arc(x0 + dx + 4, yBras - 14, 1.8, 0, Math.PI * 2); ctx.fill(); }
    }
    // flocons
    if (froid) {
      ctx.fillStyle = 'rgba(255,255,255,.7)';
      for (let k = 0; k < 40; k++) { const fx = (k * 97 + j * 3) % L, fy = (k * 61 + j * 7) % ySol; ctx.beginPath(); ctx.arc(fx, fy, 1.4, 0, Math.PI * 2); ctx.fill(); }
    }
    // étiquette
    ctx.fillStyle = 'rgba(0,0,0,.35)'; ctx.fillRect(8, 8, 118, 22);
    ctx.fillStyle = '#f3ebe4'; ctx.font = '600 11px JetBrains Mono, monospace'; ctx.textAlign = 'left';
    ctx.fillText(dateDuJour(j).toUpperCase(), 14, 23);
  }

  function maj() {
    const j = +jour.value, q = quebec.checked;
    const ph = phases(q);
    const p = phaseCourante(j, q);
    phaseNom.textContent = p.nom;
    phaseBarre.replaceChildren(...['Dormance', 'Pleurs', 'Débourrement', 'Croissance', 'Floraison', 'Nouaison', 'Véraison', 'Vendanges', 'Chute'].map((n) =>
      h('span', { class: p.nom.startsWith(n.slice(0, 5)) ? 'actif' : '' }, n)));
    pVigne.innerHTML = `<b>La vigne.</b> ${p.vigne}`;
    pVigneron.innerHTML = `<b>Le vigneron.</b> ${p.vigneron}`;
    dessiner();
    void ph;
  }
  function derouler() {
    if (anim) { cancelAnimationFrame(anim); anim = null; btnLecture.textContent = '▶ Dérouler l’année'; return; }
    btnLecture.textContent = '⏸ Arrêter';
    let dernier = performance.now();
    const pas = () => {
      const maintenant = performance.now();
      if (maintenant - dernier > 60) {
        dernier = maintenant;
        jour.value = (+jour.value % 365) + 1;
        jour.majSortie();
        maj();
      }
      anim = requestAnimationFrame(pas);
    };
    anim = requestAnimationFrame(pas);
  }
  jour.addEventListener('input', maj);
  quebec.addEventListener('change', maj);
  btnLecture.addEventListener('click', derouler);
  maj();
  return { maj, phase: () => phaseNom.textContent, jour, quebec, arreter: () => { if (anim) derouler(); } };
};

/* ---------- Maturité ---------- */

function maturite(jours, climat) {
  const d = jours * climat;
  const sucre = 60 + 190 * (1 - Math.exp(-d / 22));
  const acidite = 5 + 22 * Math.exp(-d / 18);
  const ph = 2.7 + 0.9 * (1 - Math.exp(-d / 25));
  return { sucre, acidite, ph, alcool: sucre / 16.83 };
}

ATELIERS.maturite = (boite) => {
  atelierEntete(boite, 'Atelier', 'Choisir la date de vendange');
  const canvas = h('canvas', { 'aria-label': 'Courbes du sucre et de l’acidité après la véraison' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const jours = curseur(reglages, { id: 'mJours', label: 'Jours après la véraison', min: 0, max: 70, valeur: ETAT.style === 'rouge' ? 48 : 38, affiche: (v) => `J+${v}` });
  const climat = selection(reglages, { id: 'mClimat', label: 'Climat du millésime', valeur: '1', options: [['0.8', 'Frais et pluvieux'], ['1', 'Tempéré'], ['1.25', 'Chaud et sec']] });
  const mesures = h('dl', { class: 'mesures' });
  const dSucre = h('dd', { 'data-testid': 'sucre' }), dAcide = h('dd'), dPh = h('dd'), dAlc = h('dd', { class: 'grand', 'data-testid': 'alcool-potentiel' });
  mesures.append(h('dt', {}, 'Sucre'), dSucre, h('dt', {}, 'Acidité totale (éq. H₂SO₄)'), dAcide, h('dt', {}, 'pH'), dPh, h('dt', {}, 'Alcool potentiel'), dAlc);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture' });
  boite.append(lecture);
  const btn = h('button', { class: 'bouton principal', type: 'button', id: 'btnVendanger' }, 'Vendanger à ce stade →');
  boite.append(h('div', { class: 'boutons' }, btn));

  function dessiner() {
    const L = 520, H = 200, m = { g: 36, d: 36, h: 14, b: 26 };
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#120c0e'; ctx.fillRect(0, 0, L, H);
    const c = +climat.value;
    const X = (j) => m.g + (j / 70) * (L - m.g - m.d);
    const YS = (s) => H - m.b - ((s - 40) / 220) * (H - m.h - m.b);
    const YA = (a) => H - m.b - (a / 30) * (H - m.h - m.b);
    ctx.strokeStyle = '#2f2226';
    for (let k = 0; k <= 4; k++) { const y = m.h + k * (H - m.h - m.b) / 4; ctx.beginPath(); ctx.moveTo(m.g, y); ctx.lineTo(L - m.d, y); ctx.stroke(); }
    ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.fillStyle = '#d9b25f'; ctx.textAlign = 'right';
    for (const s of [100, 150, 200, 250]) ctx.fillText(`${s}`, m.g - 4, YS(s) + 3);
    ctx.fillStyle = '#8fbf6a'; ctx.textAlign = 'left';
    for (const a of [10, 20, 30]) ctx.fillText(`${a}`, L - m.d + 4, YA(a) + 3);
    ctx.fillStyle = '#a89a95'; ctx.textAlign = 'center';
    for (let j = 0; j <= 70; j += 14) ctx.fillText(`J+${j}`, X(j), H - 8);
    ctx.textAlign = 'left'; ctx.fillStyle = '#d9b25f'; ctx.fillText('sucre g/L', m.g + 4, m.h + 8);
    ctx.textAlign = 'right'; ctx.fillStyle = '#8fbf6a'; ctx.fillText('acidité g/L', L - m.d - 4, m.h + 8);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#d9b25f'; ctx.beginPath();
    for (let j = 0; j <= 70; j++) { const p = maturite(j, c); j ? ctx.lineTo(X(j), YS(p.sucre)) : ctx.moveTo(X(j), YS(p.sucre)); }
    ctx.stroke();
    ctx.strokeStyle = '#8fbf6a'; ctx.beginPath();
    for (let j = 0; j <= 70; j++) { const p = maturite(j, c); j ? ctx.lineTo(X(j), YA(p.acidite)) : ctx.moveTo(X(j), YA(p.acidite)); }
    ctx.stroke();
    const j = +jours.value;
    ctx.strokeStyle = 'rgba(243,235,228,.6)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(j), m.h); ctx.lineTo(X(j), H - m.b); ctx.stroke(); ctx.setLineDash([]);
  }
  function maj() {
    const j = +jours.value, c = +climat.value;
    const p = maturite(j, c);
    dSucre.textContent = `${fmt(p.sucre)} g/L`;
    dAcide.textContent = `${fmt(p.acidite, 1)} g/L`;
    dPh.textContent = fmt(p.ph, 2);
    dAlc.textContent = `${fmt(p.alcool, 1)} % vol.`;
    const d = j * c;
    let l, cls = '';
    if (d < 18) { l = "<b>Trop tôt.</b> Baies vertes, dures, acides ; le sucre donnerait à peine 6 ou 7 % d'alcool. On ne vendange à ce stade que pour une base d'effervescent, qui cherche justement l'acidité, ou en vendange verte pour alléger la charge."; cls = 'alerte'; }
    else if (d < 32) { l = ETAT.style === 'rouge' ? "<b>Maturité précoce.</b> Assez de sucre pour un rouge léger, mais les tanins sont encore verts et les pépins pas bruns : le vin serait âpre, herbacé. Il faut attendre la maturité phénolique." : "<b>Fraîcheur et tension.</b> Acidité vive, sucre modéré, arômes d'agrumes et de fleurs blanches : c'est le stade des blancs vifs et des rosés nerveux, souvent vendangé de nuit."; cls = ETAT.style === 'rouge' ? 'alerte' : 'bon'; }
    else if (d < 50) { l = ETAT.style === 'rouge' ? "<b>La bonne fenêtre.</b> Sucre et acidité s'équilibrent, les pépins sont bruns, la peau lâche sa couleur au frottement. Un rouge équilibré, entre 12,5 et 14 %. C'est le stade où l'on vendange la plupart des grands rouges." : "<b>Maturité pleine.</b> Le fruit devient mûr, l'acidité reste suffisante, l'alcool sera entre 12,5 et 13,5 %. Le stade des blancs amples et ronds, chardonnays et chenins secs."; cls = 'bon'; }
    else if (d < 62) { l = "<b>Vendange tardive.</b> Beaucoup de sucre, donc d'alcool (14 à 15 %), une acidité qui manque, des arômes de fruit confit. Les vins sont riches, chaleureux, parfois lourds. C'est la maturité des rouges de climat chaud et des blancs moelleux."; }
    else { l = "<b>Surmaturité.</b> Les baies se flétrissent, le sucre dépasse ce que les levures peuvent transformer : le vin gardera du sucre résiduel. Domaine des vendanges tardives, du passerillage, et de la pourriture noble si le brouillard s'en mêle. Pour un vin sec, c'est trop tard."; cls = 'alerte'; }
    lecture.innerHTML = l;
    lecture.className = `lecture ${cls}`;
    dessiner();
  }
  jours.addEventListener('input', maj);
  climat.addEventListener('change', maj);
  btn.addEventListener('click', () => {
    const p = maturite(+jours.value, +climat.value);
    ETAT.sucre = Math.round(p.sucre);
    const f = API.ateliers.fermentation;
    if (f) { f.reglerSucre(ETAT.sucre); document.getElementById('s-fermentation')?.scrollIntoView({ behavior: 'smooth' }); }
  });
  maj();
  return { maj };
};

/* ---------- Vendanges : rendement ---------- */

ATELIERS.rendement = (boite) => {
  atelierEntete(boite, 'Atelier', 'Du raisin aux bouteilles');
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const surface = curseur(reglages, { id: 'rSurface', label: 'Surface', min: 0.5, max: 20, step: 0.5, valeur: 1, affiche: (v) => `${fmt(v, 1)} ha` });
  const rendement = curseur(reglages, { id: 'rRendement', label: 'Rendement', min: 20, max: 120, step: 5, valeur: 50, affiche: (v) => `${v} hL/ha` });
  const mesures = h('dl', { class: 'mesures' });
  const dPieds = h('dd'), dRaisin = h('dd'), dVin = h('dd'), dBout = h('dd', { class: 'grand', 'data-testid': 'bouteilles' }), dParPied = h('dd'), dCagettes = h('dd');
  mesures.append(h('dt', {}, 'Ceps (densité de l’étape 1)'), dPieds, h('dt', {}, 'Raisin récolté'), dRaisin, h('dt', {}, 'Cagettes de 15 kg'), dCagettes, h('dt', {}, 'Vin obtenu'), dVin, h('dt', {}, 'Bouteilles de 75 cL'), dBout, h('dt', {}, 'Bouteilles par cep'), dParPied);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture' });
  boite.append(lecture);
  function maj() {
    const s = +surface.value, r = +rendement.value;
    const litres = s * r * 100;
    const kg = litres / 0.7;
    const pieds = ETAT.piedsHa * s;
    const bout = litres / 0.75;
    dPieds.textContent = fmt(pieds);
    dRaisin.textContent = kg >= 10000 ? `${fmt(kg / 1000, 1)} t` : `${fmt(kg)} kg`;
    dCagettes.textContent = fmt(kg / 15);
    dVin.textContent = `${fmt(litres)} L`;
    dBout.textContent = fmt(bout);
    dParPied.textContent = fmt(bout / pieds, 2);
    let l;
    if (r <= 35) l = "<b>Rendement de grand cru.</b> Peu de grappes par cep, souvent après une vendange verte : concentration maximale, mais chaque bouteille coûte cher à produire. Les appellations les plus prestigieuses plafonnent vers 35 à 45 hL/ha.";
    else if (r <= 65) l = "<b>Rendement d'appellation.</b> La fourchette de la plupart des vins de qualité : le cep porte ce qu'il peut mûrir. Le vigneron perd environ 30 % du poids du raisin en rafles, peaux, pépins et bourbes.";
    else l = "<b>Rendement élevé.</b> Vignes vigoureuses, larges, irriguées ou très fertiles : c'est le domaine des vins de table et de marque, où le prix de la bouteille dépend du volume. Au-delà de 100 hL/ha, la maturité devient difficile à atteindre.";
    lecture.innerHTML = `${l} Ici, ${fmt(bout / pieds, 1)} bouteille${bout / pieds >= 2 ? 's' : ''} par cep.`;
  }
  surface.addEventListener('input', maj);
  rendement.addEventListener('input', maj);
  maj();
  return { maj };
};

/* ---------- Fermentation alcoolique : le simulateur ---------- */

const DEFAUTS_FERM = { rouge: { consigne: 27, depart: 20 }, blanc: { consigne: 15, depart: 12 }, rose: { consigne: 16, depart: 13 } };
const LEVURES = { indigenes: { emax: 14.5, x0: 0.004, mu: 0.12 }, selectionnees: { emax: 16, x0: 0.02, mu: 0.16 } };

/* Taux d'activité des levures selon la température (0..1). */
function fTemp(T) {
  if (T <= 5) return 0;
  if (T <= 31) return Math.exp(-(((T - 31) / 15) ** 2));
  return Math.exp(-(((T - 31) / 8) ** 2));
}

class Fermentation {
  constructor(param) { this.reinit(param); }
  reinit(p) {
    this.p = p;
    this.S = p.sucre; this.E = 0; this.X = LEVURES[p.levures].x0; this.T = p.depart; this.CO2 = 0;
    this.heure = 0; this.fini = null; this.morte = false; this.stress = 0; this.A = 0; this.Ta = 0; this.chapeauSec = 0;
    this.journal = []; this.serie = [];
    this.boostFin = -1; this.boostFacteur = 1;
    this.enregistrer();
  }
  enregistrer() { if (this.heure % 6 === 0) this.serie.push({ h: this.heure, d: this.densite(), T: this.T, E: this.E, A: this.A, Ta: this.Ta }); }
  densite() { return 0.9925 + 0.000465 * this.S - 0.00012 * this.E; }
  travailChapeau(type) {
    if (this.fini) return;
    this.boostFin = this.heure + (type === 'pigeage' ? 8 : 6);
    this.boostFacteur = type === 'pigeage' ? 3 : 2.2;
    this.chapeauSec = 0;
    this.note(type === 'pigeage' ? 'Pigeage : le chapeau est enfoncé dans le jus.' : 'Remontage : le jus est pompé par-dessus le chapeau.');
  }
  note(texte) { this.journal.unshift({ h: this.heure, texte }); if (this.journal.length > 6) this.journal.pop(); }
  pas() {
    if (this.fini) return;
    const p = this.p, L = LEVURES[p.levures];
    const dt = 1;
    // travail du chapeau programmé (rouge)
    if (p.rouge) {
      const hj = this.heure % 24;
      if (p.programme === 'remontage' && hj === 8) this.travailChapeau('remontage');
      if (p.programme === 'pigeage' && (hj === 8 || hj === 20)) this.travailChapeau('pigeage');
    }
    const fT = fTemp(this.T);
    const inhib = Math.max(0, 1 - (this.E / L.emax) ** 2);
    // croissance des levures
    if (!this.morte) {
      // Plus de multiplication au-delà de 35 °C, ni après un coup de chaud : les levures stressées ne repartent pas.
      const mu = (this.T > 35 || this.stress > 3) ? 0 : L.mu * fT * (this.S / (this.S + 10)) * inhib;
      this.X = clamp(this.X + mu * this.X * (1 - this.X) * dt, 0, 1);
      if (this.T > 35) {
        this.stress += (this.T - 35) * dt;
        this.X *= 1 - (this.T - 35) * 0.08 * dt;
        if (this.T > 38) { this.morte = true; this.note(`${fmt(this.T, 1)} °C : les levures sont tuées par la chaleur.`); }
      }
      if (this.stress > 3 && !this.morte) this.X *= 1 - 0.015 * dt;   // attrition des levures stressées, même une fois la cuve refroidie
      if (!this.morte && this.stress > 3 && this.X < 0.12 && this.S > 2) { this.morte = true; this.note('Affaiblies par le coup de chaud, les levures ne repartent plus.'); }
    }
    // consommation du sucre
    let dS = 2.2 * fT * this.X * (this.S / (this.S + 8)) * inhib * dt;
    dS = Math.min(dS, this.S);
    this.S -= dS;
    this.E += dS / 16.83;
    this.CO2 += dS * 0.267;
    // chaleur : 0,13 °C par g/L, refroidissement
    let dT = 0.13 * dS;
    if (p.thermo) dT += (p.consigne - this.T) * 0.3 * dt;
    else dT += (20 - this.T) * 0.004 * dt;   // grande cuve dans un chai à 20 °C : pertes faibles
    this.T += dT;
    // extraction (rouge)
    if (p.rouge) {
      const boost = this.heure < this.boostFin ? this.boostFacteur : 1;
      const ext = (0.25 + 0.035 * clamp(this.T - 12, 0, 25)) * boost;
      const dA = 0.9 * ext * (1 - this.A / 100) * dt - (this.E > 8 ? 0.004 * this.A * dt : 0);
      this.A = clamp(this.A + dA, 0, 100);
      const dTa = 0.22 * ext * (0.35 + this.E / 10) * (1 - this.Ta / 100) * dt;
      this.Ta = clamp(this.Ta + dTa, 0, 100);
      this.chapeauSec += dt;
    }
    this.heure += 1;
    this.enregistrer();
    // fin
    if (this.S < 2) { this.fini = 'sec'; this.note('Moins de 2 g/L de sucre : le vin est sec.'); }
    else if (this.morte && this.heure > 24) { this.fini = 'chaleur'; }
    else if (this.E >= L.emax - 0.05 && this.S > 2) { this.fini = 'alcool'; this.note(`${fmt(this.E, 1)} % vol. : les levures ne supportent plus l'alcool. Il reste ${fmt(this.S)} g/L de sucre.`); }
    else if (this.heure > 24 * 45) { this.fini = 'languissante'; this.note('Après 45 jours, la fermentation traîne toujours. On appelle l’œnologue.'); }
    if (this.fini && this.fini !== 'sec' && this.fini !== 'alcool') this.note(this.fini === 'chaleur' ? 'Fermentation arrêtée : la cuve a surchauffé.' : 'Fermentation languissante.');
  }
  etat() {
    if (this.fini === 'sec') return { texte: `Vin sec en ${fmt(this.heure / 24, 1)} jours : ${fmt(this.E, 1)} % vol.`, cls: 'bon' };
    if (this.fini === 'chaleur') return { texte: `Arrêt de fermentation : levures tuées par la chaleur. ${fmt(this.S)} g/L de sucre restent, le vin est perdu ou à relancer.`, cls: 'alerte' };
    if (this.fini === 'alcool') return { texte: `Arrêt : ${fmt(this.E, 1)} % vol., les levures ne suivent plus. ${fmt(this.S)} g/L de sucre résiduel.`, cls: 'alerte' };
    if (this.fini === 'languissante') return { texte: 'Fermentation languissante : trop froid ou levures épuisées.', cls: 'alerte' };
    if (this.heure === 0) return { texte: 'Prêt : cuve remplie, levures ensemencées.', cls: '' };
    if (this.X < 0.25) return { texte: 'Phase de latence : les levures se multiplient, rien ne bouge en apparence.', cls: '' };
    if (this.S > this.p.sucre * 0.25) return { texte: 'Fermentation tumultueuse : la cuve bout, le gaz siffle, la densité chute.', cls: '' };
    return { texte: 'Fin de fermentation : les derniers grammes de sucre partent lentement.', cls: '' };
  }
}

ATELIERS.fermentation = (boite) => {
  const rouge = ETAT.style === 'rouge';
  atelierEntete(boite, 'Simulateur', rouge ? 'Une cuve de rouge, heure par heure' : 'Une cuve au frais, heure par heure');
  const cuve = h('canvas', { 'aria-label': 'Coupe de la cuve de fermentation' });
  const graphe = h('canvas', { 'aria-label': 'Courbes de densité et de température' });
  const gauche = h('div', {}, cuve);
  const droite = h('div', {}, graphe);
  boite.append(h('div', { class: 'double' }, gauche, droite));

  const reglages = h('div', { class: 'reglages' });
  gauche.append(reglages);
  const def = DEFAUTS_FERM[ETAT.style];
  const sucre = curseur(reglages, { id: 'fSucre', label: 'Sucre du moût', min: 150, max: 280, step: 5, valeur: Math.round(ETAT.sucre / 5) * 5, affiche: (v) => `${v} g/L ≈ ${fmt(v / 16.83, 1)} %` });
  const consigne = curseur(reglages, { id: 'fConsigne', label: 'Consigne de température', min: 8, max: 34, valeur: def.consigne, affiche: (v) => `${v} °C` });
  const levures = selection(reglages, { id: 'fLevures', label: 'Levures', valeur: 'selectionnees', options: [['selectionnees', 'Sélectionnées (résistent jusqu’à 16 %)'], ['indigenes', 'Indigènes (jusqu’à 14,5 %)']] });
  let programme = null;
  if (rouge) programme = selection(reglages, { id: 'fProgramme', label: 'Travail du chapeau', valeur: 'remontage', options: [['aucun', 'Aucun'], ['remontage', 'Un remontage par jour'], ['pigeage', 'Deux pigeages par jour']] });
  const thermo = h('input', { type: 'checkbox', id: 'fThermo', checked: '' });
  gauche.append(h('label', { class: 'case' }, thermo, 'Thermorégulation de la cuve (drapeau ou double paroi)'));
  const boutons = h('div', { class: 'boutons' });
  const btnLancer = h('button', { class: 'bouton principal', type: 'button', id: 'fLancer' }, '▶ Lancer');
  const btnReinit = h('button', { class: 'bouton', type: 'button', id: 'fReinit' }, '↺ Recommencer');
  const vitesse = h('select', { id: 'fVitesse', 'aria-label': 'Vitesse' }, [[12, '×1 · 1 jour / 2 s'], [48, '×4'], [240, '×20']].map(([v, l]) => h('option', { value: v }, l)));
  vitesse.value = '48';
  boutons.append(btnLancer, btnReinit, vitesse);
  if (rouge) {
    boutons.append(h('button', { class: 'bouton', type: 'button', id: 'fRemontage', title: 'Pomper le jus du bas de la cuve sur le chapeau' }, 'Remontage'),
      h('button', { class: 'bouton', type: 'button', id: 'fPigeage', title: 'Enfoncer le chapeau de marc dans le jus' }, 'Pigeage'));
  }
  gauche.append(boutons);

  const etatEl = h('p', { class: 'etat-ferm', 'data-testid': 'etat-fermentation' });
  droite.append(etatEl);
  const mesures = h('dl', { class: 'mesures' });
  const dJour = h('dd', { 'data-testid': 'jour' }), dDens = h('dd', { 'data-testid': 'densite-ferm' }), dSucre = h('dd'), dAlc = h('dd', { class: 'grand', 'data-testid': 'alcool' }), dT = h('dd', { 'data-testid': 'temperature' }), dCO2 = h('dd'), dLev = h('dd');
  mesures.append(h('dt', {}, 'Jour'), dJour, h('dt', {}, 'Densité'), dDens, h('dt', {}, 'Sucre'), dSucre, h('dt', {}, 'Alcool'), dAlc, h('dt', {}, 'Température'), dT, h('dt', {}, 'Levures actives'), dLev, h('dt', {}, 'CO₂ dégagé, pour 1 000 L'), dCO2);
  droite.append(mesures);
  let jCouleur = null, jTanins = null, echantillon = null;
  if (rouge) {
    const jauges = h('div', { class: 'jauges' });
    jCouleur = jauge(jauges, 'Couleur extraite');
    jTanins = jauge(jauges, 'Tanins extraits', 'or');
    droite.append(jauges);
    echantillon = h('div', { class: 'echantillon' }, 'échantillon');
    droite.append(echantillon);
  }
  const journal = h('ul', { class: 'reperes', style: 'margin-top:12px', id: 'fJournal' });
  droite.append(journal);

  const lireParam = () => ({
    sucre: +sucre.value, consigne: +consigne.value, depart: def.depart, levures: levures.value,
    thermo: thermo.checked, rouge, programme: programme ? programme.value : 'aucun',
  });
  const sim = new Fermentation(lireParam());
  let anim = null, accumule = 0, dernier = 0;

  function dessinerCuve() {
    const L = 300, H = 260;
    const ctx = contexte(cuve, L, H);
    ctx.fillStyle = '#120c0e'; ctx.fillRect(0, 0, L, H);
    const x = 70, y = 20, l = 160, ht = 215;
    // cuve inox
    ctx.fillStyle = '#3a3438'; ctx.fillRect(x - 6, y - 6, l + 12, ht + 12);
    ctx.fillStyle = '#1a1416'; ctx.fillRect(x, y, l, ht);
    // liquide
    const frac = 0.86;
    const yl = y + ht * (1 - frac);
    const couleur = rouge
      ? `rgb(${Math.round(lerp(190, 70, sim.A / 100))},${Math.round(lerp(120, 12, sim.A / 100))},${Math.round(lerp(120, 45, sim.A / 100))})`
      : ETAT.style === 'rose' ? '#e9a3b0' : (sim.S > 30 ? '#c9b98a' : '#e9dfa0');
    ctx.fillStyle = couleur; ctx.fillRect(x, yl, l, y + ht - yl);
    // chapeau de marc
    if (rouge && sim.heure > 0) {
      const ep = 28 + 10 * clamp(sim.X, 0, 1);
      const enfonce = sim.heure < sim.boostFin ? 12 : 0;
      ctx.fillStyle = sim.chapeauSec > 30 && !sim.fini ? '#4a2a22' : '#5a2830';
      ctx.fillRect(x, yl + enfonce, l, ep);
      ctx.fillStyle = 'rgba(0,0,0,.25)';
      for (let k = 0; k < 40; k++) { const bx = x + ((k * 37) % l), by = yl + enfonce + ((k * 17) % ep); ctx.beginPath(); ctx.arc(bx, by, 2.2, 0, Math.PI * 2); ctx.fill(); }
    }
    // bulles
    const taux = sim.fini ? 0 : 2.2 * fTemp(sim.T) * sim.X * (sim.S / (sim.S + 8));
    const nb = Math.round(taux * 30);
    ctx.fillStyle = 'rgba(255,255,255,.55)';
    const tps = performance.now() / 1000;
    for (let k = 0; k < nb; k++) {
      const bx = x + 8 + ((k * 53) % (l - 16));
      const by = y + ht - 6 - (((tps * (30 + (k % 7) * 8)) + k * 41) % (y + ht - yl - 10));
      ctx.beginPath(); ctx.arc(bx, by, 1.2 + (k % 3) * 0.6, 0, Math.PI * 2); ctx.fill();
    }
    // vanne, thermomètre
    ctx.fillStyle = '#6a5f66'; ctx.fillRect(x + l, y + ht - 30, 18, 8);
    ctx.fillStyle = '#a89a95'; ctx.font = '600 10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
    ctx.fillText(`${fmt(sim.T, 1)} °C`, x + l / 2, y + ht + 4 + 10);
    if (sim.p.thermo) { ctx.fillStyle = '#86b7d9'; ctx.fillRect(x - 6, y + 30, 4, ht - 60); ctx.fillText('froid', x - 34, y + ht / 2); }
    ctx.fillStyle = '#f3ebe4'; ctx.textAlign = 'right';
    ctx.fillText(rouge && sim.heure > 0 ? 'chapeau de marc' : (sim.S > 30 ? 'moût' : 'vin'), x + l - 8, yl - 8);
  }
  function dessinerGraphe() {
    const L = 420, H = 210, m = { g: 44, d: 40, h: 12, b: 26 };
    const ctx = contexte(graphe, L, H);
    ctx.fillStyle = '#120c0e'; ctx.fillRect(0, 0, L, H);
    const jours = Math.max(15, Math.ceil(sim.heure / 24 / 5) * 5);
    const X = (hh) => m.g + (hh / 24 / jours) * (L - m.g - m.d);
    const YD = (d) => H - m.b - ((d - 0.985) / 0.12) * (H - m.h - m.b);
    const YT = (T) => H - m.b - (T / 42) * (H - m.h - m.b);
    ctx.strokeStyle = '#2f2226';
    for (const d of [1.0, 1.02, 1.04, 1.06, 1.08, 1.1]) { ctx.beginPath(); ctx.moveTo(m.g, YD(d)); ctx.lineTo(L - m.d, YD(d)); ctx.stroke(); }
    ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.textAlign = 'right'; ctx.fillStyle = '#d9b25f';
    for (const d of [1.0, 1.05, 1.1]) ctx.fillText(d.toFixed(3), m.g - 4, YD(d) + 3);
    ctx.textAlign = 'left'; ctx.fillStyle = '#e0705d';
    for (const T of [10, 20, 30, 40]) ctx.fillText(`${T}°`, L - m.d + 4, YT(T) + 3);
    ctx.fillStyle = '#a89a95'; ctx.textAlign = 'center';
    for (let j = 0; j <= jours; j += jours / 5) ctx.fillText(`J${j}`, X(j * 24), H - 8);
    ctx.textAlign = 'left'; ctx.fillStyle = '#d9b25f'; ctx.fillText('densité', m.g + 4, m.h + 8);
    ctx.fillStyle = '#e0705d'; ctx.fillText('température', m.g + 60, m.h + 8);
    // zone de danger
    ctx.fillStyle = 'rgba(224,112,93,.12)'; ctx.fillRect(m.g, YT(42), L - m.g - m.d, YT(35) - YT(42));
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#e0705d'; ctx.beginPath();
    sim.serie.forEach((s, i) => i ? ctx.lineTo(X(s.h), YT(s.T)) : ctx.moveTo(X(s.h), YT(s.T)));
    ctx.stroke();
    ctx.strokeStyle = '#d9b25f'; ctx.beginPath();
    sim.serie.forEach((s, i) => i ? ctx.lineTo(X(s.h), YD(s.d)) : ctx.moveTo(X(s.h), YD(s.d)));
    ctx.stroke();
    if (rouge) {
      ctx.strokeStyle = '#d4577a'; ctx.setLineDash([3, 3]); ctx.beginPath();
      sim.serie.forEach((s, i) => { const yy = H - m.b - (s.A / 100) * (H - m.h - m.b); i ? ctx.lineTo(X(s.h), yy) : ctx.moveTo(X(s.h), yy); });
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#d4577a'; ctx.fillText('- - couleur', m.g + 140, m.h + 8);
    }
  }
  function afficher() {
    const e = sim.etat();
    etatEl.textContent = e.texte; etatEl.className = `etat-ferm ${e.cls}`;
    dJour.textContent = `J${Math.floor(sim.heure / 24)} · ${sim.heure % 24} h`;
    dDens.textContent = sim.densite().toFixed(3);
    dSucre.textContent = `${fmt(sim.S)} g/L`;
    dAlc.textContent = `${fmt(sim.E, 1)} % vol.`;
    dT.textContent = `${fmt(sim.T, 1)} °C`;
    dT.className = sim.T > 35 ? 'alerte' : '';
    dLev.textContent = sim.morte ? 'mortes' : `${fmt(sim.X * 100)} %`;
    dCO2.textContent = `${fmt(sim.CO2, 1)} m³`;
    if (rouge) {
      jCouleur(sim.A); jTanins(sim.Ta);
      echantillon.style.background = `rgb(${Math.round(lerp(200, 60, sim.A / 100))},${Math.round(lerp(130, 10, sim.A / 100))},${Math.round(lerp(130, 40, sim.A / 100))})`;
      echantillon.textContent = sim.A < 15 ? 'moût rosé' : sim.A < 45 ? 'rouge clair' : sim.A < 75 ? 'rouge franc' : 'rouge profond';
      if (sim.chapeauSec > 36 && !sim.fini && sim.p.programme === 'aucun' && !sim.journal.some((n) => n.texte.startsWith('Chapeau'))) sim.note('Chapeau sec depuis plus d’un jour : les bactéries acétiques s’y installent. Remontez !');
    }
    journal.replaceChildren(...sim.journal.map((n) => h('li', {}, h('b', {}, `J${Math.floor(n.h / 24)} `), n.texte)));
    dessinerCuve(); dessinerGraphe();
    btnLancer.textContent = anim ? '⏸ Pause' : sim.fini ? '✓ Terminé' : sim.heure ? '▶ Reprendre' : '▶ Lancer';
    btnLancer.disabled = !!sim.fini;
  }
  function avancer(heures) { for (let k = 0; k < heures && !sim.fini; k++) sim.pas(); afficher(); }
  function boucle(maintenant) {
    if (!dernier) dernier = maintenant;
    accumule += (maintenant - dernier) / 1000 * (+vitesse.value);
    dernier = maintenant;
    while (accumule >= 1 && !sim.fini) { sim.pas(); accumule -= 1; }
    afficher();
    if (sim.fini) { anim = null; afficher(); return; }
    anim = requestAnimationFrame(boucle);
  }
  function lancer() {
    if (anim) { cancelAnimationFrame(anim); anim = null; afficher(); return; }
    if (sim.fini) return;
    if (sim.heure === 0) sim.reinit(lireParam());
    dernier = 0; accumule = 0;
    anim = requestAnimationFrame(boucle);
    afficher();
  }
  function reinit() { if (anim) cancelAnimationFrame(anim); anim = null; sim.reinit(lireParam()); afficher(); }
  // Les réglages changent la consigne en cours de route ; le reste s'applique au prochain départ.
  consigne.addEventListener('input', () => { sim.p.consigne = +consigne.value; });
  thermo.addEventListener('change', () => { sim.p.thermo = thermo.checked; afficher(); });
  if (programme) programme.addEventListener('change', () => { sim.p.programme = programme.value; });
  sucre.addEventListener('input', () => { if (sim.heure === 0) reinit(); });
  levures.addEventListener('change', () => { if (sim.heure === 0) reinit(); });
  btnLancer.addEventListener('click', lancer);
  btnReinit.addEventListener('click', reinit);
  if (rouge) {
    $('#fRemontage', boite).addEventListener('click', () => { sim.travailChapeau('remontage'); afficher(); });
    $('#fPigeage', boite).addEventListener('click', () => { sim.travailChapeau('pigeage'); afficher(); });
  }
  afficher();
  return {
    sim, avancer, reinit, lancer,
    reglerSucre(v) { sucre.value = Math.round(v / 5) * 5; sucre.majSortie(); reinit(); },
    regler(champ, valeur) {
      if (champ === 'thermo') thermo.checked = !!valeur;
      else if (champ === 'programme' && programme) programme.value = valeur;
      else if (champ === 'levures') levures.value = valeur;
      else if (champ === 'consigne') { consigne.value = valeur; consigne.majSortie(); }
      else if (champ === 'sucre') { sucre.value = valeur; sucre.majSortie(); }
      reinit();
    },
  };
};

/* ---------- Fermentation malolactique ---------- */

ATELIERS.malo = (boite) => {
  atelierEntete(boite, 'Atelier', "Suivre la malo");
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const malique = curseur(reglages, { id: 'mlMalique', label: 'Acide malique au départ', min: 1, max: 6, step: 0.5, valeur: ETAT.style === 'rouge' ? 3 : 4, affiche: (v) => `${fmt(v, 1)} g/L` });
  const avancement = curseur(reglages, { id: 'mlAvancement', label: 'Avancement', min: 0, max: 100, valeur: ETAT.style === 'rose' ? 0 : 60, affiche: (v) => `${v} %` });
  const jauges = h('div', { class: 'jauges' });
  const jMal = jauge(jauges, 'Acide malique', 'vert');
  const jLac = jauge(jauges, 'Acide lactique', 'or');
  const jAcid = jauge(jauges, 'Acidité totale');
  boite.append(jauges);
  const mesures = h('dl', { class: 'mesures' });
  const dPh = h('dd'), dGout = h('dd'), dStatut = h('dd', { 'data-testid': 'malo-statut' });
  mesures.append(h('dt', {}, 'pH'), dPh, h('dt', {}, 'Au nez, en bouche'), dGout, h('dt', {}, 'Statut'), dStatut);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture' });
  boite.append(lecture);
  function maj() {
    const m0 = +malique.value, a = +avancement.value / 100;
    const mal = m0 * (1 - a), lac = m0 * a * 0.67;
    const acid0 = 5.5 + m0 * 0.5, acid = acid0 - (m0 - mal - lac) * 0.5;
    const ph = 3.4 + (acid0 - acid) * 0.12;
    jMal(mal / 6 * 100, `${fmt(mal, 2)} g/L`);
    jLac(lac / 6 * 100, `${fmt(lac, 2)} g/L`);
    jAcid(acid / 8 * 100, `${fmt(acid, 2)} g/L`);
    dPh.textContent = fmt(ph, 2);
    dGout.textContent = a < 0.2 ? 'pomme verte, mordant' : a < 0.7 ? 'en transition, un peu de gaz' : ETAT.style === 'blanc' ? 'beurre, noisette, rondeur' : 'rond, souple, lacté';
    dStatut.textContent = a === 0 ? 'Non commencée' : a < 1 ? 'En cours' : 'Terminée';
    let l;
    if (ETAT.style === 'rose') l = a === 0 ? "<b>Bloquée, comme prévu.</b> Un rosé garde son acide malique : c'est lui qui donne la sensation de croquant. On le garde au froid, on soufre légèrement, et on surveille le malique au laboratoire jusqu'à la mise." : "<b>Un rosé qui fait sa malo</b> perd sa vivacité et son fruit de petits fruits rouges ; on parle d'un rosé « beurré ». Rare, et rarement voulu.";
    else if (a === 0) l = "<b>Pas encore commencée.</b> Les bactéries attendent que la fermentation alcoolique soit finie, que la température remonte vers 18 à 20 °C et que le soufre libre soit bas. Ça peut prendre des semaines ; certains chais ensemencent pour ne pas attendre.";
    else if (a < 1) l = `<b>En cours.</b> Des bulles minuscules montent dans le vin, le nez sent un peu la choucroute, puis ça passe. On suit l'acide malique par chromatographie sur papier ou par analyse enzymatique. Il en reste ${fmt(mal, 1)} g/L.`;
    else l = ETAT.style === 'blanc' ? "<b>Terminée.</b> L'acidité a baissé, le vin s'est arrondi et a pris ce goût beurré caractéristique du diacétyle, qu'un élevage sur lies atténuera. On soufre pour stabiliser. Le vin ne refermentera plus en bouteille." : "<b>Terminée.</b> Le vin a perdu son mordant et pris de la rondeur ; il est stable pour l'élevage et la bouteille. On soutire et on soufre légèrement.";
    lecture.innerHTML = l;
  }
  malique.addEventListener('input', maj);
  avancement.addEventListener('input', maj);
  maj();
  return { maj };
};

/* ---------- Élevage ---------- */

const CONTENANTS = {
  inox: { nom: 'Cuve inox', oxy: 0.05, bois: 0, evap: 0, cout: 0.05, volume: 5000 },
  beton: { nom: 'Cuve béton', oxy: 0.2, bois: 0, evap: 0.5, cout: 0.1, volume: 3000 },
  amphore: { nom: 'Amphore en terre cuite', oxy: 0.5, bois: 0, evap: 1.5, cout: 0.6, volume: 400 },
  foudre: { nom: 'Foudre de chêne ancien', oxy: 0.3, bois: 0.15, evap: 1.5, cout: 0.2, volume: 3000 },
  usagee: { nom: 'Barrique de 3 vins', oxy: 0.6, bois: 0.3, evap: 3, cout: 0.6, volume: 225 },
  neuve: { nom: 'Barrique neuve', oxy: 0.8, bois: 1, evap: 3.5, cout: 3.5, volume: 225 },
};

ATELIERS.elevage = (boite) => {
  atelierEntete(boite, 'Atelier', 'Choisir le contenant et la durée');
  const gauche = h('div'), droite = h('div');
  boite.append(h('div', { class: 'double' }, gauche, droite));
  const reglages = h('div', { class: 'reglages' });
  gauche.append(reglages);
  const contenant = selection(reglages, { id: 'eContenant', label: 'Contenant', valeur: ETAT.style === 'rouge' ? 'usagee' : 'inox', options: Object.entries(CONTENANTS).map(([k, c]) => [k, c.nom]) });
  const chauffe = selection(reglages, { id: 'eChauffe', label: 'Chauffe du bois', valeur: 'moyenne', options: [['legere', 'Légère'], ['moyenne', 'Moyenne'], ['forte', 'Forte']] });
  const duree = curseur(reglages, { id: 'eDuree', label: "Durée d'élevage", min: 0, max: 36, valeur: ETAT.style === 'rouge' ? 14 : ETAT.style === 'blanc' ? 9 : 4, affiche: (v) => `${v} mois` });
  const lies = h('input', { type: 'checkbox', id: 'eLies' });
  if (ETAT.style === 'blanc') lies.checked = true;
  gauche.append(h('label', { class: 'case' }, lies, 'Élevage sur lies avec bâtonnage'));
  const jauges = h('div', { class: 'jauges' });
  const jBois = jauge(jauges, 'Arômes du bois', 'or');
  const jRond = jauge(jauges, ETAT.style === 'rouge' ? 'Tanins fondus' : 'Rondeur, gras');
  const jFruit = jauge(jauges, 'Fruit frais préservé', 'vert');
  const jOxy = jauge(jauges, 'Oxydation', 'rougeb');
  droite.append(jauges);
  const mesures = h('dl', { class: 'mesures' });
  const dEvap = h('dd', { 'data-testid': 'part-anges' }), dCout = h('dd'), dNez = h('dd');
  mesures.append(h('dt', {}, 'Part des anges'), dEvap, h('dt', {}, 'Coût par bouteille'), dCout, h('dt', {}, 'Au nez'), dNez);
  droite.append(mesures);
  const lecture = h('p', { class: 'lecture' });
  droite.append(lecture);

  function maj() {
    const c = CONTENANTS[contenant.value], mois = +duree.value, ch = chauffe.value;
    chauffe.disabled = c.bois === 0;
    const bois = c.bois * (1 - Math.exp(-mois / 8)) * 100;
    const rond = (c.oxy * (1 - Math.exp(-mois / 10)) * 0.8 + (lies.checked ? 0.25 : 0) * (1 - Math.exp(-mois / 6))) * 100;
    const fragile = ETAT.style === 'rose' ? 1.8 : ETAT.style === 'blanc' ? 1.3 : 1;
    const fruit = 100 * Math.exp(-mois * (0.015 + c.oxy * 0.03) * fragile);
    let oxy = 100 * (1 - Math.exp(-mois * c.oxy * 0.04 * fragile));
    if (lies.checked) oxy *= 0.6;
    const evap = c.evap * mois / 12;
    const litres = evap / 100 * c.volume;
    const cout = c.cout * (12 / Math.max(mois, 1)) * Math.min(mois, 12) / 12 * (c.bois > 0 ? 1 : 1);
    jBois(bois); jRond(rond); jFruit(fruit); jOxy(oxy);
    dEvap.textContent = mois === 0 ? '—' : `${fmt(evap, 1)} % · ${fmt(litres)} L sur ${fmt(c.volume)}`;
    dCout.textContent = `${fmt(cout, 2)} $`;
    const aromes = [];
    if (bois > 15) aromes.push(ch === 'legere' ? 'vanille, bois frais' : ch === 'moyenne' ? 'vanille, caramel, épices' : 'torréfié, fumé, café');
    if (lies.checked && mois >= 3) aromes.push('brioche, gras');
    if (fruit > 60) aromes.push('fruit frais');
    else if (fruit > 30) aromes.push('fruit mûr');
    else aromes.push('fruit confit');
    if (oxy > 45) aromes.push('noix, pomme blette');
    dNez.textContent = aromes.join(', ');
    let l, cls = '';
    if (mois === 0) l = "<b>Pas d'élevage</b> : le vin va en bouteille dès qu'il est clair. C'est le cas des primeurs et de beaucoup de rosés, faits pour le fruit et pour être bus dans l'année.";
    else if (oxy > 50) { l = `<b>Trop long pour ce vin.</b> Après ${mois} mois dans ce contenant, l'oxygène a pris le dessus : le fruit s'éteint, les notes de noix et de pomme blette apparaissent. C'est voulu pour un vin jaune ou un xérès, pas ici.`; cls = 'alerte'; }
    else if (c.bois >= 1 && bois > 60 && ETAT.style !== 'rouge') { l = "<b>Le bois domine.</b> Une barrique neuve marque vite un blanc ou un rosé ; au-delà de quelques mois, on ne sent plus que la vanille et le toasté. Les vignerons mélangent souvent barriques neuves et usagées pour doser."; cls = 'alerte'; }
    else if (c.bois >= 1 && bois > 70) l = "<b>Élevage boisé classique</b> des grands rouges : la barrique neuve donne ses arômes, l'oxygène fixe la couleur et fond les tanins. Le fruit recule au profit d'un profil épicé et toasté qui demandera quelques années de bouteille pour se fondre.";
    else if (c.oxy <= 0.2) l = ETAT.style === 'rouge' ? "<b>Élevage réducteur</b>, sans bois : le fruit reste entier, les tanins restent ce qu'ils étaient à la sortie de cuve. C'est le style de nombreux rouges de soif et de vins nature ; les tanins durs y restent durs." : "<b>Élevage en cuve</b>, au frais : le vin garde son fruit et sa fraîcheur, avec du gras si on l'a laissé sur ses lies. Le profil de la plupart des blancs et rosés du monde.";
    else l = `<b>Élevage mesuré.</b> ${mois} mois dans ce contenant apportent de l'oxygène sans trop d'arômes de bois : le vin s'arrondit et garde son fruit. Il faudra ouiller régulièrement pour compenser l'évaporation${lies.checked ? ', et bâtonner les lies chaque semaine' : ''}.`;
    lecture.innerHTML = l; lecture.className = `lecture ${cls}`;
  }
  [contenant, chauffe, duree, lies].forEach((el) => el.addEventListener(el.type === 'range' ? 'input' : 'change', maj));
  maj();
  return { maj };
};

/* ---------- Bouchage et SO2 ---------- */

const BOUCHONS = {
  liege: { nom: 'Liège naturel', otr: '1 à 3 mg O₂/an, variable', tca: '≈ 1 %', garde: 1, note: "Le bouchon classique, tiré de l'écorce du chêne-liège. Il laisse passer un filet d'oxygène qui varie d'un bouchon à l'autre : deux bouteilles du même vin vieillissent différemment. C'est aussi le seul qui puisse être contaminé par le TCA, cette molécule qui sent le carton moisi." },
  technique: { nom: 'Liège micro-aggloméré', otr: '0,5 à 1 mg O₂/an, régulier', tca: 'quasi nul', garde: 0.9, note: "Des granules de liège lavées au CO₂ supercritique pour éliminer le TCA, puis collées : la régularité de la capsule à vis avec l'image du liège. Il équipe la majorité des bouteilles de milieu de gamme." },
  vis: { nom: 'Capsule à vis', otr: '0,3 à 1 mg O₂/an selon le joint', tca: 'nul', garde: 1, note: "Étanche, régulière, sans goût de bouchon ; le joint (étain ou Saranex) règle la quantité d'oxygène. Généralisée en Australie et en Nouvelle-Zélande, sur les blancs surtout. Un vin trop réducteur peut y sentir l'allumette ; on dose le soufre en conséquence." },
  synthetique: { nom: 'Bouchon synthétique', otr: '3 à 5 mg O₂/an', tca: 'nul', garde: 0.4, note: "Un polymère extrudé, bon marché, sans TCA, mais perméable : au-delà de deux ou trois ans, le vin s'oxyde. Réservé aux vins à boire jeunes." },
};

ATELIERS.bouchage = (boite) => {
  atelierEntete(boite, 'Atelier', 'Bouchon et soufre à la mise');
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const bouchon = selection(reglages, { id: 'bBouchon', label: 'Bouchon', valeur: 'liege', options: Object.entries(BOUCHONS).map(([k, b]) => [k, b.nom]) });
  const ph = curseur(reglages, { id: 'bPh', label: 'pH du vin', min: 2.9, max: 4, step: 0.05, valeur: ETAT.style === 'rouge' ? 3.6 : 3.2, affiche: (v) => fmt(v, 2) });
  const so2 = curseur(reglages, { id: 'bSo2', label: 'SO₂ libre à la mise', min: 0, max: 60, step: 5, valeur: ETAT.style === 'rouge' ? 25 : 35, affiche: (v) => `${v} mg/L` });
  const mesures = h('dl', { class: 'mesures' });
  const dOtr = h('dd'), dTca = h('dd'), dMol = h('dd', { 'data-testid': 'so2-moleculaire' }), dGarde = h('dd', { class: 'grand', 'data-testid': 'garde' });
  mesures.append(h('dt', {}, 'Oxygène qui traverse le bouchon'), dOtr, h('dt', {}, 'Risque de goût de bouchon'), dTca, h('dt', {}, 'SO₂ moléculaire (la fraction active)'), dMol, h('dt', {}, 'Potentiel de garde'), dGarde);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture' });
  boite.append(lecture);
  function maj() {
    const b = BOUCHONS[bouchon.value];
    const mol = +so2.value / (1 + 10 ** (+ph.value - 1.81));
    dOtr.textContent = b.otr; dTca.textContent = b.tca;
    dMol.textContent = `${fmt(mol, 2)} mg/L`;
    dMol.className = mol < 0.4 ? 'alerte' : mol > 1.2 ? 'alerte' : 'bon';
    const base = ETAT.style === 'rouge' ? 8 : ETAT.style === 'blanc' ? 4 : 2;
    let garde = base * b.garde;
    if (mol < 0.4) garde *= 0.6;
    if (+ph.value > 3.8) garde *= 0.7;
    dGarde.textContent = garde < 1.5 ? "l'année" : `${fmt(garde)} ans environ`;
    let l = `<b>${b.nom}.</b> ${b.note}`;
    if (mol < 0.4) l += ` <b>Soufre insuffisant</b> : à pH ${fmt(+ph.value, 2)}, ${so2.value} mg/L de SO₂ libre ne laissent que ${fmt(mol, 2)} mg/L de forme moléculaire, celle qui bloque vraiment les microbes et l'oxydation. Il faut viser 0,5 à 0,8 mg/L : plus le pH est haut, plus il faut de soufre.`;
    else if (mol > 1.2) l += ` <b>Soufre excessif</b> : au-delà de 1 mg/L moléculaire, le vin pique le nez et l'arrière-gorge. À ce pH, on peut réduire la dose.`;
    else l += ` À pH ${fmt(+ph.value, 2)}, ${so2.value} mg/L de SO₂ libre donnent ${fmt(mol, 2)} mg/L de forme moléculaire : le vin est protégé sans être marqué.`;
    lecture.innerHTML = l; lecture.className = `lecture ${mol < 0.4 || mol > 1.2 ? 'alerte' : ''}`;
  }
  [bouchon, ph, so2].forEach((el) => el.addEventListener(el.type === 'range' ? 'input' : 'change', maj));
  maj();
  return { maj };
};

/* ---------- Garde ---------- */

ATELIERS.garde = (boite) => {
  atelierEntete(boite, 'Atelier', 'Ouvrir la bouteille dans…');
  const echantillon = h('div', { class: 'echantillon', style: 'margin-top:0;height:70px', 'data-testid': 'robe' });
  boite.append(echantillon);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const annees = curseur(reglages, { id: 'gAnnees', label: 'Années en cave', min: 0, max: 30, valeur: 2, affiche: (v) => v === 0 ? 'à la mise' : `${v} an${v > 1 ? 's' : ''}` });
  const potentiel = selection(reglages, { id: 'gPotentiel', label: 'Vin de départ', valeur: 'moyen', options: [['leger', 'Léger, à boire jeune'], ['moyen', 'Vin de garde moyenne'], ['grand', 'Grand vin de garde']] });
  const jauges = h('div', { class: 'jauges' });
  const jPrim = jauge(jauges, 'Arômes primaires', 'vert');
  const jSec = jauge(jauges, 'Arômes secondaires', 'or');
  const jTer = jauge(jauges, 'Arômes tertiaires');
  const jPlaisir = jauge(jauges, 'Plaisir à l’ouverture', 'bleu');
  boite.append(jauges);
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-garde' });
  boite.append(lecture);
  function maj() {
    const a = +annees.value;
    const apogee = { leger: 1.5, moyen: 6, grand: 18 }[potentiel.value] * (ETAT.style === 'rouge' ? 1 : ETAT.style === 'blanc' ? 0.7 : 0.35);
    const x = a / apogee;
    const prim = 100 * Math.exp(-x * 1.1);
    const sec = 100 * Math.exp(-((x - 0.6) ** 2) / 0.5);
    const ter = 100 * (1 - Math.exp(-Math.max(0, x - 0.3) * 1.3));
    const plaisir = 100 * Math.exp(-((Math.log(x + 0.15) - Math.log(1.15)) ** 2) / 0.9);
    jPrim(prim); jSec(sec); jTer(ter); jPlaisir(plaisir);
    // robe
    let couleur, nom;
    if (ETAT.style === 'rouge') {
      const t2 = clamp(x / 2.5, 0, 1);
      couleur = `rgb(${Math.round(lerp(110, 150, t2))},${Math.round(lerp(20, 60, t2))},${Math.round(lerp(70, 30, t2))})`;
      nom = x < 0.3 ? 'violacé' : x < 0.9 ? 'rubis' : x < 1.6 ? 'grenat' : 'tuilé';
    } else if (ETAT.style === 'blanc') {
      const t2 = clamp(x / 2.5, 0, 1);
      couleur = `rgb(${Math.round(lerp(235, 200, t2))},${Math.round(lerp(228, 150, t2))},${Math.round(lerp(160, 60, t2))})`;
      nom = x < 0.3 ? 'jaune pâle à reflets verts' : x < 0.9 ? 'jaune paille' : x < 1.6 ? 'or' : 'ambré';
    } else {
      const t2 = clamp(x / 2.5, 0, 1);
      couleur = `rgb(${Math.round(lerp(240, 215, t2))},${Math.round(lerp(160, 150, t2))},${Math.round(lerp(170, 110, t2))})`;
      nom = x < 0.5 ? 'rose framboise' : x < 1.2 ? 'saumon' : 'pelure d’oignon';
    }
    echantillon.style.background = couleur; echantillon.textContent = `robe ${nom}`;
    const tertiaires = ETAT.style === 'rouge' ? 'sous-bois, cuir, tabac, truffe' : ETAT.style === 'blanc' ? 'miel, noix, cire, pétrole' : 'orangette, abricot sec';
    const primaires = ETAT.style === 'rouge' ? 'fruits rouges et noirs, violette' : ETAT.style === 'blanc' ? 'agrumes, fleurs blanches, pomme' : 'fraise, pamplemousse';
    let l, cls = '';
    if (x < 0.15) { l = `<b>Trop jeune, ou juste jeune.</b> Le vin sent le fruit (${primaires}) et, s'il est passé sous bois, la vanille. ${ETAT.style === 'rouge' ? 'Les tanins sont encore serrés.' : "L'acidité domine, c'est vif."} Beaucoup de vins ne demandent rien de plus.`; }
    else if (x < 0.7) { l = `<b>En montée.</b> Le fruit frais recule, les arômes d'élevage se fondent, ${ETAT.style === 'rouge' ? 'les tanins s’assouplissent' : 'le vin prend du volume'} et les premières notes tertiaires (${tertiaires}) pointent. On peut ouvrir, on peut attendre.`; cls = 'bon'; }
    else if (x < 1.5) { l = `<b>Apogée.</b> L'équilibre entre ce qui reste du fruit et ce que le temps a fabriqué : ${tertiaires}. La robe est ${nom}. C'est la fenêtre qu'on attendait, et elle dure des années pour un grand vin, des mois pour un léger.`; cls = 'bon'; }
    else if (x < 2.5) { l = `<b>Sur le déclin.</b> Le fruit a disparu, les tertiaires dominent, l'acidité et ${ETAT.style === 'rouge' ? 'les tanins' : 'l’alcool'} ressortent. Encore intéressant pour qui aime les vieux vins, décevant pour les autres. Un dépôt s'est formé : on décante.`; cls = 'alerte'; }
    else { l = `<b>Passé.</b> Robe ${nom}, arômes de madère, de pomme blette ; le vin est oxydé et fatigué. Il aurait fallu l'ouvrir vers ${fmt(apogee)} an${apogee >= 2 ? 's' : ''}.`; cls = 'alerte'; }
    lecture.innerHTML = l; lecture.className = `lecture ${cls}`;
  }
  annees.addEventListener('input', maj);
  potentiel.addEventListener('change', maj);
  maj();
  return { maj };
};

/* ======================================================================
   Démarrage
   ====================================================================== */

$$('#choixStyle button').forEach((b) => b.addEventListener('click', () => choisirStyle(b.dataset.style)));

const aide = $('#aide');
$('#btnAide').addEventListener('click', () => aide.showModal());
aide.addEventListener('click', (e) => { if (e.target === aide) aide.close(); });
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.key === 'h' || e.key === 'H') { e.preventDefault(); aide.open ? aide.close() : aide.showModal(); }
});

const bandeau = $('.bandeau');
const majBandeau = () => document.documentElement.style.setProperty('--bandeau-h', `${bandeau.offsetHeight}px`);
new ResizeObserver(majBandeau).observe(bandeau);
majBandeau();

rendre();
API.choisirStyle = choisirStyle;
API.stades = () => stadesVisibles().map((s) => s.id);
window.vin = API;
