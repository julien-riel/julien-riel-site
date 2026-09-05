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
    id: 'climat',
    titre: 'Le climat',
    duree: 'Avant tout',
    intro: "Avant le cep, il y a le ciel. La vigne cultivée pousse pour l'essentiel entre le 30<sup>e</sup> et le 50<sup>e</sup> parallèle, là où la saison est assez longue et chaude pour mûrir le raisin, et l'hiver assez marqué pour la mettre au repos. Le climat décide de ce qu'on peut planter et du vin qu'on en tirera : on le résume par la <b>somme des températures</b> de la saison (les degrés-jours d'avril à octobre, l'indice de Winkler), par la pluie et son calendrier, et par l'<b>amplitude entre le jour et la nuit</b>. Chaud le jour, le raisin fait du sucre ; frais la nuit, il garde son acidité et ses arômes.",
    reperes: [
      "<b>Degrés-jours</b> : on additionne, chaque jour d'avril à octobre, ce que la température moyenne dépasse 10 °C. Champagne ≈ 1 100, Bourgogne ≈ 1 350, Bordeaux ≈ 1 650, Napa ≈ 1 900, Jerez > 2 200. Chaque cépage a sa fenêtre.",
      "<b>Pluie</b> : 500 à 800 mm par an suffisent. Le pire n'est pas la quantité mais le moment : de la pluie à la vendange dilue le jus et fait pourrir les grappes.",
      "<b>Continental, océanique, méditerranéen</b> : hivers rudes et étés chauds ; doux et humide ; sec et ensoleillé. Chacun a ses maladies, ses cépages et son style.",
      "<b>Au Québec</b>, la saison compte moins de 1 200 degrés-jours et l'hiver descend à −30 °C : d'où les hybrides rustiques, le buttage et les toiles géotextiles.",
      "<b>Réchauffement</b> : +1 à +2 °C en quarante ans dans la plupart des vignobles. Les vendanges avancent, l'alcool monte, et la vigne gagne l'Angleterre, la Scandinavie et l'altitude.",
    ],
    atelier: 'climat',
  },
  {
    id: 'geographie',
    titre: 'La géographie : latitude, altitude, coteau',
    duree: 'Le lieu',
    intro: "À climat régional égal, le lieu précis change tout. La <b>latitude</b> règle la hauteur du soleil et la longueur des jours. L'<b>altitude</b> refroidit d'environ 0,6 °C tous les 100 mètres : monter de 300 m revient à reculer de deux degrés de latitude, c'est ainsi que Mendoza fait du vin frais à 1 500 m. La <b>pente et l'exposition</b> décident de l'énergie reçue par mètre carré et de l'écoulement de l'air froid, qui glisse vers le bas et épargne le coteau des gelées. Enfin l'<b>eau</b> à proximité, fleuve, lac ou mer, tamponne les températures et réfléchit la lumière. Ce n'est pas un hasard si les grands vignobles historiques sont des coteaux au bord d'un fleuve : Mosel, Rhin, Douro, Rhône, Loire, Côte d'Or.",
    reperes: [
      "<b>Trois échelles</b> : le macroclimat (la région), le mésoclimat (le coteau, la parcelle) et le microclimat (la grappe dans son feuillage). Le vigneron agit surtout sur les deux dernières.",
      "<b>Le tiers supérieur du coteau</b> : en Bourgogne, les grands crus sont à mi-pente, exposés est ou sud-est. Soleil du matin, sol drainé et mince, à l'abri de la poche d'air froid du bas et des sols trop riches de la plaine.",
      "<b>Les plans d'eau</b> : le lac Léman, le Rhin ou le lac Ontario retardent les gelées d'automne et adoucissent le printemps ; les brouillards matinaux du Ciron fabriquent la pourriture noble de Sauternes.",
      "<b>Le vent</b> : le mistral assèche le feuillage et limite les maladies, mais casse les rameaux ; les brises marines rafraîchissent la Californie côtière et la vallée de Casablanca au Chili.",
    ],
    atelier: 'coteau',
  },
  {
    id: 'sol',
    titre: 'Le type de sol',
    duree: 'Sous les racines',
    intro: "Le sol nourrit peu la vigne, mais il la gouverne. Ce qui compte d'abord, c'est le <b>régime hydrique</b> : un bon sol viticole est drainant, plutôt pauvre, assez profond pour que les racines trouvent l'eau en été, mais assez contraignant pour que la plante cesse de pousser à la véraison et consacre ses forces au raisin. Viennent ensuite la <b>texture</b> (argile, limon, sable, cailloux), la <b>roche-mère</b> (calcaire, schiste, granite, graves), le pH, la couleur et la pierrosité, qui règlent la température du sol. Le vin n'a pas le goût du caillou : la vigne absorbe des ions dissous, pas de la roche. Le sol agit par l'eau, l'azote, la chaleur et la vigueur qu'il impose.",
    reperes: [
      "<b>Argile</b> : retient l'eau et les éléments nutritifs, sols froids et lents à réchauffer, vins puissants et charnus (merlot de Pomerol). <b>Sable</b> : drainant, pauvre, vins légers et parfumés ; le phylloxéra n'y survit pas.",
      "<b>Calcaire</b> : drainant en surface, mais la roche fissurée garde une réserve d'eau en profondeur. Son pH élevé bloque le fer et provoque la <b>chlorose</b> (feuilles jaunes) : on choisit un porte-greffe qui le tolère. Craie de Champagne, marnes de Bourgogne, kimméridgien de Chablis.",
      "<b>Graves et galets</b> : cailloux roulés qui drainent et rendent la chaleur la nuit. Cabernet sauvignon du Médoc, grenache de Châteauneuf-du-Pape.",
      "<b>Schiste et ardoise</b> : pauvres, sombres, chauds ; les racines descendent dans les fissures. Riesling de la Mosel, syrah de Côte-Rôtie, Priorat, Douro. <b>Granite</b> : acide, se délite en sable ; gamay du Beaujolais, Muscadet.",
      "<b>Le porte-greffe se choisit pour le sol</b> : Fercal ou 41 B pour le calcaire actif, Riparia pour freiner la vigueur, 110 R ou 140 Ru pour résister à la sécheresse.",
    ],
    atelier: 'sol',
  },
  {
    id: 'vie-du-sol',
    titre: 'Les micro-organismes du sol',
    duree: 'Invisible, permanent',
    intro: "Un gramme de sol vivant contient environ un milliard de bactéries et plusieurs kilomètres de filaments de champignons. Autour des racines, dans la <b>rhizosphère</b>, ils font le travail que la vigne ne sait pas faire. Les <b>mycorhizes</b>, des champignons qui pénètrent les racines fines, prolongent le système racinaire de plusieurs mètres, vont chercher le phosphore, le zinc et l'eau dans des pores trop fins pour les racines, et se paient en sucres : jusqu'à 20 % de la photosynthèse. Les <b>bactéries</b> décomposent la matière organique et libèrent l'azote sous une forme que la plante absorbe ; celles des légumineuses semées entre les rangs le fixent depuis l'air. D'autres champignons (<i>Trichoderma</i>) tiennent en respect les parasites du bois. Les vers de terre creusent, aèrent, enfouissent. De cette vie dépendent la vigueur du cep, sa résistance à la sécheresse et l'azote du moût, dont les levures auront besoin au chai.",
    reperes: [
      "<b>Mycorhization</b> : dans un sol vivant, 80 à 90 % des racines fines de vigne sont colonisées. Le labour profond, les engrais phosphatés et les sols nus la font chuter.",
      "<b>Azote</b> : la minéralisation fournit de 20 à 60 kg par hectare et par an selon la matière organique. Trop d'azote, c'est la vigueur et la pourriture ; pas assez, c'est un moût carencé (moins de 140 mg/L d'azote assimilable), une fermentation languissante et des odeurs de soufre.",
      "<b>Cuivre</b> : la bouillie bordelaise protège du mildiou depuis 1885, mais le cuivre ne se dégrade pas. Les vieux vignobles en accumulent 100 à 500 mg par kilo de sol, toxiques pour les vers de terre et les microbes ; l'Union européenne le plafonne à 4 kg par hectare et par an.",
      "<b>Vers de terre</b> : de 50 à 400 par mètre carré selon les pratiques. Enherbement, compost et couverts semés les multiplient ; désherbage chimique et sols nus les font disparaître.",
      "<b>Le terroir microbien</b> : le séquençage montre des communautés de bactéries et de champignons propres à chaque région, voire à chaque parcelle. Mais la levure du vin, <i>Saccharomyces cerevisiae</i>, est rare sur les baies saines : c'est au chai qu'elle abonde. L'effet du microbiote du sol sur le goût reste un sujet de recherche, pas une certitude.",
    ],
    atelier: 'vieDuSol',
  },
  {
    id: 'rechauffement',
    titre: 'Le climat qui change',
    duree: "L'horizon d'une plantation : 40 ans",
    intro: "Une vigne plantée aujourd'hui produira jusque vers 2065. La question n'est donc pas quel climat il fait ici, mais quel climat il y fera. Depuis les années 1980, la saison végétative s'est réchauffée d'environ <b>1,5 °C</b> dans les vignobles européens, et cela se lit dans les vins : vendanges avancées de <b>deux à trois semaines</b>, degré alcoolique en hausse de <b>1 à 2 % vol.</b>, acidité en baisse, pH en hausse. Le sucre s'accumule désormais plus vite que les tanins et les arômes ne mûrissent : on parle de <b>découplage</b> entre maturité technologique et maturité phénolique, et c'est le problème central de la décennie. Chaque région monte d'un cran sur l'échelle de Winkler, ce qui déplace la carte : l'Angleterre et le sud du Québec deviennent viables, le Sud méditerranéen approche de sa limite.",
    reperes: [
      "<b>+1 °C de moyenne</b> sur la saison, c'est environ <b>200 degrés-jours</b> de plus, une vendange avancée de six ou sept jours et un demi-degré d'alcool en plus à maturité égale.",
      "<b>Le paradoxe du gel</b> : le débourrement avance de cinq jours par degré, la date des dernières gelées ne recule que de deux ou trois. Le risque de gel de printemps <i>augmente</i> avec le réchauffement — 2021 en France : 2 milliards d'euros de pertes.",
      "<b>Acidité et pH</b> : l'acide malique est respiré par la baie d'autant plus vite qu'il fait chaud. Un pH qui passe de 3,4 à 3,8 rend le soufre deux fois moins efficace et ouvre la porte aux bactéries d'altération. D'où l'acidification tartrique, autorisée dans les régions chaudes.",
      "<b>Adaptations</b> : cépages tardifs ou méridionaux (Bordeaux a autorisé en 2021 six variétés dont le touriga nacional et le marselan, plafonnées à 5 % de l'assemblage), altitude et expositions fraîches, taille tardive qui retarde le débourrement, canopée haute, ombrage, porte-greffes résistants à la sécheresse, irrigation là où elle est permise.",
      "<b>Nouveaux vignobles</b> : l'Angleterre est passée de quelques hectares à plus de 4 000, le Québec de 30 à plus de 150 producteurs, et la Scandinavie plante. La vigne remonte vers le nord d'environ <b>50 km par décennie</b>.",
    ],
    atelier: 'rechauffement',
  },
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
    id: 'eau-chaleur',
    titre: "L'eau et la chaleur",
    duree: 'Juin → septembre',
    intro: "Entre la floraison et la vendange, deux variables décident du goût du vin bien avant le chai : <b>ce que la vigne boit</b> et <b>la chaleur qu'elle reçoit</b>. L'eau règle la taille de la baie, et la taille de la baie règle tout le reste : une petite baie a proportionnellement plus de peau que de jus, donc plus de couleur, plus de tanins et plus d'arômes par litre. Une <b>contrainte hydrique modérée</b>, installée après la véraison, arrête la pousse des rameaux, envoie les sucres vers les grappes et déclenche la synthèse des anthocyanes : c'est l'idéal, et c'est étroit. Trop d'eau et la baie gonfle, la vigne continue de pousser, le vin est dilué ; trop peu et la photosynthèse s'arrête, la maturité se bloque, le raisin reste vert et acide malgré le soleil. La chaleur, elle, fait monter le sucre et brûle l'acide malique — la baie le respire d'autant plus vite qu'il fait chaud. Au-delà de <b>30 à 35 °C</b>, la machine s'inverse : les anthocyanes se dégradent, les arômes fins s'évaporent, les grappes exposées s'échaudent.",
    reperes: [
      "<b>Une baie</b>, c'est 75 à 80 % d'eau. Les tanins sont dans la peau (un tiers) et dans les pépins (deux tiers) ; la couleur, uniquement dans la peau ; la pulpe n'apporte que le sucre et les acides.",
      "<b>La bonne contrainte</b> se mesure : potentiel hydrique de base entre −0,4 et −0,6 MPa, lu à la chambre à pression avant le lever du jour. Au-delà de −0,8 MPa, la vigne ferme ses stomates et cesse de fabriquer du sucre.",
      "<b>Consommation</b> : 3 à 5 mm d'eau par jour en juillet, soit 30 à 50 m³ par hectare. Un sol à 150 mm de réserve utile tient donc un mois de sec, un sol caillouteux superficiel, dix jours.",
      "<b>Acide malique</b> : sa respiration double environ à chaque +10 °C. C'est pourquoi les nuits fraîches conservent l'acidité, et pourquoi un été caniculaire donne des vins mous à pH élevé.",
      "<b>Les leviers du vigneron</b> : charge laissée à la taille et vendange verte, hauteur de la canopée (il faut de 1,2 à 1,5 m² de feuilles par kilo de raisin pour mûrir), effeuillage (côté levant pour aérer sans brûler), enherbement qui concurrence l'eau, paillage, filets d'ombrage, et irrigation là où elle est autorisée.",
    ],
    atelier: 'equilibre',
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
      rouge: "Un domaine n'a jamais une seule cuve : il en a une par parcelle, par cépage, par date de vendange, puis des barriques neuves et d'autres usagées, du vin de goutte et du vin de presse. L'<b>assemblage</b> est le moment où l'on goûte tout et où l'on compose le vin final. Ce n'est pas un mélange, c'est un calcul : chaque lot apporte une caractéristique mesurable — degré, acidité totale, pH, indice de tanins, intensité colorante, marque du bois — et le vigneron cherche la combinaison qui approche le mieux le vin qu'il a en tête. À Bordeaux, c'est le mariage du merlot (chair, alcool, rondeur) et du cabernet sauvignon (structure, couleur, garde), corrigé au cabernet franc ; en Bourgogne, un seul cépage, mais on décide quelle barrique entre dans la cuvée et laquelle part dans le second vin.",
      blanc: "Un domaine n'a jamais une seule cuve : il en a une par parcelle, par cépage, par pressée, en inox et en barrique. L'<b>assemblage</b> est le moment où l'on goûte tout et où l'on compose le vin final. Ce n'est pas un mélange, c'est un calcul : chaque lot apporte une caractéristique mesurable — degré, acidité totale, pH, gras, amertume de peau, marque du bois — et le vigneron cherche la combinaison qui approche le mieux le vin qu'il a en tête. La cuve inox pour la tension, la barrique pour le gras, les lies pour le volume, le jus de presse pour la matière ou pas du tout : c'est un exercice de dégustation, avec des éprouvettes, des proportions et une calculatrice.",
      rose: "Même pour un rosé, on assemble, et c'est même là que la couleur se décide. La cuvée de saignée, sombre et structurée, corrige la cuvée de pressurage direct, pâle et fruitée ; le grenache donne le gras, le cinsault la fraîcheur, une pointe de syrah la couleur. L'<b>assemblage</b> vise une teinte précise — souvent mesurée au spectrophotomètre, tant la couleur fait vendre — et un équilibre entre le fruit et l'acidité. Il se fait à la dégustation, avec des éprouvettes et des proportions.",
    },
    reperes: [
      "<b>Pourquoi assembler</b> : la complémentarité (ce qui manque à l'un, l'autre l'a), la régularité (offrir le même vin chaque année malgré les millésimes), la correction (remonter une acidité, diluer un excès de bois) et la sélection (ce qui n'entre pas dans la cuvée fait le second vin).",
      "<b>Les paramètres qu'on ajuste</b> : le degré, l'acidité totale et le pH, l'indice de tanins, l'intensité et la teinte de la couleur, la part de bois neuf, la part de vin de presse, et l'équilibre entre les cépages — souvent encadré par l'appellation.",
      "<b>La technique</b> : on prépare des essais en éprouvette de 100 mL, on les goûte à l'aveugle et en équipe, on ajuste au pourcent près, puis on refait analyser le mélange retenu. Une cuvée se compose <b>à 1 % près</b>, et l'ordre des essais compte : le palais s'habitue.",
      "<b>Le vin de presse</b> : plus tannique, plus coloré, plus riche en potassium que le vin de goutte. On en remet 5 à 15 % pour la structure, rarement plus, et souvent après l'avoir élevé à part.",
      "<b>Assembler tôt ou tard</b> : avant l'élevage, les lots se fondent ensemble pendant des mois mais on juge des vins encore fermés ; après l'élevage, on juge sur pièces mais il faut laisser <b>trois à quatre semaines de mariage</b> en cuve avant la mise, sinon le vin est dissocié.",
      "<b>Ailleurs</b> : le champagne non millésimé assemble jusqu'à cinquante vins clairs et 30 à 40 % de vins de réserve des années précédentes ; le porto et le xérès assemblent des âges différents ; le rioja assemble des durées de fût. À l'inverse, un vin de parcelle unique revendique de n'être assemblé avec rien.",
    ],
    atelier: 'assemblage',
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

/* ======================================================================
   Terroir : climat, coteau, sol, vie du sol
   ====================================================================== */

/* Petit générateur pseudo-aléatoire déterministe pour des dessins stables. */
function alea(graine) {
  let s = graine >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

const WINKLER = [
  { max: 1111, nom: 'Ia', qualif: 'très frais', cepages: 'hybrides rustiques, müller-thurgau, riesling à la limite', style: 'effervescents, blancs vifs, rouges légers et pâles', lieux: 'sud du Québec, Angleterre, Mosel' },
  { max: 1389, nom: 'Ib', qualif: 'frais', cepages: 'pinot noir, chardonnay, riesling, gamay, chenin', style: 'blancs tendus, rouges fins et peu colorés, bases d’effervescents', lieux: 'Champagne, Bourgogne, Alsace, Loire, Willamette' },
  { max: 1667, nom: 'II', qualif: 'tempéré', cepages: 'merlot, cabernet franc, sauvignon, syrah, nebbiolo', style: 'rouges équilibrés à 12,5–13,5 %, blancs secs amples', lieux: 'Bordeaux, Rhône nord, Rioja, Piémont' },
  { max: 1944, nom: 'III', qualif: 'chaud', cepages: 'cabernet sauvignon, syrah, sangiovese, grenache', style: 'rouges corsés et colorés, blancs ronds', lieux: 'Napa, Toscane, Rhône sud, Stellenbosch' },
  { max: 2222, nom: 'IV', qualif: 'très chaud', cepages: 'grenache, mourvèdre, zinfandel, carignan', style: 'rouges chaleureux à 14 % et plus, rosés de soif', lieux: 'Languedoc, Barossa, Mendoza' },
  { max: Infinity, nom: 'V', qualif: 'torride', cepages: 'palomino, muscat, raisins de table', style: 'vins mutés et doux, gros volumes irrigués', lieux: 'Jerez, Vallée centrale de Californie, intérieur de l’Australie' },
];
const regionWinkler = (dj) => WINKLER.find((r) => dj < r.max);
const COULEURS_WINKLER = ['#3f6f8f', '#5f8fa8', '#8fbf6a', '#d9b25f', '#e0705d', '#9b2f4c'];
const JOURS_SAISON = 214;   // du 1er avril au 31 octobre

/* ---------- Climat : l'échelle de Winkler ---------- */

ATELIERS.climat = (boite) => {
  atelierEntete(boite, 'Atelier', 'Situer un vignoble sur l’échelle de Winkler');
  const canvas = h('canvas', { 'aria-label': 'Échelle des degrés-jours de Winkler, avec quelques vignobles repères' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const temp = curseur(reglages, { id: 'kTemp', label: 'Température moyenne, avril → octobre', min: 12, max: 22, step: 0.5, valeur: 16.5, affiche: (v) => `${fmt(v, 1)} °C` });
  const ampl = curseur(reglages, { id: 'kAmpl', label: 'Écart jour / nuit en été', min: 5, max: 18, valeur: 10, affiche: (v) => `${v} °C` });
  const pluie = curseur(reglages, { id: 'kPluie', label: 'Pluie annuelle', min: 300, max: 1200, step: 50, valeur: 700, affiche: (v) => `${fmt(v)} mm` });
  const automne = h('input', { type: 'checkbox', id: 'kAutomne' });
  reglages.append(h('div', { class: 'reglage' }, h('span', {}, 'Vendanges'), h('label', { class: 'case' }, automne, 'Automnes souvent pluvieux')));
  const mesures = h('dl', { class: 'mesures' });
  const dDj = h('dd', { class: 'grand', 'data-testid': 'degres-jours' }), dRegion = h('dd', { 'data-testid': 'winkler' }), dCepages = h('dd', { style: 'text-align:left;grid-column:1 / -1;font-family:var(--ui)' });
  mesures.append(h('dt', {}, 'Degrés-jours (base 10 °C)'), dDj, h('dt', {}, 'Région de Winkler'), dRegion, h('dt', { style: 'grid-column:1 / -1' }, 'Cépages qui y mûrissent'), dCepages);
  boite.append(mesures);
  const jauges = h('div', { class: 'jauges' });
  boite.append(jauges);
  const jTardif = jauge(jauges, 'Mûrir un cépage tardif', 'or');
  const jAcide = jauge(jauges, 'Garder l’acidité', 'vert');
  const jPourri = jauge(jauges, 'Pression des pourritures', 'rougeb');
  const jSec = jauge(jauges, 'Manque d’eau en été', 'bleu');
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-climat' });
  boite.append(lecture);

  const REPERES = [[1050, 'Québec'], [1100, 'Champagne'], [1350, 'Bourgogne'], [1650, 'Bordeaux'], [1900, 'Napa'], [2100, 'Languedoc'], [2350, 'Jerez']];
  function dessiner(dj) {
    const L = 520, H = 118, g = 14, d = 14, y0 = 46, eh = 22;
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#120c0e'; ctx.fillRect(0, 0, L, H);
    const X = (v) => g + clamp((v - 800) / 1800, 0, 1) * (L - g - d);
    let debut = 800;
    WINKLER.forEach((r, i) => {
      const fin = Math.min(r.max, 2600);
      ctx.fillStyle = COULEURS_WINKLER[i]; ctx.globalAlpha = .85;
      ctx.fillRect(X(debut), y0, X(fin) - X(debut), eh);
      ctx.globalAlpha = 1; ctx.fillStyle = '#120c0e'; ctx.font = '600 10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
      ctx.fillText(r.nom, (X(debut) + X(fin)) / 2, y0 + 15);
      debut = fin;
    });
    ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.fillStyle = '#a89a95'; ctx.textAlign = 'center';
    for (let v = 800; v <= 2600; v += 300) { ctx.fillText(fmt(v), X(v), H - 6); ctx.fillRect(X(v) - .5, y0 + eh, 1, 4); }
    REPERES.forEach(([v, nom], i) => {
      const x = X(v), yy = i % 2 ? 22 : 10;
      ctx.fillStyle = '#55414a'; ctx.fillRect(x - .5, yy + 3, 1, y0 - yy - 3);
      ctx.fillStyle = '#dccfc9'; ctx.font = '500 9px Outfit, sans-serif'; ctx.fillText(nom, x, yy);
    });
    const x = X(dj);
    ctx.fillStyle = '#f3ebe4'; ctx.beginPath(); ctx.moveTo(x, y0 - 2); ctx.lineTo(x - 6, y0 - 10); ctx.lineTo(x + 6, y0 - 10); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.moveTo(x, y0 + eh + 2); ctx.lineTo(x - 6, y0 + eh + 10); ctx.lineTo(x + 6, y0 + eh + 10); ctx.closePath(); ctx.fill();
  }
  function maj() {
    const T = +temp.value, A = +ampl.value, P = +pluie.value, aut = automne.checked;
    const dj = Math.round((T - 10) * JOURS_SAISON);
    const r = regionWinkler(dj);
    dDj.textContent = `${fmt(dj)} °C·j`;
    dRegion.textContent = `${r.nom} · ${r.qualif}`;
    dCepages.textContent = r.cepages;
    jTardif(clamp((dj - 1150) / 9, 0, 100));
    const acide = clamp(55 + (A - 10) * 4 - (T - 16) * 9, 0, 100);
    jAcide(acide);
    const pourri = clamp(P / 15 - 15 + (aut ? 35 : 0) - (A - 10) * 1.5, 0, 100);
    jPourri(pourri);
    const sec = clamp((750 - P) / 4.5 + (T - 16) * 5, 0, 100);
    jSec(sec);
    let l = `<b>Région ${r.nom}, climat ${r.qualif}.</b> ${r.style[0].toUpperCase()}${r.style.slice(1)} ; on est dans les conditions de ${r.lieux}. `;
    if (A >= 13) l += "Les <b>nuits fraîches</b> font le reste : l'acide malique et les arômes tiennent jusqu'à la vendange, les rouges gardent leur couleur. ";
    else if (A <= 7) l += "Les <b>nuits restent chaudes</b> : la vigne respire son acide malique, les vins sont mous et il faudra vendanger tôt, souvent de nuit. ";
    if (P < 450) l += "Avec si peu de pluie, la vigne ne survit qu'<b>irriguée</b> ou sur un sol qui garde l'eau, comme en Espagne intérieure ou à Mendoza.";
    else if (P > 900) l += "Beaucoup de pluie : le <b>mildiou</b> et la pourriture grise imposent des traitements répétés et un feuillage aéré ; c'est le lot des climats océaniques.";
    else if (aut) l += "Les <b>automnes pluvieux</b> forcent la main : on vendange dès que le sucre y est, avant que la pourriture s'installe, quitte à perdre en maturité.";
    else l += "La pluie est suffisante et bien placée : la contrainte hydrique estivale est modérée, ce que la vigne préfère.";
    lecture.innerHTML = l;
    lecture.className = `lecture ${dj < 1000 || dj > 2300 ? 'alerte' : dj >= 1250 && dj <= 1950 ? 'bon' : ''}`;
    dessiner(dj);
  }
  [temp, ampl, pluie].forEach((c) => c.addEventListener('input', maj));
  automne.addEventListener('change', maj);
  maj();
  return { maj };
};

/* ---------- Réchauffement : faire vieillir un vignoble ---------- */

const REGIONS_CLIMAT = {
  quebec: { nom: 'Sud du Québec', dj: 1020, cepage: 'hybrides rustiques (frontenac, marquette, vidal)', alcool: 11.6, acidite: 5.4, ph: 3.1, vendange: [10, 2], gel: 45, sec: 8 },
  champagne: { nom: 'Champagne', dj: 1110, cepage: 'pinot noir, meunier, chardonnay', alcool: 10.4, acidite: 5.6, ph: 3.02, vendange: [9, 24], gel: 40, sec: 12 },
  bourgogne: { nom: 'Bourgogne', dj: 1340, cepage: 'pinot noir, chardonnay', alcool: 12.4, acidite: 4.4, ph: 3.28, vendange: [9, 22], gel: 35, sec: 18 },
  niagara: { nom: 'Niagara-on-the-Lake', dj: 1420, cepage: 'riesling, chardonnay, cabernet franc', alcool: 12.2, acidite: 4.6, ph: 3.22, vendange: [10, 5], gel: 30, sec: 15 },
  bordeaux: { nom: 'Bordelais', dj: 1640, cepage: 'merlot, cabernet sauvignon', alcool: 12.9, acidite: 3.9, ph: 3.42, vendange: [9, 28], gel: 25, sec: 25 },
  rhone: { nom: 'Rhône sud', dj: 2020, cepage: 'grenache, syrah, mourvèdre', alcool: 14.2, acidite: 3.3, ph: 3.62, vendange: [9, 12], gel: 12, sec: 55 },
};
const EQUIVALENTS = [[1020, 'du sud du Québec'], [1110, 'de la Champagne'], [1340, 'de la Bourgogne'], [1640, 'du Bordelais'], [1900, 'de la Napa'], [2100, 'du Languedoc'], [2350, 'de Jerez']];

ATELIERS.rechauffement = (boite) => {
  atelierEntete(boite, 'Atelier', 'Réchauffer un vignoble, degré par degré');
  const canvas = h('canvas', { 'aria-label': 'Déplacement d’un vignoble sur l’échelle de Winkler sous l’effet du réchauffement' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const region = selection(reglages, { id: 'wRegion', label: 'Vignoble', valeur: 'bourgogne', options: Object.entries(REGIONS_CLIMAT).map(([k, r]) => [k, r.nom]) });
  const hausse = curseur(reglages, { id: 'wHausse', label: 'Réchauffement depuis 1980', min: 0, max: 4, step: 0.1, valeur: 1.5, affiche: (v) => `+${fmt(v, 1)} °C` });
  const cepage = h('input', { type: 'checkbox', id: 'wCepage' });
  const altitude = h('input', { type: 'checkbox', id: 'wAltitude' });
  const conduite = h('input', { type: 'checkbox', id: 'wConduite' });
  boite.append(h('div', { class: 'reglages une' },
    h('div', { class: 'reglage' },
      h('span', {}, 'Adaptations'),
      h('label', { class: 'case' }, cepage, 'Cépage plus tardif ou méridional'),
      h('label', { class: 'case' }, altitude, 'Monter de 200 m, ou exposition fraîche'),
      h('label', { class: 'case' }, conduite, 'Taille tardive, canopée haute, ombrage'))));
  const mesures = h('dl', { class: 'mesures' });
  const dDj = h('dd', { class: 'grand', 'data-testid': 'dj-rechauffe' });
  const dEquiv = h('dd', { 'data-testid': 'equivalent' });
  const dVendange = h('dd', { 'data-testid': 'avance-vendange' });
  const dAlcool = h('dd', { 'data-testid': 'alcool-rechauffe' });
  const dAcide = h('dd');
  const dPh = h('dd');
  mesures.append(
    h('dt', {}, 'Degrés-jours de la saison'), dDj,
    h('dt', {}, 'Climat des années 1980 équivalent'), dEquiv,
    h('dt', {}, 'Date de vendange'), dVendange,
    h('dt', {}, 'Alcool à maturité égale'), dAlcool,
    h('dt', {}, 'Acidité totale (éq. H₂SO₄)'), dAcide,
    h('dt', {}, 'pH'), dPh);
  boite.append(mesures);
  const jauges = h('div', { class: 'jauges' });
  const jAdeq = jauge(jauges, 'Cépage encore adapté', 'vert');
  const jFrais = jauge(jauges, 'Fraîcheur aromatique', 'bleu');
  const jGel = jauge(jauges, 'Risque de gel de printemps', 'rougeb');
  const jSec = jauge(jauges, 'Stress hydrique estival', 'or');
  boite.append(jauges);
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-rechauffement' });
  boite.append(lecture);

  function dessiner(djBase, dj) {
    const L = 520, H = 112, g = 14, d = 14, y0 = 52, eh = 22;
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#120c0e'; ctx.fillRect(0, 0, L, H);
    const X = (v) => g + clamp((v - 800) / 1800, 0, 1) * (L - g - d);
    let debut = 800;
    WINKLER.forEach((r, i) => {
      const fin = Math.min(r.max, 2600);
      ctx.fillStyle = COULEURS_WINKLER[i]; ctx.globalAlpha = .85;
      ctx.fillRect(X(debut), y0, X(fin) - X(debut), eh);
      ctx.globalAlpha = 1; ctx.fillStyle = '#120c0e'; ctx.font = '600 10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
      ctx.fillText(r.nom, (X(debut) + X(fin)) / 2, y0 + 15);
      debut = fin;
    });
    ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.fillStyle = '#a89a95'; ctx.textAlign = 'center';
    for (let v = 800; v <= 2600; v += 300) { ctx.fillText(fmt(v), X(v), H - 6); ctx.fillRect(X(v) - .5, y0 + eh, 1, 4); }
    // flèche du climat de 1980 vers le climat actuel
    const x0 = X(djBase), x1 = X(dj), y = 28;
    ctx.strokeStyle = '#55414a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y + 6); ctx.lineTo(x0, y0 - 2); ctx.stroke();
    ctx.fillStyle = '#a89a95'; ctx.font = '500 9px Outfit, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('1980', x0, y - 10);
    if (Math.abs(x1 - x0) > 2) {
      ctx.strokeStyle = '#f3ebe4'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      const s = Math.sign(x1 - x0);
      ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x1 - 6 * s, y - 4); ctx.lineTo(x1 - 6 * s, y + 4); ctx.closePath();
      ctx.fillStyle = '#f3ebe4'; ctx.fill();
    }
    ctx.fillStyle = '#f3ebe4'; ctx.textAlign = 'center'; ctx.font = '600 9px Outfit, sans-serif';
    ctx.fillText('aujourd’hui', x1, y - 10);
    ctx.beginPath(); ctx.moveTo(x1, y0 - 2); ctx.lineTo(x1 - 6, y0 - 11); ctx.lineTo(x1 + 6, y0 - 11); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.moveTo(x1, y0 + eh + 2); ctx.lineTo(x1 - 6, y0 + eh + 11); ctx.lineTo(x1 + 6, y0 + eh + 11); ctx.closePath(); ctx.fill();
  }

  function maj() {
    const base = REGIONS_CLIMAT[region.value];
    const dT = +hausse.value;
    const dTeff = dT - (altitude.checked ? 1.2 : 0);
    const dj = Math.round(base.dj + 195 * dTeff);
    const r = regionWinkler(dj);
    const avance = 6.5 * dTeff - (cepage.checked ? 8 : 0) - (conduite.checked ? 10 : 0);
    const jour = clamp(D(base.vendange[0], base.vendange[1]) - Math.round(avance), 200, 320);
    const alcool = base.alcool + 0.55 * dTeff - (cepage.checked ? 0.5 : 0) - (conduite.checked ? 0.6 : 0);
    const acidite = Math.max(1.8, base.acidite - 0.3 * dTeff + (cepage.checked ? 0.2 : 0) + (conduite.checked ? 0.25 : 0));
    const ph = base.ph + 0.055 * dTeff - (cepage.checked ? 0.03 : 0) - (conduite.checked ? 0.03 : 0);
    const cible = base.dj + (cepage.checked ? 330 : 0);
    const adeq = clamp(100 - Math.abs(dj - cible) / 7, 0, 100);
    const frais = clamp(100 - 15 * dTeff + (conduite.checked ? 12 : 0) + (cepage.checked ? 6 : 0), 0, 100);
    const gel = clamp(base.gel + 11 * dT - (conduite.checked ? 12 : 0) + (altitude.checked ? 10 : 0), 0, 100);
    const sec = clamp(base.sec + 17 * dT - (altitude.checked ? 6 : 0) - (cepage.checked ? 8 : 0) + (conduite.checked ? 4 : 0), 0, 100);
    const equiv = EQUIVALENTS.reduce((a, b) => Math.abs(b[0] - dj) < Math.abs(a[0] - dj) ? b : a);

    dDj.textContent = `${fmt(dj)} °C·j · ${r.nom}`;
    dEquiv.textContent = `celui ${equiv[1]}`;
    dVendange.textContent = `${dateDuJour(jour)} · ${avance >= 0 ? `${fmt(avance)} j plus tôt` : `${fmt(-avance)} j plus tard`}`;
    dAlcool.textContent = `${fmt(alcool, 1)} % vol.`;
    dAlcool.className = alcool > 14.5 ? 'alerte' : '';
    dAcide.textContent = `${fmt(acidite, 1)} g/L`;
    dPh.textContent = fmt(ph, 2);
    dPh.className = ph > 3.75 ? 'alerte' : '';
    jAdeq(adeq); jFrais(frais); jGel(gel); jSec(sec);

    let l, cls = '';
    if (dT < 0.4) { l = `<b>Le climat de départ.</b> ${base.nom} des années 1980 : ${fmt(base.dj)} degrés-jours, ${base.cepage}, vendange vers le ${dateDuJour(D(base.vendange[0], base.vendange[1]))}. C'est la référence dont sont issues les appellations, les cépages autorisés et les dates de vendange qu'on croit immuables.`; }
    else if (adeq > 70) { l = `<b>Le vignoble tient.</b> Avec ${fmt(dj)} degrés-jours, ${base.nom} reste dans la fenêtre de ${cepage.checked ? 'ce cépage plus tardif' : 'ses cépages historiques'} ; le vin est un peu plus alcoolique, un peu moins acide, mais reconnaissable. C'est la situation de la plupart des régions aujourd'hui — et le moment d'agir, pas de conclure.`; cls = 'bon'; }
    else if (dj > 2200) { l = `<b>Au-delà de la vigne de qualité.</b> ${fmt(dj)} degrés-jours, c'est le climat ${equiv[1]} : région V de Winkler, celle des vins mutés et des gros volumes irrigués. Les cépages fins y perdent leurs arômes et leur acidité avant d'avoir des tanins mûrs. Il faudrait changer de métier, de lieu, ou de définition du vin.`; cls = 'alerte'; }
    else { l = `<b>Le vignoble a changé de région.</b> ${fmt(dj)} degrés-jours : ${base.nom} cultive désormais sous le climat ${equiv[1]} des années 1980. ${cepage.checked ? "Le cépage tardif rattrape une partie du décalage" : "Les cépages historiques mûrissent trop vite"} : la vendange tombe ${fmt(Math.abs(avance))} jours ${avance >= 0 ? 'plus tôt' : 'plus tard'}, en pleine chaleur, et il faut vendanger de nuit pour rentrer un raisin frais.`; cls = 'alerte'; }
    if (alcool > 14.5) l += ` À ${fmt(alcool, 1)} % vol., le vin sort de l'équilibre attendu : on désalcoolise, on vendange plus tôt au risque de tanins verts, ou on assume un autre style.`;
    if (ph > 3.75) l += ` À pH ${fmt(ph, 2)}, le soufre perd son efficacité et les bactéries d'altération prospèrent : l'acidification tartrique devient une routine de chai.`;
    if (gel > 55) l += ` Et le paradoxe reste entier : ${fmt(gel)} % de risque de gel de printemps, parce que la vigne débourre plus tôt sans que les gelées tardives aient reculé d'autant.`;
    lecture.innerHTML = l;
    lecture.className = `lecture ${cls}`;
    dessiner(base.dj, dj);
  }
  region.addEventListener('change', maj);
  hausse.addEventListener('input', maj);
  [cepage, altitude, conduite].forEach((c) => c.addEventListener('change', maj));
  maj();
  return { maj };
};

/* ---------- Géographie : le coteau ---------- */

const EXPOS = { N: 0, NE: 45, E: 90, SE: 135, S: 180, SO: 225, O: 270, NO: 315 };
const rad = (d) => d * Math.PI / 180;

/* Énergie solaire directe reçue sur la saison par une pente, rapportée au terrain plat. */
function energieRelative(lat, pente, expo) {
  const nx = Math.sin(rad(pente)) * Math.sin(rad(expo));   // composante est de la normale
  const ny = Math.sin(rad(pente)) * Math.cos(rad(expo));   // composante nord
  const nz = Math.cos(rad(pente));
  let sPente = 0, sPlat = 0;
  for (const dec of [4, 12, 20, 23, 19, 10, 0]) {           // déclinaison type d'avril à octobre
    for (let H = -180; H <= 180; H += 5) {
      const sinEl = Math.sin(rad(lat)) * Math.sin(rad(dec)) + Math.cos(rad(lat)) * Math.cos(rad(dec)) * Math.cos(rad(H));
      if (sinEl <= 0) continue;
      const el = Math.asin(sinEl), cosEl = Math.cos(el);
      let cosA = (Math.sin(rad(dec)) - sinEl * Math.sin(rad(lat))) / (cosEl * Math.cos(rad(lat)));
      cosA = clamp(cosA, -1, 1);
      const A = H < 0 ? Math.acos(cosA) : 2 * Math.PI - Math.acos(cosA);   // azimut depuis le nord, matin à l'est
      const sx = cosEl * Math.sin(A), sy = cosEl * Math.cos(A), sz = sinEl;
      sPlat += sz;
      sPente += Math.max(0, sx * nx + sy * ny + sz * nz);
    }
  }
  return sPlat ? sPente / sPlat : 1;
}

ATELIERS.coteau = (boite) => {
  atelierEntete(boite, 'Atelier', 'Incliner et orienter un coteau');
  const canvas = h('canvas', { 'aria-label': 'Coupe d’un coteau planté de vignes, avec le soleil et la poche d’air froid du bas' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const lat = curseur(reglages, { id: 'gLat', label: 'Latitude (hémisphère Nord)', min: 30, max: 52, step: 0.5, valeur: 47, affiche: (v) => `${fmt(v, 1)}°` });
  const alt = curseur(reglages, { id: 'gAlt', label: 'Altitude', min: 0, max: 1500, step: 25, valeur: 250, affiche: (v) => `${fmt(v)} m` });
  const pente = curseur(reglages, { id: 'gPente', label: 'Pente', min: 0, max: 35, valeur: 12, affiche: (v) => v === 0 ? 'plat' : `${v}° (${fmt(Math.tan(rad(v)) * 100)} %)` });
  const expo = selection(reglages, { id: 'gExpo', label: 'Exposition', valeur: 'SE', options: Object.keys(EXPOS).map((k) => [k, k === 'N' ? 'Nord' : k === 'S' ? 'Sud' : k === 'E' ? 'Est' : k === 'O' ? 'Ouest' : k.replace('N', 'Nord-').replace('S', 'Sud-').replace('E', 'est').replace('O', 'ouest')]) });
  const eau = h('input', { type: 'checkbox', id: 'gEau' });
  reglages.append(h('div', { class: 'reglage' }, h('span', {}, 'Environnement'), h('label', { class: 'case' }, eau, 'Fleuve ou lac en contrebas')));
  const mesures = h('dl', { class: 'mesures' });
  const dSoleil = h('dd'), dEnergie = h('dd', { class: 'grand', 'data-testid': 'energie' }), dTemp = h('dd'), dDj = h('dd', { 'data-testid': 'dj-coteau' }), dGel = h('dd', { 'data-testid': 'gel' });
  mesures.append(h('dt', {}, 'Hauteur du soleil à midi, mi-septembre'), dSoleil, h('dt', {}, 'Énergie reçue, par rapport au plat'), dEnergie, h('dt', {}, 'Température moyenne de la saison'), dTemp, h('dt', {}, 'Degrés-jours → région de Winkler'), dDj, h('dt', {}, 'Risque de gel de printemps'), dGel);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-coteau' });
  boite.append(lecture);

  function dessiner(p, elSoleil, gel, exp) {
    const L = 520, H = 210;
    const ctx = contexte(canvas, L, H);
    const ciel = ctx.createLinearGradient(0, 0, 0, H);
    ciel.addColorStop(0, '#1b2333'); ciel.addColorStop(1, '#2a1d21');
    ctx.fillStyle = ciel; ctx.fillRect(0, 0, L, H);
    // le coteau monte vers la droite ; le soleil vient de la gauche si l'exposition est plutôt sud/est... on simplifie : il éclaire depuis la gauche.
    const xBas = 70, yBas = H - 40, longueur = L - xBas - 20;
    const yHaut = yBas - Math.tan(rad(p)) * longueur * 0.9;
    // soleil
    const sx = 60, sy = clamp(H - 40 - Math.tan(rad(Math.min(elSoleil, 70))) * 160, 18, H - 60);
    ctx.fillStyle = '#d9b25f'; ctx.beginPath(); ctx.arc(sx, sy, 13, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(217,178,95,.25)'; ctx.lineWidth = 1;
    for (let i = 0; i < 6; i++) {
      const t0 = i / 5, x1 = xBas + t0 * longueur, y1 = yBas + (yHaut - yBas) * t0;
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(x1, y1); ctx.stroke();
    }
    // eau
    if (eau.checked) { ctx.fillStyle = '#3f6f8f'; ctx.fillRect(0, yBas + 8, xBas + 10, H - yBas - 8); }
    // terre
    ctx.fillStyle = '#4a2f27'; ctx.beginPath(); ctx.moveTo(0, yBas + 8); ctx.lineTo(xBas, yBas); ctx.lineTo(xBas + longueur, yHaut); ctx.lineTo(L, yHaut); ctx.lineTo(L, H); ctx.lineTo(0, H); ctx.closePath(); ctx.fill();
    // ceps
    ctx.strokeStyle = '#8fbf6a'; ctx.lineWidth = 2;
    for (let i = 1; i < 18; i++) {
      const t0 = i / 18, x = xBas + t0 * longueur, y = yBas + (yHaut - yBas) * t0;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - 12); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y - 15, 4, 0, Math.PI * 2); ctx.stroke();
    }
    // poche d'air froid
    const eps = gel / 100;
    const froid = ctx.createLinearGradient(0, yBas - 40, 0, yBas + 8);
    froid.addColorStop(0, 'rgba(134,183,217,0)'); froid.addColorStop(1, `rgba(134,183,217,${(.55 * eps).toFixed(2)})`);
    ctx.fillStyle = froid; ctx.beginPath(); ctx.moveTo(0, yBas - 40); ctx.lineTo(xBas + longueur * .35, yBas - 40); ctx.lineTo(xBas + longueur * .35, yBas + (yHaut - yBas) * .35); ctx.lineTo(xBas, yBas); ctx.lineTo(0, yBas + 8); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#a89a95'; ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.textAlign = 'left';
    ctx.fillText(`pente ${p}°, exposée ${exp}`, 12, 16);
    ctx.fillText(`soleil à ${fmt(elSoleil)}°`, 12, 28);
    if (gel > 25) { ctx.fillStyle = '#86b7d9'; ctx.fillText('air froid', 8, yBas - 6); }
  }
  function maj() {
    const la = +lat.value, al = +alt.value, p = +pente.value, ex = expo.value, ea = eau.checked;
    const r = energieRelative(la, p, EXPOS[ex]);
    const elSoleil = 90 - la + 3;
    const T = 24.5 - 0.45 * (la - 30) - 0.0065 * al + 1.6 * (r - 1) + (ea ? 0.3 : 0);
    const dj = Math.round((T - 10) * JOURS_SAISON);
    const reg = regionWinkler(Math.max(dj, 0));
    const gel = Math.round(clamp(6 + (la - 38) * 3 + al * 0.02 - p * 1.3 - (ea ? 14 : 0), 2, 95));
    dSoleil.textContent = `${fmt(elSoleil)}°`;
    dEnergie.textContent = `${fmt(r * 100)} %`;
    dEnergie.className = `grand ${r > 1.05 ? 'bon' : r < 0.9 ? 'alerte' : ''}`;
    dTemp.textContent = `${fmt(T, 1)} °C`;
    dDj.textContent = dj < 900 ? `${fmt(Math.max(dj, 0))} · trop froid` : `${fmt(dj)} · ${reg.nom}`;
    dGel.textContent = `${gel} %`;
    dGel.className = gel > 35 ? 'alerte' : gel < 15 ? 'bon' : '';
    let l, cls = '';
    if (dj < 900) { l = "<b>Trop froid pour la vigne.</b> Entre la latitude et l'altitude, la saison ne suffit pas à mûrir quoi que ce soit ; il faudrait descendre ou se rapprocher de l'équateur."; cls = 'alerte'; }
    else if (p === 0) { l = "<b>Terrain plat.</b> Facile à travailler et à mécaniser, mais l'air froid y stagne au printemps, les sols sont souvent profonds et fertiles, et chaque mètre carré reçoit le soleil qu'on lui doit, sans bonus. C'est la géographie des vins de volume."; }
    else if (r >= 1.08) { l = `<b>Le coteau qui fait les grands crus.</b> Exposé ${ex}, il reçoit ${fmt((r - 1) * 100)} % d'énergie de plus qu'un terrain plat, ce qui vaut ${fmt(1.6 * (r - 1), 1)} °C de plus sur la saison, et l'air froid s'écoule vers le bas au lieu de geler les bourgeons. Contrepartie : le sol est mince, l'érosion menace, et tout se fait à la main ou au treuil.`; cls = 'bon'; }
    else if (r < 0.9) { l = `<b>Versant à l'ombre.</b> Exposé ${ex}, il reçoit ${fmt((1 - r) * 100)} % d'énergie de moins que le plat. En climat chaud, c'est un refuge de fraîcheur recherché pour les blancs ; en climat frais, le raisin n'y mûrit pas.`; cls = la < 40 ? '' : 'alerte'; }
    else { l = `<b>Pente modérée.</b> Un peu plus d'énergie que le plat (${fmt(r * 100)} %), un bon drainage de l'eau et de l'air froid, et une mécanisation encore possible. La majorité des vignobles de qualité ressemblent à ça.`; cls = 'bon'; }
    if (al >= 800) l += ` À ${fmt(al)} m, la fraîcheur nocturne conserve l'acidité et le rayonnement ultraviolet épaissit les peaux : c'est la recette des Andes et de l'Etna.`;
    if (ea) l += " Le plan d'eau retarde les gelées d'automne, adoucit le printemps et renvoie de la lumière sur le feuillage.";
    lecture.innerHTML = l; lecture.className = `lecture ${cls}`;
    dessiner(p, elSoleil, gel, ex);
  }
  [lat, alt, pente].forEach((c) => c.addEventListener('input', maj));
  expo.addEventListener('change', maj);
  eau.addEventListener('change', maj);
  maj();
  return { maj, energieRelative };
};

/* ---------- Sol : composer un profil ---------- */

const SOLS = {
  'argilo-calcaire': { nom: 'Argilo-calcaire', ru: 1.5, drainage: .55, fertilite: .55, chaleur: .5, ph: 8.1, calcaire: 1, reserveRoche: 30, cailloux: 20, couleur: '#8a7a5a', roche: '#c9c0a2', vins: 'des rouges structurés et des blancs amples ; c’est le sol de la Côte d’Or, de Saint-Émilion et d’une bonne partie de la Rioja', pg: '41 B ou Fercal, qui tolèrent le calcaire actif' },
  craie: { nom: 'Craie', ru: 1.1, drainage: .8, fertilite: .3, chaleur: .5, ph: 8.3, calcaire: 2, reserveRoche: 60, cailloux: 10, couleur: '#b6ad94', roche: '#e5e0cf', vins: 'des blancs tendus et des bases d’effervescents ; la craie de Champagne rend l’eau lentement et garde les racines au frais', pg: '41 B, le seul qui supporte vraiment autant de calcaire actif' },
  graves: { nom: 'Graves et galets', ru: .7, drainage: .95, fertilite: .25, chaleur: .9, ph: 6.6, calcaire: 0, reserveRoche: 0, cailloux: 55, couleur: '#6f5a48', roche: '#8a7a6a', vins: 'des rouges mûrs et charpentés ; les galets emmagasinent la chaleur du jour et la rendent la nuit, ce qui mûrit le cabernet sauvignon du Médoc et le grenache de Châteauneuf', pg: '101-14 ou 3309 C ; 110 R si l’été est très sec' },
  schiste: { nom: 'Schiste', ru: .9, drainage: .85, fertilite: .2, chaleur: .85, ph: 6.0, calcaire: 0, reserveRoche: 25, cailloux: 40, couleur: '#4d4a55', roche: '#2f2d38', vins: 'des vins profonds et concentrés, à petits rendements ; les racines descendent dans les feuillets fissurés : Mosel, Côte-Rôtie, Priorat, Douro', pg: '110 R ou 140 Ru pour affronter la sécheresse' },
  granite: { nom: 'Granite', ru: .8, drainage: .85, fertilite: .3, chaleur: .6, ph: 5.6, calcaire: 0, reserveRoche: 10, cailloux: 25, couleur: '#7d6c62', roche: '#a09a94', vins: 'des vins fins, parfumés, plutôt légers ; le granite se délite en sable grossier acide : Beaujolais, Muscadet, Rhône nord', pg: 'Riparia Gloire ou 3309 C ; attention aux carences en magnésium' },
  sable: { nom: 'Sable', ru: .7, drainage: .95, fertilite: .15, chaleur: .7, ph: 6.5, calcaire: 0, reserveRoche: 0, cailloux: 5, couleur: '#a08a62', roche: '#8f7a55', vins: 'des vins légers, souples et aromatiques, peu colorés ; le phylloxéra ne s’y déplace pas, ce qui permet encore des vignes franches de pied', pg: 'souvent aucun : vignes franches, ou 110 R contre la sécheresse' },
  argile: { nom: 'Argile lourde', ru: 1.7, drainage: .25, fertilite: .8, chaleur: .3, ph: 7.0, calcaire: 0, reserveRoche: 0, cailloux: 5, couleur: '#5a3f3a', roche: '#4a3530', vins: 'des rouges puissants, charnus, tanniques ; l’argile retient l’eau et les nutriments, mais reste froide au printemps : merlot de Pomerol, malbec de Cahors', pg: 'SO 4 ou 3309 C, qui tolèrent l’humidité' },
  limon: { nom: 'Limon profond de plaine', ru: 2.0, drainage: .45, fertilite: .95, chaleur: .4, ph: 6.9, calcaire: 0, reserveRoche: 0, cailloux: 0, couleur: '#6b5340', roche: '#5a4535', vins: 'des vins de volume : le cep n’a jamais soif, pousse sans arrêt, et donne beaucoup de raisin dilué. Excellent pour les céréales, médiocre pour les grands vins', pg: 'Riparia Gloire pour freiner la vigueur, ou pas de vigne du tout' },
};

ATELIERS.sol = (boite) => {
  atelierEntete(boite, 'Atelier', 'Composer un profil de sol');
  const canvas = h('canvas', { 'aria-label': 'Coupe verticale du sol avec les racines de la vigne' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const type = selection(reglages, { id: 'sType', label: 'Roche-mère et texture', valeur: 'argilo-calcaire', options: Object.entries(SOLS).map(([k, s]) => [k, s.nom]) });
  const prof = curseur(reglages, { id: 'sProf', label: 'Profondeur explorée par les racines', min: 30, max: 200, step: 10, valeur: 90, affiche: (v) => `${v} cm` });
  const cailloux = curseur(reglages, { id: 'sCailloux', label: 'Pierrosité', min: 0, max: 70, step: 5, valeur: 20, affiche: (v) => `${v} %` });
  const jauges = h('div', { class: 'jauges' });
  boite.append(jauges);
  const jEau = jauge(jauges, 'Réserve d’eau utile', 'bleu');
  const jDrain = jauge(jauges, 'Drainage', 'vert');
  const jFert = jauge(jauges, 'Fertilité', 'or');
  const jChaud = jauge(jauges, 'Chaleur du sol', 'rougeb');
  const jVig = jauge(jauges, 'Vigueur attendue');
  const mesures = h('dl', { class: 'mesures' });
  const dPh = h('dd'), dRu = h('dd', { 'data-testid': 'reserve-eau' }), dContr = h('dd', { class: 'grand', 'data-testid': 'contrainte' });
  mesures.append(h('dt', {}, 'pH du sol'), dPh, h('dt', {}, 'Réserve utile'), dRu, h('dt', {}, 'Contrainte hydrique en août'), dContr);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-sol' });
  boite.append(lecture);

  function dessiner(s, p, c, ru) {
    const L = 520, H = 220, ySurf = 26;
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#1b2333'; ctx.fillRect(0, 0, L, ySurf);
    const Y = (cm) => ySurf + (cm / 220) * (H - ySurf);
    // horizon de surface (humifère), sous-sol, roche
    ctx.fillStyle = '#3a2a22'; ctx.fillRect(0, ySurf, L, Y(25) - ySurf);
    ctx.fillStyle = s.couleur; ctx.fillRect(0, Y(25), L, Y(p + 20) - Y(25));
    ctx.fillStyle = s.roche; ctx.fillRect(0, Y(p + 20), L, H - Y(p + 20));
    if (s.calcaire || s.roche === '#2f2d38') {   // fissures dans la roche
      ctx.strokeStyle = 'rgba(0,0,0,.35)'; ctx.lineWidth = 1;
      for (let x = 20; x < L; x += 34) { ctx.beginPath(); ctx.moveTo(x, Y(p + 20)); ctx.lineTo(x + 8, H); ctx.stroke(); }
    }
    // cailloux
    const r = alea(7);
    const n = Math.round(c * 1.6);
    ctx.fillStyle = 'rgba(255,255,255,.28)';
    for (let i = 0; i < n; i++) {
      const x = r() * L, y = Y(25) + r() * (Y(p + 20) - Y(25)), w = 3 + r() * 7;
      ctx.beginPath(); ctx.ellipse(x, y, w, w * .6, r() * Math.PI, 0, Math.PI * 2); ctx.fill();
    }
    // herbe et cep
    ctx.strokeStyle = '#8fbf6a'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(L / 2, ySurf); ctx.lineTo(L / 2, 4); ctx.stroke();
    ctx.lineWidth = 1.5; ctx.strokeStyle = '#a89a95';
    // racines : un pivot et des latérales, jusqu'à la profondeur choisie (un peu dans la roche si elle est fissurée)
    const fond = Y(p + (s.reserveRoche ? 25 : 0));
    ctx.beginPath(); ctx.moveTo(L / 2, ySurf); ctx.quadraticCurveTo(L / 2 + 8, (ySurf + fond) / 2, L / 2 - 4, fond); ctx.stroke();
    const rr = alea(3);
    for (let i = 0; i < 14; i++) {
      const y0 = ySurf + 6 + (i / 14) * (fond - ySurf - 10);
      const portee = (60 + rr() * 90) * (1 - (i / 14) * .6);
      const dir = i % 2 ? 1 : -1;
      ctx.beginPath(); ctx.moveTo(L / 2, y0); ctx.quadraticCurveTo(L / 2 + dir * portee * .6, y0 + 10 + rr() * 14, L / 2 + dir * portee, y0 + 18 + rr() * 22); ctx.stroke();
    }
    // eau
    ctx.fillStyle = 'rgba(134,183,217,.16)'; ctx.fillRect(0, Y(25), L, (Y(p + 20) - Y(25)) * clamp(ru / 250, .1, 1));
    ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.textAlign = 'right';
    ctx.fillStyle = '#a89a95'; ctx.fillText('0 cm', L - 8, ySurf + 11);
    ctx.fillStyle = 'rgba(18,12,14,.7)'; ctx.fillRect(L - 58, Y(p) - 8, 52, 15);
    ctx.fillStyle = '#f3ebe4'; ctx.fillText(`${p} cm`, L - 8, Y(p) + 3);
    ctx.textAlign = 'left'; ctx.fillStyle = '#dccfc9'; ctx.fillText(s.nom, 8, ySurf + 13);
  }
  function maj() {
    const s = SOLS[type.value], p = +prof.value, c = +cailloux.value;
    const ru = Math.round(s.ru * p * (1 - c / 100) + s.reserveRoche);
    const deficit = 260;   // ce que l'été (juin → septembre) évapore de plus qu'il ne pleut, en climat tempéré
    const manque = deficit - ru;
    const drainage = clamp(s.drainage + c / 250, 0, 1);
    const chaleur = clamp(s.chaleur + c / 300 - (ru > 200 ? .15 : 0), 0, 1);
    const vigueur = clamp(s.fertilite * .55 + clamp(ru / 320, 0, 1) * .45, 0, 1);
    jEau(clamp(ru / 3.5, 0, 100), `${fmt(ru)} mm`);
    jDrain(drainage * 100); jFert(s.fertilite * 100); jChaud(chaleur * 100); jVig(vigueur * 100);
    dPh.textContent = fmt(s.ph, 1);
    dRu.textContent = `${fmt(ru)} mm`;
    let contr, cls;
    if (manque <= 0) { contr = 'aucune'; cls = 'alerte'; }
    else if (manque < 90) { contr = 'faible'; cls = ''; }
    else if (manque < 170) { contr = 'modérée'; cls = 'bon'; }
    else if (manque < 230) { contr = 'forte'; cls = ''; }
    else { contr = 'sévère'; cls = 'alerte'; }
    dContr.textContent = contr; dContr.className = `grand ${cls}`;
    let l = `<b>${s.nom}.</b> Ce sol donne ${s.vins}. `;
    if (manque <= 0) l += "Avec une telle réserve d'eau, la vigne <b>ne connaît jamais la soif</b> : elle pousse jusqu'aux vendanges, ombrage ses grappes et dilue son jus. Il faudra enherber, limiter la charge, et accepter un vin simple.";
    else if (manque < 90) l += "La contrainte hydrique est <b>légère</b> : bon pour les rendements et les blancs frais, un peu court pour concentrer un grand rouge.";
    else if (manque < 170) l += "La contrainte est <b>modérée et arrive au bon moment</b>, vers la véraison : la vigne cesse de pousser, ses baies restent petites et se concentrent. C'est l'équilibre que cherchent tous les grands terroirs.";
    else if (manque < 230) l += "La contrainte est <b>forte</b> : de petits rendements très concentrés les bonnes années, mais un blocage de maturité et des feuilles grillées lors des étés caniculaires. On y plante des cépages méditerranéens et des porte-greffes résistants.";
    else l += "La réserve est <b>trop faible</b> : sans irrigation ou sans racines qui trouvent la roche fissurée, la vigne se bloque en août, perd ses feuilles et ne mûrit pas.";
    l += ` Porte-greffe : ${s.pg}.`;
    if (s.calcaire === 2) l += " Le calcaire actif est si élevé qu'un mauvais porte-greffe donnerait des ceps <b>chlorotiques</b>, aux feuilles jaunes, incapables de mûrir.";
    else if (s.calcaire === 1) l += " Surveiller la <b>chlorose</b> sur les parcelles les plus calcaires.";
    lecture.innerHTML = l; lecture.className = `lecture ${cls}`;
    dessiner(s, p, c, ru);
  }
  type.addEventListener('change', () => { cailloux.value = SOLS[type.value].cailloux; cailloux.majSortie(); maj(); });
  prof.addEventListener('input', maj);
  cailloux.addEventListener('input', maj);
  maj();
  return { maj };
};

/* ---------- Vie du sol : la rhizosphère ---------- */

const COUVERTS = {
  nu: { nom: 'Sol nu, désherbé chimiquement', f: .15 },
  travail: { nom: 'Sol travaillé (labour, griffage)', f: .4 },
  partiel: { nom: 'Enherbement un rang sur deux', f: .75 },
  total: { nom: 'Couvert semé, avec légumineuses', f: 1 },
};

ATELIERS.vieDuSol = (boite) => {
  atelierEntete(boite, 'Atelier', 'Faire vivre le sol');
  const canvas = h('canvas', { 'aria-label': 'Vue rapprochée d’une racine de vigne, de ses mycorhizes et de la vie du sol' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const mo = curseur(reglages, { id: 'vMO', label: 'Matière organique du sol', min: 0.5, max: 5, step: 0.1, valeur: 1.8, affiche: (v) => `${fmt(v, 1)} %` });
  const couvert = selection(reglages, { id: 'vCouvert', label: 'Entretien du sol', valeur: 'travail', options: Object.entries(COUVERTS).map(([k, c]) => [k, c.nom]) });
  const cuivre = curseur(reglages, { id: 'vCuivre', label: 'Cuivre épandu (mildiou)', min: 0, max: 8, step: 0.5, valeur: 3, affiche: (v) => `${fmt(v, 1)} kg/ha/an` });
  const compost = h('input', { type: 'checkbox', id: 'vCompost' });
  reglages.append(h('div', { class: 'reglage' }, h('span', {}, 'Apports'), h('label', { class: 'case' }, compost, 'Compost ou marc composté chaque hiver')));
  const jauges = h('div', { class: 'jauges' });
  boite.append(jauges);
  const jMicro = jauge(jauges, 'Activité microbienne');
  const jMyco = jauge(jauges, 'Racines mycorhizées', 'or');
  const jVers = jauge(jauges, 'Vers de terre', 'vert');
  const jSec = jauge(jauges, 'Tenue à la sécheresse', 'bleu');
  const jEro = jauge(jauges, 'Risque d’érosion', 'rougeb');
  const mesures = h('dl', { class: 'mesures' });
  const dN = h('dd'), dYan = h('dd', { class: 'grand', 'data-testid': 'azote-mout' }), dCu = h('dd');
  mesures.append(h('dt', {}, 'Azote libéré par le sol'), dN, h('dt', {}, 'Azote assimilable attendu dans le moût'), dYan, h('dt', {}, 'Cuivre accumulé en 30 ans'), dCu);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-vie' });
  boite.append(lecture);

  function dessiner(micro, myco, vers) {
    const L = 520, H = 190;
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#2a1b16'; ctx.fillRect(0, 0, L, H);
    const r = alea(11);
    // bactéries : des points
    ctx.fillStyle = 'rgba(212,87,122,.7)';
    for (let i = 0; i < micro * 4; i++) { ctx.beginPath(); ctx.arc(r() * L, r() * H, 1 + r() * 1.2, 0, Math.PI * 2); ctx.fill(); }
    // vers de terre
    ctx.strokeStyle = '#c98a7a'; ctx.lineWidth = 3; ctx.lineCap = 'round';
    for (let i = 0; i < Math.round(vers / 60); i++) {
      const x = 30 + r() * (L - 60), y = 20 + r() * (H - 40);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.bezierCurveTo(x + 15, y - 12, x + 30, y + 12, x + 45, y - 4); ctx.stroke();
    }
    // racine principale et latérales
    ctx.strokeStyle = '#e8d9c8'; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(L / 2, 0); ctx.quadraticCurveTo(L / 2 + 10, H / 2, L / 2 - 6, H); ctx.stroke();
    ctx.lineWidth = 2.5;
    const lat = [];
    for (let i = 0; i < 7; i++) {
      const y0 = 14 + i * 24, dir = i % 2 ? 1 : -1, portee = 70 + r() * 80;
      const x1 = L / 2 + dir * portee, y1 = y0 + 12 + r() * 16;
      ctx.beginPath(); ctx.moveTo(L / 2, y0); ctx.quadraticCurveTo(L / 2 + dir * portee * .5, y0 + 4, x1, y1); ctx.stroke();
      lat.push([x1, y1, dir, y0]);
    }
    // hyphes mycorhiziens : filaments fins depuis les racines latérales
    ctx.strokeStyle = 'rgba(217,178,95,.75)'; ctx.lineWidth = .8;
    const nH = Math.round(myco / 100 * 120);
    for (let i = 0; i < nH; i++) {
      const [x1, y1, dir, y0] = lat[i % lat.length];
      const t0 = r(), x = L / 2 + (x1 - L / 2) * t0, y = y0 + (y1 - y0) * t0;
      const a = r() * Math.PI * 2, len = 12 + r() * 40;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.bezierCurveTo(x + Math.cos(a) * len * .4, y + Math.sin(a) * len * .4 + 6, x + Math.cos(a) * len * .8 + dir * 4, y + Math.sin(a) * len * .8, x + Math.cos(a) * len, y + Math.sin(a) * len); ctx.stroke();
    }
    ctx.fillStyle = '#a89a95'; ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.textAlign = 'left';
    ctx.fillText('— racines   - hyphes mycorhiziens   · bactéries', 8, H - 8);
  }
  function maj() {
    const M = +mo.value, c = COUVERTS[couvert.value], f = c.f, Cu = +cuivre.value, comp = compost.checked;
    const micro = clamp((12 + 15 * M) * (0.5 + 0.5 * f) * (1 - 0.06 * Cu) + (comp ? 10 : 0), 0, 100);
    const myco = clamp(28 + 42 * f + 5 * M - 4 * Cu - (couvert.value === 'travail' ? 8 : 0), 5, 95);
    const vers = Math.round(clamp((20 + 55 * M) * (0.25 + f) * (1 - 0.08 * Cu) + (comp ? 40 : 0), 0, 500));
    const nMin = M * 14 * (0.7 + 0.3 * f) + (couvert.value === 'total' ? 22 : 0) + (comp ? 15 : 0);
    const nDispo = nMin - (couvert.value === 'total' ? 14 : couvert.value === 'partiel' ? 7 : 0);
    const yan = Math.round(clamp(70 + 3.5 * nDispo, 40, 380));
    const sec = clamp(15 + 0.5 * myco + 6 * M - (couvert.value === 'total' ? 10 : 0), 0, 100);
    const ero = clamp(82 - 72 * f - 5 * M, 0, 100);
    const cuMgKg = Cu * 30 / 3;   // 30 ans, 3 000 t de terre par hectare sur 20 cm
    jMicro(micro); jMyco(myco); jVers(clamp(vers / 4.5, 0, 100), `${fmt(vers)} /m²`); jSec(sec); jEro(ero);
    dN.textContent = `${fmt(nMin)} kg/ha/an`;
    dYan.textContent = `${fmt(yan)} mg/L`;
    dYan.className = `grand ${yan < 140 ? 'alerte' : yan > 260 ? '' : 'bon'}`;
    dCu.textContent = cuMgKg ? `+${fmt(cuMgKg)} mg/kg` : 'aucun';
    let l, cls = '';
    if (couvert.value === 'nu') { l = "<b>Sol nu et désherbé.</b> Sans racines vivantes entre les rangs, les champignons mycorhiziens n'ont plus d'hôte hors de la vigne, la matière organique se consume sans être renouvelée, les vers de terre partent et la pluie ruisselle en emportant la terre. C'était la norme des années 1970 ; c'est aujourd'hui l'exception."; cls = 'alerte'; }
    else if (couvert.value === 'travail') { l = "<b>Sol travaillé.</b> Le labour tue l'herbe, aère et libère un peu d'azote, mais il casse les réseaux d'hyphes et expose la matière organique à l'air, qui la brûle. Bon compromis en climat sec, où l'herbe concurrencerait la vigne pour l'eau."; }
    else if (couvert.value === 'partiel') { l = "<b>Un rang sur deux enherbé.</b> Le compromis le plus répandu : l'herbe nourrit la vie du sol, porte le tracteur et freine l'érosion, tandis que le rang travaillé limite la concurrence pour l'eau et l'azote."; cls = 'bon'; }
    else { l = "<b>Couvert semé avec légumineuses.</b> Trèfle, féverole ou vesce fixent l'azote de l'air grâce à leurs bactéries symbiotiques, leurs racines nourrissent les mycorhizes, le sol reste couvert toute l'année. En climat frais et humide, c'est le meilleur sol vivant possible ; en climat sec, on le détruit au printemps pour rendre l'eau à la vigne."; cls = 'bon'; }
    if (yan < 140) l += ` Avec ${fmt(yan)} mg/L d'azote assimilable, le moût sera <b>carencé</b> : fermentation languissante et odeurs de soufre au chai, à moins d'ajouter des nutriments aux levures.`;
    else if (yan > 260) l += " Le sol libère beaucoup d'azote : ceps vigoureux, grappes serrées, plus de pourriture et des vins dilués. On sème un couvert pour en consommer une partie.";
    if (Cu >= 5) l += ` À ${fmt(Cu, 1)} kg de cuivre par hectare et par an, au-delà du plafond européen, le sol en accumule ${fmt(cuMgKg)} mg/kg en trente ans : les vers de terre et une partie des microbes ne s'en remettent pas.`;
    lecture.innerHTML = l; lecture.className = `lecture ${cls}`;
    dessiner(micro, myco, vers);
  }
  [mo, cuivre].forEach((c) => c.addEventListener('input', maj));
  couvert.addEventListener('change', maj);
  compost.addEventListener('change', maj);
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

/* ---------- Eau et chaleur : ce qui fait le goût ---------- */

const EXPOSITIONS = {
  ombre: { nom: 'Canopée dense, grappes à l’ombre', soleil: 0.25 },
  levant: { nom: 'Effeuillé côté levant', soleil: 0.6 },
  double: { nom: 'Effeuillé des deux côtés', soleil: 1 },
};

/* Modèle jouet : l'eau règle la taille de la baie, la taille de la baie règle
   le rapport peau/jus, et ce rapport commande couleur, tanins et concentration.
   La chaleur ajoute du sucre et brûle l'acide malique ; au-delà de 24 °C de
   moyenne, elle dégrade aussi la couleur et les arômes. */
function equilibreBaie({ eau, temp, ampl, charge, soleil }) {
  const contrainte = clamp((300 - eau) / 190, 0, 1);          // 0 : la vigne boit à sa faim ; 1 : stress sévère
  const exces = clamp((eau - 330) / 170, 0, 1);               // sol saturé, vigne qui pousse encore
  const bloque = Math.max(0, contrainte - 0.72) / 0.28;       // stomates fermés, maturité en panne
  const poids = clamp(1 + 1.2 * ((eau - 100) / 400) ** 0.85 + 0.15 * (charge - 2) / 2, 0.8, 2.4);
  const peau = (1.55 / poids) ** (1 / 3);                     // indice peau/jus, 1 = baie de référence
  const optimum = Math.exp(-((contrainte - 0.45) ** 2) / 0.12);  // la contrainte modérée, l'idéal étroit
  const echaudage = clamp((temp - 22) / 4 * (0.35 + 0.65 * soleil) * (0.4 + 0.6 * contrainte), 0, 1);
  const sucre = clamp(196 + 8.5 * (temp - 19) - 15 * (charge - 2) + 55 * (peau - 1) - 20 * exces - 50 * bloque, 120, 285);
  const malique = clamp(6.2 - 0.4 * (temp - 14) + 0.11 * (ampl - 10) + 0.5 * (charge - 2) / 2, 0.5, 7.5);
  const tartrique = clamp(5.6 - 0.06 * (temp - 19) + 0.05 * (ampl - 10), 4.4, 6.4);
  const acidite = clamp((malique + tartrique) * 0.653 * 0.68 * (1 + 0.08 * (peau - 1)), 1.5, 6.5);
  const ph = clamp(4.55 - 0.115 * (malique + tartrique) + 0.03 * (temp - 19), 2.9, 4.1);
  const tanins = clamp(46 * peau + 18 * (soleil - 0.6) + 7 * (temp - 19) + 22 * optimum - 8 * (charge - 2) - 35 * bloque - 15 * echaudage, 5, 100);
  const couleur = clamp(50 * peau + 14 * (soleil - 0.4) + 20 * (ampl - 6) / 12 + 20 * optimum - 9 * (charge - 2) - 18 * Math.max(0, temp - 24) - 25 * echaudage - 30 * bloque, 5, 100);
  const aromes = clamp(96 - 7 * (temp - 18) + 2.6 * (ampl - 8) - 25 * bloque - 14 * exces - 22 * echaudage - 25 * Math.max(0, soleil - 0.6), 0, 100);
  return { contrainte, exces, bloque, poids, peau, echaudage, sucre, malique, tartrique, acidite, ph, tanins, couleur, aromes, alcool: sucre / 16.83 };
}

ATELIERS.equilibre = (boite) => {
  const rouge = ETAT.style !== 'blanc';
  atelierEntete(boite, 'Atelier', 'Les boutons de l’eau et de la chaleur');
  const canvas = h('canvas', { 'aria-label': 'Une grappe et une baie en coupe : taille, épaisseur de peau et couleur selon les réglages' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const eau = curseur(reglages, { id: 'qEau', label: 'Eau reçue, floraison → vendange', min: 100, max: 500, step: 10, valeur: 280, affiche: (v) => `${fmt(v)} mm` });
  const temp = curseur(reglages, { id: 'qTemp', label: 'Température moyenne, juillet → septembre', min: 14, max: 28, step: 0.5, valeur: 20, affiche: (v) => `${fmt(v, 1)} °C` });
  const ampl = curseur(reglages, { id: 'qAmpl', label: 'Écart jour / nuit', min: 4, max: 20, valeur: 12, affiche: (v) => `${v} °C` });
  const charge = curseur(reglages, { id: 'qCharge', label: 'Charge laissée sur le cep', min: 0.5, max: 4, step: 0.1, valeur: 2, affiche: (v) => `${fmt(v, 1)} kg/cep` });
  const expo = selection(reglages, { id: 'qExpo', label: 'Exposition des grappes', valeur: 'levant', options: Object.entries(EXPOSITIONS).map(([k, e]) => [k, e.nom]) });
  const mesures = h('dl', { class: 'mesures' });
  const dPoids = h('dd', { 'data-testid': 'poids-baie' }), dPeau = h('dd'), dSucre = h('dd', { 'data-testid': 'sucre-baie' }),
    dAlc = h('dd', { class: 'grand' }), dAcide = h('dd', { 'data-testid': 'acidite-baie' }), dMal = h('dd'), dPh = h('dd');
  mesures.append(
    h('dt', {}, 'Poids d’une baie'), dPoids,
    h('dt', {}, 'Rapport peau / jus'), dPeau,
    h('dt', {}, 'Sucre'), dSucre,
    h('dt', {}, 'Alcool potentiel'), dAlc,
    h('dt', {}, 'Acidité totale (éq. H₂SO₄)'), dAcide,
    h('dt', {}, 'dont acide malique (éq. tartrique)'), dMal,
    h('dt', {}, 'pH'), dPh);
  boite.append(mesures);
  const jauges = h('div', { class: 'jauges' });
  const jTan = jauge(jauges, rouge ? 'Tanins de peau et de pépins' : 'Structure, amers de peau', 'or');
  const jCoul = jauge(jauges, ETAT.style === 'blanc' ? 'Matière colorante et gras' : 'Couleur (anthocyanes)', 'rougeb');
  const jArom = jauge(jauges, 'Arômes fins et fraîcheur', 'vert');
  const jAcid = jauge(jauges, 'Acidité perçue', 'bleu');
  boite.append(jauges);
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-equilibre' });
  boite.append(lecture);
  const btn = h('button', { class: 'bouton', type: 'button', id: 'qReporter' }, 'Reporter ce millésime sur la maturité →');
  boite.append(h('div', { class: 'boutons' }, btn));

  function couleurBaie(r) {
    const c = r.couleur / 100;
    if (ETAT.style === 'blanc') return `rgb(${Math.round(lerp(205, 232, c))},${Math.round(lerp(200, 210, c))},${Math.round(lerp(120, 90, c))})`;
    if (ETAT.style === 'rose') return `rgb(${Math.round(lerp(190, 150, c))},${Math.round(lerp(90, 40, c))},${Math.round(lerp(110, 80, c))})`;
    return `rgb(${Math.round(lerp(96, 44, c))},${Math.round(lerp(40, 12, c))},${Math.round(lerp(70, 44, c))})`;
  }
  function dessiner(r) {
    const L = 520, H = 190;
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#120c0e'; ctx.fillRect(0, 0, L, H);
    const teinte = couleurBaie(r);
    // la grappe, à l'échelle : le rayon suit le poids de la baie
    const rayon = 9 + 11 * (r.poids - 0.8) / 1.6;
    const rnd = alea(7);
    ctx.save();
    ctx.translate(150, 26);
    ctx.strokeStyle = '#5a4a2c'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(0, -14); ctx.lineTo(0, 6); ctx.stroke();
    for (let i = 0; i < 6; i++) {
      const n = 4 - Math.floor(i / 2);
      for (let k = 0; k < n; k++) {
        const x = (k - (n - 1) / 2) * (rayon * 1.85) + (rnd() - 0.5) * 4;
        const y = 12 + i * rayon * 1.5 + (rnd() - 0.5) * 3;
        if (y + rayon > H - 16) continue;
        ctx.beginPath(); ctx.arc(x, y, rayon, 0, Math.PI * 2);
        ctx.fillStyle = teinte; ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,.16)';
        ctx.beginPath(); ctx.arc(x - rayon * 0.3, y - rayon * 0.35, rayon * 0.28, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.restore();
    // la baie en coupe, à droite : épaisseur de peau proportionnelle au rapport peau/jus
    const cx = 400, cy = 96, R = 62;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = teinte; ctx.fill();
    const ep = clamp(6 + 16 * (r.peau - 0.85), 4, 22);
    ctx.beginPath(); ctx.arc(cx, cy, R - ep, 0, Math.PI * 2);
    ctx.fillStyle = ETAT.style === 'blanc' ? '#d9d0a8' : '#e8dcc6'; ctx.globalAlpha = .9; ctx.fill(); ctx.globalAlpha = 1;
    const brun = clamp((r.tanins - 20) / 70, 0, 1);
    for (const [dx, dy] of [[-13, -6], [12, -8], [-2, 12]]) {
      ctx.save(); ctx.translate(cx + dx, cy + dy); ctx.rotate(dx * 0.06);
      ctx.beginPath(); ctx.ellipse(0, 0, 6, 9, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${Math.round(lerp(150, 92, brun))},${Math.round(lerp(160, 58, brun))},${Math.round(lerp(90, 26, brun))})`;
      ctx.fill(); ctx.restore();
    }
    ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.fillStyle = '#a89a95'; ctx.textAlign = 'center';
    ctx.fillText(`${fmt(r.poids, 2)} g par baie`, 150, H - 6);
    ctx.fillText(`peau ${r.peau >= 1.06 ? 'épaisse' : r.peau <= 0.94 ? 'mince' : 'moyenne'} · pépins ${brun > 0.6 ? 'bruns' : brun > 0.3 ? 'jaunes' : 'verts'}`, cx, H - 6);
  }

  function maj() {
    const r = equilibreBaie({ eau: +eau.value, temp: +temp.value, ampl: +ampl.value, charge: +charge.value, soleil: EXPOSITIONS[expo.value].soleil });
    dPoids.textContent = `${fmt(r.poids, 2)} g`;
    dPeau.textContent = `${fmt(r.peau * 100)} % de la référence`;
    dSucre.textContent = `${fmt(r.sucre)} g/L`;
    dAlc.textContent = `${fmt(r.alcool, 1)} % vol.`;
    dAcide.textContent = `${fmt(r.acidite, 1)} g/L`;
    dAcide.className = r.acidite < 2.6 ? 'alerte' : r.acidite > 5 ? 'alerte' : 'bon';
    dMal.textContent = `${fmt(r.malique, 1)} g/L`;
    dPh.textContent = fmt(r.ph, 2);
    dPh.className = r.ph > 3.75 ? 'alerte' : '';
    jTan(r.tanins); jCoul(r.couleur); jArom(r.aromes);
    jAcid(clamp((r.acidite - 1.5) / 4 * 100, 0, 100), `${fmt(r.acidite, 1)} g/L`);
    let l, cls = '';
    if (r.bloque > 0.35) { l = "<b>Blocage de maturité.</b> La vigne a fermé ses stomates pour ne pas se dessécher : elle ne photosynthétise plus, donc le sucre n'entre plus. Les feuilles jaunissent et tombent, les baies se flétrissent sans mûrir, les tanins restent verts et astringents. C'est le millésime de sécheresse extrême, où l'irrigation de sauvetage — quand elle est permise — sauve la récolte."; cls = 'alerte'; }
    else if (r.exces > 0.6) { l = "<b>Trop d'eau.</b> Les baies gonflent, la vigne continue de pousser au lieu de mûrir ses raisins, la canopée s'épaissit et ombrage les grappes. Le jus est dilué : moins de sucre, moins de couleur, moins de tout, par litre. Et l'humidité dans les grappes appelle le mildiou et la pourriture grise."; cls = 'alerte'; }
    else if (r.echaudage > 0.5) { l = `<b>Coup de chaleur.</b> À ${fmt(+temp.value, 1)} °C de moyenne, avec des grappes exposées et un sol sec, la peau au soleil dépasse 45 °C : elle brunit, se dessèche et perd ses anthocyanes ; les arômes fins partent avec. On revient à l'effeuillage minimal, côté levant seulement, et on pose des filets d'ombrage.`; cls = 'alerte'; }
    else if (+temp.value >= 24 && r.acidite < 3) { l = `<b>Millésime solaire.</b> Beaucoup de sucre (${fmt(r.sucre)} g/L, soit ${fmt(r.alcool, 1)} % d'alcool potentiel), peu d'acidité (${fmt(r.acidite, 1)} g/L) et un pH de ${fmt(r.ph, 2)} : le vin sera chaleureux, rond, un peu mou, et fragile en cave parce que le soufre y travaille mal. On vendange plus tôt, de nuit, et on acidifie si la loi locale le permet.`; cls = 'alerte'; }
    else if (+temp.value <= 17 && r.sucre < 185) { l = `<b>Millésime frais.</b> ${fmt(r.sucre)} g/L de sucre, soit ${fmt(r.alcool, 1)} % d'alcool potentiel, et ${fmt(r.acidite, 1)} g/L d'acidité : c'est vif, tendu, parfois maigre. ${rouge ? "Les tanins n'ont pas fini de mûrir : le vin sera herbacé si l'on extrait trop." : 'Le profil des blancs de climat froid et des bases d’effervescents, qui cherchent justement cette acidité.'}`; cls = rouge ? 'alerte' : ''; }
    else if (r.contrainte > 0.28 && r.contrainte < 0.62 && r.aromes > 55) { l = `<b>La contrainte modérée, l'idéal étroit.</b> La vigne a juste assez soif pour arrêter de pousser sans cesser de mûrir : baies de ${fmt(r.poids, 2)} g, peau proportionnellement épaisse, ${fmt(r.sucre)} g/L de sucre, ${fmt(r.acidite, 1)} g/L d'acidité. ${rouge ? 'La couleur et les tanins se concentrent' : 'Les arômes se concentrent'} sans que rien ne se dégrade. C'est ce que les grands terroirs offrent naturellement : un sol qui donne l'eau au compte-gouttes.`; cls = 'bon'; }
    else { l = `<b>Équilibre correct.</b> ${fmt(r.sucre)} g/L de sucre pour ${fmt(r.acidite, 1)} g/L d'acidité et un pH de ${fmt(r.ph, 2)} : rien ne cloche, rien ne brille. Essayez de descendre l'eau vers 220 mm ou la charge sous 1,5 kg par cep pour voir la concentration monter — et poussez la température au-delà de 25 °C pour voir l'acidité fondre.`; }
    if (+charge.value > 3.2) l += ` Avec ${fmt(+charge.value, 1)} kg par cep, la charge dépasse ce que la canopée peut mûrir : il faudrait 1,2 à 1,5 m² de feuilles par kilo de raisin.`;
    if (EXPOSITIONS[expo.value].soleil >= 1 && +temp.value >= 22) l += " L'effeuillage des deux côtés est un pari : il fait mûrir les tanins, mais il expose la face ouest au soleil de l'après-midi, le plus brûlant.";
    if (EXPOSITIONS[expo.value].soleil <= 0.25) l += ' À l’ombre d’une canopée dense, les pépins restent verts et les peaux pauvres en couleur : la maturité phénolique a besoin de lumière, pas seulement de chaleur.';
    lecture.innerHTML = l;
    lecture.className = `lecture ${cls}`;
    dessiner(r);
  }
  [eau, temp, ampl, charge].forEach((c) => c.addEventListener('input', maj));
  expo.addEventListener('change', maj);
  btn.addEventListener('click', () => {
    const T = +temp.value, E = +eau.value;
    const m = API.ateliers.maturite;
    if (!m) return;
    m.reglerClimat(T >= 22 && E < 300 ? '1.25' : T <= 17 || E > 380 ? '0.8' : '1');
    document.getElementById('s-maturite')?.scrollIntoView({ behavior: 'smooth' });
  });
  maj();
  return { maj };
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
  return { maj, reglerClimat: (v) => { climat.value = v; maj(); } };
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

/* ---------- Assemblage : composer la cuvée ---------- */

/* Chaque lot est une cuve du chai, avec son volume et son profil mesuré :
   degré, acidité totale, pH, indice de tanins, couleur, fruit, marque du bois. */
const LOTS = {
  rouge: [
    { nom: 'Merlot, argile, barrique de 1 vin', court: 'Merlot', hl: 420, alcool: 14.2, acidite: 3.3, ph: 3.7, tanin: 45, couleur: 60, fruit: 80, bois: 35, presse: 0, mot: 'la chair et le volume, mais du degré et peu d’acidité' },
    { nom: 'Cabernet sauvignon, graves, barrique neuve', court: 'Cabernet sauvignon', hl: 260, alcool: 13.2, acidite: 3.9, ph: 3.5, tanin: 85, couleur: 85, fruit: 55, bois: 75, presse: 0, mot: 'la structure, la couleur et la garde' },
    { nom: 'Cabernet franc, jeunes vignes, cuve inox', court: 'Cabernet franc', hl: 190, alcool: 12.4, acidite: 4.2, ph: 3.4, tanin: 55, couleur: 50, fruit: 85, bois: 0, presse: 0, mot: 'la fraîcheur et le fruit, sans bois' },
    { nom: 'Vin de presse, élevé à part', court: 'Presse', hl: 90, alcool: 13, acidite: 4, ph: 3.6, tanin: 95, couleur: 95, fruit: 30, bois: 10, presse: 1, mot: 'de la matière brute, vite astringente au-delà de 15 %' },
  ],
  blanc: [
    { nom: 'Chardonnay, cuve inox, vendange précoce', court: 'Inox', hl: 380, alcool: 12.6, acidite: 4.6, ph: 3.15, tanin: 20, couleur: 25, fruit: 90, bois: 0, presse: 0, mot: 'la tension et les agrumes' },
    { nom: 'Chardonnay, barrique neuve, bâtonné', court: 'Barrique', hl: 170, alcool: 13.4, acidite: 3.6, ph: 3.4, tanin: 38, couleur: 55, fruit: 45, bois: 80, presse: 0, mot: 'le gras, la vanille et le toasté' },
    { nom: 'Vieilles vignes, foudre, sur lies', court: 'Foudre', hl: 220, alcool: 13, acidite: 4, ph: 3.3, tanin: 30, couleur: 40, fruit: 65, bois: 25, presse: 0, mot: 'le volume et la longueur, sans marquer le bois' },
    { nom: 'Jus de presse, débourbé serré', court: 'Presse', hl: 110, alcool: 12.2, acidite: 3.8, ph: 3.5, tanin: 62, couleur: 60, fruit: 40, bois: 0, presse: 1, mot: 'de la matière et de l’amertume de peau' },
  ],
  rose: [
    { nom: 'Grenache, pressurage direct', court: 'Grenache', hl: 400, alcool: 12.8, acidite: 3.7, ph: 3.4, tanin: 15, couleur: 20, fruit: 88, bois: 0, presse: 0, mot: 'la pâleur, le gras et le fruit blanc' },
    { nom: 'Syrah de saignée', court: 'Saignée', hl: 150, alcool: 13.2, acidite: 3.5, ph: 3.5, tanin: 42, couleur: 72, fruit: 70, bois: 0, presse: 0, mot: 'la couleur et un peu de structure' },
    { nom: 'Cinsault vendangé de nuit', court: 'Cinsault', hl: 230, alcool: 12.2, acidite: 4.5, ph: 3.2, tanin: 12, couleur: 25, fruit: 92, bois: 0, presse: 0, mot: 'l’acidité et la fraîcheur' },
    { nom: 'Cuvée élevée en barrique', court: 'Barrique', hl: 90, alcool: 13.5, acidite: 3.4, ph: 3.5, tanin: 36, couleur: 45, fruit: 45, bois: 70, presse: 0, mot: 'du gras et du bois, à doser au pourcent' },
  ],
};
const DEFAUTS_ASSEMBLAGE = { rouge: [40, 28, 22, 10], blanc: [45, 15, 30, 10], rose: [45, 15, 32, 8] };
const COULEURS_LOTS = ['#9b2f4c', '#d9b25f', '#8fbf6a', '#86b7d9'];

function butsAssemblage(style) {
  if (style === 'blanc') return {
    garde: { nom: 'Un blanc de garde, ample et boisé', cible: { alcool: 13.2, acidite: 4.2, tanin: 32, couleur: 50, bois: 45, fruit: 55 }, prime: 6 },
    fruit: { nom: 'Un blanc de soif, vif et fruité', cible: { alcool: 12.5, acidite: 4.5, tanin: 20, couleur: 28, bois: 5, fruit: 90 }, prime: 3 },
    regularite: { nom: 'La cuvée de marque, la même chaque année', cible: { alcool: 12.8, acidite: 4.3, tanin: 25, couleur: 38, bois: 20, fruit: 75 }, prime: 10 },
  };
  if (style === 'rose') return {
    garde: { nom: 'Un rosé de gastronomie, vineux', cible: { alcool: 13.2, acidite: 3.8, tanin: 32, couleur: 55, bois: 35, fruit: 60 }, prime: 6 },
    fruit: { nom: 'Un rosé de terrasse, pâle et fruité', cible: { alcool: 12.5, acidite: 4.3, tanin: 15, couleur: 24, bois: 3, fruit: 90 }, prime: 3 },
    regularite: { nom: 'La cuvée de marque, la même chaque année', cible: { alcool: 12.8, acidite: 4.1, tanin: 22, couleur: 35, bois: 12, fruit: 80 }, prime: 10 },
  };
  return {
    garde: { nom: 'Le grand vin, fait pour dix ans', cible: { alcool: 13.5, acidite: 3.9, tanin: 75, couleur: 80, bois: 45, fruit: 60 }, prime: 6 },
    fruit: { nom: 'Une cuvée de plaisir immédiat', cible: { alcool: 12.8, acidite: 4.1, tanin: 38, couleur: 58, bois: 8, fruit: 88 }, prime: 3 },
    regularite: { nom: 'La cuvée de marque, la même chaque année', cible: { alcool: 13.2, acidite: 4, tanin: 55, couleur: 65, bois: 25, fruit: 72 }, prime: 10 },
  };
}

const AXES = ['alcool', 'acidite', 'ph', 'tanin', 'couleur', 'fruit', 'bois'];

/* Mélange pondéré des lots + volume réalisable : la cuvée ne peut pas dépasser
   ce que le plus sollicité des lots permet de tirer. */
function melanger(lots, parts) {
  const somme = parts.reduce((a, b) => a + b, 0) || 1;
  const p = parts.map((v) => v / somme);
  const mix = {};
  for (const a of AXES) mix[a] = lots.reduce((s, l, i) => s + p[i] * l[a], 0);
  mix.parts = p;
  mix.presse = lots.reduce((s, l, i) => s + p[i] * l.presse * 100, 0);
  mix.complexite = clamp((1 - p.reduce((s, v) => s + v * v, 0)) / 0.75 * 100, 0, 100);
  mix.total = lots.reduce((s, l) => s + l.hl, 0);
  mix.volume = Math.min(...lots.map((l, i) => p[i] > 0.001 ? l.hl / p[i] : Infinity), mix.total);
  return mix;
}

function noterAssemblage(mix, but) {
  const c = but.cible;
  const ecart = 0.9 * Math.abs(mix.tanin - c.tanin) + 0.9 * Math.abs(mix.couleur - c.couleur)
    + 0.7 * Math.abs(mix.bois - c.bois) + 0.8 * Math.abs(mix.fruit - c.fruit)
    + 12 * Math.abs(mix.alcool - c.alcool) + 15 * Math.abs(mix.acidite - c.acidite);
  let note = 100 - ecart / 1.6 + mix.complexite / 100 * but.prime;
  if (mix.presse > 15) note -= (mix.presse - 15) * 1.2;
  note -= Math.max(0, 55 - mix.volume / mix.total * 100) * (but.prime >= 10 ? 0.35 : 0.12);
  return clamp(note, 0, 100);
}

ATELIERS.assemblage = (boite) => {
  const lots = LOTS[ETAT.style];
  const buts = butsAssemblage(ETAT.style);
  atelierEntete(boite, 'Atelier', 'Composer la cuvée à l’éprouvette');
  const canvas = h('canvas', { 'aria-label': 'Proportions de l’assemblage et volumes disponibles dans la cave' });
  boite.append(canvas);
  const reglages = h('div', { class: 'reglages' });
  boite.append(reglages);
  const parts = lots.map((l, i) => curseur(reglages, {
    id: `asLot${i}`, label: `${l.court} · ${fmt(l.hl)} hL`, min: 0, max: 100, valeur: DEFAUTS_ASSEMBLAGE[ETAT.style][i], affiche: (v) => `${v} %`,
  }));
  const sorties = lots.map((l, i) => boite.querySelector(`output[for="asLot${i}"]`));
  const but = selection(reglages, { id: 'asBut', label: 'Ce que l’on cherche', valeur: 'garde', options: Object.entries(buts).map(([k, b]) => [k, b.nom]) });
  const echantillon = h('div', { class: 'echantillon', 'data-testid': 'robe-assemblage' });
  boite.append(echantillon);
  const jauges = h('div', { class: 'jauges' });
  const jTan = jauge(jauges, ETAT.style === 'rouge' ? 'Tanins' : 'Structure', 'or');
  const jCoul = jauge(jauges, 'Couleur', 'rougeb');
  const jFruit = jauge(jauges, 'Fruit', 'vert');
  const jBois = jauge(jauges, 'Marque du bois');
  boite.append(jauges);
  const mesures = h('dl', { class: 'mesures' });
  const dAlc = h('dd', { 'data-testid': 'assemblage-alcool' }), dAcide = h('dd'), dPh = h('dd'),
    dPresse = h('dd'), dCplx = h('dd'), dVol = h('dd', { 'data-testid': 'volume-cuvee' }), dSecond = h('dd'),
    dNote = h('dd', { class: 'grand', 'data-testid': 'assemblage-note' });
  mesures.append(
    h('dt', {}, 'Degré de la cuvée'), dAlc,
    h('dt', {}, 'Acidité totale (éq. H₂SO₄)'), dAcide,
    h('dt', {}, 'pH'), dPh,
    h('dt', {}, 'Part de vin de presse'), dPresse,
    h('dt', {}, 'Complexité (répartition entre les lots)'), dCplx,
    h('dt', {}, 'Volume réalisable'), dVol,
    h('dt', {}, 'Ce qui part au second vin'), dSecond,
    h('dt', {}, 'Proximité de l’objectif'), dNote);
  boite.append(mesures);
  const lecture = h('p', { class: 'lecture', 'data-testid': 'lecture-assemblage' });
  boite.append(lecture);
  const btnAuto = h('button', { class: 'bouton principal', type: 'button', id: 'asAuto' }, 'Chercher le meilleur assemblage');
  const btnRaz = h('button', { class: 'bouton', type: 'button', id: 'asRaz' }, 'Proportions du chai');
  boite.append(h('div', { class: 'boutons' }, btnAuto, btnRaz));

  function robe(mix) {
    const c = mix.couleur / 100;
    if (ETAT.style === 'blanc') return `rgb(${Math.round(lerp(238, 214, c))},${Math.round(lerp(232, 186, c))},${Math.round(lerp(178, 96, c))})`;
    if (ETAT.style === 'rose') return `rgb(${Math.round(lerp(247, 214, c))},${Math.round(lerp(196, 96, c))},${Math.round(lerp(190, 112, c))})`;
    return `rgb(${Math.round(lerp(150, 80, c))},${Math.round(lerp(60, 14, c))},${Math.round(lerp(90, 46, c))})`;
  }

  function dessiner(mix) {
    const L = 520, H = 150, g = 12, d = 12, l0 = L - g - d;
    const ctx = contexte(canvas, L, H);
    ctx.fillStyle = '#120c0e'; ctx.fillRect(0, 0, L, H);
    ctx.font = '600 9.5px JetBrains Mono, monospace'; ctx.fillStyle = '#a89a95'; ctx.textAlign = 'left';
    ctx.fillText('LA CUVÉE', g, 14);
    let x = g;
    mix.parts.forEach((p, i) => {
      const w = p * l0;
      ctx.fillStyle = COULEURS_LOTS[i];
      ctx.fillRect(x, 20, Math.max(0, w - 1), 30);
      if (w > 42) {
        ctx.fillStyle = '#120c0e'; ctx.font = '600 10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
        ctx.fillText(`${fmt(p * 100)} %`, x + w / 2, 39);
      }
      x += w;
    });
    ctx.font = '500 9.5px JetBrains Mono, monospace'; ctx.fillStyle = '#a89a95'; ctx.textAlign = 'left';
    ctx.fillText('LA CAVE — CE QUI RESTE APRÈS AVOIR TIRÉ LA CUVÉE', g, 74);
    const bh = 13, ecart = 18;
    lots.forEach((l, i) => {
      const y = 82 + i * ecart;
      const w = (l.hl / mix.total) * (l0 - 160);
      const pris = clamp(mix.parts[i] * mix.volume / l.hl, 0, 1);
      ctx.fillStyle = '#241a1d'; ctx.fillRect(g + 148, y, w, bh);
      ctx.fillStyle = COULEURS_LOTS[i]; ctx.fillRect(g + 148, y, w * pris, bh);
      ctx.fillStyle = '#dccfc9'; ctx.font = '500 9.5px Outfit, sans-serif'; ctx.textAlign = 'right';
      ctx.fillText(l.court, g + 142, y + 10);
      ctx.fillStyle = '#a89a95'; ctx.font = '500 9px JetBrains Mono, monospace'; ctx.textAlign = 'left';
      ctx.fillText(`${fmt(l.hl * (1 - pris))} hL`, g + 152 + w, y + 10);
    });
  }

  function maj() {
    const valeurs = parts.map((p) => +p.value);
    const mix = melanger(lots, valeurs);
    const b = buts[but.value];
    const note = noterAssemblage(mix, b);
    mix.parts.forEach((p, i) => { sorties[i].textContent = `${fmt(p * 100)} %`; });
    jTan(mix.tanin); jCoul(mix.couleur); jFruit(mix.fruit); jBois(mix.bois);
    dAlc.textContent = `${fmt(mix.alcool, 1)} % vol.`;
    dAcide.textContent = `${fmt(mix.acidite, 2)} g/L`;
    dPh.textContent = fmt(mix.ph, 2);
    dPresse.textContent = `${fmt(mix.presse, 1)} %`;
    dPresse.className = mix.presse > 15 ? 'alerte' : '';
    dCplx.textContent = `${fmt(mix.complexite)} / 100`;
    dVol.textContent = `${fmt(mix.volume)} hL · ${fmt(mix.volume / 0.75 * 100)} bouteilles`;
    dSecond.textContent = `${fmt(100 - mix.volume / mix.total * 100)} % du chai`;
    dNote.textContent = `${fmt(note)} / 100`;
    dNote.className = `grand ${note >= 80 ? 'bon' : note < 55 ? 'alerte' : ''}`;
    echantillon.style.background = robe(mix);
    echantillon.textContent = `${fmt(mix.alcool, 1)} % vol. · ${fmt(mix.acidite, 2)} g/L · pH ${fmt(mix.ph, 2)}`;

    // Le plus gros écart à l'objectif, et le lot qui le corrigerait.
    const axes = [['tanin', 'de tanins', 0.9], ['couleur', 'de couleur', 0.9], ['bois', 'de bois', 0.7], ['fruit', 'de fruit', 0.8]];
    let pire = null;
    for (const [a, mot, poids] of axes) {
      const e = (mix[a] - b.cible[a]) * poids;
      if (!pire || Math.abs(e) > Math.abs(pire.e)) pire = { a, mot, e };
    }
    const trie = lots.map((l, i) => ({ l, i })).sort((x, y) => y.l[pire.a] - x.l[pire.a]);
    const conseil = pire.e < 0
      ? `il manque ${pire.mot} : montez <b>${trie[0].l.court}</b> (${fmt(trie[0].l[pire.a])} sur cet axe) ou baissez <b>${trie[trie.length - 1].l.court}</b>`
      : `il y a trop ${pire.mot} : baissez <b>${trie[0].l.court}</b> au profit de <b>${trie[trie.length - 1].l.court}</b>`;
    let l, cls = '';
    if (note >= 82) { l = `<b>Cet assemblage y est.</b> ${fmt(mix.alcool, 1)} % vol., ${fmt(mix.acidite, 2)} g/L d'acidité, un profil qui colle à l'objectif « ${b.nom.toLowerCase()} ». On refait analyser le mélange, on le monte en cuve et on le laisse se marier trois à quatre semaines avant la mise : un assemblage frais est toujours dissocié, chaque lot se goûte encore séparément.`; cls = 'bon'; }
    else if (note >= 60) { l = `<b>On approche.</b> Sur les six paramètres, ${conseil}. En cave, on ne bouge pas la cuve pour autant : on refait deux ou trois éprouvettes de 100 mL à 5 % près, on les goûte à l'aveugle avec l'équipe, et on ne tranche qu'ensuite.`; }
    else { l = `<b>Pas encore la cuvée.</b> ${conseil[0].toUpperCase()}${conseil.slice(1)}. Rappel : un assemblage n'est pas une moyenne, c'est un choix — ce qui n'entre pas ici fera le second vin, vendu moins cher mais vendu quand même.`; cls = 'alerte'; }
    if (mix.presse > 15) l += ` Avec ${fmt(mix.presse, 1)} % de vin de presse, la cuvée devient sèche et amère en finale : au-delà de 15 %, on assèche plus qu'on ne structure.`;
    if (mix.complexite < 35) l += ` Un seul lot domine : c'est un vin de parcelle, pas un assemblage. Défendable — la Bourgogne fait cela — mais on perd le filet de sécurité qui fait qu'un assemblage est plus régulier que chacune de ses parties.`;
    if (mix.volume / mix.total < 0.55) l += ` Et la contrainte de cave mord : la proportion demandée d'un petit lot limite la cuvée à ${fmt(mix.volume)} hL, soit ${fmt(100 - mix.volume / mix.total * 100)} % du chai qui part au second vin.`;
    lecture.innerHTML = l;
    lecture.className = `lecture ${cls}`;
    dessiner(mix);
  }

  /* Recherche exhaustive par pas de 5 % : 1 771 assemblages, de quoi trancher. */
  function chercher() {
    const b = buts[but.value];
    let meilleur = null;
    for (let a = 0; a <= 100; a += 5) {
      for (let c = 0; a + c <= 100; c += 5) {
        for (let d = 0; a + c + d <= 100; d += 5) {
          const e = 100 - a - c - d;
          const mix = melanger(lots, [a, c, d, e]);
          const n = noterAssemblage(mix, b);
          if (!meilleur || n > meilleur.n) meilleur = { n, v: [a, c, d, e] };
        }
      }
    }
    parts.forEach((p, i) => { p.value = meilleur.v[i]; p.majSortie(); });
    maj();
  }

  parts.forEach((p) => p.addEventListener('input', maj));
  but.addEventListener('change', maj);
  btnAuto.addEventListener('click', chercher);
  btnRaz.addEventListener('click', () => {
    parts.forEach((p, i) => { p.value = DEFAUTS_ASSEMBLAGE[ETAT.style][i]; p.majSortie(); });
    maj();
  });
  maj();
  return { maj, chercher };
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
