/* Construit les jeux de données de la carte des vignobles de Niagara-on-the-Lake.
   Tout vient de sources ouvertes ; le script est rejouable et écrit dans src/assets/data/.

     node scripts/build-notl-data.js

   Sources
   -------
   · Lots viticoles et trame des concessions : OpenStreetMap, via Overpass (ODbL).
   · Vignobles : Niagara Open Data, couche « Wineries » de la région de Niagara.
   · Limite municipale : Niagara Open Data, couche « Municipal Boundaries ».
   · Sols : Ontario GeoHub / LIO, couche « Soil Survey Complex » (Open Government Licence – Ontario).

   Le cadastre légal de l'Ontario (parcelles Teranet/MPAC) n'est pas une donnée
   ouverte : les polygones de lots affichés sont les blocs de vigne cartographiés
   dans OpenStreetMap, et la trame des concessions rappelle l'arpentage d'origine.
*/

import { writeFile, readFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const RACINE = join(dirname(fileURLToPath(import.meta.url)), "..");
const SORTIE = join(RACINE, "src", "assets", "data");
/* Les serveurs Overpass publics limitent le débit : on garde les réponses sous
   .cache/ pour pouvoir rejouer le script sans les solliciter à nouveau. */
const CACHE = join(RACINE, ".cache", "overpass");

/* Emprise généreuse autour de Niagara-on-the-Lake : sud, ouest, nord, est. */
const BBOX = { sud: 43.12, ouest: -79.42, nord: 43.29, est: -79.02 };

const MIROIRS_OVERPASS = [
  "https://overpass.kumi.systems/api/interpreter",
  "https://overpass.private.coffee/api/interpreter",
  "https://overpass-api.de/api/interpreter",
];

const ARCGIS_NIAGARA =
  "https://services1.arcgis.com/WxiLK82TWf8W3O3f/arcgis/rest/services";
const ARCGIS_LIO =
  "https://ws.lioservices.lrc.gov.on.ca/arcgis1071a/rest/services/LIO_OPEN_DATA";

/* ---------- Petites fonctions de géométrie ---------- */

const ronde = (v, d = 5) => Math.round(v * 10 ** d) / 10 ** d;

const arrondirAnneau = (anneau, d = 5) => {
  const sortie = [];
  for (const [x, y] of anneau) {
    const p = [ronde(x, d), ronde(y, d)];
    const prec = sortie[sortie.length - 1];
    if (!prec || prec[0] !== p[0] || prec[1] !== p[1]) sortie.push(p);
  }
  /* Un anneau doit rester fermé et garder au moins un triangle. */
  if (sortie.length >= 3) {
    const [a, b] = [sortie[0], sortie[sortie.length - 1]];
    if (a[0] !== b[0] || a[1] !== b[1]) sortie.push([a[0], a[1]]);
  }
  return sortie.length >= 4 ? sortie : null;
};

/* Douglas-Peucker, en degrés. À cette latitude 1e-4° ≈ 8 m. */
function simplifier(points, tolerance) {
  if (points.length <= 3 || !tolerance) return points;
  const garde = new Uint8Array(points.length);
  garde[0] = garde[points.length - 1] = 1;
  const pile = [[0, points.length - 1]];
  while (pile.length) {
    const [debut, fin] = pile.pop();
    const [x1, y1] = points[debut];
    const [x2, y2] = points[fin];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const norme = Math.hypot(dx, dy) || 1e-12;
    let pire = -1;
    let distMax = tolerance;
    for (let i = debut + 1; i < fin; i++) {
      const [x, y] = points[i];
      const d = Math.abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norme;
      if (d > distMax) {
        distMax = d;
        pire = i;
      }
    }
    if (pire !== -1) {
      garde[pire] = 1;
      pile.push([debut, pire], [pire, fin]);
    }
  }
  return points.filter((_, i) => garde[i]);
}

function simplifierAnneau(anneau, tolerance) {
  if (anneau.length <= 5) return anneau;
  const ouvert = anneau.slice(0, -1);
  const reduit = simplifier(ouvert, tolerance);
  if (reduit.length < 3) return anneau;
  return [...reduit, reduit[0]];
}

/* Rayon-croisement, sur un polygone [[anneau extérieur], [trous...]]. */
function pointDansAnneau([x, y], anneau) {
  let dedans = false;
  for (let i = 0, j = anneau.length - 1; i < anneau.length; j = i++) {
    const [xi, yi] = anneau[i];
    const [xj, yj] = anneau[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) dedans = !dedans;
  }
  return dedans;
}

function pointDansPolygone(point, polygone) {
  if (!pointDansAnneau(point, polygone[0])) return false;
  for (let t = 1; t < polygone.length; t++) {
    if (pointDansAnneau(point, polygone[t])) return false;
  }
  return true;
}

function pointDansGeometrie(point, geom) {
  const polys = geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
  return polys.some((p) => pointDansPolygone(point, p));
}

/* Aire géodésique approchée d'un anneau, en m². Formule sphérique de l'aire signée. */
const R_TERRE = 6378137;
function aireAnneau(anneau) {
  let total = 0;
  for (let i = 0, j = anneau.length - 1; i < anneau.length; j = i++) {
    const [x1, y1] = anneau[j];
    const [x2, y2] = anneau[i];
    total +=
      ((x2 - x1) * Math.PI) / 180 *
      (2 + Math.sin((y1 * Math.PI) / 180) + Math.sin((y2 * Math.PI) / 180));
  }
  return Math.abs((total * R_TERRE * R_TERRE) / 2);
}

function aireGeometrie(geom) {
  const polys = geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
  let total = 0;
  for (const poly of polys) {
    total += aireAnneau(poly[0]);
    for (let t = 1; t < poly.length; t++) total -= aireAnneau(poly[t]);
  }
  return total;
}

function centroideGeometrie(geom) {
  const polys = geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
  /* La formule du polygone est calculée autour d'un point local : à 79° de
     longitude, la faire tourner sur les coordonnées brutes perd toute la
     précision utile dans les soustractions. */
  const [ox, oy] = polys[0][0][0];
  let sx = 0;
  let sy = 0;
  let sp = 0;
  for (const poly of polys) {
    const anneau = poly[0];
    let aire2 = 0;
    let cx = 0;
    let cy = 0;
    for (let i = 0, j = anneau.length - 1; i < anneau.length; j = i++) {
      const x1 = anneau[j][0] - ox;
      const y1 = anneau[j][1] - oy;
      const x2 = anneau[i][0] - ox;
      const y2 = anneau[i][1] - oy;
      const croix = x1 * y2 - x2 * y1;
      aire2 += croix;
      cx += (x1 + x2) * croix;
      cy += (y1 + y2) * croix;
    }
    if (Math.abs(aire2) < 1e-14) continue;
    const poids = Math.abs(aire2 / 2);
    sx += (cx / (3 * aire2)) * poids;
    sy += (cy / (3 * aire2)) * poids;
    sp += poids;
  }
  if (!sp) return null;
  return [ronde(ox + sx, 5), ronde(oy + sy, 5)];
}

/* Distance approchée en mètres, suffisante à l'échelle d'une commune. */
function distanceM([x1, y1], [x2, y2]) {
  const lat = ((y1 + y2) / 2) * (Math.PI / 180);
  const dx = (x2 - x1) * Math.cos(lat) * 111320;
  const dy = (y2 - y1) * 110540;
  return Math.hypot(dx, dy);
}

/* ---------- Réseau ---------- */

async function json(url, options = {}, essais = 3) {
  let derniere;
  for (let i = 0; i < essais; i++) {
    try {
      const reponse = await fetch(url, options);
      if (!reponse.ok) throw new Error(`HTTP ${reponse.status} — ${url}`);
      return await reponse.json();
    } catch (erreur) {
      derniere = erreur;
      await new Promise((r) => setTimeout(r, 2000 * 2 ** i));
    }
  }
  throw derniere;
}

async function overpass(requete) {
  const cle = createHash("sha1").update(requete).digest("hex").slice(0, 16);
  const chemin = join(CACHE, `${cle}.json`);
  try {
    const cache = JSON.parse(await readFile(chemin, "utf8"));
    console.log(`  ⤓ cache ${cle}`);
    return cache;
  } catch {
    /* Pas encore en cache : on interroge le réseau. */
  }

  let derniere;
  for (const miroir of MIROIRS_OVERPASS) {
    try {
      const url = `${miroir}?data=${encodeURIComponent(requete)}`;
      const donnees = await json(url, {}, 2);
      if (!donnees.elements) throw new Error("réponse Overpass sans éléments");
      await mkdir(CACHE, { recursive: true });
      await writeFile(chemin, JSON.stringify(donnees));
      return donnees;
    } catch (erreur) {
      derniere = erreur;
      console.warn(`  ↻ ${miroir} : ${erreur.message}`);
    }
  }
  throw derniere;
}

async function arcgis(base, couche, parametres) {
  const url = new URL(`${base}/${couche}/query`);
  for (const [cle, valeur] of Object.entries({
    where: "1=1",
    outFields: "*",
    outSR: "4326",
    returnGeometry: "true",
    f: "geojson",
    ...parametres,
  })) {
    url.searchParams.set(cle, valeur);
  }
  return json(url.toString());
}

/* ---------- Conversion OSM → GeoJSON ---------- */

const enAnneau = (geometrie) => geometrie.map((p) => [p.lon, p.lat]);

function fermer(anneau) {
  if (anneau.length < 3) return null;
  const [a, b] = [anneau[0], anneau[anneau.length - 1]];
  if (a[0] !== b[0] || a[1] !== b[1]) return [...anneau, [a[0], a[1]]];
  return anneau;
}

/* Recolle les segments d'une relation multipolygone en anneaux fermés. */
function assemblerAnneaux(segments) {
  const restants = segments.map((s) => s.slice());
  const anneaux = [];
  while (restants.length) {
    let courant = restants.pop();
    let progresse = true;
    while (progresse && !memeSommet(courant[0], courant[courant.length - 1])) {
      progresse = false;
      for (let i = 0; i < restants.length; i++) {
        const seg = restants[i];
        const fin = courant[courant.length - 1];
        if (memeSommet(fin, seg[0])) {
          courant = courant.concat(seg.slice(1));
        } else if (memeSommet(fin, seg[seg.length - 1])) {
          courant = courant.concat(seg.slice(0, -1).reverse());
        } else if (memeSommet(courant[0], seg[seg.length - 1])) {
          courant = seg.slice(0, -1).concat(courant);
        } else if (memeSommet(courant[0], seg[0])) {
          courant = seg.slice(1).reverse().concat(courant);
        } else {
          continue;
        }
        restants.splice(i, 1);
        progresse = true;
        break;
      }
    }
    const ferme = fermer(courant);
    if (ferme) anneaux.push(ferme);
  }
  return anneaux;
}

const memeSommet = (a, b) => a && b && a[0] === b[0] && a[1] === b[1];

function geometrieOsm(element) {
  if (element.type === "way") {
    const anneau = fermer(enAnneau(element.geometry || []));
    return anneau ? { type: "Polygon", coordinates: [anneau] } : null;
  }
  const exterieurs = [];
  const interieurs = [];
  for (const membre of element.members || []) {
    if (!membre.geometry) continue;
    (membre.role === "inner" ? interieurs : exterieurs).push(enAnneau(membre.geometry));
  }
  const anneauxExt = assemblerAnneaux(exterieurs);
  const anneauxInt = assemblerAnneaux(interieurs);
  if (!anneauxExt.length) return null;
  /* Chaque trou est rattaché au premier extérieur qui le contient. */
  const polygones = anneauxExt.map((ext) => [ext]);
  for (const trou of anneauxInt) {
    const cible = polygones.find((p) => pointDansAnneau(trou[0], p[0]));
    (cible || polygones[0]).push(trou);
  }
  return polygones.length === 1
    ? { type: "Polygon", coordinates: polygones[0] }
    : { type: "MultiPolygon", coordinates: polygones };
}

function nettoyerGeometrie(geom, tolerance = 0) {
  if (!geom) return null;
  const polys = geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
  const propres = [];
  for (const poly of polys) {
    const anneaux = [];
    for (const anneau of poly) {
      const simplifie = tolerance ? simplifierAnneau(anneau, tolerance) : anneau;
      const arrondi = arrondirAnneau(simplifie);
      if (arrondi) anneaux.push(arrondi);
    }
    if (anneaux.length) propres.push(anneaux);
  }
  if (!propres.length) return null;
  return propres.length === 1
    ? { type: "Polygon", coordinates: propres[0] }
    : { type: "MultiPolygon", coordinates: propres };
}

/* ---------- Étapes ---------- */

async function limiteMunicipale() {
  const donnees = await arcgis(ARCGIS_NIAGARA, "OpenData_Municipal_Boundaries/FeatureServer/26", {
    where: "Name='Niagara-on-the-Lake'",
    outFields: "Name,Label,LandArea,POP2021",
  });
  const brute = donnees.features[0];
  if (!brute) throw new Error("limite municipale introuvable");
  return {
    type: "Feature",
    properties: {
      nom: "Niagara-on-the-Lake",
      superficie_km2: brute.properties.LandArea,
      population_2021: brute.properties.POP2021,
    },
    geometry: nettoyerGeometrie(brute.geometry, 0.00003),
  };
}

/* Noms de cépages tels qu'on les écrit en français, quand OpenStreetMap
   s'en écarte. Le reste est simplement mis en capitale initiale. */
const CEPAGES = {
  aligote: "Aligoté",
  "cabernet franc": "Cabernet franc",
  "cabernet sauvignon": "Cabernet sauvignon",
  "gamay noir": "Gamay noir",
  gewurztraminer: "Gewurztraminer",
  "petit verdot": "Petit verdot",
  "pinot gris": "Pinot gris",
  "pinot noir": "Pinot noir",
  "sauvignon blanc": "Sauvignon blanc",
};

const joliCepage = (brut) =>
  brut
    .split(";")
    .map((c) => c.trim().toLowerCase().replace(/_/g, " "))
    .filter(Boolean)
    .map((c) => CEPAGES[c] || c[0].toUpperCase() + c.slice(1))
    .filter((c, i, t) => t.indexOf(c) === i);

async function lotsViticoles(limite) {
  const requete = `[out:json][timeout:300][bbox:${BBOX.sud},${BBOX.ouest},${BBOX.nord},${BBOX.est}];
(way["landuse"="vineyard"];relation["landuse"="vineyard"];);
out geom;`;
  const donnees = await overpass(requete);
  const traits = [];
  for (const element of donnees.elements) {
    const geom = nettoyerGeometrie(geometrieOsm(element), 0.00002);
    if (!geom) continue;
    const centre = centroideGeometrie(geom);
    if (!centre || !pointDansGeometrie(centre, limite.geometry)) continue;
    const tags = element.tags || {};
    const hectares = aireGeometrie(geom) / 10000;
    if (hectares < 0.15) continue;
    traits.push({
      type: "Feature",
      id: `${element.type === "way" ? "w" : "r"}${element.id}`,
      properties: {
        nom: tags.name || null,
        exploitant: tags.operator || null,
        cepages: tags.grape_variety ? joliCepage(tags.grape_variety) : null,
        hectares: ronde(hectares, 2),
        centre,
      },
      geometry: geom,
    });
  }
  traits.sort((a, b) => b.properties.hectares - a.properties.hectares);
  return traits;
}

async function trameCadastrale(limite) {
  const requete = `[out:json][timeout:180][bbox:${BBOX.sud},${BBOX.ouest},${BBOX.nord},${BBOX.est}];
way["highway"]["name"~"^(Line [0-9]+|Concession [0-9]+)"];
out geom;`;
  const donnees = await overpass(requete);
  const parNom = new Map();
  for (const element of donnees.elements) {
    const nom = element.tags?.name;
    if (!nom) continue;
    const points = enAnneau(element.geometry || []).map(([x, y]) => [ronde(x), ronde(y)]);
    if (points.length < 2) continue;
    if (!points.some((p) => pointDansGeometrie(p, limite.geometry))) continue;
    if (!parNom.has(nom)) parNom.set(nom, []);
    parNom.get(nom).push(points);
  }
  return [...parNom.entries()]
    .sort((a, b) => a[0].localeCompare(b[0], "fr", { numeric: true }))
    .map(([nom, lignes]) => ({
      type: "Feature",
      properties: {
        nom,
        genre: nom.startsWith("Line") ? "ligne" : "concession",
      },
      geometry: { type: "MultiLineString", coordinates: lignes },
    }));
}

async function vignobles() {
  const donnees = await arcgis(ARCGIS_NIAGARA, "OpenData_Wineries/FeatureServer/5", {
    where: "Municipality='Niagara-on-the-Lake'",
    outFields: "Name,Municipality,Full_Address,Phone,URL",
  });
  return donnees.features
    .filter((f) => f.geometry?.coordinates)
    .map((f) => ({
      type: "Feature",
      properties: {
        nom: f.properties.Name,
        adresse: f.properties.Full_Address || null,
        telephone: f.properties.Phone || null,
        site: f.properties.URL || null,
      },
      geometry: {
        type: "Point",
        coordinates: f.geometry.coordinates.map((v) => ronde(v, 5)),
      },
    }))
    .sort((a, b) => a.properties.nom.localeCompare(b.properties.nom, "fr"));
}

/* Codes de la couche « Soil Survey Complex » de l'Ontario. */
const TEXTURES = {
  S: "sable", LS: "sable loameux", SL: "loam sableux", L: "loam",
  SIL: "loam limoneux", SI: "limon", SCL: "loam sablo-argileux",
  CL: "loam argileux", SICL: "loam limono-argileux", SC: "argile sableuse",
  SIC: "argile limoneuse", C: "argile", HC: "argile lourde",
  O: "matière organique", M: "terreau noir", R: "roc",
};
const DRAINAGES = {
  R: "rapide", W: "bon", MW: "modérément bon", I: "imparfait",
  P: "pauvre", VP: "très pauvre", V: "très pauvre",
};

const joliNom = (brut) =>
  brut
    ? brut
        .toLowerCase()
        .split(/([ -])/)
        .map((m) => (m.length > 1 ? m[0].toUpperCase() + m.slice(1) : m))
        .join("")
    : null;

async function sols(limite) {
  const donnees = await arcgis(ARCGIS_LIO, "LIO_Open05/MapServer/9", {
    geometry: `${BBOX.ouest},${BBOX.sud},${BBOX.est},${BBOX.nord}`,
    geometryType: "esriGeometryEnvelope",
    inSR: "4326",
    spatialRel: "esriSpatialRelIntersects",
    outFields: "OGF_ID,SOIL_NAME1,ATEXTURE1,DRAINAGE1,SLOPE1,CLI1,HECTARES",
  });
  const traits = [];
  for (const brut of donnees.features) {
    const geom = nettoyerGeometrie(brut.geometry, 0.00006);
    if (!geom) continue;
    const polys = geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
    const touche =
      polys.some((p) => p[0].some((pt) => pointDansGeometrie(pt, limite.geometry))) ||
      pointDansGeometrie(centroideGeometrie(geom) || [0, 0], limite.geometry);
    if (!touche) continue;
    const p = brut.properties;
    const texture = (p.ATEXTURE1 || "").toUpperCase().trim();
    traits.push({
      type: "Feature",
      properties: {
        serie: joliNom(p.SOIL_NAME1),
        texture: TEXTURES[texture] || (texture || null),
        code_texture: texture || null,
        drainage: DRAINAGES[(p.DRAINAGE1 || "").toUpperCase().trim()] || null,
        pente: p.SLOPE1 == null ? null : ronde(p.SLOPE1, 1),
        classe_agricole: p.CLI1 || null,
      },
      geometry: geom,
    });
  }
  return traits;
}

/* Rattache chaque lot au vignoble le plus proche, et chaque vignoble à ses lots. */
function relier(lots, vignobles) {
  const normalise = (s) =>
    (s || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/\b(winery|wines|wine|estate|estates|vineyards|vineyard|cellars|the|co|inc|niagara)\b/g, "")
      .replace(/[^a-z0-9]/g, "");

  const index = vignobles.map((v) => ({
    nom: v.properties.nom,
    cle: normalise(v.properties.nom),
    point: v.geometry.coordinates,
    hectares: 0,
    lots: 0,
    nommes: 0,
  }));

  for (const lot of lots) {
    const centre = lot.properties.centre;
    let rattache = null;
    const cle = normalise(lot.properties.exploitant || lot.properties.nom);
    if (cle.length > 3) {
      rattache = index.find((v) => v.cle && (v.cle.includes(cle) || cle.includes(v.cle))) || null;
    }
    const parNom = !!rattache;
    if (!rattache) {
      let meilleure = 900;
      for (const v of index) {
        const d = distanceM(centre, v.point);
        if (d < meilleure) {
          meilleure = d;
          rattache = v;
        }
      }
    }
    lot.properties.vignoble = rattache ? rattache.nom : null;
    lot.properties.rattachement = rattache ? (parNom ? "exploitant" : "proximite") : null;
    if (rattache) {
      rattache.hectares += lot.properties.hectares;
      rattache.lots += 1;
      if (parNom) rattache.nommes += 1;
    }
    delete lot.properties.centre;
  }

  for (const v of vignobles) {
    const fiche = index.find((i) => i.nom === v.properties.nom);
    v.properties.lots = fiche.lots;
    v.properties.lots_nommes = fiche.nommes;
    v.properties.hectares = ronde(fiche.hectares, 1);
  }
}

const collection = (traits, extra = {}) => ({
  type: "FeatureCollection",
  ...extra,
  features: traits,
});

async function ecrire(nom, contenu) {
  const chemin = join(SORTIE, nom);
  await writeFile(chemin, JSON.stringify(contenu));
  const taille = Buffer.byteLength(JSON.stringify(contenu));
  console.log(`  ✓ ${nom} — ${(taille / 1024).toFixed(0)} Ko`);
}

async function principal() {
  await mkdir(SORTIE, { recursive: true });

  console.log("Limite municipale…");
  const limite = await limiteMunicipale();

  console.log("Vignobles…");
  const listeVignobles = await vignobles();

  console.log("Lots viticoles (OpenStreetMap)…");
  const lots = await lotsViticoles(limite);

  console.log("Trame des concessions (OpenStreetMap)…");
  const trame = await trameCadastrale(limite);

  console.log("Sols (Ontario GeoHub)…");
  const listeSols = await sols(limite);

  relier(lots, listeVignobles);

  const total = lots.reduce((s, l) => s + l.properties.hectares, 0);

  await ecrire("notl-limite.geojson", collection([limite]));
  await ecrire("notl-trame.geojson", collection(trame));
  await ecrire("notl-vignobles.geojson", collection(listeVignobles));
  await ecrire(
    "notl-lots.geojson",
    collection(lots, {
      meta: {
        genere: new Date().toISOString().slice(0, 10),
        lots: lots.length,
        hectares: ronde(total, 1),
        source: "OpenStreetMap (ODbL)",
      },
    })
  );
  await ecrire("notl-sols.geojson", collection(listeSols));

  console.log(
    `\n${lots.length} lots, ${ronde(total, 1)} ha, ${listeVignobles.length} vignobles, ${listeSols.length} polygones de sol.`
  );
}

principal().catch((erreur) => {
  console.error(erreur);
  process.exit(1);
});
