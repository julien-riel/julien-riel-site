/* Carte des lots viticoles de Niagara-on-the-Lake.

   Trois jeux de données, tous ouverts, préparés par scripts/build-notl-data.js :
   les lots de vigne et la trame des concessions viennent d'OpenStreetMap, les
   vignobles et la limite municipale de Niagara Open Data, les sols du GeoHub
   de l'Ontario. Le cadastre légal de l'Ontario n'étant pas ouvert, les
   polygones affichés sont les blocs de vigne, pas les parcelles du registre.
*/

import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

/* MapLibre déduit l'adresse de son worker de celle de son propre module ;
   une fois le tout empaqueté, cette déduction tombe à côté. On la lui donne. */
import urlWorker from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

/* Les fichiers sont importés en ?url pour que Vite les empreinte et les
   publie ; un simple chemin en dur serait retiré du build. */
import urlLimite from "../data/notl-limite.geojson?url";
import urlTrame from "../data/notl-trame.geojson?url";
import urlVignobles from "../data/notl-vignobles.geojson?url";
import urlLots from "../data/notl-lots.geojson?url";
import urlSols from "../data/notl-sols.geojson?url";

maplibregl.setWorkerUrl(urlWorker);

/* ---------- Constantes ---------- */

const CENTRE = [-79.12, 43.205];

const FONDS = {
  sombre: {
    nom: "Sombre",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    clair: false,
  },
  clair: {
    nom: "Clair",
    style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    clair: true,
  },
  photo: {
    nom: "Photo",
    style: {
      version: 8,
      glyphs: "https://tiles.basemaps.cartocdn.com/fonts/{fontstack}/{range}.pbf",
      sources: {
        photo: {
          type: "raster",
          tiles: [
            "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          ],
          tileSize: 256,
          maxzoom: 19,
          attribution:
            "Imagerie <a href=\"https://www.esri.com/\">Esri</a>, Maxar, Earthstar Geographics",
        },
      },
      layers: [
        { id: "fond", type: "background", paint: { "background-color": "#0d1210" } },
        { id: "photo", type: "raster", source: "photo" },
      ],
    },
    clair: false,
  },
};

/* Si le fond distant ne répond pas, la carte reste utilisable sur un fond uni :
   les lots, eux, sont servis avec la page. */
const STYLE_SECOURS = {
  version: 8,
  sources: {},
  layers: [{ id: "fond", type: "background", paint: { "background-color": "#141b1e" } }],
};

/* Palette qualitative : assez de teintes pour distinguer les voisins, toutes
   lisibles sur fond sombre comme sur photo aérienne. */
const PALETTE = [
  "#e0577f", "#f0a63f", "#79c46a", "#5fb8d6", "#b98ae0",
  "#e8d35c", "#ef8560", "#4fc3a1", "#7f9ff0", "#d96fb8",
  "#a8cf8a", "#c9a227",
];

/* Les blancs vers le jaune-vert, les rouges vers le grenat. */
const COULEUR_CEPAGE = {
  Riesling: "#e8d35c",
  Chardonnay: "#f0c14b",
  "Sauvignon blanc": "#a9cf6a",
  Vidal: "#c9d46a",
  Aligoté: "#d6e08a",
  Savagnin: "#bcd07f",
  Gewurztraminer: "#e6a3c4",
  "Pinot gris": "#d8a0b4",
  Viognier: "#efb06a",
  Gamay: "#d1607f",
  "Gamay noir": "#c95c86",
  "Pinot noir": "#c2547e",
  Merlot: "#b5567a",
  "Cabernet franc": "#a8425f",
  "Cabernet sauvignon": "#7f2f4a",
  "Petit verdot": "#8d3b62",
  Syrah: "#6e2a48",
};

const ECHELLE_SURFACE = [
  [0.5, "#3d2f52"],
  [1.5, "#4f4b86"],
  [3, "#3f7fa6"],
  [6, "#4aa88b"],
  [12, "#8fc45f"],
  [25, "#e8d35c"],
];

const COULEUR_SOL = {
  sable: "#e3c98d",
  "sable loameux": "#dcc084",
  "loam sableux": "#d1b077",
  loam: "#bf9f6d",
  "loam limoneux": "#ab9a72",
  limon: "#9d9a7c",
  "loam sablo-argileux": "#b08a6b",
  "loam argileux": "#9d7c66",
  "loam limono-argileux": "#8a7570",
  "argile sableuse": "#8a6a5f",
  "argile limoneuse": "#7a6265",
  argile: "#6d5a60",
  "argile lourde": "#5c4c55",
  "matière organique": "#4a4438",
  "terreau noir": "#3f3b33",
  roc: "#807d7a",
};

const nf0 = new Intl.NumberFormat("fr-CA", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("fr-CA", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat("fr-CA", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const ha = (v) => (v >= 10 ? nf1.format(v) : nf2.format(v)) + " ha";
const acres = (v) => nf1.format(v * 2.47105) + " acres";

const $ = (sel) => document.querySelector(sel);
const creer = (balise, classe) => {
  const el = document.createElement(balise);
  if (classe) el.className = classe;
  return el;
};

const sansAccent = (s) =>
  (s || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");

/* ---------- État ---------- */

const etat = {
  carte: null,
  fond: "sombre",
  teinte: "vignoble",
  couches: { lots: true, vignobles: true, trame: false, sols: false },
  lots: null,
  vignobles: null,
  limite: null,
  trame: null,
  sols: null,
  solsEnCours: null,
  selection: null,
  survol: null,
  filtre: "",
  secours: false,
  empriseMuni: null,
  mesureAvis: null,
  couleurs: new Map(),
  cepages: [],
  pret: false,
};

/* ---------- Chargement des données ---------- */

async function charger(url) {
  const reponse = await fetch(url);
  if (!reponse.ok) throw new Error(`HTTP ${reponse.status}`);
  return reponse.json();
}

/* Chaque vignoble reçoit une teinte stable, et chaque lot celle du sien. */
function preparerCouleurs() {
  const noms = [...new Set(etat.lots.features.map((f) => f.properties.vignoble).filter(Boolean))].sort(
    (a, b) => a.localeCompare(b, "fr")
  );
  noms.forEach((nom, i) => etat.couleurs.set(nom, PALETTE[i % PALETTE.length]));

  const comptes = new Map();
  for (const lot of etat.lots.features) {
    const p = lot.properties;
    p.couleurVignoble = etat.couleurs.get(p.vignoble) || "#6b5a63";
    const cepage = p.cepages && p.cepages.length ? p.cepages[0] : null;
    p.cepage = cepage;
    p.couleurCepage = (cepage && COULEUR_CEPAGE[cepage]) || "#4a3b44";
    p.etiquette = p.nom || p.vignoble || "Lot de vigne";
    if (cepage) comptes.set(cepage, (comptes.get(cepage) || 0) + 1);
  }
  etat.cepages = [...comptes.entries()].sort((a, b) => b[1] - a[1]);
}

/* ---------- Style des lots selon le mode de teinte ---------- */

function opaciteLots() {
  const base = etat.fond === "photo" ? 0.5 : 0.68;
  const ordinaire =
    etat.teinte === "cepage"
      ? ["case", ["==", ["get", "cepage"], null], base * 0.4, base]
      : base;
  return [
    "case",
    ["boolean", ["feature-state", "selection"], false], 0.92,
    ["boolean", ["feature-state", "survol"], false], 0.82,
    ordinaire,
  ];
}

function couleurLots() {
  if (etat.teinte === "cepage") return ["get", "couleurCepage"];
  if (etat.teinte === "surface") {
    const stops = ECHELLE_SURFACE.flatMap(([seuil, couleur]) => [seuil, couleur]);
    return ["interpolate", ["linear"], ["get", "hectares"], ...stops];
  }
  return ["get", "couleurVignoble"];
}

/* ---------- Construction des couches ---------- */

function ajouterSource(id, donnees, options = {}) {
  const carte = etat.carte;
  if (carte.getSource(id)) carte.getSource(id).setData(donnees);
  else carte.addSource(id, { type: "geojson", data: donnees, ...options });
}

/* Dans l'ordre de dessin : les sols dessous, les vignobles dessus. */
const MES_COUCHES = [
  "sols-fond", "sols-contour", "limite-halo", "trame-ligne", "trame-nom",
  "lots-fond", "lots-contour", "lots-nom", "vignobles-point", "vignobles-nom",
];

function construireCouches() {
  const carte = etat.carte;
  /* Reposer les couches plutôt que les empiler : c'est ce qui garde l'ordre
     quand les sols arrivent après coup ou qu'on change de fond. */
  for (const id of MES_COUCHES) if (carte.getLayer(id)) carte.removeLayer(id);
  const clair = !etat.secours && FONDS[etat.fond].clair;
  /* Un style sans glyphes ne sait pas dessiner de texte : on saute les
     étiquettes plutôt que d'inonder la console d'avertissements. */
  const etiquettes = !!carte.getStyle().glyphs;

  ajouterSource("limite", etat.limite);
  ajouterSource("trame", etat.trame);
  ajouterSource("lots", etat.lots);
  ajouterSource("vignobles", etat.vignobles);
  if (etat.sols) ajouterSource("sols", etat.sols);

  if (etat.sols) {
    carte.addLayer({
      id: "sols-fond",
      type: "fill",
      source: "sols",
      layout: { visibility: etat.couches.sols ? "visible" : "none" },
      paint: {
        "fill-color": ["coalesce", ["get", "couleur"], "#6d5a60"],
        "fill-opacity": clair ? 0.5 : 0.38,
      },
    });
    carte.addLayer({
      id: "sols-contour",
      type: "line",
      source: "sols",
      layout: { visibility: etat.couches.sols ? "visible" : "none" },
      paint: { "line-color": clair ? "#ffffff" : "#000000", "line-opacity": 0.25, "line-width": 0.6 },
    });
  }

  carte.addLayer({
    id: "limite-halo",
    type: "line",
    source: "limite",
    paint: {
      "line-color": clair ? "#7a5c66" : "#f0a3b8",
      "line-width": 1.4,
      "line-opacity": 0.55,
      "line-dasharray": [3, 2],
    },
  });

  carte.addLayer({
    id: "trame-ligne",
    type: "line",
    source: "trame",
    layout: { visibility: etat.couches.trame ? "visible" : "none" },
    paint: {
      "line-color": clair ? "#8a6a3c" : "#d9b25f",
      "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.6, 15, 2.2],
      "line-opacity": 0.75,
    },
  });
  if (etiquettes) carte.addLayer({
    id: "trame-nom",
    type: "symbol",
    source: "trame",
    minzoom: 11.5,
    layout: {
      visibility: etat.couches.trame ? "visible" : "none",
      "symbol-placement": "line",
      "text-field": ["get", "nom"],
      "text-size": 11,
      "text-letter-spacing": 0.08,
      "text-font": ["Open Sans Regular", "Arial Unicode MS Regular"],
    },
    paint: {
      "text-color": clair ? "#7a5b30" : "#e8cf94",
      "text-halo-color": clair ? "#ffffff" : "#1a1114",
      "text-halo-width": 1.4,
    },
  });

  carte.addLayer({
    id: "lots-fond",
    type: "fill",
    source: "lots",
    layout: { visibility: etat.couches.lots ? "visible" : "none" },
    paint: {
      "fill-color": couleurLots(),
      "fill-opacity": opaciteLots(),
    },
  });
  carte.addLayer({
    id: "lots-contour",
    type: "line",
    source: "lots",
    layout: { visibility: etat.couches.lots ? "visible" : "none" },
    paint: {
      "line-color": [
        "case",
        ["boolean", ["feature-state", "selection"], false], clair ? "#1c0a10" : "#ffffff",
        clair ? "#4a2f38" : "#f3ebe4",
      ],
      /* « zoom » n'est accepté qu'à la racine : le cas passe donc à l'intérieur. */
      "line-width": [
        "interpolate", ["linear"], ["zoom"],
        11, [
          "case",
          ["boolean", ["feature-state", "selection"], false], 2.2,
          ["boolean", ["feature-state", "survol"], false], 1.2,
          0.25,
        ],
        16, [
          "case",
          ["boolean", ["feature-state", "selection"], false], 3.4,
          ["boolean", ["feature-state", "survol"], false], 2,
          0.9,
        ],
      ],
      "line-opacity": 0.85,
    },
  });
  if (etiquettes) carte.addLayer({
    id: "lots-nom",
    type: "symbol",
    source: "lots",
    minzoom: 14,
    layout: {
      visibility: etat.couches.lots ? "visible" : "none",
      "text-field": ["get", "etiquette"],
      "text-size": 11,
      "text-max-width": 9,
      "text-font": ["Open Sans Regular", "Arial Unicode MS Regular"],
      "text-optional": true,
    },
    paint: {
      "text-color": clair ? "#2a1d21" : "#f3ebe4",
      "text-halo-color": clair ? "#ffffff" : "#160f11",
      "text-halo-width": 1.5,
    },
  });

  carte.addLayer({
    id: "vignobles-point",
    type: "circle",
    source: "vignobles",
    layout: { visibility: etat.couches.vignobles ? "visible" : "none" },
    paint: {
      "circle-radius": [
        "interpolate", ["linear"], ["zoom"],
        10, ["case", ["boolean", ["feature-state", "selection"], false], 6.5, 3.2],
        15, ["case", ["boolean", ["feature-state", "selection"], false], 10.5, 7],
      ],
      "circle-color": "#f3ebe4",
      "circle-stroke-color": "#9b2f4c",
      "circle-stroke-width": 2.2,
    },
  });
  if (etiquettes) carte.addLayer({
    id: "vignobles-nom",
    type: "symbol",
    source: "vignobles",
    minzoom: 12,
    layout: {
      visibility: etat.couches.vignobles ? "visible" : "none",
      "text-field": ["get", "nom"],
      "text-size": 12,
      "text-max-width": 10,
      "text-offset": [0, 1.1],
      "text-anchor": "top",
      "text-font": ["Open Sans Regular", "Arial Unicode MS Regular"],
      "text-optional": true,
    },
    paint: {
      "text-color": clair ? "#1c0a10" : "#ffffff",
      "text-halo-color": clair ? "#ffffff" : "#160f11",
      "text-halo-width": 1.6,
    },
  });

  appliquerSelection();
}

/* ---------- Survol et sélection ---------- */

function poserEtat(id, cle, valeur) {
  if (id == null) return;
  const source = typeof id === "string" && id.startsWith("v:") ? "vignobles" : "lots";
  const identifiant = source === "vignobles" ? Number(id.slice(2)) : id;
  try {
    etat.carte.setFeatureState({ source, id: identifiant }, { [cle]: valeur });
  } catch {
    /* La source n'est pas encore prête après un changement de fond. */
  }
}

function appliquerSelection() {
  if (!etat.selection) return;
  poserEtat(etat.selection.id, "selection", true);
}

function selectionner(genre, identifiant, options = {}) {
  if (etat.selection) poserEtat(etat.selection.id, "selection", false);

  if (!genre) {
    etat.selection = null;
    rendrePanneau();
    return;
  }

  const id = genre === "vignoble" ? `v:${identifiant}` : identifiant;
  etat.selection = { genre, identifiant, id };
  poserEtat(id, "selection", true);
  rendrePanneau();

  if (options.recadrer) {
    const trait =
      genre === "vignoble"
        ? etat.vignobles.features.find((f) => f.id === identifiant)
        : etat.lots.features.find((f) => f.id === identifiant);
    if (trait) {
      if (trait.geometry.type === "Point") {
        etat.carte.easeTo({ center: trait.geometry.coordinates, zoom: Math.max(etat.carte.getZoom(), 14) });
      } else {
        etat.carte.fitBounds(emprise(trait.geometry), { padding: 120, maxZoom: 16.5 });
      }
    }
  }
  rendreListe();
}

/* L'emprise de la municipalité, calculée une fois à partir de sa limite. */
function empriseMunicipale() {
  if (!etat.empriseMuni) etat.empriseMuni = emprise(etat.limite.features[0].geometry);
  return etat.empriseMuni;
}

function emprise(geometrie) {
  const polys = geometrie.type === "MultiPolygon" ? geometrie.coordinates : [geometrie.coordinates];
  const b = new maplibregl.LngLatBounds();
  for (const poly of polys) for (const point of poly[0]) b.extend(point);
  return b;
}

/* ---------- Sols : chargement paresseux ---------- */

async function assurerSols() {
  if (etat.sols) return etat.sols;
  if (etat.solsEnCours) return etat.solsEnCours;
  etat.solsEnCours = charger(urlSols)
    .then((donnees) => {
      for (const trait of donnees.features) {
        trait.properties.couleur = COULEUR_SOL[trait.properties.texture] || "#6d5a60";
      }
      etat.sols = donnees;
      etat.solsEnCours = null;
      if (etat.carte.isStyleLoaded() && !etat.carte.getSource("sols")) {
        reconstruire();
      }
      return donnees;
    })
    .catch((erreur) => {
      etat.solsEnCours = null;
      console.warn("sols :", erreur.message);
      return null;
    });
  return etat.solsEnCours;
}

/* Le sol d'un lot : le polygone pédologique qui contient son centre. */
function solSous(point) {
  if (!etat.sols) return null;
  for (const trait of etat.sols.features) {
    const polys =
      trait.geometry.type === "MultiPolygon" ? trait.geometry.coordinates : [trait.geometry.coordinates];
    for (const poly of polys) {
      if (dansAnneau(point, poly[0]) && !poly.slice(1).some((trou) => dansAnneau(point, trou))) {
        return trait.properties;
      }
    }
  }
  return null;
}

function dansAnneau([x, y], anneau) {
  let dedans = false;
  for (let i = 0, j = anneau.length - 1; i < anneau.length; j = i++) {
    const [xi, yi] = anneau[i];
    const [xj, yj] = anneau[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) dedans = !dedans;
  }
  return dedans;
}

function centre(geometrie) {
  const polys = geometrie.type === "MultiPolygon" ? geometrie.coordinates : [geometrie.coordinates];
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const poly of polys) {
    for (const [x, y] of poly[0]) {
      sx += x;
      sy += y;
      n += 1;
    }
  }
  return n ? [sx / n, sy / n] : null;
}

/* ---------- Panneau de détail ---------- */

function ligne(cle, valeur) {
  if (valeur == null || valeur === "") return null;
  const li = creer("div", "fiche-ligne");
  const dt = creer("span", "fiche-cle");
  dt.textContent = cle;
  const dd = creer("span", "fiche-valeur");
  if (valeur instanceof Node) dd.appendChild(valeur);
  else dd.textContent = valeur;
  li.append(dt, dd);
  return li;
}

function rendrePanneau() {
  const boite = $("#fiche");
  boite.textContent = "";

  if (!etat.selection) {
    boite.classList.add("vide");
    const p = creer("p", "invite");
    p.textContent =
      "Touchez un lot ou un vignoble sur la carte — ou choisissez-en un dans la liste — pour voir sa fiche.";
    boite.appendChild(p);
    return;
  }
  boite.classList.remove("vide");

  if (etat.selection.genre === "vignoble") rendreFicheVignoble(boite);
  else rendreFicheLot(boite);
}

function boutonFermer() {
  const b = creer("button", "fermer-fiche");
  b.type = "button";
  b.setAttribute("aria-label", "Fermer la fiche");
  b.textContent = "×";
  b.addEventListener("click", () => selectionner(null));
  return b;
}

function rendreFicheVignoble(boite) {
  const trait = etat.vignobles.features.find((f) => f.id === etat.selection.identifiant);
  if (!trait) return;
  const p = trait.properties;

  const entete = creer("div", "fiche-entete");
  const h = creer("h2");
  h.textContent = p.nom;
  const genre = creer("p", "fiche-genre");
  genre.textContent = "Vignoble";
  entete.append(h, genre, boutonFermer());
  boite.appendChild(entete);

  const corps = creer("div", "fiche-corps");
  const lots = etat.lots.features.filter((f) => f.properties.vignoble === p.nom);
  const total = lots.reduce((s, f) => s + f.properties.hectares, 0);

  const pastille = creer("span", "pastille");
  pastille.style.background = etat.couleurs.get(p.nom) || "#6b5a63";
  const teinteLigne = ligne("Couleur sur la carte", pastille);

  const cepages = [...new Set(lots.flatMap((f) => f.properties.cepages || []))].sort((a, b) =>
    a.localeCompare(b, "fr")
  );

  const lignes = [
    ligne("Adresse", p.adresse),
    ligne("Municipalité", "Niagara-on-the-Lake"),
    lots.length
      ? ligne("Lots rattachés", `${lots.length} · ${ha(total)} (${acres(total)})`)
      : ligne("Lots rattachés", "aucun lot cartographié à proximité"),
    cepages.length ? ligne("Cépages relevés", cepages.join(", ")) : null,
    p.telephone ? ligne("Téléphone", p.telephone) : null,
    teinteLigne,
  ].filter(Boolean);
  corps.append(...lignes);

  if (p.site) {
    const a = creer("a", "fiche-lien");
    a.href = p.site;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = "Site du vignoble ↗";
    corps.appendChild(a);
  }

  if (lots.length) {
    const details = creer("details", "fiche-lots");
    const somm = creer("summary");
    somm.textContent = `${lots.length} lot${lots.length > 1 ? "s" : ""} rattaché${lots.length > 1 ? "s" : ""}`;
    details.appendChild(somm);
    const ul = creer("ul");
    for (const lot of [...lots].sort((a, b) => b.properties.hectares - a.properties.hectares)) {
      const li = creer("li");
      const b = creer("button");
      b.type = "button";
      b.innerHTML = `<span>${echapper(lot.properties.nom || "Lot sans nom")}</span><span class="mono">${ha(
        lot.properties.hectares
      )}</span>`;
      b.addEventListener("click", () => selectionner("lot", lot.id, { recadrer: true }));
      li.appendChild(b);
      ul.appendChild(li);
    }
    details.appendChild(ul);
    corps.appendChild(details);
  }

  boite.appendChild(corps);
}

function rendreFicheLot(boite) {
  const trait = etat.lots.features.find((f) => f.id === etat.selection.identifiant);
  if (!trait) return;
  const p = trait.properties;

  const entete = creer("div", "fiche-entete");
  const h = creer("h2");
  h.textContent = p.nom || "Lot de vigne";
  const genre = creer("p", "fiche-genre");
  genre.textContent = "Lot viticole";
  entete.append(h, genre, boutonFermer());
  boite.appendChild(entete);

  const corps = creer("div", "fiche-corps");
  const lignes = [
    ligne("Superficie", `${ha(p.hectares)} · ${acres(p.hectares)}`),
    p.exploitant ? ligne("Exploitant", p.exploitant) : null,
  ];

  if (p.vignoble) {
    const b = creer("button", "fiche-renvoi");
    b.type = "button";
    b.textContent = p.vignoble;
    const cible = etat.vignobles.features.find((f) => f.properties.nom === p.vignoble);
    b.addEventListener("click", () => cible && selectionner("vignoble", cible.id, { recadrer: true }));
    lignes.push(
      ligne(p.rattachement === "exploitant" ? "Vignoble (exploitant déclaré)" : "Vignoble le plus proche", b)
    );
  }

  if (p.cepages && p.cepages.length) lignes.push(ligne("Cépages", p.cepages.join(", ")));
  corps.append(...lignes.filter(Boolean));

  const bloc = creer("div", "fiche-sol");
  bloc.textContent = "Sol : lecture en cours…";
  corps.appendChild(bloc);
  assurerSols().then(() => {
    if (!etat.selection || etat.selection.identifiant !== trait.id) return;
    const point = centre(trait.geometry);
    const sol = point && solSous(point);
    bloc.textContent = "";
    if (!sol) {
      bloc.textContent = "Sol : non renseigné à cet endroit.";
      return;
    }
    const titre = creer("p", "fiche-sol-titre");
    const pastille = creer("span", "pastille");
    pastille.style.background = COULEUR_SOL[sol.texture] || "#6d5a60";
    titre.append(pastille, document.createTextNode(` Sol ${sol.serie || "sans nom"}`));
    bloc.appendChild(titre);
    const details = [
      sol.texture ? ligne("Texture", sol.texture) : null,
      sol.drainage ? ligne("Drainage", sol.drainage) : null,
      sol.pente != null ? ligne("Pente", `${nf1.format(sol.pente)} %`) : null,
      sol.classe_agricole ? ligne("Classe agricole (ICT)", sol.classe_agricole) : null,
    ].filter(Boolean);
    bloc.append(...details);
  });

  boite.appendChild(corps);
}

const echapper = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/* ---------- Liste latérale ---------- */

function rendreListe() {
  const ul = $("#liste");
  ul.textContent = "";
  const filtre = sansAccent(etat.filtre).trim();

  const entrees = etat.vignobles.features
    .map((f) => {
      const lots = etat.lots.features.filter((l) => l.properties.vignoble === f.properties.nom);
      return {
        trait: f,
        lots: lots.length,
        hectares: lots.reduce((s, l) => s + l.properties.hectares, 0),
      };
    })
    .filter((e) => !filtre || sansAccent(e.trait.properties.nom).includes(filtre))
    .sort((a, b) => b.hectares - a.hectares || a.trait.properties.nom.localeCompare(b.trait.properties.nom, "fr"));

  $("#compte-liste").textContent =
    entrees.length === etat.vignobles.features.length
      ? `${entrees.length} vignobles`
      : `${entrees.length} sur ${etat.vignobles.features.length}`;

  for (const entree of entrees) {
    const li = creer("li");
    const b = creer("button", "entree");
    b.type = "button";
    const choisi =
      etat.selection && etat.selection.genre === "vignoble" && etat.selection.identifiant === entree.trait.id;
    if (choisi) b.classList.add("choisi");
    b.setAttribute("aria-pressed", String(!!choisi));

    const pastille = creer("span", "pastille");
    pastille.style.background = etat.couleurs.get(entree.trait.properties.nom) || "#6b5a63";
    const nom = creer("span", "entree-nom");
    nom.textContent = entree.trait.properties.nom;
    const chiffre = creer("span", "entree-chiffre mono");
    chiffre.textContent = entree.lots ? ha(entree.hectares) : "—";
    b.append(pastille, nom, chiffre);
    b.addEventListener("click", () => selectionner("vignoble", entree.trait.id, { recadrer: true }));
    li.appendChild(b);
    ul.appendChild(li);
  }

  if (!entrees.length) {
    const li = creer("li", "vide");
    li.textContent = "Aucun vignoble ne porte ce nom.";
    ul.appendChild(li);
  }
}

/* ---------- Légende ---------- */

function rendreLegende() {
  const boite = $("#legende");
  boite.textContent = "";

  const ajouter = (couleur, texte) => {
    const li = creer("li");
    const pastille = creer("span", "pastille");
    pastille.style.background = couleur;
    const t = creer("span");
    t.textContent = texte;
    li.append(pastille, t);
    boite.appendChild(li);
  };

  if (etat.teinte === "surface") {
    for (const [seuil, couleur] of ECHELLE_SURFACE) ajouter(couleur, `${nf0.format(seuil)} ha`);
  } else if (etat.teinte === "cepage") {
    for (const [cepage, n] of etat.cepages.slice(0, 9)) {
      ajouter(COULEUR_CEPAGE[cepage] || "#4a3b44", `${cepage} · ${n}`);
    }
    ajouter("#4a3b44", "Cépage non relevé");
  } else {
    ajouter("#f3ebe4", "Vignoble (point)");
    ajouter("#e0577f", "Lot rattaché à un vignoble");
    ajouter("#6b5a63", "Lot sans vignoble connu");
  }

  if (etat.couches.sols) {
    const titre = creer("li", "legende-titre");
    titre.textContent = "Texture du sol";
    boite.appendChild(titre);
    for (const texture of ["sable loameux", "loam sableux", "loam", "loam argileux", "loam limono-argileux", "argile"]) {
      ajouter(COULEUR_SOL[texture], texture);
    }
  }
}

/* ---------- Chiffres d'en-tête ---------- */

function rendreChiffres() {
  const lots = etat.lots.features;
  const total = lots.reduce((s, f) => s + f.properties.hectares, 0);
  $("#chiffre-lots").textContent = nf0.format(lots.length);
  $("#chiffre-hectares").textContent = nf0.format(total);
  $("#chiffre-vignobles").textContent = nf0.format(etat.vignobles.features.length);
  $("#resume").textContent = `${nf0.format(lots.length)} lots · ${nf0.format(total)} ha · ${nf0.format(
    etat.vignobles.features.length
  )} vignobles`;
}

/* ---------- Mise en service ---------- */

/* Appelé au premier style qui aboutit, que ce soit le fond distant ou celui
   de secours : c'est là que la carte devient utilisable. */
function finaliser() {
  if (etat.pret) return;
  brancherCarte();
  etat.carte.fitBounds(empriseMunicipale(), { padding: 30, duration: 0 });
  $("#etat-chargement")?.remove();
  document.body.classList.add("carte-prete");
  etat.pret = true;
  /* Les sols arrivent en sourdine, une fois la carte posée. */
  etat.carte.once("idle", () => {
    if (!etat.sols && !etat.solsEnCours) assurerSols();
  });
}

/* ---------- Fond indisponible ---------- */

function afficherAvis(texte) {
  const avis = $("#avis");
  avis.textContent = texte;
  avis.hidden = false;
  document.body.classList.add("avis-visible");
  /* Sur un écran étroit le message passe à la ligne : on mesure plutôt que
     de parier sur une hauteur fixe. */
  const mesurer = () => {
    document.body.style.setProperty("--decalage-avis", `${avis.offsetHeight}px`);
    etat.carte?.resize();
  };
  mesurer();
  if (typeof ResizeObserver === "function") {
    etat.mesureAvis?.disconnect();
    etat.mesureAvis = new ResizeObserver(mesurer);
    etat.mesureAvis.observe(avis);
  }
}

function masquerAvis() {
  $("#avis").hidden = true;
  etat.mesureAvis?.disconnect();
  document.body.classList.remove("avis-visible");
  document.body.style.removeProperty("--decalage-avis");
  etat.carte?.resize();
}

/* Le fond de carte vient d'un service tiers ; s'il ne répond pas, on retombe
   sur un fond uni au lieu de laisser la page vide. */
function surveillerFond() {
  let retombe = false;
  const basculer = () => {
    if (retombe || etat.secours) return;
    retombe = true;
    etat.secours = true;
    afficherAvis("Le fond de carte n'a pas pu être chargé. Les lots restent affichés sur fond uni.");
    etat.carte.setStyle(STYLE_SECOURS);
    etat.carte.once("styledata", () => reconstruire());
  };

  /* Seul un style qui ne charge pas déclenche le repli : une tuile manquante
     ici ou là ne doit pas faire disparaître le fond. */
  etat.carte.on("error", (e) => {
    if (!etat.carte.isStyleLoaded()) {
      basculer();
      return;
    }
    console.warn("carte :", e?.error?.message || e);
  });
  const minuteur = setTimeout(basculer, 9000);
  etat.carte.on("load", () => clearTimeout(minuteur));
}

/* ---------- Reconstruction après changement de fond ---------- */

function reconstruire() {
  if (!etat.carte.isStyleLoaded()) {
    etat.carte.once("idle", reconstruire);
    return;
  }
  construireCouches();
  rendreLegende();
  finaliser();
}

function changerFond(nom) {
  if (etat.fond === nom && !etat.secours) return;
  etat.fond = nom;
  etat.secours = false;
  masquerAvis();
  for (const bouton of document.querySelectorAll("[data-fond]")) {
    bouton.setAttribute("aria-pressed", String(bouton.dataset.fond === nom));
  }
  etat.carte.setStyle(FONDS[nom].style);
  etat.carte.once("styledata", () => reconstruire());
}

function changerTeinte(nom) {
  etat.teinte = nom;
  for (const bouton of document.querySelectorAll("[data-teinte]")) {
    bouton.setAttribute("aria-pressed", String(bouton.dataset.teinte === nom));
  }
  if (etat.carte.getLayer("lots-fond")) {
    etat.carte.setPaintProperty("lots-fond", "fill-color", couleurLots());
    etat.carte.setPaintProperty("lots-fond", "fill-opacity", opaciteLots());
  }
  rendreLegende();
}

async function basculerCouche(nom, actif) {
  etat.couches[nom] = actif;
  const bouton = document.querySelector(`[data-couche="${nom}"]`);
  if (bouton) bouton.setAttribute("aria-pressed", String(actif));

  if (nom === "sols" && actif) {
    bouton?.classList.add("chargement");
    await assurerSols();
    bouton?.classList.remove("chargement");
    if (!etat.carte.getSource("sols") && etat.sols) reconstruire();
  }

  const couches = {
    lots: ["lots-fond", "lots-contour", "lots-nom"],
    vignobles: ["vignobles-point", "vignobles-nom"],
    trame: ["trame-ligne", "trame-nom"],
    sols: ["sols-fond", "sols-contour"],
  }[nom];
  for (const couche of couches) {
    if (etat.carte.getLayer(couche)) {
      etat.carte.setLayoutProperty(couche, "visibility", actif ? "visible" : "none");
    }
  }
  rendreLegende();
}

/* ---------- Interactions carte ---------- */

function brancherCarte() {
  const carte = etat.carte;

  const cliquables = ["vignobles-point", "lots-fond"];
  for (const couche of cliquables) {
    carte.on("mouseenter", couche, () => {
      carte.getCanvas().style.cursor = "pointer";
    });
    carte.on("mouseleave", couche, () => {
      carte.getCanvas().style.cursor = "";
      if (etat.survol) {
        poserEtat(etat.survol, "survol", false);
        etat.survol = null;
      }
    });
  }

  carte.on("mousemove", "lots-fond", (e) => {
    const trait = e.features && e.features[0];
    if (!trait) return;
    if (etat.survol === trait.id) return;
    if (etat.survol) poserEtat(etat.survol, "survol", false);
    etat.survol = trait.id;
    poserEtat(trait.id, "survol", true);
  });

  carte.on("click", (e) => {
    const points = carte.queryRenderedFeatures(e.point, { layers: ["vignobles-point"] });
    if (points.length) {
      selectionner("vignoble", points[0].id);
      return;
    }
    const lots = carte.queryRenderedFeatures(e.point, { layers: ["lots-fond"] });
    if (lots.length) {
      selectionner("lot", lots[0].id);
      return;
    }
    selectionner(null);
  });

  carte.on("moveend", () => {
    const zoom = carte.getZoom();
    $("#zoom").textContent = zoom.toFixed(1);
  });
}

/* ---------- Commandes ---------- */

function brancherCommandes() {
  for (const bouton of document.querySelectorAll("[data-fond]")) {
    bouton.addEventListener("click", () => changerFond(bouton.dataset.fond));
  }
  for (const bouton of document.querySelectorAll("[data-teinte]")) {
    bouton.addEventListener("click", () => changerTeinte(bouton.dataset.teinte));
  }
  for (const bouton of document.querySelectorAll("[data-couche]")) {
    bouton.addEventListener("click", () => {
      basculerCouche(bouton.dataset.couche, bouton.getAttribute("aria-pressed") !== "true");
    });
  }

  $("#recherche").addEventListener("input", (e) => {
    etat.filtre = e.target.value;
    rendreListe();
  });

  $("#recadrer").addEventListener("click", () => {
    etat.carte.fitBounds(empriseMunicipale(), { padding: 30 });
  });

  $("#bascule-volet").addEventListener("click", () => {
    const ouvert = document.body.classList.toggle("volet-ouvert");
    $("#bascule-volet").setAttribute("aria-expanded", String(ouvert));
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && etat.selection) selectionner(null);
    if (e.key === "?" || (e.key === "h" && !e.metaKey && !e.ctrlKey)) {
      const cible = e.target;
      if (cible && (cible.tagName === "INPUT" || cible.tagName === "TEXTAREA")) return;
      const aide = $("#aide");
      if (aide.open) aide.close();
      else aide.showModal();
    }
  });
  $("#ouvrir-aide").addEventListener("click", () => $("#aide").showModal());
}

/* ---------- Démarrage ---------- */

async function demarrer() {
  const etatChargement = $("#etat-chargement");
  try {
    const [limite, trame, vignobles, lots] = await Promise.all([
      charger(urlLimite),
      charger(urlTrame),
      charger(urlVignobles),
      charger(urlLots),
    ]);
    etat.limite = limite;
    etat.trame = trame;
    etat.vignobles = vignobles;
    etat.lots = lots;
    /* feature-state veut des identifiants numériques : on garde l'identifiant
       OpenStreetMap dans les propriétés et on numérote les traits. */
    vignobles.features.forEach((f, i) => {
      f.id = i;
    });
    lots.features.forEach((f, i) => {
      f.properties.osm = f.id;
      f.id = i;
    });
  } catch (erreur) {
    etatChargement.textContent = `Les données n'ont pas pu être chargées (${erreur.message}).`;
    etatChargement.classList.add("erreur");
    return;
  }

  preparerCouleurs();
  rendreChiffres();
  rendreListe();
  rendrePanneau();

  etat.carte = new maplibregl.Map({
    container: "carte",
    style: FONDS.sombre.style,
    center: CENTRE,
    zoom: 11.1,
    maxZoom: 18,
    minZoom: 8,
    maxBounds: [
      [-79.9, 42.8],
      [-78.6, 43.6],
    ],
    attributionControl: false,
    cooperativeGestures: true,
  });

  etat.carte.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  etat.carte.addControl(new maplibregl.ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-right");
  etat.carte.addControl(
    new maplibregl.AttributionControl({
      compact: true,
      customAttribution:
        "<a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> · <a href=\"https://niagaraopendata.ca/\">Niagara Open Data</a> · <a href=\"https://geohub.lio.gov.on.ca/\">Ontario GeoHub</a>",
    }),
    "bottom-right"
  );

  surveillerFond();

  etat.carte.on("load", () => {
    construireCouches();
    rendreLegende();
    finaliser();
  });

  brancherCommandes();
}

demarrer();

/* Surface d'inspection pour les tests de bout en bout. */
window.vignobles = {
  pret: () => etat.pret,
  lots: () => etat.lots.features.length,
  hectares: () => etat.lots.features.reduce((s, f) => s + f.properties.hectares, 0),
  vignobles: () => etat.vignobles.features.map((f) => f.properties.nom),
  selection: () => etat.selection,
  teinte: () => etat.teinte,
  fond: () => etat.fond,
  couches: () => ({ ...etat.couches }),
  selectionner,
  solsCharges: () => !!etat.sols,
  secours: () => etat.secours,
  carte: () => etat.carte,
  couchesPosees: () => etat.carte.getStyle().layers.map((c) => c.id),
};
