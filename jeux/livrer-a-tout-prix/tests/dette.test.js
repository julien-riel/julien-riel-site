// Dette cognitive (23) : compréhension, MTTR, bogues et incidents (CONCEPTION §4.5).
import { test } from "node:test";
import assert from "node:assert/strict";
import { LTP, partie, avancer, moyenneGraines, GRAINES, somme, ilots, devs, carnetPlein } from "./production-outils.js";

const comprehension = (e) => somme(devs(e), (d) => d.comprehension) / devs(e).length;
function compApres(leviers, n = 16) {
  return moyenneGraines(GRAINES, (g) => comprehension(avancer(partie(g, leviers), n)));
}

test("au départ, la compréhension reste à peu près stable", () => {
  const avant = comprehension(partie(1, []));
  const apres = compApres([], 16);
  assert.ok(Math.abs(apres - avant) < 0.06, `${avant} → ${apres}`);
});

test("plus d'autonomie : moins de compréhension", () => {
  const a1 = compApres([{ id: "autonomie", valeur: 1 }]);
  const a3 = compApres([{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 3 }]);
  const a4 = compApres([{ id: "autonomie", valeur: 4 }, { id: "sessionsMax", valeur: 6 }]);
  assert.ok(a3 < a1 && a4 < a3, `${a1} ${a3} ${a4}`);
});

test("pair humain et revues d'explication restaurent la compréhension", () => {
  const ia = [{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 3 }];
  const sans = compApres(ia);
  const avec = compApres([...ia, { id: "pair", valeur: "humain" }, { id: "revuesExplication", valeur: "majeures" }]);
  assert.ok(avec > sans + 0.03, `${sans} → ${avec}`);
});

test("MTTR : ~6 à 10 h au départ, 30 h et plus avec une compréhension effondrée et un gros volume", () => {
  const e = partie(1, []);
  for (const il of ilots(e)) {
    const h = LTP.systemes.dette.mttr(e, il.id, 0.5);
    assert.ok(h >= 5 && h <= 11, `${il.id} : ${h} h`);
  }
  const x = JSON.parse(JSON.stringify(e));
  for (const d of devs(x)) if (d.ilot === "portail") d.comprehension = 0.1;
  x.ilots.portail.codebase.volume = 600;
  assert.ok(LTP.systemes.dette.mttr(x, "portail", 0.6) >= 30);
  // Sens de variation : compréhension, volume, observabilité, mods du registre.
  const base = LTP.systemes.dette.mttr(e, "portail", 0.5);
  assert.ok(LTP.systemes.dette.mttr(x, "portail", 0.5) > base);
  const obs = JSON.parse(JSON.stringify(e));
  obs.org.observabilite = 1;
  assert.ok(LTP.systemes.dette.mttr(obs, "portail", 0.5) < base);
  const mod = JSON.parse(JSON.stringify(e));
  mod.ilots.portail.mods.mttr = 0.5;
  assert.ok(LTP.systemes.dette.mttr(mod, "portail", 0.5) < base);
  assert.ok(LTP.systemes.dette.mttr(e, "portail", 0.9) > LTP.systemes.dette.mttr(e, "portail", 0.2));
});

test("drapeau « personne ne comprend » sous 0,3, retiré au-dessus de 0,4", () => {
  let e = avancer(partie(2, [], { preparer: (x) => { for (const d of Object.values(x.devs)) if (d.ilot === "iot") d.comprehension = 0.2; } }), 1);
  assert.equal(e.org.drapeaux.personneNeComprend_iot, true);
  assert.ok(!e.org.drapeaux.personneNeComprend_web);
  assert.ok(e.journal.some((j) => j.cle === "dette.personneNeComprend" && j.ilot === "iot"));
  e = avancer(e, 1, { avant: (x) => { for (const d of Object.values(x.devs)) if (d.ilot === "iot") d.comprehension = 0.6; } });
  assert.ok(!e.org.drapeaux.personneNeComprend_iot);
});

test("plus de défauts latents : plus de bogues ; les bogues retirent des défauts latents", () => {
  const bogues = (latents) => moyenneGraines(GRAINES, (g) => {
    let b = 0;
    avancer(partie(g, [], { preparer: (x) => { for (const il of Object.values(x.ilots)) il.codebase.defautsLatents = latents; } }), 2,
      { apres: (e) => { b += somme(ilots(e), (i) => i.derniere.bogues); } });
    return b;
  });
  assert.ok(bogues(60) > 2 * bogues(5));
});

test("incidents : créés selon la criticité, développeurs mobilisés, résolus à échéance", () => {
  // Beaucoup de défauts latents dans un îlot critique : des incidents finissent par éclater.
  let vus = 0, mobilises = 0, resolus = 0;
  for (const g of GRAINES) {
    let e = partie(g, [], { preparer: (x) => { x.ilots.iot.codebase.defautsLatents = 120; } });
    e = avancer(e, 6, {
      apres: (x) => {
        const actifs = LTP.q.incidentsActifs(x, "iot");
        if (actifs.length) {
          vus++;
          if (devs(x).some((d) => d.ilot === "iot" && d.etat === "incident")) mobilises++;
        }
      },
    });
    resolus += e.incidents.filter((i) => i.resolu && i.finSemaine < e.semaine).length;
    for (const i of e.incidents) if (!i.resolu) assert.ok(i.finSemaine >= e.semaine - 1, `incident ${i.id} échu non résolu`);
  }
  assert.ok(vus > 0, "au moins un incident");
  assert.equal(mobilises, vus, "pendant un incident, des développeurs de l'îlot sont mobilisés");
  assert.ok(resolus > 0);
});

test("le code fusionné sans lecture coûte plus de compréhension que le code relu", () => {
  const lev = [{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 4 }];
  const lu = compApres([...lev, { id: "reviseurs", valeur: 1 }], 12);
  const auto = compApres([...lev, { id: "approbation", valeur: "auto" }], 12);
  assert.ok(auto < lu, `${lu} contre ${auto}`);
});

test("un incident en cours sur l'îlot prend du temps de production la semaine suivante", () => {
  const pts = (avecIncident) => moyenneGraines(GRAINES, (g) => {
    const e = avancer(partie(g, [], {
      preparer: (x) => {
        if (avecIncident) x.incidents.push({ id: "t1", ilot: "web", gravite: 0.8, cause: "bogue", debut: x.semaine, mttrHeures: 40, finSemaine: x.semaine + 1, resolu: false });
      },
    }), 1, { avant: carnetPlein });
    return e.ilots.web.derniere.points;
  });
  assert.ok(pts(true) < pts(false) * 0.85);
});
