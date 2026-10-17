/* 10-donnees-ilots.js */
/* Les sept lignes d'affaires (îlots de l'open space) et leurs paramètres de départ.
   Valeurs normalisées 0–1 sauf mention contraire. */
(function (LTP) {
  'use strict';

  // ordre : position dans l'open space (lecture de gauche à droite, de haut en bas)
  // criticite : coût d'un incident ; sensibilite : part de données personnelles ou financières
  // volatilite : fréquence des changements de besoins ; adequationPile : qualité de l'agent sur cette pile
  // volumeInitial : taille du code existant, en milliers de lignes (kLOC)
  // demandeBase : points de demande ajoutés au backlog par semaine (un développeur inter livre ≈ 5 points)
  // exposition : branches de la triade létale allumées par la nature même du travail
  LTP.donnees.ilots = [
    {
      id: 'appro', ordre: 0, nom: 'Approvisionnement', court: 'Appro', teinte: 32,
      pile: 'AS/400 (RPG), SQL Server, Excel',
      mission: 'Achats, fournisseurs, bons de commande et factures.',
      criticite: 0.6, sensibilite: 0.7, volatilite: 0.3, adequationPile: 0.45,
      volumeInitial: 180, backlogInitial: 30, demandeBase: 14, satisfactionInitiale: 0.55,
      exposition: { prive: true, nonFiable: true, sortie: false },
      direction: { nom: 'Francine Gagnon', titre: 'Directrice de l\'approvisionnement' },
    },
    {
      id: 'operations', ordre: 1, nom: 'Opérations', court: 'Opérations', teinte: 4,
      pile: 'Python, Bash, Ansible, PostgreSQL',
      mission: 'Ordonnancement, garde, déploiements et outils internes des opérations.',
      criticite: 0.9, sensibilite: 0.4, volatilite: 0.25, adequationPile: 0.8,
      volumeInitial: 90, backlogInitial: 25, demandeBase: 13, satisfactionInitiale: 0.6,
      exposition: { prive: true, nonFiable: false, sortie: false },
      direction: { nom: 'Réjean Tremblay', titre: 'Directeur des opérations' },
    },
    {
      id: 'bureautique', ordre: 2, nom: 'Bureautique', court: 'Bureautique', teinte: 140,
      pile: 'VBA, Power Automate, SharePoint',
      mission: 'Macros, formulaires et automatisations pour les services administratifs, dont la paie.',
      criticite: 0.5, sensibilite: 0.8, volatilite: 0.5, adequationPile: 0.5,
      volumeInitial: 60, backlogInitial: 35, demandeBase: 15, satisfactionInitiale: 0.5,
      exposition: { prive: true, nonFiable: false, sortie: false },
      direction: { nom: 'Josée Lapointe', titre: 'Directrice des services administratifs' },
    },
    {
      id: 'inspections', ordre: 3, nom: 'Inspections', court: 'Inspections', teinte: 200,
      pile: 'Kotlin, Swift, synchronisation hors ligne',
      mission: 'L\'application mobile des inspecteurs terrain : photos, formulaires, rapports.',
      criticite: 0.6, sensibilite: 0.75, volatilite: 0.6, adequationPile: 0.7,
      volumeInitial: 70, backlogInitial: 30, demandeBase: 14, satisfactionInitiale: 0.55,
      exposition: { prive: true, nonFiable: false, sortie: false },
      direction: { nom: 'Nathalie Bouchard', titre: 'Directrice des inspections' },
    },
    {
      id: 'iot', ordre: 4, nom: 'Capteurs et IoT', court: 'IoT', teinte: 265,
      pile: 'C embarqué, MQTT, Python',
      mission: 'Firmware des capteurs et pipelines de télémétrie.',
      criticite: 0.95, sensibilite: 0.3, volatilite: 0.2, adequationPile: 0.4,
      volumeInitial: 40, backlogInitial: 20, demandeBase: 12, satisfactionInitiale: 0.6,
      exposition: { prive: false, nonFiable: false, sortie: false },
      direction: { nom: 'Alain Côté', titre: 'Directeur du réseau de capteurs' },
    },
    {
      id: 'web', ordre: 5, nom: 'Sites web', court: 'Web', teinte: 320,
      pile: 'TypeScript, Astro, WordPress',
      mission: 'Les sites publics et les microsites de campagne.',
      criticite: 0.3, sensibilite: 0.2, volatilite: 0.85, adequationPile: 0.9,
      volumeInitial: 110, backlogInitial: 40, demandeBase: 16, satisfactionInitiale: 0.5,
      exposition: { prive: false, nonFiable: true, sortie: true },
      direction: { nom: 'Isabelle Roy', titre: 'Directrice des communications' },
    },
    {
      id: 'portail', ordre: 6, nom: 'Portail client', court: 'Portail', teinte: 230,
      pile: 'TypeScript, React, Java, OAuth',
      mission: 'Le portail des clients et des partenaires : comptes, demandes, paiements.',
      criticite: 0.8, sensibilite: 0.85, volatilite: 0.5, adequationPile: 0.85,
      volumeInitial: 150, backlogInitial: 35, demandeBase: 15, satisfactionInitiale: 0.55,
      exposition: { prive: true, nonFiable: true, sortie: false },
      direction: { nom: 'Daniel Morin', titre: 'Directeur des services aux clients' },
    },
  ];
})(globalThis.LTP = globalThis.LTP || {});
