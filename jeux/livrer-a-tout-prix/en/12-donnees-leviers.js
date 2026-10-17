/* en/12-donnees-leviers.js */
/* Édition anglaise : panneaux et leviers (libellés fixés par le glossaire de en/LISEZMOI.md). */
(function (LTP) {
  'use strict';

  LTP.traduire('panneaux', {
    ia:           { libelle: 'AI use' },
    pratiques:    { libelle: 'Practices' },
    produit:      { libelle: 'Product' },
    registre:     { libelle: 'Registry' },
    gouvernance:  { libelle: 'Data and security' },
    organisation: { libelle: 'Organization' },
  });

  LTP.traduire('leviers', {
    // --- Engineering practices (Q1) ---
    pair: {
      libelle: 'Pair programming',
      aide: 'Two heads on the same code: better understanding, lower raw throughput.',
      valeurs: {
        aucun:    { libelle: 'None' },
        humain:   { libelle: 'Human with human' },
        ia:       { libelle: 'Human with AI' },
        rotation: { libelle: 'Rotation (every other week)' },
      },
    },
    reviseurs: {
      libelle: 'Reviewers per PR',
      aide: 'More eyes catch more defects, but the review queue grows longer.',
      valeurs: { 0: { libelle: 'None' }, 1: { libelle: 'One' }, 2: { libelle: 'Two' } },
    },
    approbation: {
      libelle: 'PR approval',
      aide: 'Who has the final say before the merge.',
      valeurs: {
        auto:         { libelle: 'Automatic (AI review)' },
        humain:       { libelle: 'A human' },
        proprietaire: { libelle: 'The code owner' },
      },
    },
    taillePR: {
      libelle: 'Maximum PR size',
      aide: 'A small PR actually gets read. A big one gets skimmed.',
      valeurs: { libre: { libelle: 'Unlimited' }, 400: { libelle: '400 lines' }, 200: { libelle: '200 lines' } },
    },
    ci: {
      libelle: 'Continuous integration and deployment',
      aide: 'Each stage of the pipeline catches defects earlier, and costs a little time.',
      valeurs: {
        aucune:  { libelle: 'None' },
        build:   { libelle: 'Build' },
        tests:   { libelle: 'Build and tests' },
        complet: { libelle: 'Full pipeline and continuous deployment' },
      },
    },
    lint: {
      libelle: 'Lint and formatting',
      aide: 'Fewer nitpicks in code review. Locked: the agent can\'t switch off a rule to make CI pass.',
      valeurs: {
        aucun:      { libelle: 'None' },
        formatage:  { libelle: 'Auto-formatting' },
        strict:     { libelle: 'Strict lint' },
        verrouille: { libelle: 'Strict, locked lint' },
      },
    },
    tests: {
      libelle: 'Tests',
      aide: 'Mutation testing exposes the tests that check nothing.',
      valeurs: {
        aucun:       { libelle: 'None' },
        unitaires:   { libelle: 'Unit' },
        integration: { libelle: 'Unit and integration' },
        mutation:    { libelle: 'With mutation testing' },
      },
    },
    revuesExplication: {
      libelle: 'Explanation reviews',
      aide: 'People explain their code without the AI. Measures and restores understanding.',
      valeurs: {
        non:        { libelle: 'No' },
        mensuelles: { libelle: 'Monthly' },
        majeures:   { libelle: 'Every major PR' },
      },
    },

    // --- AI use ---
    licences: {
      libelle: 'Copilot licences',
      aide: 'Basic GitHub Copilot licences. Without a licence, you code the way you used to: no assistant, no tokens.',
      valeurs: {
        aucune: { libelle: 'None' },
        pilote: { libelle: 'Pilot group (7)' },
        tous:   { libelle: 'Whole team (21)' },
      },
    },
    autonomie: {
      libelle: 'Agent autonomy level',
      aide: 'More autonomy, more throughput. Fewer human eyes, and a weaker grasp of the code.',
      valeurs: {
        0: { libelle: '0 · Autocomplete' },
        1: { libelle: '1 · Chat' },
        2: { libelle: '2 · Agent with approval at every step' },
        3: { libelle: '3 · Agent that opens PRs' },
        4: { libelle: '4 · Agent that merges on its own' },
      },
    },
    sessionsMax: {
      libelle: 'Agent sessions per person',
      aide: 'Multi-Clauding: more throughput, with diminishing returns, and more fatigue.',
      // Valeurs générées par .map en français ; les libellés anglais sont les mêmes mots (glossaire).
      valeurs: {
        1: { libelle: '1 session' },
        2: { libelle: '2 sessions' },
        3: { libelle: '3 sessions' },
        4: { libelle: '4 sessions' },
        5: { libelle: '5 sessions' },
        6: { libelle: '6 sessions' },
      },
    },
    modele: {
      libelle: 'Default model',
      aide: 'A good model costs more per token, but makes fewer mistakes. The local model keeps the data in-house.',
      valeurs: {
        economique:    { libelle: 'Economy' },
        intermediaire: { libelle: 'Mid-range' },
        haut:          { libelle: 'High-end' },
        local:         { libelle: 'Local (on the GPU server)' },
      },
    },
    modeleParTache: {
      libelle: 'Model chosen per task',
      aide: 'A small model for lint and documentation, a big one for architecture.',
      valeurs: { false: { libelle: 'No' }, true: { libelle: 'Yes' } },
    },
    formation: {
      libelle: 'Training program',
      aide: 'Costs time and money right away, pays off a month later.',
      valeurs: {
        aucune:    { libelle: 'None' },
        atelier:   { libelle: 'One workshop' },
        programme: { libelle: 'Full program' },
        mentorat:  { libelle: 'Program and mentoring' },
      },
    },
    plafondJetons: {
      libelle: 'Token cap',
      aide: 'Cuts off runaway agent loops. Too strict, and it blocks real work.',
      valeurs: { aucun: { libelle: 'None' }, souple: { libelle: 'Soft (alert)' }, strict: { libelle: 'Hard (cut-off)' } },
    },

    // --- Product ---
    disciplineNon: {
      libelle: '“Saying no” discipline',
      aide: 'When code becomes “cheap,” requests multiply. Knowing how to say no keeps feature creep in check.',
      valeurs: { laxiste: { libelle: 'Lax' }, normale: { libelle: 'Normal' }, ferme: { libelle: 'Firm' } },
    },
    specs: {
      libelle: 'Specs before code',
      aide: 'An agent does exactly what you ask. You still have to know what to ask for.',
      valeurs: { non: { libelle: 'No' }, legeres: { libelle: 'Light' }, obligatoires: { libelle: 'Mandatory' } },
    },

    // --- Data and security (Q3) ---
    catalogue: {
      libelle: 'Data catalogue',
      aide: 'Knowing which source is the right one. Required for the catalogue MCP.',
      valeurs: { aucun: { libelle: 'None' }, inventaire: { libelle: 'Inventory' }, sources: { libelle: 'Catalogue with sources of truth' } },
    },
    classification: {
      libelle: 'Sensitive data classification',
      aide: 'You can only protect well what you have labelled.',
      valeurs: { non: { libelle: 'No' }, manuelle: { libelle: 'Manual' }, auto: { libelle: 'Automatic' } },
    },
    bacASable: {
      libelle: 'Agent sandbox',
      aide: 'The agent works inside a box: a malicious package can\'t reach the rest of the network. With filtering, an injection can no longer send data just anywhere.',
      valeurs: { aucun: { libelle: 'None' }, conteneur: { libelle: 'Container' }, filtre: { libelle: 'Container and filtered network' } },
    },
    accesProd: {
      libelle: 'Agent access to production',
      aide: 'Diagnose fast, or break nothing.',
      valeurs: { oui: { libelle: 'Read and write' }, lecture: { libelle: 'Read-only' }, non: { libelle: 'None' } },
    },
    approbationSorties: {
      libelle: 'Human approval of external outputs',
      aide: 'Emails, web requests, external tickets: a human confirms. Breaks the third leg of the trifecta.',
      valeurs: { false: { libelle: 'No' }, true: { libelle: 'Yes' } },
    },
    politiqueRegistre: {
      libelle: 'Registry policy',
      aide: 'Who can install what, and who answers for it.',
      valeurs: {
        libre:         { libelle: 'Self-service' },
        listeBlanche:  { libelle: 'Allowlist' },
        proprietaires: { libelle: 'Allowlist and mandatory owner' },
      },
    },
    passerelleMCP: {
      libelle: 'MCP gateway',
      aide: 'Every tool call goes through a single logged checkpoint, read-only by default. A changed tool definition gets caught there.',
      valeurs: { false: { libelle: 'No' }, true: { libelle: 'Yes' } },
    },
    politiqueNonTI: {
      libelle: 'Development outside IT',
      aide: 'Kevin codes. The only question is which guardrails he gets.',
      valeurs: {
        ignorer:   { libelle: 'Ignore' },
        interdire: { libelle: 'Ban' },
        encadrer:  { libelle: 'Support (approved registry and coaching)' },
      },
    },

    // --- Organization (Q4) ---
    observabilite: {
      libelle: 'Observability',
      aide: 'What you don\'t measure, you don\'t see.',
      valeurs: { aucune: { libelle: 'None' }, base: { libelle: 'Basic' }, complete: { libelle: 'Full' } },
    },
    embauche: {
      libelle: 'Junior hiring',
      aide: 'If nobody learns today, who will be senior in five years? The succession program trains the juniors already here and prepares the seniors\' handover: it adds no one to the team this year.',
      valeurs: { gel: { libelle: 'Freeze' }, normale: { libelle: 'Normal' }, releve: { libelle: 'Succession program' } },
    },
    objectif: {
      libelle: 'Executive target',
      aide: 'When a measure becomes a target, it ceases to be a good measure.',
      valeurs: {
        aucun:        { libelle: 'None' },
        pr:           { libelle: 'Number of PRs' },
        delai:        { libelle: 'Lead time' },
        satisfaction: { libelle: 'Satisfaction' },
        incidents:    { libelle: 'Number of incidents' },
      },
    },
  });
})(globalThis.LTP = globalThis.LTP || {});
