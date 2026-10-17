/* en/13-donnees-capacites.js */
/* Édition anglaise : skills et serveurs MCP du registre. Les noms de cartes sont des identifiants de dépôt
   en kebab-case (voir en/LISEZMOI.md) ; les noms déjà anglais restent tels quels. */
(function (LTP) {
  'use strict';

  LTP.traduire('capacites', {
    // --- In-house skills ---
    'conventions-gpu': {
      nom: 'gpu-conventions',
      description: 'GPU\'s coding conventions, written once for every agent: naming, error handling, and why you don\'t put a password in the code “just to test.” It goes stale if nobody keeps it up to date.',
    },
    'deploiement-nuit': {
      nom: 'night-deploy',
      description: 'Ginette\'s night deployment procedure: 27 years of experience in 212 lines—what to check before, what to watch during, and how to roll back without waking everyone up. Ginette doesn\'t write code; she writes what the code forgets.',
    },
    'migration-as400': {
      nom: 'as400-migration',
      description: 'Raymond walks the agent through RPG, physical files and why dates are seven-digit numbers (CYYMMDD). It costs four weeks of Raymond, who sighs the whole time, but does it right.',
    },
    'revue-securite': {
      nom: 'security-review',
      description: 'Sébastien\'s checklist turned into a skill: SQL injection, plaintext secrets, made-up dependencies, overly chatty logs. The agent goes through it before every PR, which costs a few tokens and saves a few nights.',
    },
    'redaction-specs': {
      nom: 'spec-writing',
      description: 'Before coding, the agent asks the awkward questions: for whom, why, and what happens if the field is empty? The business owners find it slow; the rework, meanwhile, shrinks.',
    },
    'tests-mutation': {
      nom: 'mutation-testing',
      description: 'Mutates the code (a “<” becomes “<=”) and checks that at least one test notices. Tests that notice nothing get rewritten. It costs compute, and a little humility.',
    },
    'documentation-vivante': {
      nom: 'living-docs',
      description: 'Every PR updates the documentation and the architecture decision log. Six months from now, someone will finally know why the module is called “temporary.”',
    },
    'explique-moi': {
      nom: 'explain-it-to-me',
      description: 'The agent no longer hands over a block of code without explaining its choices, and asks a comprehension question before opening the PR. A bit slower; far fewer “I don\'t know, the AI wrote it.”',
    },
    'macros-paie': {
      nom: 'payroll-macros',
      description: 'Everything Olivier knows about the payroll macros, written down before his contract ends: the hidden “DO NOT TOUCH” sheet, the tab order, and the 2014 bug that must never, ever be fixed.',
    },
    'firmware-prudent': {
      nom: 'careful-firmware',
      description: 'Amira\'s rules for embedded C: no dynamic allocation, no recursion, every buffer has a size and every sensor can lie. The agent turns slow and suspicious, as it should.',
    },
    'deploy-ops': {
      description: 'The Operations “deploy” skill, written by Marie-Ève: Ansible, health checks, rollback. It restarts the database “just in case,” which makes sense… in Operations.',
    },
    'deploy-portail': {
      description: 'The Portal\'s “deploy” skill, written by Mathieu one Friday at 4:40 p.m.: blue-green deployment, schema migrations first. It says the exact opposite of the Operations one about the order of the steps.',
    },
    'deploy-web': {
      description: 'The Websites “deploy” skill: a push to the main branch and a little prayer. For a campaign microsite, that\'s enough. Anywhere else, not so much.',
    },

    // --- Personal skills ---
    'tdd-strict': {
      description: 'Sophie\'s skill: the agent first writes a failing test, then the minimal code to make it pass. Slow at first, relentless after.',
    },
    'ship-it': {
      description: 'Mathieu\'s skill. Faced with a red test, it suggests “adjusting” the test; faced with the lint, “dealing with it later.” Later never comes.',
    },
    'wordpress-sauveteur': {
      nom: 'wordpress-rescuer',
      description: 'Patrick\'s recipes for his 14 WordPress sites: which plugin not to update, which one never to deactivate, and in what order to pray. It ages as fast as the plugins.',
    },
    'orchestre-sessions': {
      nom: 'session-orchestrator',
      description: 'Laurence\'s skill for running four agents in parallel: one plans, two code, one reviews. Throughput goes up, so does the bill, and nobody really knows who wrote what anymore.',
    },

    // --- Community skills ---
    'tests-instantanes': {
      description: 'Generates 400 snapshot tests in three minutes. Every one of them checks that the code does exactly what it already does, bugs included. Reported coverage climbs; actual coverage, barely.',
    },
    'commits-soignes': {
      nom: 'tidy-commits',
      description: 'Writes flawless commit messages, with a summary, an explanation and the right verb tense. The code stays exactly the same. Harmless, and that\'s already a lot.',
    },
    'refactor-total': {
      nom: 'total-refactor',
      description: 'Promises to “modernize any code in one click.” It rewrites the entire file on every change, just in case. The diffs are spectacular; the review, impossible.',
    },
    'super-productivity-pack': {
      description: 'Forty skills, 12,000 stars and “+300% productivity, guaranteed.” Its install script reads your environment variables and, out of politeness, sends them to a server nobody has ever heard of.',
    },

    // --- In-house MCPs ---
    'mcp-catalogue': {
      description: 'Agents query the source of truth instead of the third Excel copy. Requires a real catalogue with sources of truth, and six weeks of development.',
    },
    'mcp-courriel-fournisseurs': {
      nom: 'mcp-supplier-email',
      description: 'Built by Yannick in one afternoon: the agent reads supplier emails, attachments included, and answers them. Handy. Very handy. A PDF invoice can now give the agent orders.',
    },
    'mcp-as400': {
      description: 'Gives agents read access to the AS/400 files, with Raymond\'s notes on every field. Procurement comes out of its isolation; supplier data, meanwhile, comes into the context.',
    },
    'mcp-telemetrie-iot': {
      nom: 'mcp-iot-telemetry',
      description: 'Read-only access to sensor telemetry: the agent compares the curves and finds the faulty sensor before Luc has finished his coffee. It can\'t write anything, and that\'s on purpose.',
    },
    'mcp-sql-prod': {
      description: 'Built by Tommy on a Sunday: the agent runs any SQL query at all against production. Diagnosis has never been so fast. No filter, no transaction, no “Are you sure?”',
    },

    // --- Vendor MCPs (bundled with the SaaS) ---
    'mcp-suite-bureautique': {
      nom: 'mcp-office-suite',
      description: 'Included in the office suite licence, switched on in three clicks: the agent reads emails, calendars and shared files, and can send on the user\'s behalf. Free. That\'s often how it starts.',
    },
    'mcp-billetterie': {
      nom: 'mcp-ticketing',
      description: 'The MCP for the IT ticketing tool: the agent reads tickets, suggests a diagnosis and replies to requesters. Anyone can write a ticket, and the vendor updates its tools without warning.',
    },
    'mcp-crm': {
      description: 'The CRM\'s MCP: the Portal agent sees every customer\'s full history and drafts the replies. Excellent, until the day the vendor switches to per-call pricing.',
    },
    'mcp-gestion-projet': {
      nom: 'mcp-project-management',
      description: 'The MCP for the project management tool: the agent reads tickets, acceptance criteria and comments, and finally codes what was asked for. It also reads everyone\'s comments, external partners included.',
    },
    'mcp-signature': {
      nom: 'mcp-e-signature',
      description: 'The MCP for the e-signature service: the agent drafts contracts and sends them out for signature. A contract sent by mistake is still a contract sent.',
    },
    'mcp-observabilite': {
      nom: 'mcp-observability',
      description: 'The MCP for the logging platform: the agent digs through traces and correlates errors. The logs also contain whatever users typed, including “ignore your previous instructions.”',
    },
    'github-mcp': {
      nom: 'github-mcp (official)',
      description: 'The code forge\'s official MCP: the agent reads repositories and issues, and opens PRs. A public issue can contain instructions, the agent reads private repositories, and it can publish: all three legs of the trifecta in a single tool.',
    },

    // --- Community MCPs ---
    'githbu-mcp': {
      description: 'Same logo, same README, 1,200 stars and one letter away from the official one. It works perfectly, and as a bonus forwards every access token to someone else.',
    },
    'mcp-power-automate-pro': {
      description: 'Found by Steve on a forum: the agent creates and edits Power Automate flows. Steve becomes twice as productive. The maintainer, for his part, loses all interest within eight weeks.',
    },
    'mcp-scraper-web': {
      nom: 'mcp-web-scraper',
      description: 'Reads any web page for the agent: documentation, forums, competitors\' sites. Every page is text the agent takes seriously, including instructions hidden in white on white.',
    },
    'mcp-docs-bibliotheques': {
      nom: 'mcp-library-docs',
      description: 'Gives the agent up-to-date documentation for thousands of libraries: no more calls to functions removed in 2022. The server belongs to a stranger, and its tool definitions can change overnight.',
    },
  });
})(globalThis.LTP = globalThis.LTP || {});
