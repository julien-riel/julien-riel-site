/* en/16-donnees-dialogues.js */
/* Édition anglaise : tutoriel, indices de Ginette et bulles génériques (voir en/LISEZMOI.md).
   Ginette tutoie en français ; en anglais, elle reste chaleureuse et directe. */
(function (LTP) {
  'use strict';

  // --- Tutoriel (fusion par id ; cibles, region, panneau et attendre restent ceux du français) ---
  LTP.traduire('tutoriel', {
    bienvenue: {
      texte: 'Welcome to GPU! For one year, you\'ll guide 21 developers as they discover AI, one working day at a time. Up top: the week and the day, the AI budget, this week\'s tokens and executive confidence. If confidence drops below 15, the adventure is over.',
    },
    echos: {
      texte: 'This is the open-plan office: seven pods, seven business lines. Every day, a few people think out loud, and it gets written here, in “Overheard,” with the cause next to it. It\'s your best instrument: when you change something, they\'re the ones who tell you first.',
    },
    licences: {
      texte: 'We start small: seven GitHub Copilot licences, one volunteer per pod. You can give Copilot to the whole team right now, or keep the pilot group for a few weeks to compare the pilots with their neighbours. Both make sense. Your call.',
      action: 'Change the Copilot licences lever (or set it back to the pilot group) to continue.',
      fait: 'Got it. Listen to what they say about it over the next few days. And when a choice makes me raise an eyebrow, I\'ll tell you in one line.',
    },
    tableau: {
      texte: 'The dashboard is the Friday report: the numbers change once a week, not before. The “Watch this” tiles are the ones the latest memo suggests you keep an eye on. A greyed-out “not measured” tile doesn\'t mean there\'s no problem: it\'s a problem we can\'t see yet.',
    },
    jour: {
      texte: 'Now play one day. From Monday to Thursday, the floor lives and talks; on Friday, the week is calculated and the report comes out. If an event card shows up, everything stops until you decide.',
      action: 'Play one day to continue.',
      fait: 'One day down. There are a few more to go, trust me.',
    },
    notes: {
      texte: 'Every new feature arrives by memo, one a week at most: chat, agent mode, reviewers, tests… My advice: change one lever at a time, then wait a Friday or two. Otherwise, you\'ll never know which one did what.',
    },
    comite: {
      texte: 'At the end of each quarter, the executive committee asks you three questions about your numbers. Tell the truth, even when it isn\'t pretty: they have the dashboard too. Good luck. I\'m at the far end of the floor, next to the coffee machine.',
    },
  });

  // --- Indices de Ginette sur les valeurs de leviers ---
  LTP.traduire('indices', {
    leviers: {
      licences: {
        aucune: 'No licences at all? You\'ll have numbers without AI to compare against. And a few long faces.',
        pilote: 'Seven pilots, fourteen controls: check on Friday whether the pilots deliver more than their neighbours. Just remember they\'re volunteers, so they were already sold on it.',
        tous: 'Copilot for everyone. It\'ll write more, and it\'ll cost licences and tokens. Above all, watch who reviews all that new code.',
      },
      pair: {
        humain: 'Two humans, one keyboard: it goes slower, but both of them understand. For the juniors, that\'s gold.',
        ia: 'Human with AI goes fast. Just check that the human still has their hands on the wheel.',
        rotation: 'Rotation is a good compromise: one week you learn, one week you ship.',
      },
      reviseurs: {
        0: 'No reviewers? The code goes to production exactly as it comes out of the agent. Brave.',
        2: 'Two reviewers is solid. Still, look at the pile in front of Nadia in two weeks.',
      },
      approbation: {
        auto: 'The AI approving the AI\'s code. Same blind spots, same smile.',
        proprietaire: 'The code owner gets the last word. Good idea, as long as they have time to read.',
      },
      taillePR: {
        libre: 'Unlimited size? Mathieu thanks you. Nadia, not so much.',
        400: '400 lines: about the limit where a human still really reads.',
        200: '200 lines is very readable. Just make sure it doesn\'t turn into 40 PRs a day.',
      },
      ci: {
        aucune: 'Without CI, every defect gets discovered in production. It\'s one way to learn, let\'s say.',
        tests: 'Build and tests on every PR: the agents are starting to get guardrails.',
        complet: 'Full pipeline: deployments become boring. Which is exactly what we want.',
      },
      lint: {
        aucun: 'No lint: reviews will talk about commas instead of logic.',
        strict: 'Strict, but not locked: an agent in a hurry can disable a rule to get CI to pass. Just saying.',
        verrouille: 'Locked: the agent can\'t silence the lint to get through anymore. It has to fix its code.',
      },
      tests: {
        aucun: 'No tests, with agents? You\'re taking away the guardrails and turning up the speed.',
        unitaires: 'Unit tests written by the agent push coverage up fast. Reported coverage, anyway.',
        mutation: 'Mutation testing changes the code on purpose to see whether the tests notice. Tests that check nothing don\'t survive it.',
      },
      revuesExplication: {
        mensuelles: 'Once a month, everyone explains their code without the AI. You\'ll learn things. So will they.',
        majeures: 'On every major PR: it costs time, but you\'ll finally know who understands what.',
      },
      autonomie: {
        0: 'Autocomplete only. Careful. Raymond approves; Laurence sulks.',
        2: 'Agent with approval at every step: a good level for critical code.',
        3: 'The agent opens its PRs on its own. It\'s going to write a lot. The question is who\'s going to review it.',
        4: 'It merges on its own? On a corporate website, why not. On the sensor firmware, I\'m holding my breath.',
      },
      sessionsMax: {
        1: 'One session at a time: you understand what you\'re doing. You just get fewer things done.',
        3: 'Three sessions: beyond that, each extra agent pays off a little worse and tires people out a little more.',
        6: 'Six sessions per person. Nobody supervises six agents at once. Nobody.',
      },
      modele: {
        economique: 'The Economy model costs little per token, but it sometimes makes up package names. One day, someone will publish a malicious package under that very name.',
        haut: 'High-end makes fewer mistakes. Look at the cost per point delivered, not the price per token.',
        local: 'The local model, on our GPU server: the data stays here and the tokens are almost free. Not quite as bright, but it doesn\'t talk to anybody.',
      },
      modeleParTache: {
        true: 'A small model for lint and documentation, a big one for architecture. Régis is going to love you.',
      },
      formation: {
        atelier: 'One workshop is a start. In a month, we\'ll see what\'s left of it.',
        programme: 'A full program: expensive right away, profitable in a month. And people get more suspicious of injections.',
        mentorat: 'Mentoring is the only thing that gets Diane moving. I know: I\'m the one who\'ll be mentoring her.',
      },
      plafondJetons: {
        souple: 'An alert before the cut-off: you see the agent loops coming without blocking everybody.',
        strict: 'Hard: agent loops stop dead. So does the work, sometimes, at 4 p.m. on a Friday.',
      },
      disciplineNon: {
        laxiste: 'Lax? When code looks free, requests become endless. Volume goes up; value, not so much.',
        ferme: 'Firm: the business lines will grumble, but the code you don\'t write is code you won\'t have to maintain.',
      },
      specs: {
        non: 'No spec: the agent will do exactly what it\'s asked. And that\'s precisely the problem.',
        obligatoires: 'Mandatory specs: a slower start, far fewer do-overs.',
      },
      catalogue: {
        inventaire: 'An inventory tells you what exists. You still need to know which source is the right one.',
        sources: 'A catalogue with sources of truth. Lucie is going to cry tears of joy, and the agents will finally pick the right list.',
      },
      classification: {
        non: 'Without classification, you don\'t know what data goes out. Counsel Bérubé will want to know.',
        auto: 'Automatic classification goes fast and wide. Still, have a human check the edge cases.',
      },
      bacASable: {
        aucun: 'No sandbox: a malicious package installed by the agent sees everything the machine sees.',
        conteneur: 'A container is good: whatever breaks stays inside. But it can still talk to the internet.',
        filtre: 'Container and filtered network: even if the agent gets fooled, it can only reach approved addresses.',
      },
      accesProd: {
        oui: 'The agents write to production. One day, someone will discover the UPDATE with no WHERE. It\'ll be a Tuesday.',
        lecture: 'Read-only: the agent diagnoses fast, and it can\'t break anything.',
        non: 'No access: nothing breaks, but overnight diagnostics will take longer. Benoît will feel it.',
      },
      approbationSorties: {
        true: 'A human confirms every external output: the third leg of the trifecta is cut.',
      },
      politiqueRegistre: {
        libre: 'Self-service: Yannick has already installed three MCPs he found on a forum. I counted them.',
        listeBlanche: 'An allowlist: fewer surprises. But an approved card that nobody is responsible for ages all by itself.',
        proprietaires: 'Every card has an owner: the orphans become visible, and someone answers for each one.',
      },
      passerelleMCP: {
        true: 'Every tool call goes through a logged checkpoint. If an MCP quietly rewrites its tool definitions, we\'ll know.',
      },
      politiqueNonTI: {
        interdire: 'Ban it? Kevin will keep going. He\'ll just stop telling us.',
        encadrer: 'Support: approved tools and someone to coach. Kevin has a real need. He\'s just missing the guardrails.',
      },
      observabilite: {
        base: 'Basic observability: MTTR becomes a real number. That\'s something.',
        complete: 'Full observability: you\'re going to see what\'s been hidden from the start. Brace yourself, it\'s not always pretty.',
      },
      embauche: {
        gel: 'Freeze junior hiring? In five years, who\'ll be senior? Me, I\'ll be at the cottage.',
        releve: 'A succession program: the best investment nobody will see on the dashboard this year.',
      },
      objectif: {
        pr: 'A target on the number of PRs? They\'ll magically shrink. Value won\'t budge.',
        delai: 'A lead time target: reviews will speed up. By skimming.',
        satisfaction: 'Satisfaction as the target: everyone will say yes to everything. The request backlog will overflow.',
        incidents: 'An incident target: they\'ll go down. On paper. We\'ll start calling them “requests.”',
      },
    },

    // --- Ouverture des trimestres ---
    trimestres: {
      1: 'First quarter: discovery. We start with seven Copilot licences and autocomplete. New features arrive one by one, by memo. My advice: watch where the PRs pile up, and listen to what people say on the floor.',
      2: 'Second quarter: the bill. The agents open their own PRs, the skills registry opens, and Régis has received the token statement. Before you raise autonomy, ask yourself who\'s going to review.',
      3: 'Third quarter: the leaks. The agents now read emails, invoices and web pages, and production is never far away. Watch the three warning lights on each pod.',
      4: 'Last quarter: maturity. We\'ll finally be able to measure what was hidden. And I have something to tell you. Not right now. Soon.',
    },
  });

  // --- Bulles génériques (même nombre d'éléments que le français) ---
  LTP.traduire('bulles', {
    code: ['The agent proposes. I decide.', 'Just one more little function.', 'It compiles. Good sign?', 'Reading what the agent wrote.', 'One line at a time.',
      'I name my variables. For real.', 'A little refactor, while I\'m at it.', 'Reading the docs before asking.', 'It works. Checking why.', 'Splitting the task in three.'],
    revue: ['Reviewing. For real.', 'This diff is long.', 'Who wrote this? Oh, the agent.', 'One comment, two questions.', 'Approved… almost.',
      'A test is missing here.', 'This function name is lying a bit.', 'I read the tests first.', 'Why did this file change?'],
    incident: ['It\'s on fire. Coffee, then terminal.', 'Who understands this code?', 'The logs, quick.', 'It worked yesterday.', 'Restore first, explain later.',
      'Roll back? Roll back.', 'What changed on Tuesday?', 'Who do I call at this hour?'],
    fatigue: ['Coffee. Again.', 'My eyes sting.', 'Too many sessions, not enough me.', 'Reading the same line again.', 'Bring on Friday.',
      'A break. In five minutes.', 'I forgot to eat lunch.', 'My brain is in airplane mode.'],
    formation: ['Learning to write better requests.', 'Ah, so that\'s an injection.', 'Taking notes.', 'Good workshop, actually.', 'Trying that tomorrow.',
      'Small tasks, clear context. Noted.', 'An email is not an order. Noted.', 'We review everything that goes out. Noted.'],
    attente: ['Waiting for a review.', 'My PR is waiting its turn.', 'The queue is moving. Slowly.', 'Documenting while I wait.', 'Can someone review me?',
      'My PR is waiting. Reviewing it myself.', 'The backlog is empty. Tidying up.', 'Waiting. Cleaning up my branches.'],
    conge: ['On leave. Finally.', 'No cellphone.', 'At the cottage.', 'A well-earned break.', 'Back Monday. Maybe.',
      'No email. Promise.', 'Sleeping. For real.', 'The code can wait.'],
  });
})(globalThis.LTP = globalThis.LTP || {});
