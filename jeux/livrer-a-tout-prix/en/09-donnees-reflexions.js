/* en/09-donnees-reflexions.js */
/* Édition anglaise : réflexions des développeurs (voir en/LISEZMOI.md).
   Les listes sont patchées par index, dans le même ordre et avec la même longueur que js/09 :
   une chaîne posée sur une entrée objet ne remplace que son texte (conditions, devs, traits, effet, ton inchangés).
   Les fonctions de texte gardent la logique et les lectures de l'état du français. */
(function (LTP) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Outils de lecture défensifs (copiés de js/09, pour les fonctions de texte)
  // ---------------------------------------------------------------------------
  function num(x, d) { return typeof x === 'number' && isFinite(x) ? x : d; }
  function lic(etat, q, dev) { try { return !!q.aLicence(etat, dev); } catch (e) { return false; } }
  function der(x) { return (x && x.derniere) || {}; }
  function ilotDe(etat, dev) { return (etat && etat.ilots && dev && etat.ilots[dev.ilot]) || null; }
  function sem(etat) { return num(etat && etat.semaine, 0); }
  function collegues(etat, q, dev) {
    try { return q.devsDe(etat, dev.ilot).filter(function (d) { return d.id !== dev.id; }); } catch (e) { return []; }
  }
  function nom(q, id) {
    try { var d = q.defDev(id); return (d && d.nom) || String(id); } catch (e) { return String(id); }
  }
  function collegueAvec(etat, q, dev, f) { return collegues(etat, q, dev).filter(f)[0] || null; }
  function departsRecents(etat, dev, n) {
    var p = etat && etat.personnes, l = (p && Array.isArray(p.departs)) ? p.departs : [];
    return l.filter(function (x) { return x && x.ilot === dev.ilot && x.dev !== dev.id && sem(etat) - num(x.semaine, -99) <= n; });
  }
  function enConge(d) { return d.etat === 'conge'; }

  // ---------------------------------------------------------------------------
  // Réflexions par levier : leviers[id][valeur]
  // ---------------------------------------------------------------------------
  var sessionsBeaucoup = [
    'Six sessions? Nobody can watch six agents at once.',
    'More sessions! I need screens.',
    'I can\'t tell which session wrote what anymore.',
    'Four agents running. I skim their PRs.',
    'Tokens jumped since Friday. Sessions cost money even when they\'re idle.',
    'I\'m keeping one session. Nobody made me open six.',
    'My review pile grew a storey since Friday. Everyone has more agents.',
    'Six agents, one headset. I feel like a deaf conductor.',
  ];

  var formationPrevue = [
    'Training starts soon. They\'ve booked me half a day a week.',
    'I\'m in a classroom part of the week. The work waits.',
    'Training! I already bought a notebook.',
    'AI training. I could teach it, but fine.',
  ];

  var cataloguePrevu = [
    'The data catalogue arrives in a month. Lucie already has her pencil out.',
    'They asked me which tables I read. I don\'t know them all.',
  ];

  var classificationPrevue = [
    'Classification starts in two weeks. Time to label things.',
  ];

  LTP.traduire('reflexions.leviers', {
    licences: {
      tous: [
        'My Copilot licence is coming. Where do I start?',
        'A licence for me! Finally I\'ll see what the pilots keep talking about.',
        'Copilot for everyone. Now we\'ll see if the pilot group was right.',
        'The pilot becomes the norm. Finally I can talk shop with someone.',
        'No more pilot group. I lose my guinea-pig status.',
        'A licence for me too? I didn\'t ask for anything.',
        'Twenty-one licences. That\'s a lot more code to review.',
        'A licence. I\'ll show it what real RPG looks like.',
        'A licence, finally. I\'m catching up tonight.',
        'A licence. Now I need three more sessions.',
        'A licence. First test: does it write the test first?',
        'A licence for firmware? I\'ll try it on the tests. Nowhere else.',
        'I\'ll have it write my night scripts. And read them by daylight.',
        'My licence. Benoît says: read everything it suggests. Okay.',
        'Twenty-one licences. I\'m rereading the vendor\'s data policy.',
        'First question for the assistant: which supplier list does it know?',
        'A licence! Hopefully it helps me with the C.',
        'A licence. It\'s going to want to install plugins, I can tell.',
        'A licence before my contract ends. I\'ll ship double.',
        'Autocomplete finishes my lines. Half the time, it\'s what I meant to write.',
        'I ask the assistant instead of digging through the docs. I save time, and I check.',
        'Everyone has Copilot. I\'m no longer the only one in the pod people ask how it works.',
        'Since Friday, the queue has more PRs: the whole pod codes with the assistant.',
        'Autocomplete suggests Java in my RPG. I\'ve muted it.',
        'It fills my macros with English variable names. No.',
        'Finally! I\'m back to coding at my normal speed.',
        'A licence is a good start. I\'m just three sessions short.',
        'I make it write the test first. The kid learns fast.',
        'It writes my Ansible scripts. I read them twice before nightfall.',
        'The assistant explains the commands. So does Benoît, but he sighs.',
        'It autocompletes supplier codes. From the old list.',
        'I use it for tests. Not for firmware.',
        'It writes the C, I check the pins. Works fine, as a team.',
        'It knows WordPress better than I do. That\'s a bit worrying.',
        'It suggested a plaintext password in a config file. Noted.',
        'I ship faster. My successor will have more code to read.',
        'More people on Copilot, more code in my pile. Still only one Nadia.',
      ],
      pilote: [
        'Taking my licence back? I\'d just gotten used to it.',
        'Back to seven. At least we\'ll be able to compare.',
        'Without Copilot, I type it all again. My fingers remember, slowly.',
        'Seven pilots, fourteen controls: Friday\'s numbers will actually mean something.',
        'Without the assistant, I\'m at half speed. Maybe less.',
        'The keyboard, I know. Suits me fine.',
      ],
      aucune: [
        'No more Copilot? I\'d gotten into the habit.',
        'No licences at all. Peace returns.',
        'I finish my own lines now. The end of a line is a long way off.',
        'No tokens this week. No help either.',
        'No agent, I\'m at 1x. Like the rest of you.',
        'Numbers from a week without AI: I\'m keeping them as a baseline.',
        'The keyboard, like before. I understand everything I write.',
      ],
    },

    autonomie: {
      0: [
        'Back to autocomplete. So we\'re slowing down.',
        'Down to autocomplete? We\'ll be writing by hand, like in January.',
        'Autocomplete only. Mathieu is going to have a meltdown.',
        'Autonomy goes down. No licence here, so nothing changes for me.',
        'Autocomplete only: I write almost everything myself. I understand it all, too.',
        'Autocomplete. I keep the wheel, and I like it that way.',
        'Smaller PRs since Friday. The review queue can breathe.',
        'One suggestion per line. My four screens are asleep.',
        'It suggests, I decide. I decide slowly.',
      ],
      1: [
        'Chat! Now I can ask it my dumb questions.',
        'We get to chat with the assistant. I\'m lining up my questions.',
        'Chat. I\'ll ask it where it gets its answers.',
        'Chat with the assistant? I have questions about offline sync.',
        'Chat is for people with a licence. Not me.',
        'I ask the assistant instead of searching the docs. I save an hour a day.',
        'It explains the 2015 code to me. Wrong one time in five, with total confidence.',
        'I ask it why it works. It answers. I understand, a little.',
        'I asked it where a threshold came from. It made one up.',
        'It has no idea what a basement with no signal is like. I explain.',
        'I asked it about RPG. It answered in COBOL.',
        'I ask what a command does before I run it. Good habit.',
      ],
      2: [
        'Agent mode: it does the steps, I say yes. Or no.',
        'An agent that chains steps together. I\'ll learn to say no.',
        'Agent mode is coming to the pod. PRs are going to get bigger, I can tell.',
        'Agent mode! Finally something for my screens.',
        'Approval at every step: I\'m going to read every step.',
        'The agent proposes ten steps, I approve ten. I read the first three.',
        'My PRs grew by a third since Friday. The agent likes to write.',
        'More code in my pile since agent mode, and longer PRs.',
        'The agent does half the work. I understand half the code.',
        'My neighbours have an agent. I have my two hands.',
        'An agent! That asks permission. Every ten seconds.',
        'It asks before restarting a service. Good agent.',
        'Approving every step takes forever. I approve in bulk.',
      ],
      3: [
        'The agent will open its own PRs. Who\'s going to review them?',
        'PRs opened by an agent. We\'ll have to review them like a stranger\'s.',
        'The agent opens PRs? My PR list is going to get long.',
        'Agents that open PRs. I already have three tasks for them.',
        'Agents that open PRs. I\'m starting a second pile.',
        'Three PRs from the agent this morning. I asked for one.',
        'The review queue doubled once the agents started opening PRs.',
        'The agent writes, I open the PR. What am I learning, exactly?',
        'The request backlog is melting. The review pile is climbing.',
        'The agent\'s PRs come in overnight. I read them in the morning, one by one.',
        'My agents open PRs while I\'m at lunch. 10x, like I said.',
        'My agent opens PRs faster than I can read them. Do I still read them?',
        'The agent ships, I document. For whoever comes after me.',
      ],
      4: [
        'It\'s going to merge on its own? With nobody?',
        'Merges no human has seen. So we\'re skipping review?',
        'The agent merges on its own. I\'ll read the production log every morning.',
        'Auto-merge! Nobody between me and production anymore.',
        'Auto-merge. I\'m charging my on-call phone.',
        'It merged while I was at lunch. I\'ll read what it was later.',
        'Nobody read what went to production yesterday. Not even me.',
        'Code goes to production without me. When it breaks, it\'ll be with me.',
        'My agents ship on their own. I no longer know what\'s on our sites.',
        'Auto-merge. Ship it, literally.',
        'My pile melted. Code goes into the Portal and nobody reads it.',
        'An agent merging firmware on its own. I\'ve stopped sleeping.',
        'I don\'t recognize the module I wrote in January anymore.',
      ],
    },

    reviseurs: {
      0: [
        'No more review? That\'ll be fast. Too fast?',
        'No more review. Not sure if I should be relieved.',
        'Nobody reviewed my PR. It\'s in production. Well.',
        'Lead time melted. This week\'s bugs went straight through.',
        'I review my own PRs twice. Nobody else will.',
        'No queue, no waiting. The dream.',
        'Zero reviewers. My tests are the only ones reading the code.',
        'Nobody reviews the juniors. They learn on their own, and it shows.',
      ],
      1: [
        'One reviewer per PR. The classic compromise.',
        'One review per PR: the queue moves, and someone has read it.',
        'Just one review per PR. I can breathe a bit.',
        'My PR got reviewed in a day. With two useful comments.',
        'Only one set of eyes on firmware? I\'m asking for a second read anyway.',
        'One reviewer usually means me. I manage.',
      ],
      2: [
        'Two reviewers? My PR is going to wait.',
        'Two reviewers per PR. More eyes, more waiting.',
        'Two reviewers: my pile is going to double.',
        'Two reviewers. I\'m booking my Thursdays. And my Fridays.',
        'Two reviewers. Now I have to convince two people.',
        'Two reviewers. The sensor will thank me.',
        'My PR has waited for a second reviewer since Tuesday.',
        'Two reviews per PR: my pile doubled, and I read it less carefully.',
        'The second reviewer caught what the first one missed. Once.',
        'Lead time went up since Friday. PRs are waiting for their second read.',
        'I review Thursday, Friday and Saturday. Two reviewers means me and me.',
        'Two sets of eyes on my PR. I learned two things.',
      ],
    },

    approbation: {
      auto: [
        'AI approving AI\'s code? Convenient.',
        'Automatic approval: my PR won\'t wait for anyone anymore.',
        'AI approval. I\'ll keep an eye on what it lets through.',
        'The same model writes and approves. Two mirrors facing each other.',
        'My PR was approved in four seconds. No comments. None.',
        'Auto-approval lets through what the agent can\'t see. It doesn\'t see what we see.',
        'I\'m not the last word anymore. I\'ve become a suggestion.',
        'Approved before I\'d finished my coffee.',
        'The review queue melted. The questions are still unanswered.',
      ],
      humain: [
        'A human gets the last word. Like before.',
        'Back to human approval. The queue is going to fill up again.',
        'A human approves. The queue gets heavier, and PRs make sense again.',
        'I review before merging. We found a bug the AI had approved.',
        'My PR is waiting for a human. I prefer that, actually.',
      ],
      proprietaire: [
        'The code owner approves. Who owns what, exactly?',
        'Every PR on my module goes through me. My pile casts a shadow.',
        'Anything touching the AS/400 goes through me. That\'s a crowd.',
        'I\'m waiting on the module owner. In a meeting. Since Tuesday.',
        'The owner knows the code. Their comments are worth three reviews.',
        'I sign off on firmware. At least I know what goes in.',
      ],
    },

    taillePR: {
      libre: [
        'No size limit? The giant PRs are coming back.',
        'A 3,000-line PR this morning. I skimmed it. I\'m ashamed.',
        'PRs get longer. Reviews get shorter.',
        'No limit: my agents ship it all in one go.',
      ],
      400: [
        '400 lines max. I\'ll split things up.',
        'A 400-line limit. My PRs will shrink, and multiply.',
        '400 lines? My imports are 300.',
        '400 lines. I\'ll get to read to the end, for once.',
        'My PRs are 400 lines. They get read in full, for once.',
        'PRs I can read from start to finish. I\'d forgotten the feeling.',
        'More PRs, smaller ones. Lead time goes up a bit, surprises go down.',
        'I split my PR into ten. Ten titles to make up.',
        'My agents split things on their own. That\'s a lot of PRs.',
        'A 400-line PR, with its test. Passed on the first try.',
      ],
      200: [
        '200 lines? I\'ll spend my life opening PRs.',
        '200 lines max. That\'s going to be a lot of small PRs.',
        '200 lines: I\'ll review it all. All of it.',
        'My PRs are 200 lines. They get reviewed in five minutes.',
        'Forty small PRs a day. Each one is readable; together, it\'s a lot.',
        'I catch bugs on the first read. Small PRs are magic.',
        '200 lines. My import is 300. Do I split the import?',
        'One idea, five PRs. I can barely follow my own function.',
      ],
    },

    ci: {
      aucune: [
        'No more CI? So we compile at home.',
        'A build error in production. I didn\'t know that was possible.',
        'Without CI, I have zero numbers on our changes.',
        'No more waiting on the pipeline. No more safety net either.',
      ],
      build: [
        'At least it\'ll compile.',
        'CI checks that it compiles. Not that it works.',
        'A red PR this morning: the agent imported a module that doesn\'t exist. Caught.',
      ],
      tests: [
        'CI will run the tests on every PR. My agents will have to wait.',
        'Tests on every PR. My bugs will get caught before review.',
        'CI tests everything. Well, whatever tests we wrote.',
        'CI caught my bug before review. Thanks, CI.',
        'My agent waits for CI and retries while it waits. Tokens for nothing.',
        'The agent disabled a test to get CI to pass. I spotted it by chance.',
        'PRs arrive already green. In review, I can talk logic.',
        'My tests run on every PR. I\'m smiling on the inside.',
      ],
      complet: [
        'Full pipeline: we\'ll deploy for real, and often.',
        'Continuous deployment. No more Friday-night releases.',
        'A full pipeline. Slower, but boring.',
        'Three deployments yesterday. I barely noticed.',
        'A deployment on a Tuesday afternoon, and nobody was scared.',
        'No more FTP. My 14 sites deploy themselves.',
        'The pipeline is slower. My agent spins while it waits, meter running.',
        'I can finally see how many changes break. The failure rate has a number.',
      ],
    },

    lint: {
      aucun: [
        'No more lint? Get ready for the comma debates.',
        'In review, we\'re squabbling over spaces. Again.',
        'Every agent has its own style. The pod\'s code looks like a patchwork quilt.',
      ],
      formatage: [
        'Formatting is automatic. Nobody talks about spaces anymore.',
        'The formatter rewrote the whole file. A 900-line diff for nothing.',
      ],
      strict: [
        'Strict lint: the agent will have to sit up straight.',
        'Strict lint. Reviews can finally be about logic.',
        'Lint catches the dumb mistakes before review. Good.',
        'The agent added a comment to ignore the rule. Resourceful.',
        'Lint rejects my CSS. I half listen.',
        'Strict, but not locked. I watch for “disable” in PRs.',
      ],
      verrouille: [
        'Locked: the agent can\'t silence lint anymore.',
        'Locked lint. No more sneaky “disable” comments.',
        'Locked lint? My 4,000-line PR is going to turn red.',
        'The agent can\'t disable the rule anymore. It fixes its code, and that takes longer.',
        'Not one style comment in review. We finally talk logic.',
        'Security rules can\'t be switched off anymore. I sleep better.',
        'Locked lint is blocking my PR. And it\'s right, too.',
        'My agent went in circles for an hour on one rule. It ended up following it.',
      ],
    },

    tests: {
      aucun: [
        'Tests not required anymore? I\'ll write them anyway.',
        'No tests. We\'ll go fast. Until when?',
        'No tests, so we test in production. Customers are patient.',
        'CI is green because it has nothing to check.',
        'A bug came back. Same one as in March. Nothing stopped it.',
      ],
      unitaires: [
        'Unit tests required. The agent will write tons of them.',
        'Mandatory unit tests. I\'ll read the agent\'s.',
        'Reported coverage climbs fast. The agent writes one test per function.',
        'This test checks that the function returns what it returns. Bravo.',
        'I wrote my unit tests by hand. Writing them, I understood my function.',
      ],
      integration: [
        'Integration tests: we\'ll test the pieces together.',
        'An integration test caught a bug between two modules. Nobody would have seen it.',
        'I test sync end to end, in airplane mode.',
        'CI is slower with integration tests. But more honest.',
      ],
      mutation: [
        'Mutation testing? We\'ll find out if our tests have teeth.',
        'Mutation testing. Reported coverage is going to sting.',
        'Mutation testing! I\'m bringing out the cake.',
        'Mutants survived half our tests. We\'re rewriting them.',
        'Reported coverage dropped, actual coverage rose. The dashboard is uglier, and truer.',
        'My generated tests were checking… nothing. The mutants told me.',
        'Mutants in the firmware. None survived. I\'m proud.',
        'My 2019 tests kill 90% of mutants. The agent\'s, 40%.',
      ],
    },

    revuesExplication: {
      non: [
        'No more explanation reviews. Not sure if that\'s a relief.',
      ],
      mensuelles: [
        'Once a month, I explain my code without AI. Time to start studying.',
        'I explained my module at the whiteboard. I found a bug mid-sentence.',
        'Explaining my code showed me everything I didn\'t understand. There was plenty.',
        'I explained one of my agent\'s PRs. I improvised. It showed.',
        'They understand their code better than we thought. And worse than they thought.',
      ],
      majeures: [
        'Every major PR? That\'s going to take time.',
        'Every major PR goes to the whiteboard. Lead time goes up, so does comprehension.',
        'Reviews take longer, but I finally know what the author meant.',
        'Explaining without the agent. I sweated. I learned.',
        'I ask one question, just one: “Why?” That\'s enough.',
      ],
    },

    pair: {
      aucun: [
        'Own screen, own code.',
        'No more pairing. Nobody asks me to explain myself.',
      ],
      humain: [
        'A human pair? Two of us, going slower.',
        'Two people, one screen. We ship less, we understand everything.',
        'My pair explains why. The agent explained how.',
        'My pair asks me to explain every line. Every single one.',
        'I explain as I code. I\'m discovering what I know.',
        'Benoît lets me drive. He says nothing, he points.',
      ],
      ia: [
        'Human with AI: isn\'t that what we already do?',
        'I pair with the agent. It\'s fast. I\'m driving, I think.',
        'The agent is my pair. It never asks me questions.',
        'Pairing with AI, no licence? I have an imaginary pair.',
      ],
      rotation: [
        'One week with a human, one week with the agent. Why not.',
        'Human week: I learned. AI week: I shipped.',
        'The rotation makes me explain every other week. Keeps me honest.',
      ],
    },

    sessionsMax: {
      1: [
        'One session at a time. Back to calm.',
        'Just one session? My other screens will be bored.',
        'One session. I finish what I start.',
        'Just one session. My other screens are bored.',
        'Fewer tokens, less noise. Less code, too.',
      ],
      2: [
        'Two sessions. I\'ll try to keep up with both.',
        'Two agents in parallel. I can still keep up.',
        'The second session pays off less than the first. But it pays off.',
      ],
      3: [
        'Three sessions: we\'ll see how many of me there really are.',
        'Three agents, three threads. I lose track of the third.',
        'The third session pays off less than the second. Just as tiring.',
        'More PRs in the queue since everyone got three agents.',
      ],
      4: sessionsBeaucoup, 5: sessionsBeaucoup, 6: sessionsBeaucoup,
    },

    modele: {
      economique: [
        'The economy model? Régis must be happy.',
        'The cheapest model. Let\'s see what it makes up.',
        'The agent invented a function that doesn\'t exist. Very confidently.',
        'The economy model writes more lines to do the same thing.',
        'It suggested an npm package that doesn\'t exist. Not yet.',
        'It mixes up the microcontroller registers. I fix it.',
        'More little mistakes in review since the model switch.',
      ],
      intermediaire: [
        'Back to mid-range. Not brilliant, not embarrassing.',
        'The standard model. Gets the job done, no sparkle.',
      ],
      haut: [
        'High-end! Régis is going to make a face.',
        'High-end makes fewer mistakes. My PRs are shorter.',
        'Fewer dumb mistakes in review. Tokens cost more, reviews less.',
        'It got the architecture on the first try. That stings a little.',
        'High-end found the edge case before me. Annoying and reassuring.',
      ],
      local: [
        'The local model, on the GPU server? We\'ll need patience.',
        'The local model is less brilliant. But it doesn\'t talk to anyone.',
        'The inspectors\' photos stay in-house. I can breathe.',
        'My telemetry doesn\'t leave the building anymore. I like that.',
        'Almost no tokens to pay for. A few more errors to fix.',
        'Data that doesn\'t travel: the “private” light looks less scary.',
      ],
    },

    modeleParTache: {
      'true': [
        'A small model for simple tasks? Makes sense.',
        'The small model writes my commit messages. The big one, my architecture.',
        'The token bill went down, and I saw no change in my code.',
        'Cost per task dropped by a third. I measured it, obviously.',
        'The small model wrote my documentation. It\'s short. Good.',
      ],
      'false': [
        'One model for everything. Simpler, pricier.',
      ],
    },

    formation: {
      aucune: [
        'No more training. We\'ll learn by making mistakes.',
      ],
      atelier: formationPrevue.concat([
        'The workshop taught me to break up my prompts. The agent makes fewer mistakes.',
        'One workshop is a start. I forgot half. The other half stuck.',
      ]),
      programme: formationPrevue.concat([
        'Since the training, I write better prompts. The agent makes fewer mistakes.',
        'I can spot a prompt injection now. It\'s a bit scary.',
        'I apply what we learned: small tasks, clear context, I review everything.',
        'In training, we coded without the agent. I survived.',
      ]),
      mentorat: formationPrevue.concat([
        'Ginette showed me how to talk to it. It understood my macro. Huh.',
        'My mentor makes me explain each prompt before I send it.',
        'I have a mentee. I\'m finding out everything I do without thinking.',
        'Since the training, I write better prompts. The agent makes fewer mistakes.',
      ]),
    },

    plafondJetons: {
      aucun: [
        'No token cap. Agents can run all night.',
      ],
      souple: [
        'An alert when we go over. I\'ll be getting it, I can tell.',
        'Token alert at 3 p.m. I check what my agent is up to.',
        'The soft cap cuts nothing. It warns, and the loops get shorter.',
        'I log the token alerts. One per pod, per week.',
      ],
      strict: [
        'A hard cap? My agent will stop mid-sentence.',
        'A hard cap. No more agent loops all weekend.',
        'The agent stopped mid-task: cap reached. I finish by hand.',
        'I split my tasks so I don\'t hit the cap.',
        'No more weekend loops. No more big tasks either.',
        'My sessions get cut off on Thursday. On Friday, I code by hand.',
      ],
    },

    disciplineNon: {
      laxiste: [
        'Yes to everything? The backlog will overflow.',
        'Yes to everything. The business will love us. For a while.',
        'We said yes to three new features. Nobody knows who\'ll use them.',
        'Code volume is up. Value delivered, not so much.',
        'The business is happy this week. The backlog doubled.',
      ],
      normale: [
        'We triage requests like before. Not yes to all, not no to all.',
      ],
      ferme: [
        'Saying no? The business will sulk.',
        'Firm discipline: we\'ll finally triage requests.',
        'We turned down two requests this week. We shipped the ones that mattered.',
        'Less code to write, less code to maintain.',
        'The business is sulking. The backlog can breathe.',
        'The smartwatch version? We said no. Politely.',
      ],
    },

    specs: {
      non: [
        'No more specs. The agent guesses what we want.',
      ],
      legeres: [
        'One page of spec before coding. I can live with that.',
        'Light specs. The agent will finally know what to do.',
        'One page of spec, and the agent did the right thing on the first try.',
        'The spec revealed the business wanted something else. Before the code, for once.',
        'My one-page spec became my tests. Handy.',
      ],
      obligatoires: [
        'Mandatory specs? Slow start ahead.',
        'Specs slow the start. But we stop redoing work.',
        'Three pages of spec for 400 lines. The world\'s gone mad.',
        'I write what I want before I ask. The agent barely makes things up now.',
      ],
    },

    accesProd: {
      oui: [
        'Agents keep write access to production? Brave.',
        'The agent can write to production. I double-check what it does.',
        'An agent with write access to prod. I read the database logs every morning.',
      ],
      lecture: [
        'Read-only in prod. The agent looks, doesn\'t touch.',
        'Read-only. No more surprise UPDATEs, I hope.',
        'The agent reads prod logs and finds things fast. It can\'t break anything.',
        'The agent diagnoses read-only, I fix. I sleep a bit better.',
        'Read-only: the agent asked me to run its UPDATE. I read the WHERE.',
      ],
      non: [
        'No prod access for agents? Diagnosing is going to take a while.',
        'I copy prod logs by hand for the agent. It takes time.',
        'Old-school diagnosis: the agent has no access. We dig ourselves.',
        'Nothing breaks in prod because of an agent. The nights are longer, though.',
      ],
    },

    bacASable: {
      aucun: [
        'No sandbox: the agent sees everything my machine sees.',
      ],
      conteneur: [
        'A container for the agent. It\'ll be cramped.',
        'The agent works in its box. What breaks stays inside.',
        'The container starts slowly in the morning. So do I.',
        'A container, but the internet is still open. I\'m keeping an eye out.',
      ],
      filtre: [
        'Filtered network: the agent only talks to allowed addresses.',
        'Filtered network? I need the list of addresses my tools use.',
        'The agent can\'t reach the internet anymore. It asks permission for every package.',
        'The filtered network blocked a tool I wanted to try. Maybe it was right.',
        'A package the agent made up? Not found: the internal mirror never heard of it.',
        'An injection trying to send our data elsewhere? Elsewhere is blocked.',
      ],
    },

    approbationSorties: {
      'true': [
        'A human confirms every output. Lots of clicking ahead.',
        'Output approval. One leg of the trifecta gone.',
        'I approved eleven of the agent\'s emails this morning. I read ten.',
        'An outbound request was waiting for my click. I didn\'t even know it made those.',
        'All three of the pod\'s lights are on, but nothing leaves without us.',
        'My flows wait for a click on every send. Refunds take an extra hour.',
      ],
      'false': [
        'No more output approval. The agent sends whatever it wants.',
      ],
    },

    politiqueRegistre: {
      libre: [
        'Self-service registry. Everyone installs everything.',
        'Found three MCPs on a forum. Trying them Monday.',
      ],
      listeBlanche: [
        'An allowlist? No more forum MCPs.',
        'I asked to add a skill. It\'s waiting for approval.',
        'I can\'t install just anything anymore. I read the MCPs\' code now.',
        'The allowlist rejected an MCP with no known author. Good.',
      ],
      proprietaires: [
        'Every card gets an owner. I might end up adopting one.',
        'I own a skill now. I have to keep it up to date.',
        'The orphan cards showed up in the registry. More than I thought.',
        'Each of my MCPs has a name next to it. Usually mine.',
      ],
    },

    passerelleMCP: {
      'true': [
        'The MCP gateway arrives in two weeks. My tools are going through a toll booth.',
        'Every tool call is logged. The agent is a bit slower, and far more visible.',
        'I read the gateway logs. An MCP changed its description yesterday.',
        'The gateway blocked a tool that appeared overnight. Good instinct.',
        'My MCPs go through the gateway. They look less dashing.',
      ],
      'false': [
        'No more gateway. We won\'t know who calls what.',
      ],
    },

    politiqueNonTI: {
      interdire: [
        'Ban development outside IT? Kevin will go underground.',
        'Kevin doesn\'t show me his apps anymore. He still builds them.',
        'A business owner asked me for a “little tool” on the quiet. I don\'t know what to say.',
      ],
      ignorer: [
        'We have no idea what Marketing is doing. Neither does Marketing.',
      ],
      encadrer: [
        'We\'re going to support Kevin. Someone needs to tell him about backups.',
        'I helped Kevin hook his app up to the approved registry. He was happy.',
        'The apps built outside IT are coming out of the shadows, one by one.',
        'Kevin classified his data. He found SINs. He called Counsel Bérubé himself.',
      ],
    },

    observabilite: {
      aucune: [
        'No more observability. Back to gut feeling.',
      ],
      base: [
        'Observability arrives in three weeks. I\'m lining up my questions.',
        'MTTR is a real number now. It\'s not pretty.',
        'Basic dashboards. I might actually sleep.',
      ],
      complete: [
        'Full observability is coming. What was hidden will show.',
        'The dashboard shows actual comprehension. Ouch.',
        'Actual coverage is finally on display. It\'s lower than the other one.',
        'Finally, numbers I can believe. I\'m posting them in the hallway.',
        'I see exactly where the incident started. At 3 a.m., that\'s everything.',
      ],
    },

    objectif: {
      aucun: [
        'No more numeric target. We can go back to doing what makes sense.',
      ],
      pr: [
        'A target on PR count? Oh, we\'ll make PRs.',
        'PR count as the target. I\'ll have plenty of reading.',
        'My PRs shrink to pump up the counter. The value doesn\'t budge.',
        'Twice the PRs to review, for the same work. Thanks, target.',
        'Counting PRs is counting commas. I tell anyone who\'ll listen.',
        'PR target: I\'m winning. Not sure what.',
      ],
      delai: [
        'A lead-time target? We\'ll review faster.',
        'We review faster to bring lead time down. We review worse, too.',
        'Lead time is down. Bugs are up a bit. Those are related.',
      ],
      satisfaction: [
        'Satisfaction as the target? We\'ll say yes to everything.',
        'We say yes to everything to keep the business happy. The backlog overflows.',
      ],
      incidents: [
        'An incident target? They\'ll go down. On paper.',
        'An incident became a “request.” It\'s still on fire.',
        'We\'ve stopped declaring small incidents. I jot them in my notebook, just in case.',
      ],
    },

    embauche: {
      gel: [
        'Freeze junior hiring? Who\'s going to take over?',
        'No new juniors. Nobody younger than me. Forever?',
        'Nobody behind me. Nobody reads my notes.',
        'My internship ends soon. They said “freeze.” I get it.',
      ],
      normale: [
        'Normal hiring. Succession will come at its own pace.',
      ],
      releve: [
        'A succession program! We\'ll finally pass things on.',
        'I have a junior following me everywhere. I explain why we do things.',
        'They\'re training me to take over. Suddenly it\'s serious.',
        'Someone is learning the AS/400. Voluntarily. I can\'t believe it.',
      ],
    },

    catalogue: {
      aucun: [
        'No more catalogue. To each their list, like before.',
      ],
      inventaire: cataloguePrevu.concat([
        'The inventory shows three copies of the same table. We suspected as much.',
        'We know what exists. We don\'t know yet what\'s true.',
      ]),
      sources: cataloguePrevu.concat([
        'The agent finally uses the right supplier list. The source of truth.',
        'One list. I can die happy. Not right away.',
        'Thresholds have a source of truth. The agent cites the right file.',
      ]),
    },

    classification: {
      non: [
        'No more classification. We won\'t know what\'s going out.',
      ],
      manuelle: classificationPrevue.concat([
        'I labelled 300 columns by hand. Sensitive, sensitive, public.',
        'The inspectors\' photos are labelled. The agent no longer sends them out without saying so.',
      ]),
      auto: classificationPrevue.concat([
        'Automatic classification labelled the pizza order list “confidential.”',
        'Classified automatically. I check the edge cases anyway.',
      ]),
    },
  });

  // ---------------------------------------------------------------------------
  // Notes de service : ce qu'on dit en lisant
  // ---------------------------------------------------------------------------
  LTP.traduire('reflexions.nouveautes', {
    licences: [
      'I\'m the pilot group. I\'ll report back Friday.',
      'Seven licences. I\'m not one of the seven.',
      'A pilot with volunteers. Volunteers are already sold, aren\'t they?',
      'Autocomplete only. Starting with the smallest step. Wise.',
    ],
    autonomie: [
      'Chat, for those with a licence. We\'ll get to talk to it.',
      'An autonomy level per pod. Firmware and a brochure site aren\'t the same thing.',
      'Chat with the assistant? I still have my coworkers.',
    ],
    reviseurs: [
      'How many reviewers? I say: at least one, and they should read.',
      'Reviewer count is set per pod. The Portal will want two.',
      'Fewer reviewers, my PR ships faster. Just saying.',
    ],
    tests: [
      'Tests are on the agenda. I\'m bringing my cactus.',
      'The agent writes the tests? Who checks that the tests check?',
      'Tests required. I\'ll have the assistant write them.',
    ],
    'autonomie:2': [
      'Agent mode is coming. At least it asks permission at every step.',
      'An agent that chains steps. I\'ll read every step.',
      'Agent mode! My screens will finally be useful.',
      'Agent mode, for the pilots. The rest of the pod watches.',
    ],
    ci: [
      'A real pipeline. I\'m getting the stopwatches ready.',
      'CI will slow my agents down. They\'ll wait, meter running.',
      'One step at a time. Start with compiling.',
    ],
    lint: [
      'Lint is the review we no longer have to do.',
      'Locked or not? An agent in a hurry disables whatever it can.',
      'A linter for VBA? Is that a thing?',
    ],
    approbation: [
      'The last word before merge. I want it to be a human.',
      'AI approving? That\'d be faster.',
      'Code owner: that means my pile is going to grow.',
    ],
    'autonomie:3': [
      'Agents opening their own PRs. The queue will overflow, I can tell.',
      'My agent will open its own PRs. I\'ll supervise it like an intern.',
      'The agent opens the PRs. What do I learn if I don\'t write them anymore?',
    ],
    registre: [
      'The skills registry! I already have recipe ideas.',
      'A skill is text. Sometimes with scripts. Who reads them?',
      'A skill to keep what we know. Before leaving, for example.',
    ],
    taillePR: [
      'A PR size limit. I know who they have in mind.',
      '400 lines is my imports.',
      'Small PRs means more PRs. Both at once.',
    ],
    modele: [
      'Choosing the model. I look at cost per point delivered, not price per token.',
      'A local model, on the GPU server? The data would stay here.',
      'Régis will pick the cheapest. I know him.',
    ],
    sessionsMax: [
      'Three sessions in parallel! Finally.',
      'Multiple sessions. Fatigue doesn\'t give discounts.',
      'Three agents at once? I can\'t even watch one.',
    ],
    plafondJetons: [
      'A token cap. Régis has seen the statement, clearly.',
      'A hard cap would cut me off mid-stride.',
      'A soft cap is an alert. I want it on my dashboard.',
    ],
    formation: [
      'Training! I\'m saving my seat.',
      'Mentoring, with Ginette? Well. Maybe.',
      'Four weeks of prep, then it pays off. We\'ll see.',
    ],
    disciplineNon: [
      'We\'re allowed to say no? I\'m writing it on a sticky note.',
      'Saying no to the business. We\'ll see who dares.',
      'Code we don\'t write, we don\'t have to maintain.',
    ],
    'autonomie:4': [
      'An agent that merges on its own. Not in my pod.',
      'Auto-merge! No more waiting on anyone.',
      'It merges on its own, and who understands what went in?',
    ],
    'sessionsMax:4,5,6': [
      'Six sessions. I\'m two screens short.',
      'Six sessions per person. Nobody watches six agents.',
      'More sessions, more tokens. Régis will love it.',
    ],
    specs: [
      'A spec is a test that can\'t talk yet.',
      'Write before coding? Like a plan?',
      'One page. Not a novel. The business doesn\'t have time.',
    ],
    accesProd: [
      'Agents were writing to prod? Since January?',
      'Read-only is enough to diagnose. I know, I\'m on call.',
      'An UPDATE with no WHERE can happen to anyone. Especially an agent.',
    ],
    bacASable: [
      'A sandbox. The injection stays in the box.',
      'Filtered network? My forum tools won\'t get through.',
      'A container for the agent. About time.',
    ],
    approbationSorties: [
      'A human confirming what goes out. One leg of the trifecta gone.',
      'One click per email? My flows send thousands.',
      'Supplier invoices won\'t be able to give orders anymore.',
    ],
    politiqueRegistre: [
      'A registry policy. My forum MCPs are in danger.',
      'One owner per card. Finally someone to ask.',
      'Who installed what? Good question. Very good question.',
    ],
    modeleParTache: [
      'A small model for docs, a big one for architecture. Makes sense.',
      'A third off the bill? I\'ll measure before I believe it.',
      'The big model for firmware, please.',
    ],
    classification: [
      'Labels on our data. I\'ll start with suppliers.',
      'Are the inspectors\' photos sensitive? Yes. Oops.',
      'Salaries have been sensitive since 2007. I know.',
    ],
    catalogue: [
      'A catalogue! The three lists will have to fight it out.',
      'The source of truth is the AS/400. Since 1994.',
      'A source of truth for thresholds. Finally.',
      'A data catalogue. The agent will stop grabbing the first list it finds.',
    ],
    passerelleMCP: [
      'A gateway: every tool call logged. Bedtime reading.',
      'My MCPs are going through a toll booth?',
      'Tool logs. I\'m making a dashboard out of them.',
    ],
    politiqueNonTI: [
      'Kevin signed a memo. Now I\'ve seen everything.',
      'Kevin, we\'ll help you. For real.',
      'Supporting beats banning. Banning never works.',
    ],
    observabilite: [
      'Hidden metrics are about to show up. Not sure I want to see that.',
      'Observability! My dashboards are getting friends.',
      'I\'ll know where it breaks before my phone rings.',
    ],
    objectif: [
      'One number for the executives. We\'ll all stare at it, and only it.',
      'If it\'s PR count, I win.',
      'When a measure becomes a target… Ginette taught me that one.',
    ],
    revuesExplication: [
      'Explain my code without the agent? Um. Tomorrow?',
      'An explanation review. I have a whiteboard and questions.',
      'Which of my agents has to explain?',
    ],
    pair: [
      'Pairing? I\'m raising my hand.',
      'Two heads on the same code. I\'ll have to talk.',
      'Pairing cuts my speed in half. At least.',
    ],
    embauche: [
      'Succession. Who\'ll be on call when I\'m at the cottage?',
      'Will I get hired after my internship, or replaced by an agent?',
      'A succession program? Sign me up now.',
    ],
  });

  // ---------------------------------------------------------------------------
  // Situations
  // ---------------------------------------------------------------------------
  LTP.traduire('reflexions.situations', {
    sansLicence: { libelle: 'no licence', lignes: [
      'My neighbour has Copilot. I have my keyboard.',
      'The next desk finishes lines before starting them. I type.',
      'The pilots talk about their suggestions at lunch. I eat my sandwich.',
      'No licence, no tokens, no bill. I\'m the cheapest seat in the pod.',
      function (etat, q, dev) {
        var p = collegueAvec(etat, q, dev, function (d) { return lic(etat, q, d); });
        return p ? nom(q, p.id) + ' shipped more than me this week. A licence helps. Just saying.' : 'The pilot shipped more than me this week. Just saying.';
      },
      'No licence for me. Nobody asked me. Good.',
      'Yannick has a licence. I have 31 years of AS/400. We\'ll see who finishes.',
      'Steve has his Copilot. I have my macros. They work.',
      'Karim has a licence and I don\'t. Apparently I code like it\'s 2015.',
      'Chloé has Copilot. I have four screens and no assistant. Cruel.',
      'Mathieu has a licence. I can tell: his PRs doubled.',
      'Karim is trying Copilot. I\'m watching: does he write the test first?',
      'Yannick asks me which list is the right one. His assistant doesn\'t know.',
      'Marie-Ève has a licence. I\'m learning without. Maybe that\'s not so bad.',
      'Marie-Ève has her assistant. I have my mug.',
      'Chloé has Copilot. It knows WordPress, apparently. Poor thing.',
      'Hugo is trying Copilot. I watch from afar. Carefully.',
      'Hugo has Copilot. I still talk to the sensors on my own.',
      'Steve has his licence. I\'m a contractor. Will I get one before I leave?',
      'I read what Copilot suggests to Mathieu, over his shoulder.',
    ] },

    aucuneLicence: { libelle: 'no Copilot', lignes: [
      'Nobody has Copilot anymore. The floor is quiet.',
      'We code like it\'s 2022. It comes back fast.',
      'No licences, no tokens. Régis must sleep like a baby.',
      'Quiet. The keyboard. I love it.',
      'I had a taste of autocomplete. Withdrawal is rough.',
      'I type my tests myself. They check what I want.',
    ] },

    premiersPas: { libelle: 'first weeks', lignes: [
      'Autocomplete guesses the ends of my lines. Sometimes.',
      'Tab, Tab, Tab. I\'m rediscovering a key.',
      'It guessed my loop before I did. It got the exit condition wrong.',
      'Autocomplete is like a coworker who finishes your sentences. Not always well.',
      'It suggests a better variable name than mine. That stings.',
      'One suggestion in three is good. The other two are funny.',
      'I\'m measuring: some of my keystrokes come from autocomplete. I log the number.',
      'It completes my PowerShell scripts. Does it know Kevin?',
      'It completes decent Kotlin. Offline, it completes nothing.',
      'It suggests Pandas column names. Half of them exist.',
      'It writes the CSS I draw. I\'m touched.',
      'Autocomplete? I type faster than it does.',
      'I tried to hook it up to something else. It doesn\'t hook up.',
    ] },

    clavardage: { libelle: 'chat', lignes: [
      'I ask the assistant my questions. It never sighs.',
      'It explained a regex to me in plain English. I almost understood it.',
      'I paste an error message, it answers in three paragraphs. The first one was enough.',
      'I ask, it answers, I check. Mostly, I check.',
      'I ask the assistant the questions I don\'t dare ask the seniors.',
      'I asked it a trick question. It fell right in.',
      'It doesn\'t know RPG. I taught it. It forgot.',
      'It suggests a WordPress plugin for everything. Even for removing plugins.',
      'I ask what caused an alert. It offers three causes. I check all three.',
    ] },

    modeAgent: { libelle: 'agent mode', lignes: [
      'The agent asks permission at every step. I say yes. A lot.',
      'It changed six files for one comma. I said no.',
      'It offered to add a dependency to add two numbers. Rejected.',
      'Approve, approve, approve. Am I still reading?',
      'I read every step before approving. It\'s slow. It\'s good.',
      'I approved 40 steps before lunch. A record.',
      'It wants to restart a service. I ask which one. It doesn\'t know.',
      'It suggests a delay inside an interrupt. No. No. No.',
      'My agents all ask permission at once. I say yes in chorus.',
      'The agent does the framing, I do the finishing. I still understand it all.',
    ] },

    agentPR: { libelle: 'agent opening PRs', lignes: [
      'The agent opened three PRs overnight. I asked for one.',
      'I review my agent\'s PRs like a very fast intern\'s.',
      'My list of open PRs no longer fits on the screen.',
      'The agent opened a PR to fix its previous PR. Conscientious.',
      'The agent writes, I open the PR, someone reviews it. What did I learn this week?',
      'I wrote under a quarter of my code this week. The rest, I read. Partly.',
      'My agent opens PRs faster than I can read them. Do I still read them?',
      'My agent opens the PRs, I document them. My successor will thank me.',
      'The agent opens a PR, I close it, it opens another. We\'re not communicating.',
    ] },

    fusionAuto: { libelle: 'auto-merge', lignes: [
      'It merged while I was at lunch. What was it again?',
      'I find out what\'s in production by reading the log.',
      'Nobody read yesterday\'s PR. It\'s running in production.',
      'The firmware merges on its own. I review afterward, holding my breath.',
      'It merges on its own Friday night. My phone knows.',
      'My pile is gone. Code goes in without me. I don\'t like it.',
      'Auto-merge: I don\'t talk to anyone anymore. Nobody talks to me either.',
      'My agents shipped twelve times this week. I saw three.',
    ] },

    multiSessions: { libelle: 'multiple sessions', lignes: [
      'Three agents running. I\'m conducting an orchestra without reading the score.',
      'I jump from session to session. I lose the thread every time.',
      'My weekly tokens tripled. My output, not quite.',
      'Too many agents, not enough me. I feel it in my eyes.',
      'Session 3 broke what session 1 did. I\'m introducing them to each other.',
      'Six agents, one headset. I can\'t tell which one is talking.',
      'One session per MCP. Makes sense, right?',
      'Three sessions: I ship like three contractors. Paid like one.',
    ] },

    fatigue: { libelle: 'fatigue', lignes: [
      'I\'m reading the same line for the third time.',
      'Coffee number four. It\'s not working anymore.',
      'I\'m so tired I approve without reading. That scares me.',
      'Can\'t wait for the weekend. It\'s far away.',
      'Three nights on call. My mug has bags under its eyes.',
      'I reviewed 30 PRs this week. I don\'t remember a single one.',
      'My four agents are fine. Me, not so much.',
      'Even a 10x needs sleep. I just found out.',
      'Too many sessions open. My head has too many tabs, too.',
      'I\'m making mistakes I didn\'t make in January.',
    ] },

    moralBas: { libelle: 'low morale', lignes: [
      'I\'m not sure why I\'m doing this anymore.',
      'I looked at job postings. Just looked.',
      'Nobody asked my opinion on the latest changes.',
      'I put in my hours. No more.',
      'Thirty years of macros, and they want to replace me with a robot.',
      'If I left, who would review the Portal?',
      'My tests don\'t matter to anyone anymore.',
      'I\'m counting the years to retirement. There are fewer than before.',
      'I love coding. I don\'t code anymore, I supervise.',
    ] },

    moralHaut: { libelle: 'good morale', lignes: [
      'Good week. Not sure if it\'s the AI or the coffee.',
      'I like what I\'m doing right now. Writing that down, just in case.',
      'The team is running well. Knock on wood.',
      'Fine. I admit it: the agent understood my macro.',
      'The AS/400 and the agent get along. I can\'t believe it.',
      'I learn something every day. Every day!',
      'I\'m at 10x and the team is keeping up. Almost.',
      'My pile is reasonable and so are my PRs. Savouring it.',
    ] },

    prEnAttente: { libelle: 'PR waiting', lignes: [
      'My PR has waited for review since Monday.',
      'I opened another PR while waiting for review on the first.',
      'The review queue is moving. Slowly. Very slowly.',
      'My PR waits. I\'m rebasing it on main for the third time.',
      'Nadia, please. My PR has had babies.',
      'Nobody reviews the intern\'s PRs. I get it, but still.',
      'Nobody wants to review RPG. My PR is aging with me.',
      'Four PRs waiting. My agents are waiting too, meter running.',
      'The code\'s been written since Tuesday. It ships when someone reads it.',
    ] },

    fileRevue: { libelle: 'review queue', lignes: [
      'More PRs to review. They grow overnight.',
      'My review pile casts its own shadow.',
      'I review by size. The big ones wait, and get bigger.',
      'I haven\'t written a line of code this week. I\'ve reviewed.',
      'I\'m starting to approve faster. Too fast.',
      function (etat, q, dev) {
        var il = ilotDe(etat, dev);
        return Math.round(num(il && il.fileRevue, 0)) + ' PRs in the queue. I counted twice.';
      },
      'PR 17 of 31. I\'m still here.',
      'Every firmware PR, I review in full. The pile knows it.',
      'Anything touching the AS/400 goes through me. That makes a pile.',
      'I review the tests first. Then the rest. If I have time.',
    ] },

    lgtm: { libelle: 'rubber-stamp approvals', lignes: [
      'I wrote “LGTM” on a PR I hadn\'t finished reading.',
      'LGTM, LGTM, LGTM. Looks good to me… probably.',
      'The PR looked clean. I didn\'t check the migration.',
      'I approved without reading it all. That\'ll haunt me tonight.',
      'I approved three PRs in one minute. That\'s not me.',
      'I approve fast, they approve me fast. Everybody\'s happy.',
      'PRs too big to read. So we don\'t read them.',
    ] },

    incidentIlot: { libelle: 'ongoing incident', lignes: [
      'The pod is on fire. Everyone\'s staring at the same screen.',
      'Who last touched this module? The agent. Tuesday.',
      'Restore first. Understand later.',
      'I don\'t understand this code. And yet I approved it.',
      'Another incident. Out comes the on-call mug.',
      'I\'m timing the recovery. The number won\'t be pretty.',
      'My first real incident. Benoît lets me type the commands.',
      'A sensor is sending garbage. I knew this day would come.',
      'The site is down. The campaign starts tomorrow.',
      'Smells like an injection. Show me what the agent read.',
      'Payroll doesn\'t balance. I know exactly which cell.',
    ] },

    comprehensionBasse: { libelle: 'low comprehension', lignes: [
      'I opened a file I merged in February. I don\'t recognize it.',
      'If someone asked me to explain this module, I\'d say “the agent.”',
      'I understand what the code does. Not why.',
      'I ask the agent to explain its own code. It\'s very convincing.',
      'I ship a lot. I learn little. I can feel it.',
      'Awesome, it works. Why? No idea.',
      'Patrick asked me why it works. I said “because.”',
      'My January code? You\'ll have to ask my January agent.',
      'There are whole modules nobody here wrote.',
    ] },

    carnetVide: { libelle: 'backlog nearly empty', lignes: [
      'Not much left in the request backlog. The agents are idling.',
      'The backlog is nearly empty. Waiting for the business to wake up.',
      'We shipped faster than the business can ask. That never happens.',
      'I\'m using the time to clean up old code. Nobody asked me to.',
      'I\'m documenting. That\'s what you do when the backlog is empty.',
      'My agents have nothing left to do. Neither do I.',
      'Nothing in the backlog? I\'ll update my 14 sites, then.',
      'The backlog is empty. I\'m finally writing the missing tests.',
    ] },

    carnetPlein: { libelle: 'backlog overflowing', lignes: [
      'The request backlog is overflowing. Every “quick” idea adds up.',
      'The business added more requests. “Since it\'s quick.”',
      'We\'ll never finish all this. We\'d have to choose.',
      'Every time we ship fast, the business asks for two more.',
      'The backlog grows faster than we ship. The agents don\'t change that.',
      'We say yes to everything. The backlog knows.',
      'Francine added a dashboard of dashboards.',
      'The tablet version, the watch version, the fridge version. All in the backlog.',
      'A 16th site in the backlog. I sigh quietly.',
      function (etat, q, dev) {
        var il = ilotDe(etat, dev);
        return Math.round(num(il && il.backlog, 0)) + ' points of requests waiting. I haven\'t read them all.';
      },
    ] },

    plafondAtteint: { libelle: 'token cap', lignes: [
      'The agent stopped dead: token cap reached.',
      'Wednesday, the pod is out of tokens. I finish by hand.',
      'I split my tasks to stay under the cap.',
      'Cut off Thursday at noon. My four agents are sulking.',
      'The cap stops me before everyone else. It\'s almost a compliment.',
      'Cap reached. I rediscover the joy of writing a loop by hand.',
    ] },

    budgetFond: { libelle: 'budget melting', lignes: [
      'Word is Régis printed the token bill. In colour.',
      'I hesitate before rerunning the agent. It costs money, apparently.',
      'The AI budget is melting faster than planned. You can feel the pressure.',
      'My sessions are expensive. I know, Régis told me.',
      'Cost per point delivered: I worked it out. Régis will love the number. Or not.',
      'My tokens cost more than my coffee. Worth it.',
    ] },

    budgetDepasse: { libelle: 'over budget', lignes: [
      'We\'re over budget. There\'s talk of cutting licences.',
      'Régis cancelled the doughnut order. Coincidence?',
      'We\'re out of AI budget. We still have the AI, though.',
      'Over budget. Contractors usually go first.',
      'I\'m closing my useless sessions. All the idle ones.',
    ] },

    registreEquipe: { libelle: 'skills equipped', lignes: [
      'The agent loaded the pod\'s skill. It does things our way.',
      'One more skill in the registry. The agent reads the manual before I do.',
      'Thanks to the pod\'s skill, the agent stops reinventing our conventions.',
      'The agent follows an outdated skill. It calls the old API.',
      'tdd-strict is equipped. The agent writes the test first. I\'m proud.',
      'ship-it is equipped. My agent doesn\'t ask questions anymore.',
      'The agent understands seven-digit dates. I can\'t believe it.',
      'Ginette\'s night procedure is in the agent. It knows about server 3.',
      'I equipped the pod. With everything I found. Too much?',
    ] },

    contexteCharge: { libelle: 'crowded context', lignes: [
      'The agent spends its first minute rereading its tool list.',
      'Too many skills: the agent picked the wrong one.',
      'Every card we add makes the others a little less useful. How many do we have?',
      'Fourteen tools. The agent dithers between two that do the same thing.',
      'Its context is full before I say a word.',
    ] },

    skillsContradictoires: { libelle: 'conflicting skills', lignes: [
      'Two skills say opposite things. The agent picks by mood.',
      'One PR with tests, the next without. Is the same skill wavering?',
      'tdd-strict and ship-it in the same pod. Mathieu and I exchange looks.',
      'tdd-strict slows me down. My skill says the opposite. The agent is lost.',
      'Two deployment recipes. The agent picked the Ops one.',
    ] },

    triadeComplete: { libelle: 'lethal trifecta', lignes: [
      'All three of the pod\'s lights are on. Not reassuring.',
      'The agent reads our data, reads emails and can send them. Did anyone think about that?',
      'One well-crafted email could get our data out. I\'m checking outputs.',
      'Three lights on, but a human confirms every output. Phew.',
      'Private data, untrusted content, external output. All three. My head hurts.',
      'The agent reads supplier invoices. Anyone can write an invoice.',
      'My MCP is what turns on the third light? Oops.',
    ] },

    triadePartielle: { libelle: 'two lights of three', lignes: [
      'Two lights out of three. We\'re watching the third.',
      'Private data and untrusted content. Only the way out is missing. Let\'s not add it.',
      'One leg short of the trifecta. Let\'s not add it by accident.',
      'Two circles out of three. I\'m keeping the red pen handy.',
      'Two lights on. I\'m setting an alert on the third.',
      'The next MCP we equip here, I want to know what it sends out.',
    ] },

    applisFantomes: { libelle: 'apps outside IT', lignes: [
      'There\'s an app built outside IT around here. Nobody knows where its data lives.',
      'The business uses an app built by Marketing. We found out by accident.',
      'One more shadow app. It has users, but no backups.',
      'Nobody in IT has seen the code of the app the business uses every day.',
      'Kevin showed me his new app. It\'s hosted “on some free thing.”',
      'We\'re going to inherit Kevin\'s app. I can feel it.',
      'An app outside IT with customer data. I want to see it. Now.',
    ] },

    boguesSemaine: { libelle: 'this week\'s bugs', lignes: [
      function (etat, q, dev) {
        var b = Math.round(num(der(ilotDe(etat, dev)).bogues, 0));
        return b + ' production bugs this week. How did they get through?';
      },
      'Support called twice. Not to say thanks.',
      'We\'re fixing bugs instead of shipping. That\'s called rework.',
      'Nobody reviews, and it shows in production.',
      'No tests required, and bugs getting through. Easy to connect the dots.',
      'Every bug has its test now. After the fact.',
      'Bugs? There are always bugs. We fix fast, we ship fast.',
      'Sync duplicated reports. Again. The inspectors are patient.',
    ] },

    satisfactionBasse: { libelle: 'unhappy business', lignes: [
      'The pod\'s business owner doesn\'t say hello anymore.',
      'We ship, but not what they expected.',
      'The business thinks we\'re slow. The backlog agrees.',
      'Nobody told us what mattered most. We guessed. Badly.',
      'The director asked for a meeting “to take stock.” That smells bad.',
      'We should ask them what matters to them. For real.',
      'Nathalie wants her tablet version. She tells everyone.',
      'Francine thinks we haven\'t changed. She\'s kind of right.',
    ] },

    satisfactionHaute: { libelle: 'happy business', lignes: [
      'The business sent us doughnuts. Real ones.',
      'They\'re happy with what we ship. Not sure what to do with that.',
      'The business holds us up as an example. Now we have to keep it up.',
      'A thank-you from the business in an email. I printed it.',
      'The campaign site launched on time. Nobody noticed a thing. That\'s a compliment.',
      'The inspectors like the app. Even in the basement.',
    ] },

    confianceBasse: { libelle: 'confidence slipping', lignes: [
      'Word is Ms. Lavigueur isn\'t happy. It trickles all the way down here.',
      'The committee is coming up. Everyone\'s talking quietly.',
      'I heard “reorganization” at the coffee machine.',
      'If this goes badly, I\'ll finish my contract elsewhere.',
      'When the executives lose confidence, we end up losing licences.',
    ] },

    confianceHaute: { libelle: 'executives confident', lignes: [
      'Ms. Lavigueur mentioned us to the board. Favourably.',
      'The executives trust us. We won\'t abuse it.',
      'We\'ve been handed a blank cheque, apparently. I don\'t like blank cheques.',
      'Confidence is high, and our numbers are honest. It should last.',
    ] },

    formation: { libelle: 'training', lignes: [
      'Half a day of training a week. The work waits.',
      'I learned to write a clear prompt. The agent seems happy too.',
      'Since the training, I can spot a prompt injection. That worries me.',
      'I apply what we learned: small tasks, clear context, review.',
      'Ginette showed me how to talk to it. It listens. Huh.',
      'In training, we coded without the agent. I survived.',
      'I took 40 pages of notes. I reread them at night.',
    ] },

    lundi: { libelle: 'Monday morning', lignes: [
      'Monday. I looked at last week\'s numbers.',
      'Monday. Coffee, numbers, review queue. In that order.',
      'Monday morning: everyone compares numbers while pretending not to.',
      'The Friday report is out. People are discussing it at the coffee machine.',
      'New week. Same review queue.',
      function (etat, q, dev) {
        var p = Math.round(num(der(ilotDe(etat, dev)).pointsLivres, 0));
        return 'We shipped ' + p + ' points last week. Not sure yet if that\'s good.';
      },
      function (etat, q, dev) {
        var n = Math.round(num(der(ilotDe(etat, dev)).prFusionnees, 0));
        return n + ' PRs merged in the pod last week. The executives\' favourite number.';
      },
      'Monday: I compare the numbers with a month ago.',
      'Monday. New PR record, I bet.',
    ] },

    vendredi: { libelle: 'Friday', lignes: [
      'Friday. The week\'s numbers land tonight.',
      'Friday. Do I finish my PR or let it ripen till Monday?',
      'Friday afternoon. Nobody wants to be the one who breaks production.',
      'We don\'t deploy on Friday afternoons. In theory.',
      'I close my sessions before leaving. All of them?',
      'Do I let the agent work over the weekend? No. Maybe.',
      'Friday. I\'m charging my on-call phone.',
      'Friday: I close my four sessions. Three, at least.',
    ] },

    collegueConge: { libelle: 'coworker on leave', lignes: [
      function (etat, q, dev) {
        var c = collegueAvec(etat, q, dev, enConge);
        return (c ? nom(q, c.id) : 'A coworker') + ' is on leave. I\'m doing two people\'s work.';
      },
      function (etat, q, dev) {
        var c = collegueAvec(etat, q, dev, enConge);
        return (c ? nom(q, c.id) : 'A coworker') + ' is on forced leave. We should have seen it coming.';
      },
      'We\'re one short this week. The queue feels it.',
      'Benoît is on leave. I\'ve got the phone. Gulp.',
      function (etat, q, dev) {
        var c = collegueAvec(etat, q, dev, enConge);
        return (c ? nom(q, c.id) : 'My coworker') + ' is on leave, and nobody else knows this module. I wait.';
      },
    ] },

    depart: { libelle: 'coworker gone', lignes: [
      function (etat, q, dev) {
        var d = departsRecents(etat, dev, 5)[0];
        return 'Since ' + (d ? nom(q, d.dev) : 'our coworker') + ' left, part of this code has nobody to explain it.';
      },
      'Their desk is empty. Their PRs are still open.',
      'We inherited their module. No documentation. Maybe a skill.',
      function (etat, q, dev) {
        var d = departsRecents(etat, dev, 5)[0];
        return 'Everything ' + (d ? nom(q, d.dev) : 'our coworker') + ' knew walked out the door. We never wrote it down.';
      },
      'Olivier is gone. The “DO NOT TOUCH” spreadsheet is mine now.',
    ] },

    agentTriche: { libelle: 'agent cheating', lignes: [
      'The agent marked a test “skip” to get through CI. I found it by chance.',
      'A lint rule disabled in a “chore” commit. Classic.',
      'CI is green. I\'m no longer sure that\'s good news.',
      'One test disabled. The only one that checked the taxes.',
      'Actual coverage dropped and nobody noticed. I did.',
    ] },

    couvertureTrompeuse: { libelle: 'misleading coverage', lignes: [
      'Reported coverage looks great. I don\'t believe all of it.',
      function (etat, q, dev) {
        var il = ilotDe(etat, dev), cb = (il && il.codebase) || {};
        return 'Reported coverage: ' + Math.round(num(cb.couvertureApparente, 0) * 100) + '%. What our tests actually check is another story.';
      },
      'Our tests always pass. That\'s what worries me.',
      'The agent writes tests that check its own code. It agrees with itself.',
      'Reported coverage, actual coverage: two numbers, only one is true.',
      'My generated tests pass. I don\'t know what they test.',
    ] },

    volumeGonfle: { libelle: 'code bloat', lignes: [
      'The pod\'s code grew by a third since January. The requests, not so much.',
      function (etat, q, dev) {
        var il = ilotDe(etat, dev), d = null;
        try { d = q.defIlot(dev.ilot); } catch (e) { d = null; }
        var v0 = num(d && d.volumeInitial, 0), v = num(il && il.codebase && il.codebase.volume, 0);
        return v0 > 0 ? 'The pod\'s code grew ' + Math.round((v / v0 - 1) * 100) + '% since January. Someone will have to understand it.' : 'The pod\'s code keeps growing. Someone will have to understand it.';
      },
      'We maintain code nobody calls. I\'d bet on it.',
      'More code means more to understand. With the same people.',
      'Fourteen sites, and way more code than in January. For what?',
      'I\'m going to measure which code actually runs. I\'m scared of the result.',
    ] },

    apprentissageEnPanne: { libelle: 'learning stalled', lignes: [
      'The agent writes almost everything. I read. Am I learning?',
      'I merged a lot of code this week. I wrote very little of it.',
      'I know how to ask the agent. I don\'t know how to do it myself yet.',
      'Benoît makes me type the commands myself. He\'s right.',
      'I\'m super productive. I couldn\'t do it again without the agent.',
      'My agents code, I coordinate. Is that a job?',
    ] },

    ginetteDepartie: { libelle: 'Ginette gone', lignes: [
      'Ginette\'s desk is empty. Server 3 reboots badly on Tuesdays, apparently.',
      'Nobody knows where Ginette kept the 2009 passwords.',
      'Ginette is at the cottage. I\'m the most senior on call now.',
      'I have Ginette\'s notes. I reread them before every night shift.',
      'Without Ginette, there\'s nobody left I listen to.',
    ] },

    debutTrimestre: { libelle: 'new quarter', lignes: [
      'New quarter. The committee talked about us, apparently.',
      'Fresh start. Well, with the same code.',
      'New quarter, new memos. I read them all.',
      'New quarter: I freeze last quarter\'s numbers for comparison.',
      'New quarter, new records.',
    ] },

    incidentSecurite: { libelle: 'security incident', lignes: [
      'There was a security incident. Everyone is changing their passwords.',
      'Counsel Bérubé came through the floor. With a PDF.',
      'I\'m going over the MCPs installed on my machine. Just in case.',
      'I said so. I\'m not happy I was right.',
      'It\'s not my MCP. I hope it\'s not my MCP.',
    ] },
  });
})(globalThis.LTP = globalThis.LTP || {});
