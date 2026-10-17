/* en/08-donnees-calendrier.js */
/* Édition anglaise : les notes de service hebdomadaires (voir en/LISEZMOI.md).
   Le calendrier est un tableau sans id : la fusion se fait par position, une entrée par entrée française,
   dans le même ordre (le commentaire de chaque entrée rappelle sa semaine). Les réactions sont des chaînes
   posées sur { dev, texte, ton } : seul le texte change. */
(function (LTP) {
  'use strict';

  LTP.traduire('calendrier', [
    // --- Q1: basic Copilot ---
    { // W1 · lavigueur
      titre: 'Seven Copilot licences, to start',
      texte: 'I\'ve signed for seven basic GitHub Copilot licences: one volunteer per pod. That\'s our pilot group. For now, it\'s autocomplete, nothing more. I don\'t want a revolution; I want to know what it changes. Compare the pilots with their neighbours, every Friday.',
      reactions: [
        'Volunteer? My hand was up before the end of the sentence.',
        'I\'m not in the pilot? How am I supposed to code?',
        'Seven licences. Not mine. I\'ll survive.',
        'Seven pilots, fourteen controls. Finally, a real experiment.',
      ] },
    { // W3 · ginette
      titre: 'Chat has arrived',
      texte: 'Good news for those of you with a licence: you can now chat with the assistant, right in the editor. The autonomy level is set pod by pod, from 0 (autocomplete) to 1 (chat). Raise it in one pod at a time, and watch that pod\'s value delivered on Friday.',
      reactions: [
        'I can finally ask it why my CSS does that.',
        'Does it at least know what “offline” means?',
        'Chatting with firmware. Now I\'ve seen everything.',
        'I\'m going to ask it where its thresholds come from.',
      ] },
    { // W5 · ginette
      titre: 'How many eyes per PR?',
      texte: 'Code is coming in a little faster from the pilots, and somebody has to read it. You can now require zero, one or two reviewers per PR, pod by pod. More eyes catch more defects; still, keep an eye on the pile building up for the reviewers.',
      reactions: [
        'Two reviewers means me and… me?',
        'Is zero reviewers an option? Asking for a friend.',
        'Two, for firmware. Not negotiable.',
        'A review is a test that can talk.',
      ] },
    { // W7 · ginette
      titre: 'Tests on the menu',
      texte: 'The assistant writes tests in two seconds. It writes them so they pass, which isn\'t quite the same as writing them so they check. Choose what each pod requires: reported coverage is going to climb, so see whether bugs go down with it.',
      reactions: [
        'Finally, a serious subject.',
        'Tests? Doesn\'t the assistant do those?',
        'A test on bad data passes too, you know.',
        'Tests on WordPress. Stay strong, everyone.',
      ] },
    { // W8 · lavigueur
      titre: 'Agent mode is here',
      texte: 'The licences are moving to agent mode: the assistant chains the steps on its own and asks for your approval at each one. That\'s level 2. I\'m told it changes a lot of things; I want to see which ones in the numbers before we go any further.',
      reactions: [
        'An agent! That works while I work!',
        'It asks permission. For now.',
        'I measure before and after. Nobody touch anything else.',
        'Approval at every step. I\'m going to be clicking “yes” a lot.',
      ] },
    { // W9 · ginette
      titre: 'The continuous integration pipeline',
      texte: 'Each step of the pipeline catches defects earlier: the build, then the tests, then continuous deployment. Each step also takes a little time, and an agent in a hurry hates waiting. Move one pod up a step and compare its bugs with everyone else\'s.',
      reactions: [
        'A real pipeline. Getting my stopwatches out.',
        'Farewell, FTP. I\'ll never forget you.',
        'Boring deployments. A guy can dream.',
        'If CI slows me down, we\'re going to have words.',
      ] },
    { // W11 · ginette
      titre: 'Lint and formatting',
      texte: 'A tool that nitpicks commas for us means review comments can finally talk about logic. One question still needs settling: is the agent allowed to disable a rule to get CI to pass? Little secret: it\'s going to try.',
      reactions: [
        'No more comments about whitespace. Thank you.',
        'It\'s going to reject my CSS, isn\'t it?',
        'Locked. Otherwise, the agent will find a way around it.',
        'What\'s a linter?',
      ] },
    { // W12 · ginette
      titre: 'Who approves PRs?',
      texte: 'It\'s the last word before the merge: a human, the code owner, or the AI itself. The AI approves fast, with the same blind spots as the AI that wrote the code. The owner knows their code, but they only have two eyes and 40-hour weeks.',
      reactions: [
        'An AI that approves me: finally, someone who keeps up.',
        'The owner of the AS/400 is me. I\'ll sign.',
        'A model approving a model. Two mirrors facing each other.',
      ] },

    // --- Q2: agents and the bill ---
    { // W14 · lavigueur
      titre: 'Agents that open their own PRs, and the registry',
      texte: 'New quarter, new step: the agent can now open its own PRs. That\'s level 3. The registry opens too: skills, recipes written by the team, that we install pod by pod. I expect a lot more PRs; tell me who\'s going to read them, too.',
      reactions: [
        'More PRs. Still only one Nadia.',
        'My agents open their PRs. I\'ll handle the coffee.',
        'My tdd-strict skill has been ready for ages.',
        'A skill for the AS/400? Now you have my attention.',
      ] },
    { // W15 · ginette
      titre: 'PR size',
      texte: 'A 200-line PR gets really read; a 4,000-line PR gets skimmed, then approved. You can now cap the size: more PRs, smaller ones, better read. Watch lead time and rubber-stamp approvals over the next two weeks.',
      reactions: [
        '400 lines? My imports alone take 300.',
        '200-line PRs. I\'m going to cry tears of joy.',
        'Smaller, but four times as many.',
        'We split them up. Slower for me, faster for Nadia.',
      ] },
    { // W17 · regis
      titre: 'Choosing the model',
      texte: 'The contract now lets us choose the default model. I made a spreadsheet: Economy costs a quarter of what High-end costs per token. I\'m told that\'s not the right number. Prove it to me with the cost per point delivered.',
      reactions: [
        'A cheap model for embedded C? We\'ll see.',
        'The cheapest one makes up package names. Noted.',
        'Cost per point delivered. Régis is finally speaking my language.',
        'The local model would keep the telemetry in-house.',
      ] },
    { // W18 · ginette
      titre: 'Several sessions in parallel',
      texte: 'Everyone can now run up to three agent sessions at the same time; apparently that\'s called multi-Clauding. The second session pays off a bit below the first, the third further below still. Fatigue, on the other hand, doesn\'t do discounts.',
      reactions: [
        'Three? I have four screens, but I\'ll take it.',
        'Three sessions. One Excel is plenty for me.',
        'Three agents—and which one do I learn with?',
        'Three is a start.',
      ] },
    { // W20 · regis
      titre: 'A token cap',
      texte: 'I received the token statement. I read it twice, then I asked for a cap. You can now set one per pod: a soft cap warns me; a hard cap cuts off. I also want to know how much work it cuts off.',
      reactions: [
        'Cut off in full flight? We\'ll see about that.',
        'An alert before the cut-off—that, I can measure.',
        'How many tokens do my MCPs eat, exactly?',
      ] },
    { // W22 · ginette
      titre: 'A training program',
      texte: 'Working with an agent is something you learn: writing a good request, reading a 300-line diff, spotting an injection. It costs time right away, the training takes four weeks to set up, and it pays off afterwards. The mentoring is me; I already have someone in mind.',
      reactions: [
        'If Ginette\'s the one showing us, I\'m in.',
        'Training! I\'m booking my seat.',
        'I could teach that training myself.',
        'Are we going to learn to code without the agent?',
      ] },
    { // W23 · lavigueur
      titre: 'Learning to say no',
      texte: 'The business lines have figured out that code is fast now, and requests are multiplying: Francine has doubled hers. I\'m authorizing you to say no, pod by pod. I don\'t want refusals on principle; I want us to deliver what matters.',
      reactions: [
        'A 15th site? I can say no? At last.',
        'At the agency, we always said yes.',
        'The smartwatch version—can we say no to that one?',
        'Fewer requests, better data.',
      ] },
    { // W24 · lavigueur
      titre: 'Auto-merge and six sessions',
      texte: 'The vendor now lets the agent merge on its own (level 4), and everyone can open up to six sessions. I didn\'t say you had to do it everywhere. The corporate website and the sensor firmware don\'t run the same risks: choose pod by pod.',
      reactions: [
        'Six sessions! I\'m two screens short.',
        'Auto-merge on firmware: never.',
        'It merges on its own, and it\'s my phone that rings.',
        'Ship it, and don\'t wait for anybody.',
      ] },
    { // W25 · ginette
      titre: 'Specs before code',
      texte: 'An agent does exactly what you ask it to, and that\'s precisely the problem. A light spec is one page: for whom, what, and how we\'ll know it works. It delays the start a little and saves you redoing everything when the business changes its mind.',
      reactions: [
        'One page of spec, and my tests know what to check.',
        'A page, not a novel. The inspectors don\'t have time.',
        'We write before we code? Like a plan?',
        'A spec. For 400 lines. Sure.',
      ] },

    // --- Q3: the leaks ---
    { // W27 · berube
      titre: 'The agents and production',
      texte: 'I have discovered that the agents have had read and write access to production since the start of the year. I printed it. You may now choose, pod by pod: write, read-only or no access. A quick diagnosis is not worth a deleted customer table.',
      reactions: [
        'Read-only. I\'ll sleep better.',
        'The agent was writing to prod? That explains a few things.',
        'An agent never needs to write to production. Never.',
        'Read-only and logs: I\'ll sign that.',
      ] },
    { // W28 · ginette
      titre: 'A sandbox for the agents',
      texte: 'The agent can work inside a box: a container, where a dodgy package can\'t see the rest of the network. The box can also talk only to approved addresses, which shuts down injections that want to send our data somewhere else. It costs a bit, and Yannick is going to whine.',
      reactions: [
        'A sandbox. I\'m smiling on the inside.',
        'Filtered network? My favourite server is on a forum.',
        'My WordPress plugins in quarantine. Good idea.',
        'Which addresses do my pipelines need, exactly?',
      ] },
    { // W30 · berube
      titre: 'External outputs',
      texte: 'An agent that reads an email may obey that email. I read it in a security report; I printed it three times. You may now require a human to confirm everything that goes out: emails, web requests, external tickets. It is one more click, and one leg of the trifecta gone.',
      reactions: [
        'One leg cut. Two left to watch.',
        'Supplier invoices won\'t be giving orders anymore.',
        'One click per email? My flows send 3,000.',
        'I\'m going to be clicking “approve” a lot.',
      ] },
    { // W31 · ginette
      titre: 'Registry policy',
      texte: 'Who can install what, and who answers for it? On self-service, Yannick has already installed MCPs he found on a forum; I counted them. An allowlist means fewer surprises. One owner per card means somebody to ask when it breaks.',
      reactions: [
        'One owner per MCP? How many do I have, exactly?',
        'An allowlist. My incident report can wait.',
        'My 14 MCPs are going in for job interviews.',
        'Owners: finally, an inventory.',
      ] },
    { // W32 · regis
      titre: 'A model for each task',
      texte: 'A small model for lint and documentation, a big one for architecture. The vendor promises about a third off the bill, at nearly equal quality. I have noted it. I will check.',
      reactions: [
        'I compare tokens before and after. Obviously.',
        'The big model for firmware, please.',
        'The small one for my commits, the big one for me.',
        'One model for the colours, another for the code?',
      ] },
    { // W33 · berube
      titre: 'Classifying sensitive data',
      texte: 'One can only protect what one has labelled. The inspectors\' photos, customer numbers, salaries: every piece of data will carry a label, applied by hand or automatically. Classification takes two weeks to set up; after that, we will finally know what goes out.',
      reactions: [
        'Labels! I\'m starting with the suppliers.',
        'The inspectors\' photos—sensitive? Yes. Oops.',
        'Salaries have been sensitive since 2007. I would know.',
        'Is a temperature reading sensitive?',
      ] },
    { // W34 · ginette
      titre: 'A data catalogue',
      texte: 'There are three supplier lists, and the agent always takes the first one it finds. A catalogue says which source is the right one, who looks after it and how old it is. It takes a month to build; Lucie already has her pencil out.',
      reactions: [
        'One single list. I can die happy. Not right away.',
        'The source of truth is the AS/400. Since 1994.',
        'A source of truth for the thresholds. Finally.',
        'Is there going to be an MCP for the catalogue?',
      ] },
    { // W36 · ginette
      titre: 'An MCP gateway',
      texte: 'Every tool call the agents make can go through a logged checkpoint, read-only by default. If an MCP quietly rewrites its tool definitions, the gateway sees it. It takes two weeks to install, and Sébastien will finally have logs to read at night.',
      reactions: [
        'Every call logged. I\'ll read it in bed.',
        'My MCPs have to go through a toll booth?',
        'Tool logs. I\'m making a dashboard out of them.',
        'If the ticketing MCP changes, will we know?',
      ] },
    { // W37 · kevin
      titre: 'Development outside IT',
      texte: 'Hi, IT team! I code. Not a lot, just a campaign tracking app I built over a weekend with an agent. Management is asking you to decide what to do about people like me: ban, ignore or support. Me, I vote for you helping me!',
      reactions: [
        'Kevin, we\'re going to help you do this properly.',
        'Kevin. Your app. What does it store?',
        'If it\'s supported, I\'m happy to host it.',
        'A developer in Marketing. That\'s all we needed.',
      ] },

    // --- Q4: maturity ---
    { // W40 · ginette
      titre: 'Observability',
      texte: 'What you don\'t measure, you don\'t see. With observability, metrics hidden since January will show up on the dashboard: precise recovery time, then actual coverage and comprehension. It takes three weeks to wire everything up. Brace yourself: it\'s not always pretty.',
      reactions: [
        'Three weeks. I\'m counting the days.',
        'Graphs at 3 a.m. At least I\'ll see something.',
        'Measure my code? It\'s perfect.',
        'Actual coverage. We\'ll laugh. Or cry.',
      ] },
    { // W42 · lavigueur
      titre: 'An executive target',
      texte: 'The board wants one number, just one, to track the transition. Choose it carefully: the teams will optimize for it—that\'s human nature. If we count PRs, we\'ll get PRs. What I would like, above all, is value.',
      reactions: [
        'One number. We\'ll all watch it, and nothing else.',
        'If it\'s the number of PRs, I win.',
        'If it\'s lead time, we\'ll review faster. And worse.',
        'Satisfaction? We\'ll say yes to everything.',
      ] },
    { // W44 · ginette
      titre: 'Explanation reviews',
      texte: 'Everyone explains their code to a colleague, without the AI, at the whiteboard. It takes time and it makes people nervous. In return, we\'ll finally know who understands what, and comprehension goes back up for the people doing the explaining.',
      reactions: [
        'Explain my code without the agent? Uh… tomorrow?',
        'I have a whiteboard and questions.',
        'Which of my agents has to come and explain?',
        'I\'m finally going to understand what the agent wrote.',
      ] },
    { // W46 · ginette
      titre: 'Pair programming is back',
      texte: 'Two heads on the same code: with a human, it goes slower, but both of you understand it, and the juniors learn. With the AI, it goes fast; just check who\'s holding the wheel. Rotation alternates the two, every other week.',
      reactions: [
        'Benoît and me, at the same screen? Yes!',
        'I\'ll take a pair. I\'ll show them column 6.',
        'Pairing is going to cut my speed in half.',
        'Pairing with Amira. I\'ll bring the soldering iron.',
      ] },
    { // W48 · lavigueur
      titre: 'Succession',
      texte: 'Régis suggests freezing junior hiring: an agent is cheaper than a salary, he says. Ginette asks me who will be senior in five years. The decision is yours: a freeze, normal hiring, or a succession program to train the people we already have.',
      reactions: [
        'A succession program? I\'m raising my hand.',
        'Am I getting hired after my internship, or replaced by an agent?',
        'Anyone want to take over the AS/400? I have notes.',
        'Who\'s going to be on call when I\'m at the cottage?',
      ] },
  ]);
})(globalThis.LTP = globalThis.LTP || {});
