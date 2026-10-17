/* en/10-donnees-ilots.js */
/* Édition anglaise : les sept lignes d'affaires (voir en/LISEZMOI.md). */
(function (LTP) {
  'use strict';

  LTP.traduire('ilots', {
    appro: {
      nom: 'Procurement', court: 'Procurement',
      mission: 'Purchasing, suppliers, purchase orders and invoices.',
      direction: { titre: 'Director of Procurement' },
    },
    operations: {
      nom: 'Operations', court: 'Ops',
      mission: 'Scheduling, on-call, deployments and internal tools for Operations.',
      direction: { titre: 'Director of Operations' },
    },
    bureautique: {
      nom: 'Office Automation', court: 'Office',
      mission: 'Macros, forms and automations for the administrative departments, payroll included.',
      direction: { titre: 'Director of Administrative Services' },
    },
    inspections: {
      nom: 'Inspections', court: 'Inspections',
      pile: 'Kotlin, Swift, offline sync',
      mission: 'The field inspectors\' mobile app: photos, forms, reports.',
      direction: { titre: 'Director of Inspections' },
    },
    iot: {
      nom: 'Sensors and IoT', court: 'IoT',
      pile: 'Embedded C, MQTT, Python',
      mission: 'Sensor firmware and telemetry pipelines.',
      direction: { titre: 'Director of the Sensor Network' },
    },
    web: {
      nom: 'Websites', court: 'Web',
      mission: 'The public websites and campaign microsites.',
      direction: { titre: 'Director of Communications' },
    },
    portail: {
      nom: 'Customer Portal', court: 'Portal',
      mission: 'The customer and partner portal: accounts, requests, payments.',
      direction: { titre: 'Director of Customer Services' },
    },
  });
})(globalThis.LTP = globalThis.LTP || {});
