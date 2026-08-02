# Architecture Design — Aurora Atlas Reporting Portal

**Organisation:** Aurora Payments (org-a) · **Project:** proj-a1 · **Codename:** ATLAS
**Criticality:** business-critical · **Handles personal data:** yes

## Overview
Atlas is an internet-facing reporting portal that lets finance staff view aggregated
settlement data. It holds customer names and email addresses (classified: confidential)
alongside non-personal financial aggregates.

## Components
- **Atlas Web** — public, internet-facing web application. Sits behind the enterprise
  WAF. Human access is via enterprise single sign-on only.
- **Atlas API** — REST API, published and versioned. Every endpoint enforces
  authentication and role-based authorisation.
- **Reporting Store** — holds confidential customer data. Classification: confidential.
  Encrypted at rest. Personal data is held only in the approved EU region.
- **Settlement Connector** — reads settlement data from the Payments platform through its
  published, versioned API — never its database.

## Data flows
- Client → Atlas Web: TLS 1.3.
- Atlas Web → Atlas API: TLS 1.3.
- Atlas API → Reporting Store: TLS 1.3.

## Resilience
- Atlas Web and Atlas API are deployed across three availability zones in the approved
  EU region.
- A documented backup runs nightly with a stated 4-hour recovery objective for the
  Reporting Store.

## Operations
- All services emit centralised logs and health metrics to the enterprise monitoring
  platform.
- Secrets are held in the approved secrets manager.
- Service accounts are granted only the specific read scopes each service needs.

## Notes
- The technology stack is drawn entirely from the approved technology catalogue.
