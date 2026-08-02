# Architecture Design — Aurora Checkout Rebuild

**Organisation:** Aurora Payments (org-a) · **Project:** proj-a1 · **Codename:** BLUEJAY
**Criticality:** business-critical · **Handles personal data:** yes

## Overview
The Aurora Checkout Rebuild replaces the legacy checkout with a new set of services
handling consumer card payments. Data includes cardholder name, email, and billing
address (classified: restricted).

## Components
- **Checkout API** — public, internet-facing REST API. Authenticated via enterprise SSO
  and OAuth tokens. Sits behind the enterprise WAF.
- **Payment Orchestrator** — internal service coordinating payment steps. Deployed to a
  single availability zone in the approved EU region.
- **Customer Datastore** — stores cardholder details. Classification: restricted.
  Encrypted at rest. Personal data kept in the approved EU region only.
- **Reporting job** — nightly batch that reads directly from the Payment Orchestrator's
  database to build finance reports.

## Data flows
- Client → Checkout API: TLS 1.3.
- Checkout API → Payment Orchestrator: internal call over TLS 1.2.
- Reporting job → Payment Orchestrator DB: direct database connection.

## Operations
- Logs and metrics are sent to the enterprise monitoring platform.
- Secrets are stored in the approved secrets manager.
- No backup or recovery approach is documented yet for the Customer Datastore.

## Notes
- Technology stack is from the approved catalogue.
- Service accounts for the Payment Orchestrator are granted admin-level database rights
  for convenience during rollout.
