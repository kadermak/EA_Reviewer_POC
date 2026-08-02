# Architecture Design — Borealis Fleet Tracker

**Organisation:** Borealis Logistics (org-b) · **Project:** proj-b1 · **Codename:** SANDPIPER
**Criticality:** business-critical · **Handles personal data:** yes (driver details)

## Overview
The Fleet Tracker ingests GPS and status data from delivery vehicles and exposes it to
warehouse and dispatch teams. Driver personal data (name, phone, licence number) is
classified confidential.

## Components
- **Ingest API** — public, internet-facing API receiving telemetry from vehicle devices.
  No authentication is enforced on this endpoint so devices can post quickly. It does NOT
  sit behind the enterprise WAF.
- **Tracking Service** — internal, deployed across two availability zones in the approved
  region.
- **Driver Datastore** — stores driver personal data. Classification: confidential.
  Encryption at rest is not mentioned.
- **Dispatch Portal** — web app used by staff, secured with local application accounts
  and passwords managed within the app.

## Data flows
- Vehicle device → Ingest API: plaintext HTTP.
- Tracking Service → Driver Datastore: TLS 1.2.

## Operations
- Logs and metrics are sent to the enterprise monitoring platform.
- Backup and recovery approach is documented with a 4-hour recovery objective.
- Secrets are stored in the approved secrets manager.
- Uses an unapproved open-source mapping database not on the technology catalogue; no
  waiver recorded.
