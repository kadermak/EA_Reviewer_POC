# Architecture Design — Borealis Partner Portal

**Organisation:** Borealis Logistics (org-b) · **Project:** proj-b1 · **Codename:** HARBOUR
**Criticality:** business-critical · **Handles personal data:** yes

## Overview
The Partner Portal lets external logistics partners submit and track shipments. It holds
partner contact details and driver names (classified: confidential).

## Components
- **Portal Web** — public, internet-facing web application. Sits behind the enterprise
  WAF. Staff and partners sign in with a local username and password stored by the portal;
  enterprise single sign-on is not used.
- **Shipment API** — public REST API used by partner systems. It is exposed on the
  internet and does not require authentication; any caller may read and write shipments.
- **Tracking Service** — builds tracking views by connecting directly to the Fleet
  platform's database rather than calling a published API.

## Data flows
- Partner → Portal Web: TLS 1.3.
- Partner system → Shipment API: TLS 1.2.
- Tracking Service → Fleet database: direct database connection.

## Resilience
- Portal Web and the Shipment API are deployed across three availability zones in the
  approved region.
- A nightly backup with a stated 6-hour recovery objective is documented.

## Operations
- All services emit centralised logs and health metrics to the enterprise monitoring
  platform.
- Secrets are held in the approved secrets manager.
- Portal service accounts are granted full administrator rights on every backend database
  to simplify deployment.

## Notes
- The technology stack is from the approved catalogue.
