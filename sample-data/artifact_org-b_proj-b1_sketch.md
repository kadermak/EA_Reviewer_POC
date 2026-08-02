# Architecture Design — Borealis Route Planner (early draft)

**Organisation:** Borealis Logistics (org-b) · **Project:** proj-b1 · **Codename:** COMPASS
**Criticality:** business-critical · **Handles personal data:** yes

## Overview
The Route Planner suggests delivery routes for drivers. This is an early design draft and
several areas are still being worked out. It handles some driver and customer information.

## Components
- **Planner Service** — computes routes. It reads and stores data in a database.
- **Driver App** — a mobile app drivers use in the field. Users sign in to it.
- **Config** — application settings, including the database password, are kept in the
  service's configuration file that ships with the deployment.

## Data flows
- Driver App → Planner Service: all traffic uses TLS 1.3.
- Planner Service → database: internal connection.

## Operations
- Deployment and hosting details are still to be decided.

## Notes
- This is a draft; classification, hosting region, redundancy, backups and monitoring are
  not yet described and will be added in a later revision.
