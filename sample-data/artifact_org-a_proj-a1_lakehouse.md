# Architecture Design — Aurora Data Lakehouse

**Organisation:** Aurora Payments (org-a) · **Project:** proj-a1 · **Codename:** LAKESIDE
**Criticality:** business-critical · **Handles personal data:** yes

## Overview
The Data Lakehouse consolidates transaction and customer records for analytics. It holds
customer names, emails and postal addresses used for marketing segmentation.

## Components
- **Ingestion Service** — pulls records nightly from the transaction platform's published
  API. Authenticated via enterprise SSO and OAuth tokens.
- **Lake Store** — holds the consolidated records. Classification: confidential. Data is
  stored in plaintext on disk; encryption at rest has not been enabled. A copy of the
  personal-data tables is replicated to a US-East region for a data-science team.
- **Query Gateway** — internal API used by analysts. Sits behind the enterprise WAF and
  enforces authentication.

## Data flows
- Ingestion Service → transaction platform API: TLS 1.3.
- Analyst → Query Gateway: TLS 1.3.

## Resilience
- The Lake Store and Query Gateway are deployed across two availability zones in the
  approved EU region.
- A nightly backup is documented with a 12-hour recovery objective.

## Operations
- Each service writes its logs to local files on its own host. There is no centralised
  log or metric collection.
- Secrets are held in the approved secrets manager.

## Notes
- The Lake Store runs on Mongoose-DB, an open-source engine that is not on the approved
  technology catalogue, chosen by the data team for its query speed.
