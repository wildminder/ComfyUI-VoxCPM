# AutoShop Voice Front Desk — Architecture

## Assumptions

1. Each shop has a dedicated Retell agent (or agent variant) and a unique inbound DID via Twilio.
2. Retell sends a pre-call webhook before connecting the caller to the agent, and a post-call webhook after the call ends.
3. n8n is self-hosted (or cloud) and exposes webhook URLs over HTTPS.
4. Postgres (or Supabase) is the single source of truth for shops, callers, vehicles, calls, tasks, and messages.
5. Twilio Messaging API is used for all outbound SMS.
6. Shop hours are stored as JSON keyed by day-of-week with open/close times in the shop's local timezone.
7. Booking mode supports `request_only` (default) or `calcom_live` — when Cal.com is configured, the voice agent can check availability, book, cancel, and reschedule in real time via custom tool webhooks.
8. Cal.com integration is optional. Without it, the system creates appointment requests for human confirmation.
9. After-hours calls are still answered by the voice agent, but outcomes route to callback tasks instead of live transfers.
10. Multi-language support is config-driven; V1 defaults to English.

## High-Level Data Flow

```
Inbound Call (Twilio → Retell)
        │
        ▼
┌──────────────────────────┐
│  Workflow 1:             │
│  retell_inbound_router   │◄── Retell pre-call webhook
│                          │
│  • Identify shop by DID  │
│  • Look up caller        │
│  • Determine hours       │
│  • Return dynamic vars   │
│  • Return agent config   │
└──────────┬───────────────┘
           │ Retell webhook response (JSON)
           ▼
    ┌─────────────┐
    │ Retell Agent │  ← Live conversation using agent_prompt
    │ (Voice Call) │    with injected dynamic variables
    └──────┬──────┘
           │ Call ends
           ▼
┌──────────────────────────────┐
│  Workflow 2:                 │
│  retell_post_call_processor  │◄── Retell post-call webhook
│                              │
│  • Extract structured fields │
│  • Upsert caller + vehicle   │
│  • Store call record         │
│  • Classify intent + urgency │
│  • Create task / appt req    │
│  • Send SMS recap            │
│  • Alert staff if urgent     │
└──────────────────────────────┘

┌──────────────────────────┐
│  Workflow 3:             │
│  callback_sla_manager    │◄── Cron trigger (every 5 min)
│                          │
│  • Find overdue tasks    │
│  • Escalate via SMS/     │
│    Slack/email           │
│  • Increment escalation  │
│    count                 │
└──────────────────────────┘

┌──────────────────────────┐
│  Workflow 4:             │
│  shop_config_sync        │◄── Manual trigger / webhook
│                          │
│  • Read shop config      │
│  • Push to Retell agent  │
│  • Update DB             │
└──────────────────────────┘
```

## Component Map

| Component        | Technology       | Role                                       |
| ---------------- | ---------------- | ------------------------------------------ |
| Voice agent      | Retell           | Live conversational AI on inbound calls    |
| Phone/SMS        | Twilio           | Inbound DID routing, outbound SMS          |
| Scheduling       | Cal.com (opt.)   | Real-time booking, cancel, reschedule      |
| Orchestration    | n8n              | 5 workflows: routing, processing, SLA, sync, booking tools|
| Database         | Postgres/Supabase| Shops, callers, vehicles, calls, tasks     |
| Admin UI         | Next.js (opt.)   | Dashboard for shop staff                   |

## Workflow Details

### Workflow 1: retell_inbound_router

**Trigger:** Retell pre-call webhook (`POST /webhook/retell-inbound`)

**Node sequence:**
1. `Webhook` — receive Retell pre-call payload (`call_id`, `from_number`, `to_number`)
2. `Shop Lookup` — query `shops` by `main_number = to_number`
3. `Caller Lookup` — query `callers` by `phone = from_number`
4. `Hours Check` — compare current time (in shop timezone) against `hours_json`
5. `Build Response` — assemble dynamic variables and optional `override_agent_id`
6. `Respond to Webhook` — return JSON to Retell

**Output contract:** See `api/retell_inbound_webhook.md`

### Workflow 2: retell_post_call_processor

**Trigger:** Retell post-call webhook (`POST /webhook/retell-post-call`)

**Node sequence:**
1. `Webhook` — receive Retell post-call payload (transcript, call metadata, custom variables)
2. `Extract Fields` — use structured extraction (LLM or rule-based) to pull intent, urgency, customer details, vehicle info
3. `Upsert Caller` — insert or update `callers` table
4. `Upsert Vehicle` — insert or update `vehicles` table
5. `Insert Call` — write to `calls` table
6. `Route by Intent` — switch node on intent
7. `Create Task or Appointment` — write to `tasks` or `appointment_requests`
8. `Send SMS` — Twilio SMS with appropriate template
9. `Alert Staff` — if urgent, send internal alert via SMS/Slack

### Workflow 3: callback_sla_manager

**Trigger:** Cron every 5 minutes

**Node sequence:**
1. `Cron Trigger`
2. `Query Overdue Tasks` — `SELECT * FROM tasks WHERE status = 'open' AND due_at < NOW() AND priority IN ('urgent','high')`
3. `Loop` — for each overdue task
4. `Escalate` — send SMS/Slack/email to shop owner or assigned advisor
5. `Update Escalation` — increment `escalation_count`, update `escalated_at`

### Workflow 4: shop_config_sync

**Trigger:** Manual or webhook (`POST /webhook/shop-config-sync`)

**Node sequence:**
1. `Trigger` — manual or webhook with `shop_id`
2. `Load Config` — read from `shops` table
3. `Build Agent Config` — map DB fields to Retell agent dynamic variables
4. `Push to Retell` — PATCH Retell agent via API
5. `Log Sync` — record sync timestamp

### Workflow 5: retell_calcom_booking_tools

**Trigger:** Retell custom tool webhooks (4 separate webhook endpoints)

This workflow provides real-time booking capabilities that the Retell voice agent calls as custom tools during a live conversation. Modeled after the AmplifyAutomation/n8n-templates receptionist pattern.

**Endpoints:**
1. `POST /webhook/find-appointment` — look up existing booking by attendee email
2. `POST /webhook/cancel-appointment` — cancel a booking by ID with reason
3. `POST /webhook/reschedule-appointment` — reschedule a booking to a new time
4. `POST /webhook/check-availability` — check Cal.com slot availability for a date range

**Node patterns:**
- Each endpoint uses `respondToWebhook` with success/failure branches
- Cal.com API v2 with Bearer auth and `cal-api-version` header
- `onError: continueErrorOutput` for graceful failure handling
- Date formatting via `dateTime` node for ISO 8601 conversion
- Slot formatting via Code node for voice-friendly output

**When to enable:** Set `booking_mode = 'calcom_live'` in shop config and configure `CALCOM_API_KEY` + `CALCOM_EVENT_TYPE_ID` environment variables.

## Security Considerations

- All webhook endpoints should validate Retell signatures.
- Twilio webhook validation should be enabled.
- Database credentials are stored in n8n credentials, never in workflow JSON.
- SMS sender numbers must be verified in Twilio.
- PII (phone numbers, names) is stored in the database; consider encryption at rest.

## Scaling Notes

- Each shop gets its own row in `shops` with a unique DID → supports multi-tenant from day one.
- n8n workflows are stateless per execution → horizontal scaling is straightforward.
- Retell agent per shop allows independent prompt tuning.
- Cron-based SLA workflow runs globally across all shops.
