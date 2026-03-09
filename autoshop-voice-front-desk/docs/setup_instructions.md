# Setup Instructions

## Prerequisites

- n8n instance (self-hosted or cloud) with webhook access over HTTPS
- Retell account with at least one agent created
- Twilio account with a phone number provisioned for SMS
- Postgres database (or Supabase project)
- A shop phone number (Twilio DID) pointed at your Retell agent

## Step 1: Database Setup

1. Connect to your Postgres instance.
2. Run the schema file:
   ```bash
   psql $DATABASE_URL -f db/database_schema.sql
   ```
3. Insert your shop's configuration:
   ```sql
   INSERT INTO shops (
     name, main_number, address, timezone, hours_json,
     diag_fee_text, services_text, makes_serviced_text,
     tow_policy_text, transfer_number, sms_from_number,
     retell_agent_id
   ) VALUES (
     'Mike''s Auto Repair',
     '+15551234567',
     '123 Main St, Springfield, IL 62701',
     'America/Chicago',
     '{
       "monday":    {"open": "08:00", "close": "17:00"},
       "tuesday":   {"open": "08:00", "close": "17:00"},
       "wednesday": {"open": "08:00", "close": "17:00"},
       "thursday":  {"open": "08:00", "close": "17:00"},
       "friday":    {"open": "08:00", "close": "17:00"},
       "saturday":  {"open": "09:00", "close": "13:00"},
       "sunday":    null
     }',
     'Our diagnostic fee starts at $125.',
     'General repair, brakes, AC, suspension, diagnostics',
     'All makes and models',
     'We can arrange a tow. Let us know your location.',
     '+15559876543',
     '+15551112222',
     'agent_YOUR_RETELL_AGENT_ID'
   );
   ```

## Step 2: Retell Agent Configuration

1. Log into your Retell dashboard.
2. Create a new agent (or use an existing one).
3. Set the agent's system prompt to the contents of `docs/agent_prompt.md`.
4. Configure **Post Call Analysis** using the schema in `api/webhook_contracts.md` (Section 4).
5. Set the agent's **Pre-Call Webhook URL** to:
   ```
   https://your-n8n.example.com/webhook/retell-inbound
   ```
6. Set the agent's **Post-Call Webhook URL** to:
   ```
   https://your-n8n.example.com/webhook/retell-post-call
   ```
7. Note the agent ID and update the `shops.retell_agent_id` in the database.

## Step 3: Twilio Setup

1. Purchase or assign a phone number in Twilio for the shop.
2. Configure the phone number to forward calls to Retell (follow Retell's Twilio integration docs).
3. Enable SMS on the number (or use a separate number for SMS).
4. Note the SMS-capable number and set it as `sms_from_number` in the shop config.

## Step 4: n8n Workflow Import

1. Open your n8n instance.
2. Import each workflow from the `n8n-workflows/` directory:
   - `retell_inbound_router.json`
   - `retell_post_call_processor.json`
   - `callback_sla_manager.json`
   - `shop_config_sync.json`
3. For each workflow, configure credentials:
   - **Postgres**: Create a Postgres credential with your database connection details.
   - **Twilio**: Create a Twilio credential with your Account SID and Auth Token.
   - **Retell API** (for `shop_config_sync`): Create an HTTP Header Auth credential with your Retell API key (`Authorization: Bearer YOUR_KEY`).
4. Update credential IDs in each workflow's Postgres and Twilio nodes to match your configured credentials.
5. Set environment variables in n8n:
   - `TWILIO_SMS_FROM` — your Twilio SMS number
   - `SHOP_ALERT_PHONE` — fallback alert number for urgent escalations
   - `N8N_BASE_URL` — your n8n instance's base URL
6. Activate all 4 workflows.

## Step 5: Environment Variables

Copy `config/.env.example` to `.env` and fill in all values. These are used by n8n and any supporting services.

## Step 6: Test the System

### Test 1: Pre-Call Webhook
```bash
curl -X POST https://your-n8n.example.com/webhook/retell-inbound \
  -H "Content-Type: application/json" \
  -d '{
    "event": "call_started",
    "call_id": "test_call_001",
    "from_number": "+15559876543",
    "to_number": "+15551234567",
    "direction": "inbound",
    "agent_id": "agent_test"
  }'
```
Expected: 200 response with dynamic variables.

### Test 2: Post-Call Webhook
```bash
curl -X POST https://your-n8n.example.com/webhook/retell-post-call \
  -H "Content-Type: application/json" \
  -d '{
    "event": "call_ended",
    "call_id": "test_call_001",
    "from_number": "+15559876543",
    "to_number": "+15551234567",
    "start_timestamp": "2025-03-10T14:30:00Z",
    "end_timestamp": "2025-03-10T14:35:22Z",
    "duration_ms": 322000,
    "transcript": "Test transcript",
    "call_analysis": {
      "intent": "new_appointment",
      "urgency": "low",
      "customer_name": "Test User",
      "vehicle_year": 2019,
      "vehicle_make": "Toyota",
      "vehicle_model": "Camry",
      "issue_summary": "Check engine light",
      "vehicle_drivable": "yes",
      "preferred_day": "Thursday",
      "preferred_time": "morning",
      "needs_tow": false,
      "human_transfer_requested": false,
      "sentiment": "positive",
      "follow_up_type": "appointment_confirmation"
    },
    "dynamic_variables": {
      "shop_name": "Mike'\''s Auto Repair",
      "caller_phone": "+15559876543",
      "returning_customer": "no",
      "shop_is_open": "true"
    }
  }'
```
Expected: Caller created, vehicle created, call logged, appointment request created, SMS sent.

### Test 3: Config Sync
```bash
curl -X POST https://your-n8n.example.com/webhook/shop-config-sync \
  -H "Content-Type: application/json" \
  -d '{"shop_id": "YOUR_SHOP_UUID"}'
```
Expected: Retell agent updated, sync logged.

## Step 7: Go Live

1. Verify all test calls produce correct DB records.
2. Make a real test call to the shop number.
3. Confirm the voice agent answers, asks the right questions, and ends correctly.
4. Confirm SMS is received after the call.
5. Confirm task is created in the database.
6. Wait 5+ minutes and confirm SLA manager escalates overdue tasks.
7. Monitor the first few real calls and review transcripts.

## Troubleshooting

| Issue | Check |
|-------|-------|
| Webhook returns 404 | Verify n8n workflow is active and webhook path matches |
| No dynamic variables | Check that `shops.main_number` matches the Twilio DID exactly (E.164) |
| SMS not sending | Verify Twilio credentials and that the `from` number is SMS-capable |
| Caller not found | This is expected for new callers — they'll be created post-call |
| Agent doesn't transfer | Check `transfer_number` in shop config and that `shop_is_open` is true |
| SLA not escalating | Verify the cron workflow is active and tasks have `due_at` in the past |
