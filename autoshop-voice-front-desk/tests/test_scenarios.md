# Test Scenarios & Acceptance Criteria

## Test Matrix

### TC-01: New Appointment — Happy Path

**Scenario:** New caller requests an appointment during business hours.

**Input:**
- Caller: new (not in DB)
- Time: within business hours
- Intent: new_appointment
- Vehicle: 2019 Toyota Camry
- Issue: Check engine light, rough idle
- Drivable: yes
- Preferred: Thursday morning

**Expected Results:**
- [ ] Pre-call webhook returns shop dynamic variables with `customer_name = ""`
- [ ] Pre-call webhook returns `returning_customer = "no"`
- [ ] Agent greets with generic greeting (no name)
- [ ] Agent asks for name, vehicle, issue, drivable status, preferred time
- [ ] Agent confirms details before ending
- [ ] Post-call creates new `callers` record
- [ ] Post-call creates new `vehicles` record (2019 Toyota Camry)
- [ ] Post-call creates `calls` record with `intent = 'new_appointment'`
- [ ] Post-call creates `appointment_requests` record with status `requested`
- [ ] Post-call creates `tasks` record with type `appointment_confirm`
- [ ] SMS sent to caller with booking confirmation template
- [ ] SMS logged in `messages` table

### TC-02: Returning Customer — Appointment

**Scenario:** Known caller (already in DB) requests a new appointment.

**Input:**
- Caller: existing (phone +15559876543, name "John Smith")
- Time: within business hours

**Expected Results:**
- [ ] Pre-call webhook returns `customer_name = "John Smith"`
- [ ] Pre-call webhook returns `returning_customer = "yes"`
- [ ] Agent greets by name: "Hi John, thanks for calling..."
- [ ] Post-call updates existing `callers` record (not duplicate)
- [ ] New vehicle record if different vehicle

### TC-03: Existing Vehicle Status

**Scenario:** Customer calls to check on a vehicle already in the shop.

**Input:**
- Intent: existing_status
- Shop is open, transfer number available

**Expected Results:**
- [ ] Agent verifies name and vehicle
- [ ] Agent does NOT invent status information
- [ ] Agent attempts transfer to `transfer_number`
- [ ] `calls.intent = 'existing_status'`
- [ ] `tasks` record created with type `status_followup`
- [ ] SMS sent with status_request_received template

### TC-04: Existing Vehicle Status — Transfer Unavailable

**Scenario:** Status call but shop is closed or no transfer number.

**Expected Results:**
- [ ] Agent says advisor will call back
- [ ] Callback task created with appropriate priority
- [ ] No transfer attempted

### TC-05: Price Question — Diagnostic Fee

**Scenario:** Caller asks how much a diagnostic costs.

**Expected Results:**
- [ ] Agent quotes `diag_fee_text` from shop config
- [ ] Agent steers toward booking a diagnostic appointment
- [ ] Agent captures vehicle info if caller is willing
- [ ] `calls.intent = 'price_question'`

### TC-06: Price Question — Specific Repair

**Scenario:** Caller asks "How much to replace brake pads?"

**Expected Results:**
- [ ] Agent does NOT quote a specific price
- [ ] Agent says something like "Costs depend on what we find during diagnostic"
- [ ] Agent offers to schedule a diagnostic

### TC-07: Urgent Breakdown — Brakes

**Scenario:** Caller reports brake failure.

**Expected Results:**
- [ ] Agent detects urgency keyword "brakes not working"
- [ ] Agent acknowledges urgency
- [ ] Agent asks if caller is safe
- [ ] If shop open + transfer available: agent transfers
- [ ] If not: urgent callback task created
- [ ] `calls.urgency = 'urgent'`
- [ ] `tasks.priority = 'urgent'`
- [ ] `tasks.due_at` is 15 minutes from now
- [ ] Urgent SMS sent to caller
- [ ] Staff alert SMS sent to shop owner/advisor

### TC-08: Urgent Breakdown — Overheating

**Scenario:** Caller says car is overheating on the road.

**Expected Results:**
- [ ] Same as TC-07 with overheating keywords
- [ ] Agent asks about location for tow

### TC-09: Tow Request

**Scenario:** Caller needs a tow.

**Expected Results:**
- [ ] Agent asks for location
- [ ] Agent asks for vehicle info
- [ ] Agent quotes tow policy from config
- [ ] `calls.intent = 'tow_request'`
- [ ] Urgent callback task created
- [ ] `tasks.task_type = 'urgent_callback'`

### TC-10: After-Hours Call — New Appointment

**Scenario:** Caller calls at 8 PM (shop closes at 5 PM).

**Expected Results:**
- [ ] Pre-call webhook returns `shop_is_open = "false"`, `transfer_available = "false"`
- [ ] Agent informs caller shop is closed, gives next open hours
- [ ] Agent still captures intake (name, vehicle, issue)
- [ ] Callback task created (not appointment confirm)
- [ ] SMS sent with after_hours_intake_confirmation template
- [ ] `calls.after_hours = true`

### TC-11: After-Hours — Urgent

**Scenario:** Urgent call after hours.

**Expected Results:**
- [ ] Agent captures details, marks urgent
- [ ] Urgent callback task created
- [ ] Staff alert sent even though after hours
- [ ] No transfer attempted (shop closed)

### TC-12: Vendor/Sales Call

**Scenario:** Caller is a vendor trying to sell parts/services.

**Expected Results:**
- [ ] Agent politely declines
- [ ] `calls.intent = 'vendor_sales'`
- [ ] No task created
- [ ] No SMS sent
- [ ] No appointment created

### TC-13: Wrong Number

**Scenario:** Caller didn't intend to call the shop.

**Expected Results:**
- [ ] Agent identifies it's a wrong number
- [ ] `calls.intent = 'wrong_number'`
- [ ] No task or SMS

### TC-14: Angry Customer

**Scenario:** Customer is upset about a previous repair.

**Expected Results:**
- [ ] Agent stays calm, empathetic
- [ ] Agent does NOT argue
- [ ] Callback task created with `priority = 'high'`
- [ ] Staff alert triggered
- [ ] If caller demands human: transfer attempted

### TC-15: Blocked Caller

**Scenario:** Caller's phone is in the `callers` table with `blocked_flag = true`.

**Expected Results:**
- [ ] Pre-call webhook returns 403 with `reject_call: true`
- [ ] Call is not connected to the agent

### TC-16: Unknown Shop Number

**Scenario:** Call comes in on a number not in the `shops` table.

**Expected Results:**
- [ ] Pre-call webhook returns 404
- [ ] Retell handles gracefully (fallback agent or error message)

### TC-17: Same-Day Appointment Request

**Scenario:** Caller wants to come in today.

**Expected Results:**
- [ ] `appointment_requests.requested_day` indicates today
- [ ] `tasks.priority = 'high'` (same-day = high priority)

### TC-18: Callback SLA — Urgent Overdue

**Scenario:** Urgent task is 20 minutes past `due_at`, still open.

**Expected Results:**
- [ ] SLA manager picks up the task
- [ ] Escalation SMS sent to shop phone
- [ ] `tasks.escalation_count` incremented
- [ ] `tasks.escalated_at` updated
- [ ] Escalation message logged in `messages`

### TC-19: Callback SLA — Multiple Escalations

**Scenario:** Same task overdue through 3 SLA cycles.

**Expected Results:**
- [ ] `escalation_count` reaches 3
- [ ] 3 escalation messages in `messages` table
- [ ] Task remains open until manually resolved

### TC-20: Shop Config Sync

**Scenario:** Shop updates their hours and diagnostic fee in the database.

**Expected Results:**
- [ ] POST to `/webhook/shop-config-sync` with `shop_id`
- [ ] Retell agent updated via API
- [ ] `shops.retell_agent_version` updated
- [ ] `shops.updated_at` updated

## Acceptance Criteria Summary

The system passes acceptance if ALL of the following are true:

1. **Routing**: Inbound webhook correctly identifies shop by DID and returns appropriate dynamic variables.
2. **Caller Recognition**: Returning callers are identified by phone number; new callers are created post-call.
3. **Structured Intake**: Every completed call produces a `calls` record with extracted intent, urgency, and summary.
4. **Vehicle Capture**: When vehicle info is provided, a `vehicles` record is created and linked.
5. **Appointment Flow**: `new_appointment` calls create both an `appointment_requests` record and a confirmation task.
6. **Urgent Handling**: Urgent calls create urgent tasks with 15-minute SLA, trigger staff alerts, and send urgent SMS.
7. **SMS Recaps**: Appropriate SMS template is sent for each intent type; SMS is logged in `messages`.
8. **SLA Escalation**: `callback_sla_manager` finds overdue tasks every 5 minutes and sends escalation alerts.
9. **After-Hours**: After-hours calls are handled gracefully — intake is captured, callback tasks created, no transfers attempted.
10. **Blocked Callers**: Blocked callers are rejected at the pre-call webhook level.
11. **Agent Guardrails**: Agent never quotes unauthorized prices, never invents status, never argues.
12. **Multi-Shop**: Schema and workflows support multiple shops operating independently on the same system.
13. **Config Sync**: Shop config changes can be pushed to the Retell agent via the sync workflow.
