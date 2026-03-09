# Webhook & API Contracts

## 1. Retell Pre-Call Webhook (Inbound Router)

### Endpoint
```
POST {N8N_BASE_URL}/webhook/retell-inbound
```

### Request (from Retell)
```json
{
  "event": "call_started",
  "call_id": "call_abc123def456",
  "from_number": "+15559876543",
  "to_number": "+15551234567",
  "direction": "inbound",
  "agent_id": "agent_xyz789",
  "metadata": {}
}
```

### Response (to Retell) — Success
```json
{
  "dynamic_variables": {
    "shop_name": "Mike's Auto Repair",
    "address": "123 Main St, Springfield, IL 62701",
    "hours_today": "Open today 8:00 AM – 5:00 PM",
    "diag_fee": "Our diagnostic fee starts at $125.",
    "services": "General repair, brakes, AC, suspension, diagnostics",
    "makes_serviced": "All makes and models",
    "tow_policy": "We can arrange a tow. Let us know your location.",
    "customer_name": "John",
    "caller_phone": "+15559876543",
    "returning_customer": "yes",
    "shop_is_open": "true",
    "transfer_available": "true",
    "transfer_number": "+15559876543"
  },
  "override_agent_id": null
}
```

### Response — Shop Not Found (404)
```json
{
  "error": "shop_not_found",
  "message": "No shop configured for this number"
}
```

### Response — Blocked Caller (403)
```json
{
  "error": "caller_blocked",
  "reject_call": true
}
```

---

## 2. Retell Post-Call Webhook (Post-Call Processor)

### Endpoint
```
POST {N8N_BASE_URL}/webhook/retell-post-call
```

### Request (from Retell)
```json
{
  "event": "call_ended",
  "call_id": "call_abc123def456",
  "from_number": "+15559876543",
  "to_number": "+15551234567",
  "direction": "inbound",
  "agent_id": "agent_xyz789",
  "start_timestamp": "2025-03-10T14:30:00Z",
  "end_timestamp": "2025-03-10T14:35:22Z",
  "duration_ms": 322000,
  "call_status": "ended",
  "transcript": "Agent: Thanks for calling Mike's Auto Repair...\nUser: Hi, I need to schedule...",
  "recording_url": "https://storage.retellai.com/recordings/call_abc123.wav",
  "call_analysis": {
    "intent": "new_appointment",
    "urgency": "low",
    "customer_name": "John Smith",
    "vehicle_year": 2019,
    "vehicle_make": "Toyota",
    "vehicle_model": "Camry",
    "issue_summary": "Check engine light on, rough idle",
    "vehicle_drivable": "yes",
    "preferred_day": "Thursday",
    "preferred_time": "morning",
    "needs_tow": false,
    "human_transfer_requested": false,
    "sentiment": "positive",
    "follow_up_type": "appointment_confirmation"
  },
  "dynamic_variables": {
    "shop_name": "Mike's Auto Repair",
    "caller_phone": "+15559876543",
    "customer_name": "John",
    "returning_customer": "no",
    "shop_is_open": "true"
  }
}
```

### Response
```json
{
  "status": "processed",
  "call_id": "call_abc123def456",
  "actions_taken": [
    "caller_upserted",
    "vehicle_created",
    "call_logged",
    "appointment_request_created",
    "sms_sent"
  ]
}
```

---

## 3. Shop Config Sync Webhook

### Endpoint
```
POST {N8N_BASE_URL}/webhook/shop-config-sync
```

### Request
```json
{
  "shop_id": "uuid-of-shop",
  "trigger": "manual"
}
```

### Response — Success
```json
{
  "success": true,
  "shop_id": "uuid-of-shop",
  "agent_id": "agent_xyz789",
  "synced_at": "2025-03-10T14:30:00Z"
}
```

### Response — Shop Not Found
```json
{
  "success": false,
  "error": "shop_not_found",
  "shop_id": "uuid-of-shop"
}
```

---

## 4. Retell Call Analysis Schema

Configure this in Retell's agent settings under "Post Call Analysis" to have Retell automatically extract structured fields from the transcript.

```json
{
  "intent": {
    "type": "enum",
    "description": "The primary reason for the call",
    "enum": [
      "new_appointment",
      "existing_status",
      "price_question",
      "hours_address",
      "urgent_breakdown",
      "tow_request",
      "vendor_sales",
      "wrong_number",
      "angry_customer"
    ]
  },
  "urgency": {
    "type": "enum",
    "description": "How urgent is this call",
    "enum": ["low", "medium", "high", "urgent"]
  },
  "customer_name": {
    "type": "string",
    "description": "The caller's full name"
  },
  "vehicle_year": {
    "type": "number",
    "description": "The vehicle year"
  },
  "vehicle_make": {
    "type": "string",
    "description": "The vehicle make (e.g. Toyota, Ford)"
  },
  "vehicle_model": {
    "type": "string",
    "description": "The vehicle model (e.g. Camry, F-150)"
  },
  "issue_summary": {
    "type": "string",
    "description": "Brief summary of the reported issue or reason for calling"
  },
  "vehicle_drivable": {
    "type": "enum",
    "description": "Whether the vehicle is currently drivable",
    "enum": ["yes", "no", "unknown"]
  },
  "preferred_day": {
    "type": "string",
    "description": "Caller's preferred day for appointment"
  },
  "preferred_time": {
    "type": "string",
    "description": "Caller's preferred time for appointment"
  },
  "needs_tow": {
    "type": "boolean",
    "description": "Whether the caller needs a tow"
  },
  "human_transfer_requested": {
    "type": "boolean",
    "description": "Whether the caller asked to speak with a human"
  },
  "sentiment": {
    "type": "enum",
    "description": "Overall caller sentiment",
    "enum": ["positive", "neutral", "negative", "angry"]
  },
  "follow_up_type": {
    "type": "enum",
    "description": "What follow-up action is needed",
    "enum": [
      "appointment_confirmation",
      "callback_requested",
      "urgent_callback",
      "status_update",
      "none"
    ]
  }
}
```

---

## 5. Twilio SMS API (Outbound)

Used internally by n8n via the Twilio node. No external API endpoint exposed.

### SMS Send Parameters
| Field  | Source                              |
| ------ | ----------------------------------- |
| `from` | `shops.sms_from_number` or env var  |
| `to`   | `callers.phone`                     |
| `body` | Generated from SMS template         |

### Twilio Webhook (Delivery Status — Optional)
```
POST {N8N_BASE_URL}/webhook/twilio-sms-status
```
```json
{
  "MessageSid": "SM...",
  "MessageStatus": "delivered",
  "To": "+15559876543",
  "From": "+15551112222"
}
```

Used to update `messages.delivery_status` if configured.
