# Dynamic Variables Specification

## Overview

Dynamic variables are injected into the Retell agent at the start of each call via the pre-call webhook response from `retell_inbound_router`. These variables are used in the agent prompt to personalize the conversation.

## Variable Reference

| Variable              | Type    | Source           | Description                                                | Example                                                |
| --------------------- | ------- | ---------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| `shop_name`           | string  | `shops.name`     | Shop business name                                         | `"Mike's Auto Repair"`                                 |
| `address`             | string  | `shops.address`  | Full street address                                        | `"123 Main St, Springfield, IL 62701"`                 |
| `hours_today`         | string  | Computed         | Human-readable hours for today based on `shops.hours_json` | `"Open today 8:00 AM – 5:00 PM"`                      |
| `diag_fee`            | string  | `shops.diag_fee_text` | Diagnostic fee quote text                             | `"Our diagnostic fee starts at $125."`                 |
| `services`            | string  | `shops.services_text` | Comma-separated list of services                      | `"General repair, brakes, AC, suspension, diagnostics"`|
| `makes_serviced`      | string  | `shops.makes_serviced_text` | Makes the shop works on                          | `"All makes and models"`                               |
| `tow_policy`          | string  | `shops.tow_policy_text` | Tow policy explanation                               | `"We can arrange a tow. Let us know your location."`   |
| `customer_name`       | string  | `callers.name`   | Known customer name, or empty string                       | `"John"` or `""`                                       |
| `caller_phone`        | string  | Webhook payload  | Caller's phone number in E.164                             | `"+15559876543"`                                       |
| `returning_customer`  | string  | Computed         | `"yes"` if caller exists in DB, `"no"` otherwise           | `"yes"`                                                |
| `shop_is_open`        | string  | Computed         | `"true"` or `"false"` based on current time vs hours_json  | `"true"`                                               |
| `transfer_available`  | string  | Computed         | `"true"` if shop is open AND transfer_number exists        | `"true"`                                               |
| `transfer_number`     | string  | `shops.transfer_number` | Phone number for live transfers                      | `"+15559876543"`                                       |

## Computation Logic

### `hours_today`

```javascript
// Pseudocode for n8n Function node
const now = DateTime.now().setZone(shop.timezone);
const dayName = now.toFormat('EEEE').toLowerCase(); // e.g. "monday"
const todayHours = shop.hours_json[dayName];

if (!todayHours) {
  return "Closed today";
}

// Format: "Open today 8:00 AM – 5:00 PM"
return `Open today ${formatTime(todayHours.open)} – ${formatTime(todayHours.close)}`;
```

### `shop_is_open`

```javascript
const now = DateTime.now().setZone(shop.timezone);
const dayName = now.toFormat('EEEE').toLowerCase();
const todayHours = shop.hours_json[dayName];

if (!todayHours) return "false";

const currentTime = now.toFormat('HH:mm');
return (currentTime >= todayHours.open && currentTime < todayHours.close) ? "true" : "false";
```

### `transfer_available`

```javascript
return (shop_is_open === "true" && shop.transfer_number) ? "true" : "false";
```

### `returning_customer`

```javascript
// Caller lookup result from DB
return (caller !== null && caller.name) ? "yes" : "no";
```

## Retell Webhook Response Format

The `retell_inbound_router` webhook must return this JSON structure:

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

## Notes

- All variables are strings. Retell template syntax `{{variable_name}}` does not support non-string types.
- Empty string `""` is used for unknown/unavailable values, not `null`.
- `override_agent_id` can be set to route to a different Retell agent (e.g., a Spanish-language agent) based on shop config or caller preference.
- Variables are read-only during the call. Post-call data is extracted from the transcript, not from modified variables.
