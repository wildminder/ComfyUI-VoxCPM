# SMS Templates

## 1. Booking Request Captured

**Trigger:** `new_appointment` intent processed successfully

```
Hi {{customer_name}}, thanks for calling {{shop_name}}! We've received your appointment request for your {{year}} {{make}} {{model}}. We'll confirm your time shortly. If you need to reach us, call {{shop_number}}.
```

**Example:**
> Hi John, thanks for calling Mike's Auto Repair! We've received your appointment request for your 2019 Toyota Camry. We'll confirm your time shortly. If you need to reach us, call (555) 123-4567.

---

## 2. Urgent Callback

**Trigger:** `urgent_breakdown` or `tow_request` intent, or urgency = `urgent`

```
Hi {{customer_name}}, we received your call about an urgent issue with your {{year}} {{make}} {{model}}. A team member will call you back as soon as possible. If this is a life-threatening emergency, please call 911.
```

**Example:**
> Hi Sarah, we received your call about an urgent issue with your 2017 Honda Civic. A team member will call you back as soon as possible. If this is a life-threatening emergency, please call 911.

---

## 3. Status Request Received

**Trigger:** `existing_status` intent

```
Hi {{customer_name}}, thanks for calling {{shop_name}}. We've noted your request for a status update on your {{year}} {{make}} {{model}}. A service advisor will reach out to you shortly.
```

**Example:**
> Hi Mike, thanks for calling Mike's Auto Repair. We've noted your request for a status update on your 2020 Ford F-150. A service advisor will reach out to you shortly.

---

## 4. Hours & Address Reply

**Trigger:** `hours_address` intent

```
{{shop_name}} is located at {{address}}. Our hours today: {{hours_today}}. We look forward to seeing you!
```

**Example:**
> Mike's Auto Repair is located at 123 Main St, Springfield, IL 62701. Our hours today: Open 8:00 AM – 5:00 PM. We look forward to seeing you!

---

## 5. Missed Call Follow-Up

**Trigger:** Call disconnected before completion or no intake captured

```
Hi, we noticed you called {{shop_name}} but we weren't able to connect. How can we help? Reply to this message or call us back at {{shop_number}}. We're here {{hours_today}}.
```

**Example:**
> Hi, we noticed you called Mike's Auto Repair but we weren't able to connect. How can we help? Reply to this message or call us back at (555) 123-4567. We're here Open 8:00 AM – 5:00 PM.

---

## 6. After-Hours Intake Confirmation

**Trigger:** Any call processed while `shop_is_open = false`

```
Hi {{customer_name}}, thanks for calling {{shop_name}} after hours. We've captured your information and someone will reach out when the shop opens. Our next open hours: {{next_open_hours}}. Thanks for your patience!
```

**Example:**
> Hi David, thanks for calling Mike's Auto Repair after hours. We've captured your information and someone will reach out when the shop opens. Our next open hours: Monday 8:00 AM – 5:00 PM. Thanks for your patience!

---

## Template Variables

| Variable           | Source                        |
| ------------------ | ----------------------------- |
| `{{customer_name}}`| `callers.name` or "there"     |
| `{{shop_name}}`    | `shops.name`                  |
| `{{shop_number}}`  | `shops.main_number` formatted |
| `{{address}}`      | `shops.address`               |
| `{{hours_today}}`  | Computed from `hours_json`    |
| `{{next_open_hours}}`| Computed from `hours_json`  |
| `{{year}}`         | Extracted vehicle year        |
| `{{make}}`         | Extracted vehicle make        |
| `{{model}}`        | Extracted vehicle model       |

## Notes

- If `customer_name` is empty, substitute "there" (e.g., "Hi there, thanks for calling...").
- If vehicle info is incomplete, omit the vehicle portion rather than sending blanks.
- All SMS should be under 320 characters (2 SMS segments) to avoid carrier issues.
- SMS are logged in the `messages` table with `message_type = 'sms_outbound'`.
