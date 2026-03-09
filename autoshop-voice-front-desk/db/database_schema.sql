-- AutoShop Voice Front Desk — Database Schema
-- Target: Postgres 14+ / Supabase

-- ============================================================
-- EXTENSIONS
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- ENUMS
-- ============================================================
CREATE TYPE call_intent AS ENUM (
  'new_appointment',
  'existing_status',
  'price_question',
  'hours_address',
  'urgent_breakdown',
  'tow_request',
  'vendor_sales',
  'wrong_number',
  'angry_customer',
  'unknown'
);

CREATE TYPE urgency_level AS ENUM (
  'low',
  'medium',
  'high',
  'urgent'
);

CREATE TYPE task_type AS ENUM (
  'callback',
  'urgent_callback',
  'appointment_confirm',
  'status_followup',
  'tow_dispatch',
  'escalation'
);

CREATE TYPE task_status AS ENUM (
  'open',
  'in_progress',
  'completed',
  'cancelled'
);

CREATE TYPE task_priority AS ENUM (
  'low',
  'medium',
  'high',
  'urgent'
);

CREATE TYPE appointment_status AS ENUM (
  'requested',
  'confirmed',
  'cancelled',
  'completed',
  'no_show'
);

CREATE TYPE message_type AS ENUM (
  'sms_outbound',
  'sms_inbound',
  'email_outbound',
  'slack_alert'
);

CREATE TYPE booking_mode AS ENUM (
  'request_only',
  'auto_confirm',
  'disabled'
);

CREATE TYPE sentiment_level AS ENUM (
  'positive',
  'neutral',
  'negative',
  'angry'
);

-- ============================================================
-- SHOPS
-- ============================================================
CREATE TABLE shops (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name            TEXT NOT NULL,
  main_number     TEXT NOT NULL UNIQUE,          -- Twilio DID, E.164 format
  address         TEXT,
  timezone        TEXT NOT NULL DEFAULT 'America/Chicago',
  hours_json      JSONB NOT NULL DEFAULT '{}'::jsonb,
  diag_fee_text   TEXT DEFAULT 'Our diagnostic fee starts at $125.',
  services_text   TEXT DEFAULT 'We offer general repair, brakes, AC, suspension, and diagnostics.',
  makes_serviced_text TEXT DEFAULT 'We service all makes and models.',
  tow_policy_text TEXT DEFAULT 'We can arrange a tow. Please let us know your location.',
  after_hours_enabled BOOLEAN NOT NULL DEFAULT true,
  retell_agent_id TEXT,
  retell_agent_version TEXT,
  transfer_number TEXT,                          -- human fallback number
  booking_mode    booking_mode NOT NULL DEFAULT 'request_only',
  sms_from_number TEXT,                          -- Twilio sender number
  supported_languages TEXT[] DEFAULT ARRAY['en'],
  banned_promises TEXT[] DEFAULT ARRAY[]::text[],
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_shops_main_number ON shops(main_number);

-- ============================================================
-- CALLERS
-- ============================================================
CREATE TABLE callers (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  phone           TEXT NOT NULL UNIQUE,          -- E.164
  name            TEXT,
  last_vehicle_id UUID,                          -- FK set after vehicles table
  vip_flag        BOOLEAN NOT NULL DEFAULT false,
  blocked_flag    BOOLEAN NOT NULL DEFAULT false,
  notes           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_callers_phone ON callers(phone);

-- ============================================================
-- VEHICLES
-- ============================================================
CREATE TABLE vehicles (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  caller_id       UUID NOT NULL REFERENCES callers(id) ON DELETE CASCADE,
  year            INT,
  make            TEXT,
  model           TEXT,
  vin             TEXT,
  plate           TEXT,
  drivable_status TEXT DEFAULT 'unknown',        -- 'yes', 'no', 'unknown'
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_vehicles_caller ON vehicles(caller_id);

-- Add FK for callers.last_vehicle_id
ALTER TABLE callers
  ADD CONSTRAINT fk_callers_last_vehicle
  FOREIGN KEY (last_vehicle_id) REFERENCES vehicles(id)
  ON DELETE SET NULL;

-- ============================================================
-- CALLS
-- ============================================================
CREATE TABLE calls (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  shop_id         UUID NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
  caller_id       UUID REFERENCES callers(id) ON DELETE SET NULL,
  retell_call_id  TEXT,
  from_number     TEXT,
  to_number       TEXT,
  started_at      TIMESTAMPTZ,
  ended_at        TIMESTAMPTZ,
  duration_sec    INT,
  intent          call_intent DEFAULT 'unknown',
  urgency         urgency_level DEFAULT 'low',
  summary         TEXT,
  transcript      TEXT,
  transferred     BOOLEAN NOT NULL DEFAULT false,
  sentiment       sentiment_level DEFAULT 'neutral',
  recording_url   TEXT,
  human_transfer_requested BOOLEAN NOT NULL DEFAULT false,
  vehicle_id      UUID REFERENCES vehicles(id) ON DELETE SET NULL,
  extracted_data  JSONB DEFAULT '{}'::jsonb,      -- raw extracted fields
  after_hours     BOOLEAN NOT NULL DEFAULT false,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_calls_shop ON calls(shop_id);
CREATE INDEX idx_calls_caller ON calls(caller_id);
CREATE INDEX idx_calls_retell ON calls(retell_call_id);
CREATE INDEX idx_calls_intent ON calls(intent);
CREATE INDEX idx_calls_created ON calls(created_at DESC);

-- ============================================================
-- TASKS
-- ============================================================
CREATE TABLE tasks (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  shop_id         UUID NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
  caller_id       UUID REFERENCES callers(id) ON DELETE SET NULL,
  call_id         UUID REFERENCES calls(id) ON DELETE SET NULL,
  task_type       task_type NOT NULL,
  priority        task_priority NOT NULL DEFAULT 'medium',
  assigned_to     TEXT,                          -- staff name or phone
  status          task_status NOT NULL DEFAULT 'open',
  due_at          TIMESTAMPTZ,
  escalation_count INT NOT NULL DEFAULT 0,
  escalated_at    TIMESTAMPTZ,
  notes           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tasks_shop ON tasks(shop_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_due ON tasks(due_at);
CREATE INDEX idx_tasks_priority ON tasks(priority);

-- ============================================================
-- APPOINTMENT REQUESTS
-- ============================================================
CREATE TABLE appointment_requests (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  shop_id         UUID NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
  caller_id       UUID REFERENCES callers(id) ON DELETE SET NULL,
  call_id         UUID REFERENCES calls(id) ON DELETE SET NULL,
  vehicle_id      UUID REFERENCES vehicles(id) ON DELETE SET NULL,
  requested_service TEXT,
  requested_day   TEXT,                          -- e.g. 'Monday', '2025-03-10'
  requested_time  TEXT,                          -- e.g. 'morning', '9:00 AM'
  status          appointment_status NOT NULL DEFAULT 'requested',
  source          TEXT DEFAULT 'voice_intake',
  notes           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_appt_shop ON appointment_requests(shop_id);
CREATE INDEX idx_appt_status ON appointment_requests(status);

-- ============================================================
-- MESSAGES
-- ============================================================
CREATE TABLE messages (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  shop_id         UUID REFERENCES shops(id) ON DELETE SET NULL,
  caller_id       UUID REFERENCES callers(id) ON DELETE SET NULL,
  call_id         UUID REFERENCES calls(id) ON DELETE SET NULL,
  message_type    message_type NOT NULL,
  sms_body        TEXT,
  to_number       TEXT,
  from_number     TEXT,
  twilio_sid      TEXT,
  sent_at         TIMESTAMPTZ DEFAULT now(),
  delivery_status TEXT DEFAULT 'queued',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_messages_caller ON messages(caller_id);
CREATE INDEX idx_messages_call ON messages(call_id);

-- ============================================================
-- UPDATED_AT TRIGGER
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_shops_updated_at
  BEFORE UPDATE ON shops FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_callers_updated_at
  BEFORE UPDATE ON callers FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_vehicles_updated_at
  BEFORE UPDATE ON vehicles FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_tasks_updated_at
  BEFORE UPDATE ON tasks FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_appt_updated_at
  BEFORE UPDATE ON appointment_requests FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- SAMPLE SHOP INSERT (for testing)
-- ============================================================
-- INSERT INTO shops (name, main_number, address, timezone, hours_json, transfer_number, sms_from_number)
-- VALUES (
--   'Mike''s Auto Repair',
--   '+15551234567',
--   '123 Main St, Springfield, IL 62701',
--   'America/Chicago',
--   '{
--     "monday":    {"open": "08:00", "close": "17:00"},
--     "tuesday":   {"open": "08:00", "close": "17:00"},
--     "wednesday": {"open": "08:00", "close": "17:00"},
--     "thursday":  {"open": "08:00", "close": "17:00"},
--     "friday":    {"open": "08:00", "close": "17:00"},
--     "saturday":  {"open": "09:00", "close": "13:00"},
--     "sunday":    null
--   }',
--   '+15559876543',
--   '+15551112222'
-- );
