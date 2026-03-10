-- Email Campaign System for 100K Auto Shop Outreach
-- Run after the main database_schema.sql

-- ============================================================
-- CAMPAIGN ENUMS
-- ============================================================
CREATE TYPE lead_status AS ENUM (
  'new',
  'contacted',
  'opened',
  'clicked',
  'replied',
  'converted',
  'unsubscribed',
  'bounced'
);

CREATE TYPE campaign_status AS ENUM (
  'draft',
  'scheduled',
  'sending',
  'paused',
  'completed'
);

-- ============================================================
-- LEADS (your 100K email list)
-- ============================================================
CREATE TABLE leads (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email           TEXT NOT NULL,
  shop_name       TEXT,
  owner_name      TEXT,
  phone           TEXT,
  city            TEXT,
  state           TEXT,
  zip             TEXT,
  source          TEXT DEFAULT 'import',          -- 'import', 'website', 'referral'
  status          lead_status NOT NULL DEFAULT 'new',
  tags            TEXT[] DEFAULT ARRAY[]::text[],
  -- Campaign tracking
  last_campaign_id UUID,
  sent_count      INT DEFAULT 0,
  last_sent_at    TIMESTAMPTZ,
  last_opened_at  TIMESTAMPTZ,
  last_clicked_at TIMESTAMPTZ,
  replied_at      TIMESTAMPTZ,
  converted_at    TIMESTAMPTZ,
  converted_shop_id UUID REFERENCES shops(id),    -- links to shop if they sign up
  -- Compliance
  unsubscribed_at TIMESTAMPTZ,
  bounce_count    INT DEFAULT 0,
  last_bounce_at  TIMESTAMPTZ,
  -- Meta
  custom_data     JSONB DEFAULT '{}'::jsonb,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_leads_email ON leads(email);
CREATE INDEX idx_leads_status ON leads(status);
CREATE INDEX idx_leads_state ON leads(state);
CREATE INDEX idx_leads_source ON leads(source);
CREATE INDEX idx_leads_last_sent ON leads(last_sent_at);

-- ============================================================
-- CAMPAIGNS
-- ============================================================
CREATE TABLE campaigns (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name            TEXT NOT NULL,
  subject_line    TEXT NOT NULL,
  subject_line_b  TEXT,                           -- A/B test variant
  preview_text    TEXT,
  template_id     TEXT NOT NULL,                  -- references EMAIL_TEMPLATES key
  status          campaign_status NOT NULL DEFAULT 'draft',
  -- Targeting
  target_states   TEXT[],                         -- filter leads by state
  target_tags     TEXT[],                         -- filter leads by tags
  exclude_statuses lead_status[] DEFAULT ARRAY['unsubscribed', 'bounced', 'converted']::lead_status[],
  -- Send config
  send_rate       INT DEFAULT 100,                -- emails per hour
  batch_size      INT DEFAULT 50,                 -- emails per batch
  -- Stats
  total_targeted  INT DEFAULT 0,
  sent            INT DEFAULT 0,
  delivered       INT DEFAULT 0,
  opened          INT DEFAULT 0,
  clicked         INT DEFAULT 0,
  replied         INT DEFAULT 0,
  converted       INT DEFAULT 0,
  bounced         INT DEFAULT 0,
  unsubscribed    INT DEFAULT 0,
  -- Timing
  scheduled_at    TIMESTAMPTZ,
  started_at      TIMESTAMPTZ,
  completed_at    TIMESTAMPTZ,
  paused_at       TIMESTAMPTZ,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_campaigns_status ON campaigns(status);

-- ============================================================
-- CAMPAIGN SENDS (individual send tracking)
-- ============================================================
CREATE TABLE campaign_sends (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  campaign_id     UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
  lead_id         UUID NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
  email           TEXT NOT NULL,
  subject_variant TEXT DEFAULT 'a',               -- 'a' or 'b' for A/B testing
  email_provider_id TEXT,                         -- ID from Resend/SendGrid
  status          TEXT DEFAULT 'pending',         -- 'pending', 'sent', 'delivered', 'opened', 'clicked', 'bounced', 'complained'
  sent_at         TIMESTAMPTZ,
  delivered_at    TIMESTAMPTZ,
  opened_at       TIMESTAMPTZ,
  clicked_at      TIMESTAMPTZ,
  bounced_at      TIMESTAMPTZ,
  error_message   TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_sends_campaign ON campaign_sends(campaign_id);
CREATE INDEX idx_sends_lead ON campaign_sends(lead_id);
CREATE INDEX idx_sends_status ON campaign_sends(status);

-- ============================================================
-- UNSUBSCRIBE LIST (compliance — CAN-SPAM)
-- ============================================================
CREATE TABLE email_unsubscribes (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email           TEXT NOT NULL UNIQUE,
  reason          TEXT,
  unsubscribed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_unsub_email ON email_unsubscribes(email);

-- ============================================================
-- TRIGGER: updated_at
-- ============================================================
CREATE TRIGGER trg_leads_updated_at
  BEFORE UPDATE ON leads FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_campaigns_updated_at
  BEFORE UPDATE ON campaigns FOR EACH ROW EXECUTE FUNCTION update_updated_at();
