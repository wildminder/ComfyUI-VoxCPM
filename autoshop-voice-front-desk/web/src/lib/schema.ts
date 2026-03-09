import {
  pgTable,
  uuid,
  text,
  varchar,
  boolean,
  integer,
  timestamp,
  jsonb,
  pgEnum,
} from "drizzle-orm/pg-core";

// ── Enums ──────────────────────────────────────────────

export const subscriptionStatusEnum = pgEnum("subscription_status", [
  "trialing",
  "active",
  "past_due",
  "canceled",
  "unpaid",
]);

export const onboardingStatusEnum = pgEnum("onboarding_status", [
  "pending",
  "step1_complete",
  "step2_complete",
  "step3_complete",
  "complete",
]);

export const callIntentEnum = pgEnum("call_intent", [
  "new_appointment",
  "existing_status",
  "price_question",
  "hours_address",
  "urgent_breakdown",
  "tow_request",
  "vendor_sales",
  "wrong_number",
  "angry_customer",
  "unknown",
]);

export const urgencyEnum = pgEnum("urgency_level", [
  "low",
  "medium",
  "high",
  "urgent",
]);

export const taskStatusEnum = pgEnum("task_status", [
  "open",
  "in_progress",
  "completed",
  "canceled",
]);

export const taskPriorityEnum = pgEnum("task_priority", [
  "low",
  "medium",
  "high",
  "urgent",
]);

export const dmsProviderEnum = pgEnum("dms_provider", [
  "tekmetric",
  "mitchell1",
  "shopware",
  "none",
]);

export const dmsSyncStatusEnum = pgEnum("dms_sync_status", [
  "pending",
  "synced",
  "failed",
  "skipped",
]);

// ── Users & Auth ───────────────────────────────────────

export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  name: varchar("name", { length: 255 }),
  role: varchar("role", { length: 50 }).notNull().default("owner"),
  shopId: uuid("shop_id").references(() => shops.id),
  emailVerified: boolean("email_verified").default(false),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// ── Shops ──────────────────────────────────────────────

export const shops = pgTable("shops", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: varchar("name", { length: 255 }).notNull(),
  slug: varchar("slug", { length: 100 }).unique(),
  mainNumber: varchar("main_number", { length: 20 }),
  address: text("address"),
  city: varchar("city", { length: 100 }),
  state: varchar("state", { length: 50 }),
  zip: varchar("zip", { length: 20 }),
  timezone: varchar("timezone", { length: 50 }).default("America/Chicago"),
  hoursJson: jsonb("hours_json").default({}),
  diagFeeText: text("diag_fee_text"),
  servicesText: text("services_text"),
  makesServicedText: text("makes_serviced_text"),
  towPolicyText: text("tow_policy_text"),
  afterHoursEnabled: boolean("after_hours_enabled").default(true),
  retellAgentId: varchar("retell_agent_id", { length: 100 }),
  retellAgentVersion: varchar("retell_agent_version", { length: 50 }),
  transferNumber: varchar("transfer_number", { length: 20 }),
  bookingMode: varchar("booking_mode", { length: 20 }).default("request_only"),
  smsFromNumber: varchar("sms_from_number", { length: 20 }),
  supportedLanguages: jsonb("supported_languages").default(["en"]),
  onboardingStatus: onboardingStatusEnum("onboarding_status").default("pending"),
  stripeCustomerId: varchar("stripe_customer_id", { length: 100 }),
  stripeSubscriptionId: varchar("stripe_subscription_id", { length: 100 }),
  subscriptionStatus: subscriptionStatusEnum("subscription_status"),
  planId: varchar("plan_id", { length: 50 }),
  trialEndsAt: timestamp("trial_ends_at"),
  provisionedAt: timestamp("provisioned_at"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// ── DMS Integrations ──────────────────────────────────

export const dmsIntegrations = pgTable("dms_integrations", {
  id: uuid("id").primaryKey().defaultRandom(),
  shopId: uuid("shop_id").references(() => shops.id).notNull(),
  provider: dmsProviderEnum("provider").notNull(),
  apiKey: text("api_key"),
  apiUrl: text("api_url"),
  shopExternalId: text("shop_external_id"),
  oauthToken: text("oauth_token"),
  oauthRefresh: text("oauth_refresh"),
  oauthExpiresAt: timestamp("oauth_expires_at"),
  webhookSecret: text("webhook_secret"),
  enabled: boolean("enabled").default(true),
  syncCustomers: boolean("sync_customers").default(true),
  syncVehicles: boolean("sync_vehicles").default(true),
  syncRepairOrders: boolean("sync_repair_orders").default(true),
  syncAppointments: boolean("sync_appointments").default(true),
  lastSyncAt: timestamp("last_sync_at"),
  lastSyncError: text("last_sync_error"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// ── DMS Sync Log ──────────────────────────────────────

export const dmsSyncLog = pgTable("dms_sync_log", {
  id: uuid("id").primaryKey().defaultRandom(),
  shopId: uuid("shop_id").references(() => shops.id).notNull(),
  integrationId: uuid("integration_id").references(() => dmsIntegrations.id).notNull(),
  entityType: varchar("entity_type", { length: 50 }).notNull(),
  entityId: uuid("entity_id"),
  externalId: text("external_id"),
  direction: varchar("direction", { length: 20 }).default("outbound"),
  status: dmsSyncStatusEnum("status").default("pending"),
  requestPayload: jsonb("request_payload"),
  responsePayload: jsonb("response_payload"),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow(),
});

// ── Callers ────────────────────────────────────────────

export const callers = pgTable("callers", {
  id: uuid("id").primaryKey().defaultRandom(),
  phone: varchar("phone", { length: 20 }).notNull().unique(),
  name: varchar("name", { length: 255 }),
  lastVehicleId: uuid("last_vehicle_id"),
  vipFlag: boolean("vip_flag").default(false),
  blockedFlag: boolean("blocked_flag").default(false),
  notes: text("notes"),
  dmsExternalId: text("dms_external_id"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// ── Vehicles ───────────────────────────────────────────

export const vehicles = pgTable("vehicles", {
  id: uuid("id").primaryKey().defaultRandom(),
  callerId: uuid("caller_id").references(() => callers.id),
  year: integer("year"),
  make: varchar("make", { length: 100 }),
  model: varchar("model", { length: 100 }),
  vin: varchar("vin", { length: 20 }),
  plate: varchar("plate", { length: 20 }),
  drivableStatus: varchar("drivable_status", { length: 20 }).default("unknown"),
  dmsExternalId: text("dms_external_id"),
  createdAt: timestamp("created_at").defaultNow(),
});

// ── Calls ──────────────────────────────────────────────

export const calls = pgTable("calls", {
  id: uuid("id").primaryKey().defaultRandom(),
  shopId: uuid("shop_id").references(() => shops.id),
  callerId: uuid("caller_id").references(() => callers.id),
  retellCallId: varchar("retell_call_id", { length: 100 }),
  fromNumber: varchar("from_number", { length: 20 }),
  toNumber: varchar("to_number", { length: 20 }),
  startedAt: timestamp("started_at"),
  endedAt: timestamp("ended_at"),
  durationSec: integer("duration_sec"),
  intent: callIntentEnum("intent").default("unknown"),
  urgency: urgencyEnum("urgency").default("low"),
  summary: text("summary"),
  transcript: text("transcript"),
  transferred: boolean("transferred").default(false),
  sentiment: varchar("sentiment", { length: 20 }),
  recordingUrl: text("recording_url"),
  humanTransferRequested: boolean("human_transfer_requested").default(false),
  afterHours: boolean("after_hours").default(false),
  extractedData: jsonb("extracted_data"),
  createdAt: timestamp("created_at").defaultNow(),
});

// ── Tasks ──────────────────────────────────────────────

export const tasks = pgTable("tasks", {
  id: uuid("id").primaryKey().defaultRandom(),
  shopId: uuid("shop_id").references(() => shops.id),
  callerId: uuid("caller_id").references(() => callers.id),
  callId: uuid("call_id").references(() => calls.id),
  taskType: varchar("task_type", { length: 50 }),
  priority: taskPriorityEnum("priority").default("medium"),
  assignedTo: varchar("assigned_to", { length: 255 }),
  status: taskStatusEnum("status").default("open"),
  dueAt: timestamp("due_at"),
  escalationCount: integer("escalation_count").default(0),
  escalatedAt: timestamp("escalated_at"),
  notes: text("notes"),
  dmsExternalId: text("dms_external_id"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// ── Appointment Requests ───────────────────────────────

export const appointmentRequests = pgTable("appointment_requests", {
  id: uuid("id").primaryKey().defaultRandom(),
  shopId: uuid("shop_id").references(() => shops.id),
  callerId: uuid("caller_id").references(() => callers.id),
  callId: uuid("call_id").references(() => calls.id),
  vehicleId: uuid("vehicle_id").references(() => vehicles.id),
  requestedService: text("requested_service"),
  requestedDay: varchar("requested_day", { length: 50 }),
  requestedTime: varchar("requested_time", { length: 50 }),
  status: varchar("status", { length: 20 }).default("requested"),
  source: varchar("source", { length: 50 }).default("voice_intake"),
  dmsExternalId: text("dms_external_id"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// ── Messages ───────────────────────────────────────────

export const messages = pgTable("messages", {
  id: uuid("id").primaryKey().defaultRandom(),
  shopId: uuid("shop_id").references(() => shops.id),
  callerId: uuid("caller_id").references(() => callers.id),
  callId: uuid("call_id").references(() => calls.id),
  messageType: varchar("message_type", { length: 50 }),
  smsBody: text("sms_body"),
  toNumber: varchar("to_number", { length: 20 }),
  fromNumber: varchar("from_number", { length: 20 }),
  twilioSid: varchar("twilio_sid", { length: 100 }),
  deliveryStatus: varchar("delivery_status", { length: 20 }),
  sentAt: timestamp("sent_at").defaultNow(),
});
