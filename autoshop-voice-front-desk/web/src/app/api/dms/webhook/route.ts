import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { dmsIntegrations, dmsSyncLog, tasks, appointmentRequests } from "@/lib/schema";
import { eq, and } from "drizzle-orm";
import crypto from "crypto";

/**
 * DMS Webhook Receiver
 *
 * Receives callbacks from Tekmetric, Mitchell 1, or Shop-Ware
 * when entities are updated in their system (e.g., repair order status changes,
 * appointment confirmations, etc.).
 *
 * Signature verification is mandatory for all webhooks.
 */
export async function POST(req: NextRequest) {
  const rawBody = await req.text();
  let body: Record<string, unknown>;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const webhookSecret = req.headers.get("x-webhook-secret");
  const signature = req.headers.get("x-webhook-signature");

  // Find integration — require webhook secret header for lookup
  let integration;
  if (webhookSecret) {
    const [found] = await db
      .select()
      .from(dmsIntegrations)
      .where(and(eq(dmsIntegrations.webhookSecret, webhookSecret), eq(dmsIntegrations.enabled, true)))
      .limit(1);
    integration = found;
  }

  if (!integration) {
    return NextResponse.json({ error: "Unknown integration" }, { status: 404 });
  }

  // Mandatory signature verification
  if (!integration.webhookSecret) {
    console.error(`Integration ${integration.id} missing webhook secret — rejecting`);
    return NextResponse.json({ error: "Webhook not configured" }, { status: 400 });
  }

  if (!signature) {
    return NextResponse.json({ error: "Missing signature" }, { status: 403 });
  }

  const expected = crypto
    .createHmac("sha256", integration.webhookSecret)
    .update(rawBody)
    .digest("hex");

  if (
    !crypto.timingSafeEqual(
      Buffer.from(signature.replace(/^sha256=/, "")),
      Buffer.from(expected)
    )
  ) {
    return NextResponse.json({ error: "Invalid signature" }, { status: 403 });
  }

  // Normalize the webhook event
  const event = normalizeWebhookEvent(integration.provider, body);

  // Log the inbound sync
  await db.insert(dmsSyncLog).values({
    shopId: integration.shopId,
    integrationId: integration.id,
    entityType: event.entityType,
    externalId: event.externalId,
    direction: "inbound",
    status: "synced",
    requestPayload: body,
    responsePayload: event,
  });

  // Process status updates
  if (event.entityType === "repair_order" && event.status) {
    const completedStatuses = ["completed", "closed", "invoiced", "paid"];
    if (completedStatuses.includes(event.status.toLowerCase())) {
      await db
        .update(tasks)
        .set({ status: "completed", updatedAt: new Date() })
        .where(eq(tasks.dmsExternalId, event.externalId));
    }
  }

  if (event.entityType === "appointment" && event.status) {
    const statusMap: Record<string, string> = {
      confirmed: "confirmed",
      cancelled: "cancelled",
      completed: "completed",
      "no-show": "no_show",
      "no_show": "no_show",
    };
    const mappedStatus = statusMap[event.status.toLowerCase()];
    if (mappedStatus) {
      await db
        .update(appointmentRequests)
        .set({ status: mappedStatus, updatedAt: new Date() })
        .where(eq(appointmentRequests.dmsExternalId, event.externalId));
    }
  }

  return NextResponse.json({ status: "processed", event_type: event.entityType });
}

interface NormalizedEvent {
  entityType: string;
  externalId: string;
  status?: string;
  data?: Record<string, unknown>;
}

function normalizeWebhookEvent(provider: string, body: Record<string, unknown>): NormalizedEvent {
  switch (provider) {
    case "tekmetric": {
      const type = String(body.type || "").toLowerCase();
      const data = (body.data || {}) as Record<string, unknown>;
      let entityType = "unknown";
      if (type.includes("repair_order")) entityType = "repair_order";
      else if (type.includes("appointment")) entityType = "appointment";
      else if (type.includes("customer")) entityType = "customer";
      else if (type.includes("vehicle")) entityType = "vehicle";

      return {
        entityType,
        externalId: String(data.id || ""),
        status: String(data.status || (data as Record<string, unknown>).repairOrderStatus || ""),
        data,
      };
    }

    case "mitchell1": {
      const eventType = String(body.eventType || "").toLowerCase();
      let entityType = "unknown";
      if (eventType.includes("job") || eventType.includes("repairorder")) entityType = "repair_order";
      else if (eventType.includes("appointment")) entityType = "appointment";
      else if (eventType.includes("customer")) entityType = "customer";

      return {
        entityType,
        externalId: String(body.jobId || body.appointmentId || body.customerId || ""),
        status: String(body.status || ""),
        data: body as Record<string, unknown>,
      };
    }

    case "shopware": {
      const event = String(body.event || "");
      const payload = (body.payload || {}) as Record<string, unknown>;
      const entityType = event.split(".")[0] || "unknown";

      return {
        entityType: entityType === "repair_order" ? "repair_order" : entityType,
        externalId: String(payload.id || ""),
        status: String(payload.state || payload.status || ""),
        data: payload,
      };
    }

    default:
      return {
        entityType: "unknown",
        externalId: String(body.id || ""),
        status: String(body.status || ""),
        data: body as Record<string, unknown>,
      };
  }
}
