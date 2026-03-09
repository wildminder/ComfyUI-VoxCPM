import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { dmsIntegrations } from "@/lib/schema";
import { eq } from "drizzle-orm";
import { z } from "zod";

export async function GET() {
  const session = await getSession();
  if (!session || !session.user.shopId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const [integration] = await db
    .select({
      id: dmsIntegrations.id,
      provider: dmsIntegrations.provider,
      shopExternalId: dmsIntegrations.shopExternalId,
      enabled: dmsIntegrations.enabled,
      syncCustomers: dmsIntegrations.syncCustomers,
      syncVehicles: dmsIntegrations.syncVehicles,
      syncRepairOrders: dmsIntegrations.syncRepairOrders,
      syncAppointments: dmsIntegrations.syncAppointments,
      lastSyncAt: dmsIntegrations.lastSyncAt,
      lastSyncError: dmsIntegrations.lastSyncError,
    })
    .from(dmsIntegrations)
    .where(eq(dmsIntegrations.shopId, session.user.shopId))
    .limit(1);

  return NextResponse.json({ integration: integration || null });
}

const updateSchema = z.object({
  provider: z.enum(["tekmetric", "mitchell1", "shopware", "none"]).optional(),
  apiKey: z.string().optional(),
  apiUrl: z.string().optional(),
  shopExternalId: z.string().optional(),
  enabled: z.boolean().optional(),
  syncCustomers: z.boolean().optional(),
  syncVehicles: z.boolean().optional(),
  syncRepairOrders: z.boolean().optional(),
  syncAppointments: z.boolean().optional(),
});

export async function PUT(req: NextRequest) {
  const session = await getSession();
  if (!session || !session.user.shopId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const parsed = updateSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
  }

  const shopId = session.user.shopId;

  // Check if integration exists
  const [existing] = await db
    .select()
    .from(dmsIntegrations)
    .where(eq(dmsIntegrations.shopId, shopId))
    .limit(1);

  if (parsed.data.provider === "none") {
    // Disable or remove integration
    if (existing) {
      await db
        .update(dmsIntegrations)
        .set({ enabled: false, updatedAt: new Date() })
        .where(eq(dmsIntegrations.shopId, shopId));
    }
    return NextResponse.json({ success: true });
  }

  const updateFields: Record<string, unknown> = { updatedAt: new Date() };
  if (parsed.data.provider) updateFields.provider = parsed.data.provider;
  if (parsed.data.apiKey) updateFields.apiKey = parsed.data.apiKey;
  if (parsed.data.apiUrl !== undefined) updateFields.apiUrl = parsed.data.apiUrl || null;
  if (parsed.data.shopExternalId !== undefined) updateFields.shopExternalId = parsed.data.shopExternalId || null;
  if (parsed.data.enabled !== undefined) updateFields.enabled = parsed.data.enabled;
  if (parsed.data.syncCustomers !== undefined) updateFields.syncCustomers = parsed.data.syncCustomers;
  if (parsed.data.syncVehicles !== undefined) updateFields.syncVehicles = parsed.data.syncVehicles;
  if (parsed.data.syncRepairOrders !== undefined) updateFields.syncRepairOrders = parsed.data.syncRepairOrders;
  if (parsed.data.syncAppointments !== undefined) updateFields.syncAppointments = parsed.data.syncAppointments;

  if (existing) {
    await db
      .update(dmsIntegrations)
      .set(updateFields)
      .where(eq(dmsIntegrations.shopId, shopId));
  } else {
    await db.insert(dmsIntegrations).values({
      shopId,
      provider: parsed.data.provider || "tekmetric",
      apiKey: parsed.data.apiKey || "",
      apiUrl: parsed.data.apiUrl || null,
      shopExternalId: parsed.data.shopExternalId || null,
      enabled: parsed.data.enabled ?? true,
    });
  }

  return NextResponse.json({ success: true });
}
