import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { shops, calls, users } from "@/lib/schema";
import { eq, desc, sql, and, gte } from "drizzle-orm";

/**
 * Admin endpoint — lists all shops with usage stats.
 * Requires admin role.
 */
export async function GET(req: NextRequest) {
  const session = await getSession();
  if (!session || session.user.role !== "admin") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const url = new URL(req.url);
  const page = Math.max(1, parseInt(url.searchParams.get("page") || "1"));
  const limit = Math.min(100, Math.max(1, parseInt(url.searchParams.get("limit") || "50")));
  const status = url.searchParams.get("status"); // active, trialing, past_due, canceled
  const search = url.searchParams.get("search");
  const offset = (page - 1) * limit;

  const startOfMonth = new Date();
  startOfMonth.setDate(1);
  startOfMonth.setHours(0, 0, 0, 0);

  // Build where conditions
  const conditions = [];
  if (status) {
    conditions.push(eq(shops.subscriptionStatus, status as any));
  }
  if (search) {
    conditions.push(
      sql`(${shops.name} ILIKE ${"%" + search + "%"} OR ${shops.mainNumber} ILIKE ${"%" + search + "%"})`
    );
  }

  const whereClause = conditions.length > 0 ? and(...conditions) : undefined;

  // Get total count
  const [{ total }] = await db
    .select({ total: sql<number>`count(*)` })
    .from(shops)
    .where(whereClause);

  // Get shops with monthly call counts
  const shopList = await db
    .select({
      id: shops.id,
      name: shops.name,
      mainNumber: shops.mainNumber,
      planId: shops.planId,
      subscriptionStatus: shops.subscriptionStatus,
      onboardingStatus: shops.onboardingStatus,
      provisionedAt: shops.provisionedAt,
      createdAt: shops.createdAt,
      city: shops.city,
      state: shops.state,
      retellAgentId: shops.retellAgentId,
      stripeCustomerId: shops.stripeCustomerId,
      trialEndsAt: shops.trialEndsAt,
    })
    .from(shops)
    .where(whereClause)
    .orderBy(desc(shops.createdAt))
    .limit(limit)
    .offset(offset);

  // Get call counts for this month for all returned shops
  const shopIds = shopList.map((s) => s.id);
  const callCounts =
    shopIds.length > 0
      ? await db
          .select({
            shopId: calls.shopId,
            count: sql<number>`count(*)`,
          })
          .from(calls)
          .where(
            and(
              sql`${calls.shopId} = ANY(${shopIds})`,
              gte(calls.createdAt, startOfMonth)
            )
          )
          .groupBy(calls.shopId)
      : [];

  const callCountMap = new Map(callCounts.map((c) => [c.shopId, c.count]));

  // Get owner emails
  const ownerEmails =
    shopIds.length > 0
      ? await db
          .select({
            shopId: users.shopId,
            email: users.email,
            name: users.name,
          })
          .from(users)
          .where(
            and(
              sql`${users.shopId} = ANY(${shopIds})`,
              eq(users.role, "owner")
            )
          )
      : [];

  const ownerMap = new Map(
    ownerEmails.map((o) => [o.shopId, { email: o.email, name: o.name }])
  );

  const enrichedShops = shopList.map((shop) => ({
    ...shop,
    callsThisMonth: callCountMap.get(shop.id) ?? 0,
    owner: ownerMap.get(shop.id) || null,
  }));

  // Aggregate stats
  const [aggStats] = await db
    .select({
      totalShops: sql<number>`count(*)`,
      activeShops: sql<number>`count(*) filter (where ${shops.subscriptionStatus} in ('active', 'trialing'))`,
      provisionedShops: sql<number>`count(*) filter (where ${shops.provisionedAt} is not null)`,
      canceledShops: sql<number>`count(*) filter (where ${shops.subscriptionStatus} = 'canceled')`,
      pastDueShops: sql<number>`count(*) filter (where ${shops.subscriptionStatus} = 'past_due')`,
    })
    .from(shops);

  return NextResponse.json({
    shops: enrichedShops,
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
    },
    stats: aggStats,
  });
}
