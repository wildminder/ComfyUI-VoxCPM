import { db } from "./db";
import { calls, shops } from "./schema";
import { eq, and, sql, gte } from "drizzle-orm";
import { PLANS, type PlanId } from "./stripe";

/**
 * Get the current billing period call count for a shop.
 * Billing periods reset monthly from the subscription start date.
 */
export async function getMonthlyCallCount(shopId: string): Promise<number> {
  const startOfMonth = new Date();
  startOfMonth.setDate(1);
  startOfMonth.setHours(0, 0, 0, 0);

  const [result] = await db
    .select({
      count: sql<number>`count(*)`,
    })
    .from(calls)
    .where(
      and(eq(calls.shopId, shopId), gte(calls.createdAt, startOfMonth))
    );

  return result?.count ?? 0;
}

/**
 * Check if a shop has exceeded their plan's call limit.
 * Returns { allowed, used, limit, planId }
 */
export async function checkCallLimit(shopId: string): Promise<{
  allowed: boolean;
  used: number;
  limit: number;
  planId: string;
}> {
  const [shop] = await db
    .select({
      planId: shops.planId,
      subscriptionStatus: shops.subscriptionStatus,
    })
    .from(shops)
    .where(eq(shops.id, shopId))
    .limit(1);

  if (!shop) {
    return { allowed: false, used: 0, limit: 0, planId: "none" };
  }

  const planId = (shop.planId || "starter") as PlanId;
  const plan = PLANS[planId];

  if (!plan) {
    return { allowed: false, used: 0, limit: 0, planId };
  }

  // Unlimited plan
  if (plan.limits.callsPerMonth === -1) {
    const used = await getMonthlyCallCount(shopId);
    return { allowed: true, used, limit: -1, planId };
  }

  // Check subscription is active or trialing
  const activeStatuses = ["active", "trialing"];
  if (!shop.subscriptionStatus || !activeStatuses.includes(shop.subscriptionStatus)) {
    return { allowed: false, used: 0, limit: plan.limits.callsPerMonth, planId };
  }

  const used = await getMonthlyCallCount(shopId);
  const allowed = used < plan.limits.callsPerMonth;

  return { allowed, used, limit: plan.limits.callsPerMonth, planId };
}

/**
 * Get usage stats for a shop — used by the dashboard and admin panel.
 */
export async function getShopUsageStats(shopId: string) {
  const [shop] = await db
    .select()
    .from(shops)
    .where(eq(shops.id, shopId))
    .limit(1);

  if (!shop) return null;

  const callCount = await getMonthlyCallCount(shopId);
  const planId = (shop.planId || "starter") as PlanId;
  const plan = PLANS[planId];
  const limit = plan?.limits.callsPerMonth ?? 0;

  return {
    shopId,
    shopName: shop.name,
    planId,
    planName: plan?.name || "Unknown",
    subscriptionStatus: shop.subscriptionStatus,
    callsThisMonth: callCount,
    callLimit: limit,
    callsRemaining: limit === -1 ? -1 : Math.max(0, limit - callCount),
    usagePercent: limit === -1 ? 0 : limit > 0 ? Math.round((callCount / limit) * 100) : 100,
    provisionedAt: shop.provisionedAt,
    retellAgentId: shop.retellAgentId,
    mainNumber: shop.mainNumber,
  };
}
