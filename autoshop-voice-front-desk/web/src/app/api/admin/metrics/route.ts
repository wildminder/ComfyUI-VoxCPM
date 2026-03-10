import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { shops, calls, tasks, appointmentRequests, users } from "@/lib/schema";
import { sql, and, gte } from "drizzle-orm";

/**
 * Admin metrics endpoint — platform-wide health & monitoring.
 * For monitoring 500+ concurrent shops.
 */
export async function GET() {
  const session = await getSession();
  if (!session || session.user.role !== "admin") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const now = new Date();
  const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
  const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

  // Run all queries in parallel
  const [
    shopStats,
    callStats24h,
    callStats7d,
    callStats30d,
    taskStats,
    appointmentStats,
    userStats,
    callsByIntent,
    callsByUrgency,
    revenueByPlan,
    recentSignups,
  ] = await Promise.all([
    // Shop stats
    db
      .select({
        total: sql<number>`count(*)`,
        active: sql<number>`count(*) filter (where ${shops.subscriptionStatus} in ('active', 'trialing'))`,
        trialing: sql<number>`count(*) filter (where ${shops.subscriptionStatus} = 'trialing')`,
        pastDue: sql<number>`count(*) filter (where ${shops.subscriptionStatus} = 'past_due')`,
        canceled: sql<number>`count(*) filter (where ${shops.subscriptionStatus} = 'canceled')`,
        provisioned: sql<number>`count(*) filter (where ${shops.provisionedAt} is not null)`,
        onboarding: sql<number>`count(*) filter (where ${shops.onboardingStatus} != 'complete')`,
      })
      .from(shops),

    // Calls last 24h
    db
      .select({
        total: sql<number>`count(*)`,
        avgDuration: sql<number>`avg(${calls.durationSec})`,
        urgent: sql<number>`count(*) filter (where ${calls.urgency} = 'urgent')`,
        transferred: sql<number>`count(*) filter (where ${calls.transferred} = true)`,
        uniqueShops: sql<number>`count(distinct ${calls.shopId})`,
      })
      .from(calls)
      .where(gte(calls.createdAt, oneDayAgo)),

    // Calls last 7d
    db
      .select({
        total: sql<number>`count(*)`,
        avgDuration: sql<number>`avg(${calls.durationSec})`,
        uniqueShops: sql<number>`count(distinct ${calls.shopId})`,
      })
      .from(calls)
      .where(gte(calls.createdAt, sevenDaysAgo)),

    // Calls last 30d
    db
      .select({
        total: sql<number>`count(*)`,
        uniqueShops: sql<number>`count(distinct ${calls.shopId})`,
      })
      .from(calls)
      .where(gte(calls.createdAt, thirtyDaysAgo)),

    // Task stats
    db
      .select({
        open: sql<number>`count(*) filter (where ${tasks.status} = 'open')`,
        inProgress: sql<number>`count(*) filter (where ${tasks.status} = 'in_progress')`,
        completed7d: sql<number>`count(*) filter (where ${tasks.status} = 'completed' and ${tasks.updatedAt} > ${sevenDaysAgo})`,
        escalated: sql<number>`count(*) filter (where ${tasks.escalationCount} > 0 and ${tasks.status} = 'open')`,
      })
      .from(tasks),

    // Appointment stats
    db
      .select({
        requested7d: sql<number>`count(*) filter (where ${appointmentRequests.createdAt} > ${sevenDaysAgo})`,
        pending: sql<number>`count(*) filter (where ${appointmentRequests.status} = 'requested')`,
      })
      .from(appointmentRequests),

    // User stats
    db
      .select({
        total: sql<number>`count(*)`,
        last7d: sql<number>`count(*) filter (where ${users.createdAt} > ${sevenDaysAgo})`,
      })
      .from(users),

    // Calls by intent (last 7d)
    db
      .select({
        intent: calls.intent,
        count: sql<number>`count(*)`,
      })
      .from(calls)
      .where(gte(calls.createdAt, sevenDaysAgo))
      .groupBy(calls.intent),

    // Calls by urgency (last 7d)
    db
      .select({
        urgency: calls.urgency,
        count: sql<number>`count(*)`,
      })
      .from(calls)
      .where(gte(calls.createdAt, sevenDaysAgo))
      .groupBy(calls.urgency),

    // Revenue estimate by plan
    db
      .select({
        planId: shops.planId,
        count: sql<number>`count(*)`,
      })
      .from(shops)
      .where(
        sql`${shops.subscriptionStatus} in ('active', 'trialing')`
      )
      .groupBy(shops.planId),

    // Recent signups (last 7 days)
    db
      .select({
        day: sql<string>`date_trunc('day', ${shops.createdAt})::date::text`,
        count: sql<number>`count(*)`,
      })
      .from(shops)
      .where(gte(shops.createdAt, sevenDaysAgo))
      .groupBy(sql`date_trunc('day', ${shops.createdAt})`)
      .orderBy(sql`date_trunc('day', ${shops.createdAt})`),
  ]);

  // Calculate estimated MRR
  const planPrices: Record<string, number> = {
    starter: 149,
    professional: 299,
    enterprise: 599,
  };
  const estimatedMRR = revenueByPlan.reduce((sum, r) => {
    return sum + (planPrices[r.planId || "starter"] || 0) * r.count;
  }, 0);

  return NextResponse.json({
    shops: shopStats[0],
    calls: {
      last24h: callStats24h[0],
      last7d: callStats7d[0],
      last30d: callStats30d[0],
      byIntent: callsByIntent,
      byUrgency: callsByUrgency,
    },
    tasks: taskStats[0],
    appointments: appointmentStats[0],
    users: userStats[0],
    revenue: {
      estimatedMRR,
      byPlan: revenueByPlan,
    },
    signupTrend: recentSignups,
    generatedAt: now.toISOString(),
  });
}
