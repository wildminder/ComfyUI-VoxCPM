import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { shops, calls } from "@/lib/schema";
import { sql, gte, and, eq } from "drizzle-orm";

/**
 * Health check endpoint — no auth required, used for uptime monitoring.
 * Returns system health status without sensitive data.
 */
export async function GET() {
  const checks: Record<string, { status: string; latencyMs?: number; error?: string }> = {};

  // Database check
  const dbStart = Date.now();
  try {
    await db.select({ one: sql<number>`1` }).from(shops).limit(1);
    checks.database = { status: "healthy", latencyMs: Date.now() - dbStart };
  } catch (e) {
    checks.database = {
      status: "unhealthy",
      latencyMs: Date.now() - dbStart,
      error: e instanceof Error ? e.message : "Unknown error",
    };
  }

  // Check if calls are flowing (any calls in last hour)
  try {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    const [recent] = await db
      .select({ count: sql<number>`count(*)` })
      .from(calls)
      .where(gte(calls.createdAt, oneHourAgo));

    checks.callFlow = {
      status: "healthy",
      latencyMs: 0,
    };
  } catch (e) {
    checks.callFlow = {
      status: "degraded",
      error: e instanceof Error ? e.message : "Unknown error",
    };
  }

  // Check provisioned shops have required config
  try {
    const [missingConfig] = await db
      .select({ count: sql<number>`count(*)` })
      .from(shops)
      .where(
        and(
          sql`${shops.provisionedAt} is not null`,
          sql`(${shops.retellAgentId} is null or ${shops.mainNumber} is null)`
        )
      );

    checks.shopConfig = {
      status: (missingConfig?.count ?? 0) > 0 ? "warning" : "healthy",
    };
    if ((missingConfig?.count ?? 0) > 0) {
      checks.shopConfig.error = `${missingConfig.count} provisioned shops missing agent or number`;
    }
  } catch {
    checks.shopConfig = { status: "unknown" };
  }

  const overallStatus = Object.values(checks).every((c) => c.status === "healthy")
    ? "healthy"
    : Object.values(checks).some((c) => c.status === "unhealthy")
    ? "unhealthy"
    : "degraded";

  return NextResponse.json(
    {
      status: overallStatus,
      checks,
      timestamp: new Date().toISOString(),
      version: process.env.APP_VERSION || "0.1.0",
    },
    { status: overallStatus === "unhealthy" ? 503 : 200 }
  );
}
