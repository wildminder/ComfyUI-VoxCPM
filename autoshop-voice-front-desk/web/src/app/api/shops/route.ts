import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { shops, calls, tasks, appointmentRequests } from "@/lib/schema";
import { eq, desc, and, sql } from "drizzle-orm";

export async function GET() {
  const session = await getSession();
  if (!session || !session.user.shopId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const shopId = session.user.shopId;

  const [shop] = await db
    .select()
    .from(shops)
    .where(eq(shops.id, shopId))
    .limit(1);

  const recentCalls = await db
    .select()
    .from(calls)
    .where(eq(calls.shopId, shopId))
    .orderBy(desc(calls.createdAt))
    .limit(20);

  const openTasks = await db
    .select()
    .from(tasks)
    .where(and(eq(tasks.shopId, shopId), eq(tasks.status, "open")))
    .orderBy(desc(tasks.createdAt))
    .limit(20);

  const pendingAppointments = await db
    .select()
    .from(appointmentRequests)
    .where(
      and(
        eq(appointmentRequests.shopId, shopId),
        eq(appointmentRequests.status, "requested")
      )
    )
    .orderBy(desc(appointmentRequests.createdAt))
    .limit(20);

  // Stats
  const [stats] = await db
    .select({
      totalCalls: sql<number>`count(*)`.as("total_calls"),
      urgentCalls: sql<number>`count(*) filter (where ${calls.urgency} = 'urgent')`.as("urgent_calls"),
      appointmentCalls: sql<number>`count(*) filter (where ${calls.intent} = 'new_appointment')`.as("appt_calls"),
    })
    .from(calls)
    .where(
      and(
        eq(calls.shopId, shopId),
        sql`${calls.createdAt} > now() - interval '7 days'`
      )
    );

  return NextResponse.json({
    shop,
    recentCalls,
    openTasks,
    pendingAppointments,
    stats: stats || { totalCalls: 0, urgentCalls: 0, appointmentCalls: 0 },
  });
}
