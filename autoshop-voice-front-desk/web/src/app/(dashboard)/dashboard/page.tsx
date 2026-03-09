import { getSession } from "@/lib/auth";
import { redirect } from "next/navigation";
import { db } from "@/lib/db";
import { shops, calls, tasks, appointmentRequests, dmsIntegrations, dmsSyncLog } from "@/lib/schema";
import { eq, desc, and, sql } from "drizzle-orm";

export default async function DashboardPage() {
  const session = await getSession();
  if (!session) redirect("/login");
  if (!session.user.shopId) redirect("/onboarding/step1");

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
    .limit(10);

  const openTasks = await db
    .select()
    .from(tasks)
    .where(and(eq(tasks.shopId, shopId), eq(tasks.status, "open")))
    .orderBy(desc(tasks.createdAt))
    .limit(10);

  const pendingAppts = await db
    .select()
    .from(appointmentRequests)
    .where(
      and(
        eq(appointmentRequests.shopId, shopId),
        eq(appointmentRequests.status, "requested")
      )
    )
    .orderBy(desc(appointmentRequests.createdAt))
    .limit(10);

  // DMS Integration status
  const [dmsIntegration] = await db
    .select({
      id: dmsIntegrations.id,
      provider: dmsIntegrations.provider,
      enabled: dmsIntegrations.enabled,
      lastSyncAt: dmsIntegrations.lastSyncAt,
      lastSyncError: dmsIntegrations.lastSyncError,
    })
    .from(dmsIntegrations)
    .where(and(eq(dmsIntegrations.shopId, shopId), eq(dmsIntegrations.enabled, true)))
    .limit(1);

  const [syncStats] = dmsIntegration
    ? await db
        .select({
          totalSyncs: sql<number>`count(*)`,
          failedSyncs: sql<number>`count(*) filter (where ${dmsSyncLog.status} = 'failed')`,
        })
        .from(dmsSyncLog)
        .where(
          and(
            eq(dmsSyncLog.shopId, shopId),
            sql`${dmsSyncLog.createdAt} > now() - interval '7 days'`
          )
        )
    : [{ totalSyncs: 0, failedSyncs: 0 }];

  const [weekStats] = await db
    .select({
      totalCalls: sql<number>`count(*)`,
      urgentCalls: sql<number>`count(*) filter (where ${calls.urgency} = 'urgent')`,
      appointmentCalls: sql<number>`count(*) filter (where ${calls.intent} = 'new_appointment')`,
    })
    .from(calls)
    .where(
      and(
        eq(calls.shopId, shopId),
        sql`${calls.createdAt} > now() - interval '7 days'`
      )
    );

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold">{shop?.name || "Dashboard"}</h1>
        <p className="text-gray-600 text-sm mt-1">
          {shop?.mainNumber
            ? `AI phone: ${shop.mainNumber}`
            : "Phone number being provisioned..."}
          {" | "}
          {shop?.subscriptionStatus === "trialing"
            ? "Free trial active"
            : shop?.subscriptionStatus || "Setup in progress"}
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {[
          {
            label: "Calls (7d)",
            value: weekStats?.totalCalls ?? 0,
            color: "bg-blue-50 text-blue-700",
          },
          {
            label: "Appointments",
            value: weekStats?.appointmentCalls ?? 0,
            color: "bg-green-50 text-green-700",
          },
          {
            label: "Urgent",
            value: weekStats?.urgentCalls ?? 0,
            color: "bg-red-50 text-red-700",
          },
          {
            label: "Open Tasks",
            value: openTasks.length,
            color: "bg-yellow-50 text-yellow-700",
          },
        ].map((s) => (
          <div
            key={s.label}
            className={`rounded-xl p-4 ${s.color}`}
          >
            <div className="text-3xl font-bold">{s.value}</div>
            <div className="text-sm font-medium mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Recent Calls */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="font-semibold text-lg mb-4">Recent Calls</h2>
          {recentCalls.length === 0 ? (
            <p className="text-gray-500 text-sm">
              No calls yet. Once your AI line is live, calls will appear here.
            </p>
          ) : (
            <div className="space-y-3">
              {recentCalls.map((call) => (
                <div
                  key={call.id}
                  className="flex items-start justify-between py-2 border-b border-gray-100 last:border-0"
                >
                  <div>
                    <div className="text-sm font-medium">
                      {call.fromNumber}
                    </div>
                    <div className="text-xs text-gray-500">
                      {call.intent} | {call.summary?.slice(0, 60) || "—"}
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        call.urgency === "urgent"
                          ? "bg-red-100 text-red-700"
                          : call.urgency === "high"
                          ? "bg-yellow-100 text-yellow-700"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {call.urgency}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Open Tasks + Pending Appointments */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="font-semibold text-lg mb-4">Open Tasks</h2>
            {openTasks.length === 0 ? (
              <p className="text-gray-500 text-sm">No open tasks.</p>
            ) : (
              <div className="space-y-2">
                {openTasks.map((task) => (
                  <div
                    key={task.id}
                    className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0"
                  >
                    <div className="text-sm">
                      <span className="font-medium">{task.taskType}</span>
                      <span className="text-gray-500 ml-2">
                        {task.notes?.slice(0, 40)}
                      </span>
                    </div>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        task.priority === "urgent"
                          ? "bg-red-100 text-red-700"
                          : task.priority === "high"
                          ? "bg-yellow-100 text-yellow-700"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {task.priority}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* DMS Integration Status */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="font-semibold text-lg mb-4">Shop Management Integration</h2>
            {dmsIntegration ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Provider</span>
                  <span className="text-sm font-medium capitalize">
                    {dmsIntegration.provider === "mitchell1"
                      ? "Mitchell 1"
                      : dmsIntegration.provider === "shopware"
                      ? "Shop-Ware"
                      : "Tekmetric"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Status</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      dmsIntegration.lastSyncError
                        ? "bg-red-100 text-red-700"
                        : "bg-green-100 text-green-700"
                    }`}
                  >
                    {dmsIntegration.lastSyncError ? "Error" : "Connected"}
                  </span>
                </div>
                {dmsIntegration.lastSyncAt && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Last Sync</span>
                    <span className="text-sm text-gray-500">
                      {new Date(dmsIntegration.lastSyncAt).toLocaleString()}
                    </span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Syncs (7d)</span>
                  <span className="text-sm font-medium">
                    {syncStats?.totalSyncs ?? 0}
                    {(syncStats?.failedSyncs ?? 0) > 0 && (
                      <span className="text-red-600 ml-1">
                        ({syncStats?.failedSyncs} failed)
                      </span>
                    )}
                  </span>
                </div>
                {dmsIntegration.lastSyncError && (
                  <div className="bg-red-50 rounded-lg p-3 mt-2">
                    <p className="text-xs text-red-700">
                      {dmsIntegration.lastSyncError}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-4">
                <p className="text-gray-500 text-sm mb-3">
                  No shop management system connected.
                </p>
                <a
                  href="/onboarding/step-dms"
                  className="text-sm text-blue-600 font-medium hover:underline"
                >
                  Connect Tekmetric, Mitchell 1, or Shop-Ware
                </a>
              </div>
            )}
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="font-semibold text-lg mb-4">
              Pending Appointments
            </h2>
            {pendingAppts.length === 0 ? (
              <p className="text-gray-500 text-sm">
                No pending appointment requests.
              </p>
            ) : (
              <div className="space-y-2">
                {pendingAppts.map((appt) => (
                  <div
                    key={appt.id}
                    className="py-2 border-b border-gray-100 last:border-0"
                  >
                    <div className="text-sm font-medium">
                      {appt.requestedService?.slice(0, 50) || "Service TBD"}
                    </div>
                    <div className="text-xs text-gray-500">
                      {appt.requestedDay} {appt.requestedTime}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
