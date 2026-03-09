import { getSession } from "@/lib/auth";
import { redirect } from "next/navigation";
import { db } from "@/lib/db";
import { shops, calls, tasks, appointmentRequests } from "@/lib/schema";
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
