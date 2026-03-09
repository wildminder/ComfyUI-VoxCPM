import { getSession } from "@/lib/auth";
import { redirect } from "next/navigation";
import { db } from "@/lib/db";
import { calls } from "@/lib/schema";
import { eq, desc } from "drizzle-orm";

export default async function CallsPage() {
  const session = await getSession();
  if (!session) redirect("/login");

  const allCalls = await db
    .select()
    .from(calls)
    .where(eq(calls.shopId, session.user.shopId!))
    .orderBy(desc(calls.createdAt))
    .limit(100);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Call History</h1>

      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 bg-gray-50">
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Time
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                From
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Intent
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Urgency
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Summary
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Duration
              </th>
            </tr>
          </thead>
          <tbody>
            {allCalls.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-500">
                  No calls yet.
                </td>
              </tr>
            ) : (
              allCalls.map((call) => (
                <tr
                  key={call.id}
                  className="border-b border-gray-100 hover:bg-gray-50"
                >
                  <td className="px-4 py-3 text-gray-600">
                    {call.createdAt
                      ? new Date(call.createdAt).toLocaleString()
                      : "—"}
                  </td>
                  <td className="px-4 py-3 font-mono">{call.fromNumber}</td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs">
                      {call.intent}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`px-2 py-0.5 rounded text-xs ${
                        call.urgency === "urgent"
                          ? "bg-red-100 text-red-700"
                          : call.urgency === "high"
                          ? "bg-yellow-100 text-yellow-700"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {call.urgency}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-600 max-w-xs truncate">
                    {call.summary || "—"}
                  </td>
                  <td className="px-4 py-3 text-gray-600">
                    {call.durationSec
                      ? `${Math.floor(call.durationSec / 60)}:${String(
                          call.durationSec % 60
                        ).padStart(2, "0")}`
                      : "—"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
