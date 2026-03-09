import { getSession } from "@/lib/auth";
import { redirect } from "next/navigation";
import { db } from "@/lib/db";
import { tasks } from "@/lib/schema";
import { eq, desc } from "drizzle-orm";

export default async function TasksPage() {
  const session = await getSession();
  if (!session) redirect("/login");

  const allTasks = await db
    .select()
    .from(tasks)
    .where(eq(tasks.shopId, session.user.shopId!))
    .orderBy(desc(tasks.createdAt))
    .limit(100);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Tasks</h1>

      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 bg-gray-50">
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Type
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Priority
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Status
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Due
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Escalations
              </th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">
                Notes
              </th>
            </tr>
          </thead>
          <tbody>
            {allTasks.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-500">
                  No tasks yet.
                </td>
              </tr>
            ) : (
              allTasks.map((task) => (
                <tr
                  key={task.id}
                  className="border-b border-gray-100 hover:bg-gray-50"
                >
                  <td className="px-4 py-3 font-medium">{task.taskType}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`px-2 py-0.5 rounded text-xs ${
                        task.priority === "urgent"
                          ? "bg-red-100 text-red-700"
                          : task.priority === "high"
                          ? "bg-yellow-100 text-yellow-700"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {task.priority}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`px-2 py-0.5 rounded text-xs ${
                        task.status === "open"
                          ? "bg-blue-50 text-blue-700"
                          : task.status === "completed"
                          ? "bg-green-50 text-green-700"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {task.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-600">
                    {task.dueAt
                      ? new Date(task.dueAt).toLocaleString()
                      : "—"}
                  </td>
                  <td className="px-4 py-3 text-center">
                    {task.escalationCount || 0}
                  </td>
                  <td className="px-4 py-3 text-gray-600 max-w-xs truncate">
                    {task.notes || "—"}
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
