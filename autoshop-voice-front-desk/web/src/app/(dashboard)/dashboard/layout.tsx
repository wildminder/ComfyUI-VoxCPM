import Link from "next/link";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="border-b border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/dashboard" className="text-lg font-bold text-blue-700">
              AutoShop Voice AI
            </Link>
            <div className="flex gap-6">
              <Link
                href="/dashboard"
                className="text-sm text-gray-600 hover:text-gray-900 font-medium"
              >
                Overview
              </Link>
              <Link
                href="/dashboard/calls"
                className="text-sm text-gray-600 hover:text-gray-900 font-medium"
              >
                Calls
              </Link>
              <Link
                href="/dashboard/tasks"
                className="text-sm text-gray-600 hover:text-gray-900 font-medium"
              >
                Tasks
              </Link>
              <Link
                href="/dashboard/settings"
                className="text-sm text-gray-600 hover:text-gray-900 font-medium"
              >
                Settings
              </Link>
            </div>
          </div>
          <form action="/api/auth/logout" method="POST">
            <button
              type="submit"
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Log Out
            </button>
          </form>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-6 py-8">{children}</main>
    </div>
  );
}
