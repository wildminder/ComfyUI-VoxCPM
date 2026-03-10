"use client";

import { useState, useEffect } from "react";

interface ShopData {
  id: string;
  name: string;
  mainNumber: string | null;
  planId: string | null;
  subscriptionStatus: string | null;
  onboardingStatus: string | null;
  provisionedAt: string | null;
  createdAt: string | null;
  city: string | null;
  state: string | null;
  callsThisMonth: number;
  owner: { email: string; name: string | null } | null;
}

interface Metrics {
  shops: {
    total: number;
    active: number;
    trialing: number;
    pastDue: number;
    canceled: number;
    provisioned: number;
    onboarding: number;
  };
  calls: {
    last24h: { total: number; avgDuration: number; urgent: number; transferred: number; uniqueShops: number };
    last7d: { total: number; avgDuration: number; uniqueShops: number };
    last30d: { total: number; uniqueShops: number };
  };
  tasks: { open: number; inProgress: number; completed7d: number; escalated: number };
  revenue: { estimatedMRR: number };
  users: { total: number; last7d: number };
  signupTrend: { day: string; count: number }[];
}

export default function AdminDashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [shops, setShops] = useState<ShopData[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  useEffect(() => {
    loadData();
  }, [page, statusFilter, search]);

  async function loadData() {
    setLoading(true);
    try {
      const params = new URLSearchParams({ page: String(page), limit: "25" });
      if (statusFilter) params.set("status", statusFilter);
      if (search) params.set("search", search);

      const [metricsRes, shopsRes] = await Promise.all([
        fetch("/api/admin/metrics"),
        fetch(`/api/admin/shops?${params}`),
      ]);

      if (metricsRes.status === 403 || shopsRes.status === 403) {
        setError("Admin access required. Your account must have admin role.");
        return;
      }

      const metricsData = await metricsRes.json();
      const shopsData = await shopsRes.json();

      setMetrics(metricsData);
      setShops(shopsData.shops || []);
      setStats(shopsData.stats);
      setTotalPages(shopsData.pagination?.totalPages || 1);
    } catch (e) {
      setError("Failed to load admin data");
    } finally {
      setLoading(false);
    }
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-50 text-red-700 rounded-xl p-6 text-center">
          <h2 className="font-bold text-lg mb-2">Access Denied</h2>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (loading && !metrics) {
    return <div className="p-8 text-gray-500">Loading admin dashboard...</div>;
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Admin Dashboard</h1>
      <p className="text-gray-500 text-sm mb-6">Platform-wide monitoring for all shops</p>

      {/* Revenue + Top-Level Stats */}
      {metrics && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
            <StatCard label="Est. MRR" value={`$${(metrics.revenue.estimatedMRR || 0).toLocaleString()}`} color="bg-green-50 text-green-700" />
            <StatCard label="Total Shops" value={metrics.shops.total} color="bg-blue-50 text-blue-700" />
            <StatCard label="Active" value={metrics.shops.active} color="bg-emerald-50 text-emerald-700" />
            <StatCard label="Trialing" value={metrics.shops.trialing} color="bg-purple-50 text-purple-700" />
            <StatCard label="Past Due" value={metrics.shops.pastDue} color="bg-red-50 text-red-700" />
            <StatCard label="Canceled" value={metrics.shops.canceled} color="bg-gray-100 text-gray-600" />
          </div>

          {/* Call Volume */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard label="Calls (24h)" value={metrics.calls.last24h?.total ?? 0} color="bg-blue-50 text-blue-700" />
            <StatCard label="Calls (7d)" value={metrics.calls.last7d?.total ?? 0} color="bg-blue-50 text-blue-700" />
            <StatCard label="Calls (30d)" value={metrics.calls.last30d?.total ?? 0} color="bg-blue-50 text-blue-700" />
            <StatCard label="Active Shops (24h)" value={metrics.calls.last24h?.uniqueShops ?? 0} color="bg-indigo-50 text-indigo-700" />
          </div>

          {/* Tasks + Users */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard label="Open Tasks" value={metrics.tasks.open} color="bg-yellow-50 text-yellow-700" />
            <StatCard label="Escalated" value={metrics.tasks.escalated} color="bg-red-50 text-red-700" />
            <StatCard label="Users" value={metrics.users.total} color="bg-gray-100 text-gray-700" />
            <StatCard label="Signups (7d)" value={metrics.users.last7d} color="bg-green-50 text-green-700" />
          </div>
        </>
      )}

      {/* Shop List */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="p-4 border-b border-gray-200 flex flex-wrap gap-3 items-center">
          <h2 className="font-semibold text-lg mr-4">All Shops</h2>
          <input
            type="text"
            placeholder="Search shops..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm w-48"
          />
          <select
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
          >
            <option value="">All Statuses</option>
            <option value="active">Active</option>
            <option value="trialing">Trialing</option>
            <option value="past_due">Past Due</option>
            <option value="canceled">Canceled</option>
          </select>
          <button
            onClick={loadData}
            className="text-sm text-blue-600 font-medium hover:underline ml-auto"
          >
            Refresh
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="text-left px-4 py-3 font-medium text-gray-500">Shop</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Owner</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Plan</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Status</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Phone</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Calls/mo</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Location</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500">Joined</th>
              </tr>
            </thead>
            <tbody>
              {shops.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                    {loading ? "Loading..." : "No shops found"}
                  </td>
                </tr>
              ) : (
                shops.map((shop) => (
                  <tr key={shop.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="px-4 py-3">
                      <div className="font-medium">{shop.name}</div>
                      <div className="text-xs text-gray-400">{shop.id.slice(0, 8)}...</div>
                    </td>
                    <td className="px-4 py-3">
                      {shop.owner ? (
                        <div>
                          <div className="text-sm">{shop.owner.name || "—"}</div>
                          <div className="text-xs text-gray-400">{shop.owner.email}</div>
                        </div>
                      ) : (
                        <span className="text-gray-400">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs capitalize">
                        {shop.planId || "—"}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={shop.subscriptionStatus} />
                    </td>
                    <td className="px-4 py-3 font-mono text-xs">
                      {shop.mainNumber || <span className="text-gray-400">Not provisioned</span>}
                    </td>
                    <td className="px-4 py-3 font-medium">{shop.callsThisMonth}</td>
                    <td className="px-4 py-3 text-gray-600">
                      {[shop.city, shop.state].filter(Boolean).join(", ") || "—"}
                    </td>
                    <td className="px-4 py-3 text-gray-500 text-xs">
                      {shop.createdAt ? new Date(shop.createdAt).toLocaleDateString() : "—"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="p-4 border-t border-gray-200 flex items-center justify-between">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="text-sm text-blue-600 font-medium disabled:text-gray-400"
            >
              Previous
            </button>
            <span className="text-sm text-gray-500">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
              className="text-sm text-blue-600 font-medium disabled:text-gray-400"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className={`rounded-xl p-4 ${color}`}>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs font-medium mt-1">{label}</div>
    </div>
  );
}

function StatusBadge({ status }: { status: string | null }) {
  const styles: Record<string, string> = {
    active: "bg-green-100 text-green-700",
    trialing: "bg-purple-100 text-purple-700",
    past_due: "bg-red-100 text-red-700",
    canceled: "bg-gray-100 text-gray-600",
    unpaid: "bg-red-100 text-red-700",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs ${styles[status || ""] || "bg-gray-100 text-gray-500"}`}>
      {status || "pending"}
    </span>
  );
}
