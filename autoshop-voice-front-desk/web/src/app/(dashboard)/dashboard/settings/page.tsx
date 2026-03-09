"use client";

import { useState } from "react";

export default function SettingsPage() {
  const [billingLoading, setBillingLoading] = useState(false);

  async function openBillingPortal() {
    setBillingLoading(true);
    try {
      const res = await fetch("/api/stripe/portal", { method: "POST" });
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        alert(data.error || "Could not open billing portal");
      }
    } finally {
      setBillingLoading(false);
    }
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Settings</h1>

      <div className="space-y-6 max-w-2xl">
        {/* Billing */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="font-semibold text-lg mb-2">Billing</h2>
          <p className="text-sm text-gray-600 mb-4">
            Manage your subscription, update payment method, or download
            invoices.
          </p>
          <button
            onClick={openBillingPortal}
            disabled={billingLoading}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-blue-700 disabled:opacity-50"
          >
            {billingLoading ? "Opening..." : "Open Billing Portal"}
          </button>
        </div>

        {/* Shop Info */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="font-semibold text-lg mb-2">Shop Configuration</h2>
          <p className="text-sm text-gray-600 mb-4">
            To update your shop details, hours, or services, go through the
            onboarding flow again or contact support.
          </p>
          <a
            href="/onboarding/step1"
            className="text-blue-600 hover:underline text-sm font-medium"
          >
            Update shop info
          </a>
        </div>

        {/* Danger Zone */}
        <div className="bg-white rounded-xl border border-red-200 p-6">
          <h2 className="font-semibold text-lg mb-2 text-red-700">
            Danger Zone
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Cancel your subscription through the billing portal above. Your AI
            line will be deactivated at the end of your billing period.
          </p>
        </div>
      </div>
    </div>
  );
}
