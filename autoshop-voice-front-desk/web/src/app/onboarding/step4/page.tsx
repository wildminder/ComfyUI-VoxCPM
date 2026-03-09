"use client";

import { useState } from "react";

const PLANS = [
  {
    id: "starter",
    name: "Starter",
    price: 149,
    features: [
      "1 phone line",
      "Up to 200 calls/month",
      "SMS recaps",
      "Appointment requests",
      "Basic dashboard",
    ],
  },
  {
    id: "professional",
    name: "Professional",
    price: 299,
    popular: true,
    features: [
      "Up to 3 phone lines",
      "Up to 500 calls/month",
      "SMS recaps",
      "Live Cal.com booking",
      "Urgent escalation",
      "Full dashboard",
    ],
  },
  {
    id: "enterprise",
    name: "Enterprise",
    price: 599,
    features: [
      "Unlimited phone lines",
      "Unlimited calls",
      "Priority support",
      "Live booking",
      "Custom agent prompt",
      "API access",
    ],
  },
];

export default function Step4() {
  const [loading, setLoading] = useState<string | null>(null);

  async function selectPlan(planId: string) {
    setLoading(planId);

    const res = await fetch("/api/stripe/checkout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ planId }),
    });

    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      setLoading(null);
      alert(data.error || "Failed to start checkout");
    }
  }

  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
          <span className="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold">
            4
          </span>
          <span>Step 4 of 4</span>
        </div>
        <h1 className="text-2xl font-bold">Choose Your Plan</h1>
        <p className="text-gray-600 mt-1">
          All plans include a 14-day free trial. Cancel anytime.
        </p>
      </div>

      <div className="space-y-4">
        {PLANS.map((plan) => (
          <div
            key={plan.id}
            className={`bg-white p-6 rounded-xl border ${
              plan.popular
                ? "border-blue-600 ring-2 ring-blue-600"
                : "border-gray-200"
            }`}
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                {plan.popular && (
                  <span className="text-xs font-semibold text-blue-600 uppercase">
                    Most Popular
                  </span>
                )}
                <h3 className="text-lg font-bold">{plan.name}</h3>
              </div>
              <div className="text-right">
                <span className="text-3xl font-bold">${plan.price}</span>
                <span className="text-gray-500">/mo</span>
              </div>
            </div>

            <ul className="space-y-2 mb-6">
              {plan.features.map((f) => (
                <li key={f} className="flex items-center gap-2 text-sm">
                  <span className="text-green-500">✓</span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>

            <button
              onClick={() => selectPlan(plan.id)}
              disabled={loading !== null}
              className={`w-full py-3 rounded-lg font-semibold text-sm disabled:opacity-50 ${
                plan.popular
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "border border-gray-300 text-gray-700 hover:bg-gray-50"
              }`}
            >
              {loading === plan.id
                ? "Redirecting to payment..."
                : "Start Free Trial"}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
