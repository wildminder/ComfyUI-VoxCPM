"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const DMS_PROVIDERS = [
  {
    value: "tekmetric",
    label: "Tekmetric",
    description: "Cloud-based shop management with real-time analytics",
    authLabel: "API Key / OAuth Token",
    helpUrl: "https://www.tekmetric.com/integrations",
  },
  {
    value: "mitchell1",
    label: "Mitchell 1 Manager SE",
    description: "Industry-standard shop management by Snap-on",
    authLabel: "API Key",
    helpUrl: "https://mitchell1.com/resources/api-request/",
  },
  {
    value: "shopware",
    label: "Shop-Ware",
    description: "Modern cloud shop management with digital inspections",
    authLabel: "API Key",
    helpUrl: "https://support.shop-ware.com/s/article/API-Partner-Integration",
  },
];

export default function StepDms() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [provider, setProvider] = useState<string>("none");
  const [apiKey, setApiKey] = useState("");
  const [shopExternalId, setShopExternalId] = useState("");
  const [testStatus, setTestStatus] = useState<"idle" | "testing" | "success" | "failed">("idle");

  const selectedProvider = DMS_PROVIDERS.find((p) => p.value === provider);

  async function testConnection() {
    if (!apiKey) return;
    setTestStatus("testing");
    try {
      const res = await fetch("/api/dms/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, apiKey, shopExternalId }),
      });
      const data = await res.json();
      setTestStatus(data.ok ? "success" : "failed");
      if (!data.ok) setError(data.error || "Connection test failed");
    } catch {
      setTestStatus("failed");
      setError("Connection test failed");
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");

    const res = await fetch("/api/onboarding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        step: "step_dms",
        dmsProvider: provider,
        apiKey: provider !== "none" ? apiKey : undefined,
        shopExternalId: provider !== "none" ? shopExternalId : undefined,
      }),
    });

    const data = await res.json();
    setLoading(false);

    if (!res.ok) {
      setError(data.error || "Failed to save");
      return;
    }
    router.push(data.next);
  }

  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
          <span className="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold">
            4
          </span>
          <span>Step 4 of 5</span>
        </div>
        <h1 className="text-2xl font-bold">Shop Management Integration</h1>
        <p className="text-gray-600 mt-1">
          Connect your shop management software to automatically sync customers,
          vehicles, and repair orders from AI calls. This step is optional.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="bg-white p-8 rounded-xl border border-gray-200 space-y-6"
      >
        {error && (
          <div className="bg-red-50 text-red-700 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Select Your Shop Management System
          </label>
          <div className="space-y-3">
            <label
              className={`flex items-start gap-3 p-4 rounded-lg border cursor-pointer ${
                provider === "none"
                  ? "border-blue-300 bg-blue-50"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              <input
                type="radio"
                name="provider"
                value="none"
                checked={provider === "none"}
                onChange={() => setProvider("none")}
                className="mt-0.5"
              />
              <div>
                <div className="font-medium text-sm">Skip for now</div>
                <div className="text-xs text-gray-500">
                  You can connect your DMS later from the dashboard settings.
                </div>
              </div>
            </label>

            {DMS_PROVIDERS.map((p) => (
              <label
                key={p.value}
                className={`flex items-start gap-3 p-4 rounded-lg border cursor-pointer ${
                  provider === p.value
                    ? "border-blue-300 bg-blue-50"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <input
                  type="radio"
                  name="provider"
                  value={p.value}
                  checked={provider === p.value}
                  onChange={() => {
                    setProvider(p.value);
                    setTestStatus("idle");
                    setError("");
                  }}
                  className="mt-0.5"
                />
                <div>
                  <div className="font-medium text-sm">{p.label}</div>
                  <div className="text-xs text-gray-500">{p.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {provider !== "none" && selectedProvider && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {selectedProvider.authLabel}
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                  setTestStatus("idle");
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                placeholder="Enter your API key"
                required
              />
              <p className="text-xs text-gray-500 mt-1">
                Get your API key from{" "}
                <a
                  href={selectedProvider.helpUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 underline"
                >
                  {selectedProvider.label} integrations page
                </a>
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Shop ID in {selectedProvider.label}
                <span className="text-gray-400"> (optional)</span>
              </label>
              <input
                type="text"
                value={shopExternalId}
                onChange={(e) => setShopExternalId(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                placeholder="Your shop ID in the DMS system"
              />
            </div>

            <button
              type="button"
              onClick={testConnection}
              disabled={!apiKey || testStatus === "testing"}
              className={`w-full py-2 rounded-lg text-sm font-medium border ${
                testStatus === "success"
                  ? "border-green-300 bg-green-50 text-green-700"
                  : testStatus === "failed"
                  ? "border-red-300 bg-red-50 text-red-700"
                  : "border-gray-300 text-gray-700 hover:bg-gray-50"
              } disabled:opacity-50`}
            >
              {testStatus === "testing"
                ? "Testing connection..."
                : testStatus === "success"
                ? "Connection successful"
                : testStatus === "failed"
                ? "Connection failed - check credentials"
                : "Test Connection"}
            </button>
          </>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50"
        >
          {loading
            ? "Saving..."
            : provider === "none"
            ? "Skip & Continue to Plan Selection"
            : "Save & Continue to Plan Selection"}
        </button>
      </form>
    </div>
  );
}
