"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const COMMON_SERVICES = [
  "General repair",
  "Diagnostics",
  "Brakes",
  "AC / Heating",
  "Suspension",
  "Engine repair",
  "Transmission",
  "Electrical",
  "Oil changes",
  "Tire service",
  "Alignment",
  "State inspection",
  "Diesel repair",
  "European vehicles",
];

const COMMON_MAKES = [
  "All makes and models",
  "Domestic only (Ford, GM, Chrysler)",
  "Japanese (Toyota, Honda, Nissan, etc.)",
  "European (BMW, Mercedes, Audi, VW)",
  "Diesel trucks",
];

export default function Step3() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedServices, setSelectedServices] = useState<string[]>([
    "General repair",
    "Diagnostics",
    "Brakes",
  ]);
  const [makesServiced, setMakesServiced] = useState("All makes and models");
  const [diagFee, setDiagFee] = useState("");
  const [towPolicy, setTowPolicy] = useState("");

  function toggleService(s: string) {
    setSelectedServices((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]
    );
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");

    const res = await fetch("/api/onboarding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        step: "step3",
        servicesText: selectedServices.join(", "),
        makesServicedText: makesServiced,
        diagFeeText: diagFee || undefined,
        towPolicyText: towPolicy || undefined,
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
            3
          </span>
          <span>Step 3 of 4</span>
        </div>
        <h1 className="text-2xl font-bold">Services & Policies</h1>
        <p className="text-gray-600 mt-1">
          Tell us what your shop does so the AI can answer customers accurately.
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
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Services Offered
          </label>
          <div className="flex flex-wrap gap-2">
            {COMMON_SERVICES.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => toggleService(s)}
                className={`px-3 py-1.5 rounded-full text-sm border ${
                  selectedServices.includes(s)
                    ? "bg-blue-50 border-blue-300 text-blue-700"
                    : "border-gray-200 text-gray-600 hover:border-gray-300"
                }`}
              >
                {selectedServices.includes(s) && "✓ "}
                {s}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Makes Serviced
          </label>
          <select
            value={makesServiced}
            onChange={(e) => setMakesServiced(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          >
            {COMMON_MAKES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Diagnostic Fee{" "}
            <span className="text-gray-400">(what the AI can quote)</span>
          </label>
          <input
            type="text"
            value={diagFee}
            onChange={(e) => setDiagFee(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            placeholder="Our diagnostic fee starts at $125"
          />
          <p className="text-xs text-gray-500 mt-1">
            Leave blank if you don&apos;t want the AI quoting any fee.
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Tow Policy{" "}
            <span className="text-gray-400">(optional)</span>
          </label>
          <input
            type="text"
            value={towPolicy}
            onChange={(e) => setTowPolicy(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            placeholder="We can arrange a tow. Let us know your location."
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Saving..." : "Continue to Plan Selection"}
        </button>
      </form>
    </div>
  );
}
