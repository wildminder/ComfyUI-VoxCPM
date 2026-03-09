"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const DAYS = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
];

type DayHours = { open: string; close: string } | null;

export default function Step2() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [afterHoursEnabled, setAfterHoursEnabled] = useState(true);
  const [hours, setHours] = useState<Record<string, DayHours>>({
    monday: { open: "08:00", close: "17:00" },
    tuesday: { open: "08:00", close: "17:00" },
    wednesday: { open: "08:00", close: "17:00" },
    thursday: { open: "08:00", close: "17:00" },
    friday: { open: "08:00", close: "17:00" },
    saturday: null,
    sunday: null,
  });

  function toggleDay(day: string) {
    setHours((prev) => ({
      ...prev,
      [day]: prev[day] ? null : { open: "08:00", close: "17:00" },
    }));
  }

  function updateHours(day: string, field: "open" | "close", value: string) {
    setHours((prev) => ({
      ...prev,
      [day]: prev[day] ? { ...prev[day]!, [field]: value } : null,
    }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");

    const res = await fetch("/api/onboarding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        step: "step2",
        hoursJson: hours,
        afterHoursEnabled,
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
            2
          </span>
          <span>Step 2 of 4</span>
        </div>
        <h1 className="text-2xl font-bold">Business Hours</h1>
        <p className="text-gray-600 mt-1">
          Set your shop hours so the AI knows when to offer live transfers vs.
          after-hours mode.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="bg-white p-8 rounded-xl border border-gray-200 space-y-4"
      >
        {error && (
          <div className="bg-red-50 text-red-700 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        <div className="space-y-3">
          {DAYS.map((day) => (
            <div key={day} className="flex items-center gap-4">
              <label className="flex items-center gap-2 w-28">
                <input
                  type="checkbox"
                  checked={hours[day] !== null}
                  onChange={() => toggleDay(day)}
                  className="rounded"
                />
                <span className="text-sm font-medium capitalize">{day}</span>
              </label>

              {hours[day] ? (
                <div className="flex items-center gap-2">
                  <input
                    type="time"
                    value={hours[day]!.open}
                    onChange={(e) => updateHours(day, "open", e.target.value)}
                    className="px-2 py-1 border border-gray-300 rounded text-sm"
                  />
                  <span className="text-gray-400">to</span>
                  <input
                    type="time"
                    value={hours[day]!.close}
                    onChange={(e) => updateHours(day, "close", e.target.value)}
                    className="px-2 py-1 border border-gray-300 rounded text-sm"
                  />
                </div>
              ) : (
                <span className="text-sm text-gray-400">Closed</span>
              )}
            </div>
          ))}
        </div>

        <div className="pt-4 border-t">
          <label className="flex items-center gap-3">
            <input
              type="checkbox"
              checked={afterHoursEnabled}
              onChange={(e) => setAfterHoursEnabled(e.target.checked)}
              className="rounded"
            />
            <div>
              <span className="text-sm font-medium">
                Enable after-hours answering
              </span>
              <p className="text-xs text-gray-500">
                AI still answers calls outside hours and creates callback tasks.
              </p>
            </div>
          </label>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Saving..." : "Continue to Services"}
        </button>
      </form>
    </div>
  );
}
