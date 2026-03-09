import Link from "next/link";

export default function OnboardingComplete() {
  return (
    <div className="text-center">
      <div className="mb-6">
        <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-3xl">✓</span>
        </div>
        <h1 className="text-2xl font-bold">You&apos;re All Set!</h1>
        <p className="text-gray-600 mt-2">
          Your AI front desk is being provisioned. This takes about 2 minutes.
        </p>
      </div>

      <div className="bg-white p-6 rounded-xl border border-gray-200 text-left space-y-4 mb-8">
        <h3 className="font-semibold">What&apos;s happening now:</h3>
        <ul className="space-y-3 text-sm">
          <li className="flex items-start gap-3">
            <span className="bg-blue-100 text-blue-700 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
              1
            </span>
            <span>
              Creating your AI voice agent with your shop details, hours, and
              services.
            </span>
          </li>
          <li className="flex items-start gap-3">
            <span className="bg-blue-100 text-blue-700 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
              2
            </span>
            <span>
              Provisioning a dedicated phone number for your shop.
            </span>
          </li>
          <li className="flex items-start gap-3">
            <span className="bg-blue-100 text-blue-700 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
              3
            </span>
            <span>
              Configuring call routing, SMS, and escalation workflows.
            </span>
          </li>
        </ul>
      </div>

      <Link
        href="/dashboard"
        className="inline-block bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700"
      >
        Go to Dashboard
      </Link>
    </div>
  );
}
