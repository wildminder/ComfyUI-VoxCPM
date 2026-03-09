import Link from "next/link";

export default function OnboardingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="border-b border-gray-200 bg-white">
        <div className="max-w-3xl mx-auto px-6 py-4">
          <Link href="/" className="text-xl font-bold text-blue-700">
            AutoShop Voice AI
          </Link>
        </div>
      </nav>
      <div className="max-w-2xl mx-auto px-6 py-12">{children}</div>
    </div>
  );
}
