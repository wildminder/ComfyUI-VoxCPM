import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AutoShop Voice AI - Never Miss a Customer Call Again",
  description:
    "AI-powered front desk for independent auto repair shops. Answer every call, book appointments, and capture leads 24/7.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased bg-white text-gray-900">{children}</body>
    </html>
  );
}
