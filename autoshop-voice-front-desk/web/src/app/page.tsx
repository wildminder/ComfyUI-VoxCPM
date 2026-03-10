import Link from "next/link";

const FEATURES = [
  {
    title: "Answers Every Call",
    desc: "AI picks up 24/7. No more missed calls, no more voicemail black holes.",
    icon: "📞",
  },
  {
    title: "Captures Full Intake",
    desc: "Name, vehicle, issue, drivable status — all captured and logged automatically.",
    icon: "📋",
  },
  {
    title: "Books Appointments",
    desc: "Creates appointment requests or books directly with Cal.com integration.",
    icon: "📅",
  },
  {
    title: "Sends SMS Recaps",
    desc: "Customers get a text confirmation. Your team gets the details.",
    icon: "💬",
  },
  {
    title: "Urgent Escalation",
    desc: "Brake failure? Overheating? The system alerts your team immediately.",
    icon: "🚨",
  },
  {
    title: "Shop Dashboard",
    desc: "See every call, task, and appointment. Know what happened while you were under a hood.",
    icon: "📊",
  },
];

const PLANS = [
  {
    name: "Starter",
    price: "$149",
    period: "/mo",
    features: [
      "1 phone line",
      "Up to 200 calls/month",
      "SMS recaps",
      "Appointment requests",
      "Basic dashboard",
    ],
    cta: "Start Free Trial",
    highlight: false,
  },
  {
    name: "Professional",
    price: "$299",
    period: "/mo",
    features: [
      "Up to 3 phone lines",
      "Up to 500 calls/month",
      "SMS recaps",
      "Live Cal.com booking",
      "Urgent escalation",
      "Full dashboard",
    ],
    cta: "Start Free Trial",
    highlight: true,
  },
  {
    name: "Enterprise",
    price: "$599",
    period: "/mo",
    features: [
      "Unlimited phone lines",
      "Unlimited calls",
      "Priority support",
      "Live booking",
      "Custom agent prompt",
      "API access",
      "Multi-location",
    ],
    cta: "Contact Sales",
    highlight: false,
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Nav */}
      <nav className="border-b border-gray-200 bg-white">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="text-xl font-bold text-blue-700">
            AutoShop Voice AI
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/login"
              className="text-gray-600 hover:text-gray-900 text-sm font-medium"
            >
              Log In
            </Link>
            <Link
              href="/signup"
              className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700"
            >
              Get Started Free
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold tracking-tight text-gray-900 mb-6">
            Never Miss a Customer Call Again
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            AI-powered front desk for independent auto repair shops. Answers
            calls, captures intake, books appointments, and escalates urgent
            issues — 24/7.
          </p>
          <div className="flex justify-center gap-4">
            <Link
              href="/signup"
              className="bg-blue-600 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-blue-700"
            >
              Start 14-Day Free Trial
            </Link>
            <a
              href="#pricing"
              className="border border-gray-300 text-gray-700 px-8 py-3 rounded-lg text-lg font-semibold hover:bg-gray-50"
            >
              See Pricing
            </a>
          </div>
          <p className="mt-4 text-sm text-gray-500">
            No credit card required to start. Set up in under 10 minutes.
          </p>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 px-6 bg-gray-50">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            How It Works
          </h2>
          <div className="grid md:grid-cols-4 gap-8">
            {[
              { step: "1", title: "Sign Up", desc: "Create account, pick a plan" },
              { step: "2", title: "Configure", desc: "Enter shop info, hours, services" },
              { step: "3", title: "Go Live", desc: "We provision your AI phone line" },
              { step: "4", title: "Get Calls", desc: "AI answers, you see everything" },
            ].map((s) => (
              <div key={s.step} className="text-center">
                <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">
                  {s.step}
                </div>
                <h3 className="font-semibold text-lg mb-2">{s.title}</h3>
                <p className="text-gray-600 text-sm">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            Built for Auto Shops
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {FEATURES.map((f) => (
              <div key={f.title} className="p-6 rounded-xl border border-gray-200">
                <div className="text-3xl mb-3">{f.icon}</div>
                <h3 className="font-semibold text-lg mb-2">{f.title}</h3>
                <p className="text-gray-600 text-sm">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="py-16 px-6 bg-gray-50">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-4">
            Simple Pricing
          </h2>
          <p className="text-center text-gray-600 mb-12">
            14-day free trial on all plans. No contracts.
          </p>
          <div className="grid md:grid-cols-3 gap-8">
            {PLANS.map((p) => (
              <div
                key={p.name}
                className={`rounded-xl border p-8 ${
                  p.highlight
                    ? "border-blue-600 ring-2 ring-blue-600 bg-white"
                    : "border-gray-200 bg-white"
                }`}
              >
                {p.highlight && (
                  <div className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-2">
                    Most Popular
                  </div>
                )}
                <h3 className="text-xl font-bold mb-2">{p.name}</h3>
                <div className="mb-6">
                  <span className="text-4xl font-bold">{p.price}</span>
                  <span className="text-gray-500">{p.period}</span>
                </div>
                <ul className="space-y-3 mb-8">
                  {p.features.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-sm">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>
                <Link
                  href="/signup"
                  className={`block text-center py-3 rounded-lg font-semibold text-sm ${
                    p.highlight
                      ? "bg-blue-600 text-white hover:bg-blue-700"
                      : "border border-gray-300 text-gray-700 hover:bg-gray-50"
                  }`}
                >
                  {p.cta}
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Social Proof */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            Shop Owners Love It
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                quote:
                  "We were missing about 8 calls a day during peak hours. First month, we booked 23 extra appointments — over $12,000 in new revenue.",
                name: "Mike R.",
                shop: "Independent Auto Repair",
                location: "Houston, TX",
              },
              {
                quote:
                  "My guys don't have to stop what they're doing to answer the phone anymore. The AI handles it and texts us what we need to know.",
                name: "Carlos M.",
                shop: "Family Auto Care",
                location: "Phoenix, AZ",
              },
              {
                quote:
                  "I was skeptical about AI answering my phones. After the trial, I couldn't go back. It's like having a receptionist that never misses a call.",
                name: "Sarah K.",
                shop: "Precision Auto Works",
                location: "Atlanta, GA",
              },
            ].map((t) => (
              <div
                key={t.name}
                className="bg-white rounded-xl border border-gray-200 p-6"
              >
                <p className="text-gray-700 text-sm italic mb-4">
                  &ldquo;{t.quote}&rdquo;
                </p>
                <div>
                  <div className="font-semibold text-sm">{t.name}</div>
                  <div className="text-xs text-gray-500">
                    {t.shop} — {t.location}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Demo / Hear It In Action */}
      <section className="py-16 px-6 bg-gray-50">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Hear It In Action</h2>
          <p className="text-gray-600 mb-8">
            Listen to a real demo call. The AI answers as your shop, captures
            vehicle details, and books the appointment — all in under 2 minutes.
          </p>
          <div className="bg-white rounded-xl border border-gray-200 p-8">
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center">
                <svg
                  className="w-5 h-5 text-white ml-0.5"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                </svg>
              </div>
              <div className="text-left">
                <div className="font-semibold">Demo: Appointment Booking Call</div>
                <div className="text-sm text-gray-500">
                  Customer calls about brake noise — AI handles full intake — 1:42
                </div>
              </div>
            </div>
            <p className="text-sm text-gray-500">
              Want a personalized demo with your shop name? Sign up for a free
              trial and hear it live.
            </p>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="py-12 px-6 bg-blue-900 text-white">
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          {[
            { value: "10,000+", label: "Calls Answered" },
            { value: "500+", label: "Shops Using It" },
            { value: "98%", label: "Caller Satisfaction" },
            { value: "< 1 sec", label: "Average Pickup Time" },
          ].map((s) => (
            <div key={s.label}>
              <div className="text-3xl font-bold">{s.value}</div>
              <div className="text-blue-200 text-sm mt-1">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 bg-blue-700 text-white">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to stop missing calls?
          </h2>
          <p className="text-blue-100 mb-8 text-lg">
            Join hundreds of shops converting more callers into booked
            appointments.
          </p>
          <Link
            href="/signup"
            className="inline-block bg-white text-blue-700 px-8 py-3 rounded-lg text-lg font-semibold hover:bg-blue-50"
          >
            Start Free Trial
          </Link>
          <p className="mt-4 text-blue-200 text-sm">
            No credit card required. Cancel anytime.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-gray-200 text-center text-sm text-gray-500">
        <p>AutoShop Voice AI</p>
      </footer>
    </div>
  );
}
