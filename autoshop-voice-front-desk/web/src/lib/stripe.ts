import Stripe from "stripe";

let _stripe: Stripe | null = null;

export function getStripe(): Stripe {
  if (!_stripe) {
    const key = process.env.STRIPE_SECRET_KEY;
    if (!key) {
      throw new Error(
        "FATAL: STRIPE_SECRET_KEY environment variable is not set."
      );
    }
    _stripe = new Stripe(key, {
      apiVersion: "2025-02-24.acacia",
      typescript: true,
    });
  }
  return _stripe;
}

/** Lazy proxy for convenience */
export const stripe = new Proxy({} as Stripe, {
  get(_, prop) {
    return (getStripe() as any)[prop];
  },
});

export const PLANS = {
  starter: {
    name: "Starter",
    priceMonthly: 149,
    get priceId() {
      return process.env.STRIPE_STARTER_PRICE_ID || "";
    },
    features: [
      "1 phone line",
      "Up to 200 calls/month",
      "SMS recaps",
      "Appointment requests",
      "Basic dashboard",
    ],
    limits: { lines: 1, callsPerMonth: 200 },
  },
  professional: {
    name: "Professional",
    priceMonthly: 299,
    get priceId() {
      return process.env.STRIPE_PRO_PRICE_ID || "";
    },
    features: [
      "Up to 3 phone lines",
      "Up to 500 calls/month",
      "SMS recaps",
      "Cal.com live booking",
      "Urgent escalation",
      "Full dashboard",
    ],
    limits: { lines: 3, callsPerMonth: 500 },
  },
  enterprise: {
    name: "Enterprise",
    priceMonthly: 599,
    get priceId() {
      return process.env.STRIPE_ENTERPRISE_PRICE_ID || "";
    },
    features: [
      "Unlimited phone lines",
      "Unlimited calls",
      "Priority support",
      "Cal.com live booking",
      "Custom agent prompt",
      "API access",
      "Multi-location",
    ],
    limits: { lines: -1, callsPerMonth: -1 },
  },
} as const;

export type PlanId = keyof typeof PLANS;
