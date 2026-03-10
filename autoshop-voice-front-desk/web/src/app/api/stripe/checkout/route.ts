import { NextRequest, NextResponse } from "next/server";
import { stripe, PLANS, type PlanId } from "@/lib/stripe";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { shops } from "@/lib/schema";
import { eq } from "drizzle-orm";

function getValidatedAppUrl(): string {
  const appUrl = process.env.NEXT_PUBLIC_APP_URL;
  if (!appUrl) {
    throw new Error("NEXT_PUBLIC_APP_URL is not configured");
  }
  try {
    const url = new URL(appUrl);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      throw new Error("NEXT_PUBLIC_APP_URL must use http or https");
    }
    return url.origin;
  } catch {
    throw new Error("NEXT_PUBLIC_APP_URL is not a valid URL");
  }
}

export async function POST(req: NextRequest) {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { planId } = (await req.json()) as { planId: PlanId };
  const plan = PLANS[planId];
  if (!plan) {
    return NextResponse.json({ error: "Invalid plan" }, { status: 400 });
  }

  if (!session.user.shopId) {
    return NextResponse.json({ error: "No shop associated" }, { status: 400 });
  }

  const [shop] = await db
    .select()
    .from(shops)
    .where(eq(shops.id, session.user.shopId))
    .limit(1);

  // Create or retrieve Stripe customer
  let customerId = shop?.stripeCustomerId;
  if (!customerId) {
    const customer = await stripe.customers.create({
      email: session.user.email,
      name: shop?.name || session.user.name || undefined,
      metadata: {
        shopId: session.user.shopId,
        userId: session.user.id,
      },
    });
    customerId = customer.id;

    await db
      .update(shops)
      .set({ stripeCustomerId: customerId })
      .where(eq(shops.id, session.user.shopId));
  }

  const appUrl = getValidatedAppUrl();

  const checkoutSession = await stripe.checkout.sessions.create({
    customer: customerId,
    mode: "subscription",
    payment_method_types: ["card"],
    line_items: [{ price: plan.priceId, quantity: 1 }],
    subscription_data: {
      trial_period_days: 14,
      metadata: {
        shopId: session.user.shopId,
        planId,
      },
    },
    success_url: `${appUrl}/onboarding/complete?session_id={CHECKOUT_SESSION_ID}`,
    cancel_url: `${appUrl}/signup?canceled=true`,
    metadata: {
      shopId: session.user.shopId,
      planId,
    },
  });

  return NextResponse.json({ url: checkoutSession.url });
}
