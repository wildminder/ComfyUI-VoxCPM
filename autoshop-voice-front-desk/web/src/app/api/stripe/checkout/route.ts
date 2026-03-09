import { NextRequest, NextResponse } from "next/server";
import { stripe, PLANS, type PlanId } from "@/lib/stripe";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { shops } from "@/lib/schema";
import { eq } from "drizzle-orm";

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

  const [shop] = await db
    .select()
    .from(shops)
    .where(eq(shops.id, session.user.shopId!))
    .limit(1);

  // Create or retrieve Stripe customer
  let customerId = shop?.stripeCustomerId;
  if (!customerId) {
    const customer = await stripe.customers.create({
      email: session.user.email,
      name: shop?.name || session.user.name || undefined,
      metadata: {
        shopId: session.user.shopId || "",
        userId: session.user.id,
      },
    });
    customerId = customer.id;

    if (session.user.shopId) {
      await db
        .update(shops)
        .set({ stripeCustomerId: customerId })
        .where(eq(shops.id, session.user.shopId));
    }
  }

  const checkoutSession = await stripe.checkout.sessions.create({
    customer: customerId,
    mode: "subscription",
    payment_method_types: ["card"],
    line_items: [{ price: plan.priceId, quantity: 1 }],
    subscription_data: {
      trial_period_days: 14,
      metadata: {
        shopId: session.user.shopId || "",
        planId,
      },
    },
    success_url: `${process.env.NEXT_PUBLIC_APP_URL}/onboarding/complete?session_id={CHECKOUT_SESSION_ID}`,
    cancel_url: `${process.env.NEXT_PUBLIC_APP_URL}/signup?canceled=true`,
    metadata: {
      shopId: session.user.shopId || "",
      planId,
    },
  });

  return NextResponse.json({ url: checkoutSession.url });
}
