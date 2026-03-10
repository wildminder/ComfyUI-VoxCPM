import { NextRequest, NextResponse } from "next/server";
import { stripe } from "@/lib/stripe";
import { db } from "@/lib/db";
import { shops } from "@/lib/schema";
import { eq } from "drizzle-orm";
import { provisionShop } from "@/lib/provision";
import Stripe from "stripe";

// Simple idempotency: track processed event IDs in memory
const processedEvents = new Map<string, number>();
const MAX_EVENT_AGE_SEC = 300; // 5 minutes

// Periodic cleanup of old entries (every 10 minutes)
setInterval(() => {
  const cutoff = Date.now() - MAX_EVENT_AGE_SEC * 2 * 1000;
  for (const [id, ts] of processedEvents) {
    if (ts < cutoff) processedEvents.delete(id);
  }
}, 10 * 60 * 1000).unref?.();

export async function POST(req: NextRequest) {
  const body = await req.text();
  const sig = req.headers.get("stripe-signature");

  if (!sig) {
    return NextResponse.json({ error: "Missing signature" }, { status: 400 });
  }

  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!webhookSecret) {
    console.error("STRIPE_WEBHOOK_SECRET not configured");
    return NextResponse.json({ error: "Not configured" }, { status: 500 });
  }

  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(body, sig, webhookSecret);
  } catch (err) {
    console.error("Stripe webhook signature verification failed:", err instanceof Error ? err.message : err);
    return NextResponse.json({ error: "Invalid signature" }, { status: 400 });
  }

  // Reject stale events (older than 5 minutes)
  const eventAge = Date.now() / 1000 - event.created;
  if (eventAge > MAX_EVENT_AGE_SEC) {
    return NextResponse.json({ error: "Event too old" }, { status: 400 });
  }

  // Idempotency check
  if (processedEvents.has(event.id)) {
    return NextResponse.json({ received: true });
  }
  processedEvents.set(event.id, Date.now());

  switch (event.type) {
    case "checkout.session.completed": {
      const session = event.data.object as Stripe.Checkout.Session;
      const shopId = session.metadata?.shopId;
      const planId = session.metadata?.planId;

      if (shopId && session.subscription) {
        const subscription = await stripe.subscriptions.retrieve(
          session.subscription as string
        );

        await db
          .update(shops)
          .set({
            stripeSubscriptionId: subscription.id,
            subscriptionStatus: "trialing",
            planId: planId || "starter",
            trialEndsAt: subscription.trial_end
              ? new Date(subscription.trial_end * 1000)
              : null,
            updatedAt: new Date(),
          })
          .where(eq(shops.id, shopId));

        // Auto-provision the shop
        try {
          await provisionShop(shopId);
        } catch (e) {
          console.error("Provision failed for shop", shopId, e);
        }
      }
      break;
    }

    case "customer.subscription.updated": {
      const subscription = event.data.object as Stripe.Subscription;
      const shopId = subscription.metadata?.shopId;
      if (shopId) {
        const statusMap: Record<string, string> = {
          active: "active",
          trialing: "trialing",
          past_due: "past_due",
          canceled: "canceled",
          unpaid: "unpaid",
        };
        await db
          .update(shops)
          .set({
            subscriptionStatus: (statusMap[subscription.status] || subscription.status) as any,
            updatedAt: new Date(),
          })
          .where(eq(shops.id, shopId));
      }
      break;
    }

    case "customer.subscription.deleted": {
      const subscription = event.data.object as Stripe.Subscription;
      const shopId = subscription.metadata?.shopId;
      if (shopId) {
        await db
          .update(shops)
          .set({
            subscriptionStatus: "canceled",
            updatedAt: new Date(),
          })
          .where(eq(shops.id, shopId));
      }
      break;
    }

    case "invoice.payment_failed": {
      const invoice = event.data.object as Stripe.Invoice;
      const subId = invoice.subscription as string;
      if (subId) {
        const [shop] = await db
          .select()
          .from(shops)
          .where(eq(shops.stripeSubscriptionId, subId))
          .limit(1);

        if (shop) {
          await db
            .update(shops)
            .set({ subscriptionStatus: "past_due", updatedAt: new Date() })
            .where(eq(shops.id, shop.id));
        }
      }
      break;
    }
  }

  return NextResponse.json({ received: true });
}
