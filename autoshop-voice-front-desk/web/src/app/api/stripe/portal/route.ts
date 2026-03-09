import { NextResponse } from "next/server";
import { stripe } from "@/lib/stripe";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { shops } from "@/lib/schema";
import { eq } from "drizzle-orm";

export async function POST() {
  const session = await getSession();
  if (!session || !session.user.shopId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const [shop] = await db
    .select()
    .from(shops)
    .where(eq(shops.id, session.user.shopId))
    .limit(1);

  if (!shop?.stripeCustomerId) {
    return NextResponse.json({ error: "No billing account" }, { status: 400 });
  }

  const portalSession = await stripe.billingPortal.sessions.create({
    customer: shop.stripeCustomerId,
    return_url: `${process.env.NEXT_PUBLIC_APP_URL}/dashboard/settings`,
  });

  return NextResponse.json({ url: portalSession.url });
}
