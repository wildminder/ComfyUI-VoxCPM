/**
 * Run this script once to create Stripe products and prices.
 * Usage: npx tsx src/lib/stripe-setup.ts
 *
 * After running, copy the price IDs into your .env file.
 */

import Stripe from "stripe";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: "2025-02-24.acacia",
});

async function setup() {
  console.log("Creating Stripe products and prices...\n");

  // Starter
  const starter = await stripe.products.create({
    name: "AutoShop Voice AI - Starter",
    description: "1 phone line, 200 calls/month, SMS recaps, basic dashboard",
  });
  const starterPrice = await stripe.prices.create({
    product: starter.id,
    unit_amount: 14900,
    currency: "usd",
    recurring: { interval: "month" },
  });
  console.log(`Starter: ${starter.id}`);
  console.log(`  Price: ${starterPrice.id} ($149/mo)\n`);

  // Professional
  const pro = await stripe.products.create({
    name: "AutoShop Voice AI - Professional",
    description:
      "3 phone lines, 500 calls/month, Cal.com booking, urgent escalation",
  });
  const proPrice = await stripe.prices.create({
    product: pro.id,
    unit_amount: 29900,
    currency: "usd",
    recurring: { interval: "month" },
  });
  console.log(`Professional: ${pro.id}`);
  console.log(`  Price: ${proPrice.id} ($299/mo)\n`);

  // Enterprise
  const enterprise = await stripe.products.create({
    name: "AutoShop Voice AI - Enterprise",
    description:
      "Unlimited lines, unlimited calls, custom prompt, API access, multi-location",
  });
  const enterprisePrice = await stripe.prices.create({
    product: enterprise.id,
    unit_amount: 59900,
    currency: "usd",
    recurring: { interval: "month" },
  });
  console.log(`Enterprise: ${enterprise.id}`);
  console.log(`  Price: ${enterprisePrice.id} ($599/mo)\n`);

  console.log("=== Add these to your .env ===");
  console.log(`STRIPE_STARTER_PRICE_ID=${starterPrice.id}`);
  console.log(`STRIPE_PRO_PRICE_ID=${proPrice.id}`);
  console.log(`STRIPE_ENTERPRISE_PRICE_ID=${enterprisePrice.id}`);
}

setup().catch(console.error);
