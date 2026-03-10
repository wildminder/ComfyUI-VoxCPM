import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { shops } from "@/lib/schema";
import { eq } from "drizzle-orm";
import { checkCallLimit } from "@/lib/usage";
import crypto from "crypto";

/**
 * Retell inbound webhook handler — routes calls to the correct shop config.
 * This replaces direct n8n webhooks with per-shop auth and limit enforcement.
 *
 * Retell sends a POST with the call data. We:
 * 1. Verify the webhook signature
 * 2. Look up the shop by the called number (to_number)
 * 3. Check call limits
 * 4. Return the dynamic variables for this shop's agent
 */
export async function POST(req: NextRequest) {
  const body = await req.text();

  // Verify Retell webhook signature if configured
  const signature = req.headers.get("x-retell-signature");
  const webhookSecret = process.env.RETELL_WEBHOOK_SECRET;

  if (webhookSecret && signature) {
    const expectedSig = crypto
      .createHmac("sha256", webhookSecret)
      .update(body)
      .digest("hex");

    if (signature !== expectedSig) {
      return NextResponse.json({ error: "Invalid signature" }, { status: 401 });
    }
  }

  let payload: any;
  try {
    payload = JSON.parse(body);
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const event = payload.event;

  // Handle different Retell webhook events
  if (event === "call_started" || event === "call_analyzed") {
    const callData = payload.data || payload.call;
    const toNumber = callData?.to_number;
    const agentId = callData?.agent_id;

    // Look up shop by phone number or agent ID
    let shop;
    if (toNumber) {
      [shop] = await db
        .select()
        .from(shops)
        .where(eq(shops.mainNumber, toNumber))
        .limit(1);
    }
    if (!shop && agentId) {
      [shop] = await db
        .select()
        .from(shops)
        .where(eq(shops.retellAgentId, agentId))
        .limit(1);
    }

    if (!shop) {
      return NextResponse.json(
        { error: "No shop found for this number" },
        { status: 404 }
      );
    }

    // Check call limits for call_started events
    if (event === "call_started") {
      const { allowed, used, limit } = await checkCallLimit(shop.id);
      if (!allowed) {
        // Return a response that tells Retell to play a limit-exceeded message
        return NextResponse.json({
          shop_id: shop.id,
          limit_exceeded: true,
          message:
            "We apologize, but our automated system is currently unavailable. " +
            "Please call back during business hours or leave a message.",
        });
      }
    }

    // Build dynamic variables for the shop
    const hours = shop.hoursJson as Record<string, any> || {};
    const now = new Date();
    const dayNames = [
      "sunday", "monday", "tuesday", "wednesday",
      "thursday", "friday", "saturday",
    ];
    const today = dayNames[now.getDay()];
    const todayHours = hours[today];
    const isOpen = todayHours
      ? isWithinHours(now, todayHours.open, todayHours.close, shop.timezone || "America/Chicago")
      : false;

    return NextResponse.json({
      shop_id: shop.id,
      dynamic_variables: {
        shop_name: shop.name,
        shop_address: shop.address || "",
        shop_hours_today: todayHours
          ? `${todayHours.open} to ${todayHours.close}`
          : "Closed today",
        shop_hours_json: JSON.stringify(hours),
        is_open: isOpen,
        services_text: shop.servicesText || "We offer general auto repair services.",
        makes_serviced_text: shop.makesServicedText || "We service all makes and models.",
        diag_fee_text: shop.diagFeeText || "Please call for diagnostic fee information.",
        tow_policy_text: shop.towPolicyText || "",
        transfer_number: shop.transferNumber || "",
        booking_mode: shop.bookingMode || "request_only",
        after_hours: !isOpen,
      },
    });
  }

  // Forward post_call events to n8n for processing
  if (event === "call_ended" || event === "call_analyzed") {
    if (process.env.N8N_BASE_URL) {
      try {
        await fetch(`${process.env.N8N_BASE_URL}/webhook/retell-post-call`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        });
      } catch (e) {
        console.error("Failed to forward to n8n:", e);
      }
    }
  }

  return NextResponse.json({ received: true });
}

function isWithinHours(
  now: Date,
  openStr: string,
  closeStr: string,
  timezone: string
): boolean {
  try {
    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    const nowTime = formatter.format(now);
    const [nowH, nowM] = nowTime.split(":").map(Number);
    const nowMinutes = nowH * 60 + nowM;

    const [openH, openM] = openStr.split(":").map(Number);
    const [closeH, closeM] = closeStr.split(":").map(Number);
    const openMinutes = openH * 60 + openM;
    const closeMinutes = closeH * 60 + closeM;

    return nowMinutes >= openMinutes && nowMinutes < closeMinutes;
  } catch {
    return false;
  }
}
