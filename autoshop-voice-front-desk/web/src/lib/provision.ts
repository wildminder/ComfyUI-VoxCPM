import { db } from "./db";
import { shops } from "./schema";
import { eq } from "drizzle-orm";

/**
 * Auto-provision a shop after onboarding is complete and payment is active.
 *
 * Steps:
 * 1. Create Retell agent via API
 * 2. Purchase/assign Twilio phone number
 * 3. Configure Retell webhooks pointing to n8n
 * 4. Update shop record with agent_id and phone number
 * 5. Trigger n8n shop_config_sync
 */
export async function provisionShop(shopId: string) {
  const [shop] = await db
    .select()
    .from(shops)
    .where(eq(shops.id, shopId))
    .limit(1);

  if (!shop) throw new Error(`Shop ${shopId} not found`);

  const results: string[] = [];
  let retellAgentId: string | null = null;
  let twilioNumber: string | null = null;

  // Step 1: Create Retell agent (required)
  if (!shop.retellAgentId) {
    if (!process.env.RETELL_API_KEY) {
      await markProvisioningFailed(shopId, "RETELL_API_KEY not configured");
      throw new Error("RETELL_API_KEY not configured — cannot provision");
    }

    const retellRes = await fetch("https://api.retellai.com/v2/agent", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.RETELL_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        agent_name: `${shop.name} Front Desk`,
        webhook_url: `${process.env.N8N_BASE_URL}/webhook/retell-inbound`,
        post_call_analysis_data: [
          { name: "intent", type: "enum", enum: ["new_appointment", "existing_status", "price_question", "hours_address", "urgent_breakdown", "tow_request", "vendor_sales", "wrong_number", "angry_customer"] },
          { name: "urgency", type: "enum", enum: ["low", "medium", "high", "urgent"] },
          { name: "customer_name", type: "string" },
          { name: "vehicle_year", type: "number" },
          { name: "vehicle_make", type: "string" },
          { name: "vehicle_model", type: "string" },
          { name: "issue_summary", type: "string" },
          { name: "vehicle_drivable", type: "enum", enum: ["yes", "no", "unknown"] },
          { name: "preferred_day", type: "string" },
          { name: "preferred_time", type: "string" },
          { name: "needs_tow", type: "boolean" },
          { name: "human_transfer_requested", type: "boolean" },
          { name: "sentiment", type: "enum", enum: ["positive", "neutral", "negative", "angry"] },
        ],
      }),
    });

    if (!retellRes.ok) {
      const errText = await retellRes.text().catch(() => "");
      await markProvisioningFailed(shopId, `Retell agent creation failed (${retellRes.status}): ${errText}`);
      throw new Error(`Retell agent creation failed: ${retellRes.status}`);
    }

    const agentData = await retellRes.json();
    retellAgentId = agentData.agent_id;
    results.push(`retell_agent_created:${retellAgentId}`);
  } else {
    retellAgentId = shop.retellAgentId;
  }

  // Step 2: Purchase Twilio number (required)
  if (!shop.mainNumber) {
    if (!process.env.TWILIO_ACCOUNT_SID || !process.env.TWILIO_AUTH_TOKEN) {
      // Rollback: delete the Retell agent we just created
      await rollbackRetellAgent(retellAgentId, shop.retellAgentId);
      await markProvisioningFailed(shopId, "TWILIO credentials not configured");
      throw new Error("TWILIO credentials not configured — cannot provision");
    }

    const areaCode = shop.zip?.substring(0, 3) || "312";
    const authHeader = Buffer.from(
      `${process.env.TWILIO_ACCOUNT_SID}:${process.env.TWILIO_AUTH_TOKEN}`
    ).toString("base64");

    const searchUrl = `https://api.twilio.com/2010-04-01/Accounts/${process.env.TWILIO_ACCOUNT_SID}/AvailablePhoneNumbers/US/Local.json?AreaCode=${areaCode}&Limit=1&SmsEnabled=true&VoiceEnabled=true`;
    const searchRes = await fetch(searchUrl, {
      headers: { Authorization: `Basic ${authHeader}` },
    });
    const searchData = await searchRes.json();

    if (!searchData.available_phone_numbers?.length) {
      await rollbackRetellAgent(retellAgentId, shop.retellAgentId);
      await markProvisioningFailed(shopId, `No phone numbers available for area code ${areaCode}`);
      throw new Error(`No Twilio numbers available for area code ${areaCode}`);
    }

    const phoneNumber = searchData.available_phone_numbers[0].phone_number;
    const buyUrl = `https://api.twilio.com/2010-04-01/Accounts/${process.env.TWILIO_ACCOUNT_SID}/IncomingPhoneNumbers.json`;
    const buyRes = await fetch(buyUrl, {
      method: "POST",
      headers: {
        Authorization: `Basic ${authHeader}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({ PhoneNumber: phoneNumber }),
    });

    if (!buyRes.ok) {
      await rollbackRetellAgent(retellAgentId, shop.retellAgentId);
      await markProvisioningFailed(shopId, `Twilio number purchase failed (${buyRes.status})`);
      throw new Error(`Twilio number purchase failed: ${buyRes.status}`);
    }

    twilioNumber = phoneNumber;
    results.push(`twilio_number_purchased:${phoneNumber}`);
  } else {
    twilioNumber = shop.mainNumber;
  }

  // Step 3: Update shop record (only after both steps succeed)
  await db
    .update(shops)
    .set({
      retellAgentId,
      mainNumber: twilioNumber,
      smsFromNumber: twilioNumber,
      provisionedAt: new Date(),
      onboardingStatus: "complete",
      updatedAt: new Date(),
    })
    .where(eq(shops.id, shopId));

  results.push("shop_record_updated");

  // Step 4: Trigger n8n config sync (non-critical, best-effort)
  if (process.env.N8N_BASE_URL) {
    try {
      await fetch(`${process.env.N8N_BASE_URL}/webhook/shop-config-sync`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shop_id: shopId }),
      });
      results.push("n8n_config_sync_triggered");
    } catch (e) {
      results.push(`n8n_sync_warning:${e}`);
    }
  }

  return { shopId, results };
}

/**
 * Mark a shop as failed provisioning so it can be retried.
 */
async function markProvisioningFailed(shopId: string, reason: string) {
  console.error(`Provisioning failed for shop ${shopId}: ${reason}`);
  await db
    .update(shops)
    .set({
      onboardingStatus: "step3_complete", // Revert to pre-provisioning state
      updatedAt: new Date(),
    })
    .where(eq(shops.id, shopId));
}

/**
 * Rollback a newly created Retell agent if subsequent steps fail.
 */
async function rollbackRetellAgent(newAgentId: string | null, originalAgentId: string | null) {
  if (newAgentId && newAgentId !== originalAgentId && process.env.RETELL_API_KEY) {
    try {
      await fetch(`https://api.retellai.com/v2/agent/${newAgentId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${process.env.RETELL_API_KEY}` },
      });
    } catch (e) {
      console.error(`Failed to rollback Retell agent ${newAgentId}:`, e);
    }
  }
}
