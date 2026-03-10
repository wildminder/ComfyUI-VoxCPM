import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { sql } from "drizzle-orm";
import { sendEmail, renderTemplate, EMAIL_TEMPLATES } from "@/lib/email-campaigns";

/**
 * Send a batch of campaign emails.
 * POST: { campaignId, batchSize?: number }
 *
 * This is designed to be called repeatedly (by a cron job or n8n workflow)
 * to send emails in controlled batches.
 */
export async function POST(req: NextRequest) {
  const session = await getSession();
  if (!session || session.user.role !== "admin") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { campaignId, batchSize = 50 } = await req.json();

  if (!campaignId) {
    return NextResponse.json({ error: "campaignId required" }, { status: 400 });
  }

  // Get campaign
  const [campaign] = (await db.execute(sql`
    SELECT * FROM campaigns WHERE id = ${campaignId}
  `) as any).rows || [];

  if (!campaign) {
    return NextResponse.json({ error: "Campaign not found" }, { status: 404 });
  }

  if (campaign.status !== "sending" && campaign.status !== "scheduled") {
    return NextResponse.json(
      { error: `Campaign status is '${campaign.status}', must be 'sending' or 'scheduled'` },
      { status: 400 }
    );
  }

  // Update campaign status to sending
  await db.execute(sql`
    UPDATE campaigns
    SET status = 'sending', started_at = COALESCE(started_at, now()), updated_at = now()
    WHERE id = ${campaignId}
  `);

  // Get unsent leads for this campaign
  const leads = (await db.execute(sql`
    SELECT l.*
    FROM leads l
    WHERE l.status NOT IN ('unsubscribed', 'bounced')
    AND l.email NOT IN (SELECT email FROM email_unsubscribes)
    AND l.id NOT IN (
      SELECT lead_id FROM campaign_sends WHERE campaign_id = ${campaignId}
    )
    ${campaign.target_states ? sql`AND l.state = ANY(${campaign.target_states})` : sql``}
    ORDER BY l.created_at
    LIMIT ${batchSize}
  `) as any).rows || [];

  if (leads.length === 0) {
    // Campaign complete
    await db.execute(sql`
      UPDATE campaigns
      SET status = 'completed', completed_at = now(), updated_at = now()
      WHERE id = ${campaignId}
    `);
    return NextResponse.json({ sent: 0, remaining: 0, status: "completed" });
  }

  const templateId = campaign.template_id as keyof typeof EMAIL_TEMPLATES;
  const results: { email: string; success: boolean; error?: string }[] = [];
  const appUrl = process.env.NEXT_PUBLIC_APP_URL || "https://autoshopvoice.ai";

  for (const lead of leads) {
    // Determine A/B variant
    const useVariantB = campaign.subject_line_b && Math.random() < 0.5;
    const subjectLine = useVariantB ? campaign.subject_line_b : campaign.subject_line;
    const variant = useVariantB ? "b" : "a";

    try {
      // Record send attempt FIRST to prevent duplicate sends on retry
      await db.execute(sql`
        INSERT INTO campaign_sends (campaign_id, lead_id, email, subject_variant, email_provider_id, status, sent_at)
        VALUES (${campaignId}, ${lead.id}, ${lead.email}, ${variant}, '', 'pending', now())
        ON CONFLICT DO NOTHING
      `);

      const { subject, html } = renderTemplate(templateId, {
        shop_name: lead.shop_name || "Your Shop",
        owner_name: lead.owner_name || "",
        city: lead.city || "",
        state: lead.state || "",
        cta_url: `${appUrl}/signup?ref=email&campaign=${campaignId}&lead=${lead.id}`,
        unsubscribe_url: `${appUrl}/api/unsubscribe?email=${encodeURIComponent(lead.email)}&id=${lead.id}`,
      });

      // Override subject with campaign subject line (which may have A/B variant)
      const finalSubject = subjectLine
        .replace(/\{\{shop_name\}\}/g, lead.shop_name || "Your Shop");

      const result = await sendEmail({
        to: lead.email,
        subject: finalSubject,
        html,
        tags: [`campaign:${campaignId}`, `lead:${lead.id}`],
      });

      // Update send record with result
      await db.execute(sql`
        UPDATE campaign_sends
        SET status = ${result.success ? "sent" : "failed"}, email_provider_id = ${result.id || ""}
        WHERE campaign_id = ${campaignId} AND lead_id = ${lead.id}
      `);

      // Update lead status on success
      if (result.success) {
        await db.execute(sql`
          UPDATE leads
          SET status = 'contacted', sent_count = sent_count + 1, last_sent_at = now(), last_campaign_id = ${campaignId}, updated_at = now()
          WHERE id = ${lead.id}
        `);
      }

      results.push({ email: lead.email, success: result.success, error: result.error });
    } catch (e) {
      // Mark as failed if error thrown
      await db.execute(sql`
        UPDATE campaign_sends
        SET status = 'failed'
        WHERE campaign_id = ${campaignId} AND lead_id = ${lead.id}
      `).catch(() => {}); // Best-effort update

      results.push({
        email: lead.email,
        success: false,
        error: e instanceof Error ? e.message : "Unknown error",
      });
    }
  }

  // Update campaign stats
  const successCount = results.filter((r) => r.success).length;
  await db.execute(sql`
    UPDATE campaigns
    SET sent = sent + ${successCount}, updated_at = now()
    WHERE id = ${campaignId}
  `);

  // Check remaining
  const [remaining] = (await db.execute(sql`
    SELECT count(*) as count
    FROM leads l
    WHERE l.status NOT IN ('unsubscribed', 'bounced')
    AND l.email NOT IN (SELECT email FROM email_unsubscribes)
    AND l.id NOT IN (SELECT lead_id FROM campaign_sends WHERE campaign_id = ${campaignId})
    ${campaign.target_states ? sql`AND l.state = ANY(${campaign.target_states})` : sql``}
  `) as any).rows || [];

  return NextResponse.json({
    sent: successCount,
    failed: results.length - successCount,
    remaining: remaining?.count ?? 0,
    results,
  });
}
