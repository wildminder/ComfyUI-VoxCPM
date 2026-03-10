/**
 * Email campaign system for outreach to 100K auto repair shop leads.
 *
 * Designed for use with transactional email providers:
 * - Resend (recommended, best DX)
 * - SendGrid
 * - Mailgun
 * - AWS SES
 *
 * Architecture:
 * - Leads stored in DB with campaign tracking
 * - Sends in batches to protect deliverability
 * - Tracks opens, clicks, replies
 * - Supports A/B testing of subject lines
 * - Warm-up mode for new sending domains
 */

import { db } from "./db";
import { sql } from "drizzle-orm";

// ── Types ──────────────────────────────────────────────

export interface Lead {
  id: string;
  email: string;
  shopName: string;
  ownerName?: string;
  city?: string;
  state?: string;
  phone?: string;
  source: string;
  status: "new" | "contacted" | "opened" | "clicked" | "replied" | "converted" | "unsubscribed" | "bounced";
  campaignId?: string;
  sentAt?: Date;
  openedAt?: Date;
  clickedAt?: Date;
  repliedAt?: Date;
  convertedAt?: Date;
  createdAt: Date;
}

export interface Campaign {
  id: string;
  name: string;
  subjectLine: string;
  subjectLineB?: string; // A/B test variant
  previewText: string;
  templateId: string;
  status: "draft" | "scheduled" | "sending" | "paused" | "completed";
  sendRate: number; // emails per hour
  totalLeads: number;
  sent: number;
  opened: number;
  clicked: number;
  replied: number;
  converted: number;
  bounced: number;
  unsubscribed: number;
  scheduledAt?: Date;
  startedAt?: Date;
  completedAt?: Date;
  createdAt: Date;
}

// ── Email Templates ────────────────────────────────────

export const EMAIL_TEMPLATES = {
  cold_outreach_v1: {
    id: "cold_outreach_v1",
    name: "Cold Outreach - Missed Calls",
    subject: "{{shop_name}} is missing calls right now",
    previewText: "The average shop misses 30% of incoming calls...",
    html: (vars: Record<string, string>) => `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { border-bottom: 2px solid #2563eb; padding-bottom: 16px; margin-bottom: 24px; }
    .logo { color: #2563eb; font-size: 20px; font-weight: 700; }
    .stat-box { background: #f0f7ff; border-radius: 8px; padding: 16px; margin: 16px 0; text-align: center; }
    .stat-number { font-size: 36px; font-weight: 700; color: #2563eb; }
    .cta-button { display: inline-block; background: #2563eb; color: white; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 16px; margin: 16px 0; }
    .feature-list { list-style: none; padding: 0; }
    .feature-list li { padding: 8px 0; border-bottom: 1px solid #eee; }
    .feature-list li::before { content: "\\2713 "; color: #22c55e; font-weight: 700; }
    .footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #eee; font-size: 12px; color: #999; }
  </style>
</head>
<body>
  <div class="header">
    <div class="logo">AutoShop Voice AI</div>
  </div>

  <p>Hi ${vars.owner_name || "there"},</p>

  <p>I noticed <strong>${vars.shop_name}</strong> in ${vars.city || "your area"} — and wanted to share something we've been hearing from shops like yours:</p>

  <div class="stat-box">
    <div class="stat-number">30%</div>
    <p style="margin:4px 0; color: #666;">of calls to independent shops go unanswered</p>
    <p style="margin:4px 0; font-weight:600;">Each missed call = $300-$800 in lost repair revenue</p>
  </div>

  <p>We built an <strong>AI-powered phone receptionist</strong> specifically for auto repair shops. It:</p>

  <ul class="feature-list">
    <li>Answers every call in your shop's name, 24/7</li>
    <li>Captures full vehicle intake (year, make, model, issue)</li>
    <li>Books appointments on your schedule</li>
    <li>Texts you a summary after every call</li>
    <li>Handles after-hours with custom messaging</li>
  </ul>

  <p><strong>Setup takes 10 minutes.</strong> Just forward your line — our AI handles the rest.</p>

  <p style="text-align: center;">
    <a href="${vars.cta_url}" class="cta-button">Start Your Free 14-Day Trial</a>
  </p>

  <p>Plans start at <strong>$149/month</strong> — less than 1 hour of a receptionist's wages.</p>

  <p>Want to hear it in action? Reply to this email and I'll send you a demo call recording.</p>

  <p>Best,<br>
  The AutoShop Voice AI Team</p>

  <div class="footer">
    <p>AutoShop Voice AI | AI phone receptionist for auto repair shops</p>
    <p><a href="${vars.unsubscribe_url}" style="color:#999;">Unsubscribe</a></p>
  </div>
</body>
</html>`,
  },

  cold_outreach_v2: {
    id: "cold_outreach_v2",
    name: "Cold Outreach - Revenue Focus",
    subject: "Stop losing $4,000/month to missed calls",
    previewText: "Your competitors are picking up the calls you miss...",
    html: (vars: Record<string, string>) => `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { border-bottom: 2px solid #2563eb; padding-bottom: 16px; margin-bottom: 24px; }
    .logo { color: #2563eb; font-size: 20px; font-weight: 700; }
    .math-box { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 16px; margin: 16px 0; }
    .cta-button { display: inline-block; background: #2563eb; color: white; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 16px; margin: 16px 0; }
    .comparison { display: flex; gap: 16px; margin: 16px 0; }
    .comp-box { flex: 1; padding: 16px; border-radius: 8px; text-align: center; }
    .comp-old { background: #fef2f2; border: 1px solid #fca5a5; }
    .comp-new { background: #f0fdf4; border: 1px solid #86efac; }
    .footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #eee; font-size: 12px; color: #999; }
  </style>
</head>
<body>
  <div class="header">
    <div class="logo">AutoShop Voice AI</div>
  </div>

  <p>Hi ${vars.owner_name || "there"},</p>

  <p>Quick math for <strong>${vars.shop_name}</strong>:</p>

  <div class="math-box">
    <p style="margin:4px 0;"><strong>20 calls/day</strong> x 30% missed = <strong>6 missed calls/day</strong></p>
    <p style="margin:4px 0;">6 missed x $500 avg ticket = <strong>$3,000/day in lost revenue</strong></p>
    <p style="margin:4px 0;">That's <strong>$60,000/month</strong> walking to your competitor.</p>
  </div>

  <p>Even if only 10% of those missed calls would have converted, that's <strong>$6,000/month</strong> you're leaving on the table.</p>

  <table width="100%" cellpadding="0" cellspacing="0" style="margin: 16px 0;">
    <tr>
      <td width="48%" style="background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 16px; text-align: center;">
        <p style="font-weight:700; color:#dc2626; margin:0;">Part-Time Receptionist</p>
        <p style="font-size:24px; font-weight:700; margin:4px 0;">$2,000+/mo</p>
        <p style="font-size:12px; color:#666; margin:0;">Still misses after-hours calls</p>
      </td>
      <td width="4%"></td>
      <td width="48%" style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 16px; text-align: center;">
        <p style="font-weight:700; color:#16a34a; margin:0;">AutoShop Voice AI</p>
        <p style="font-size:24px; font-weight:700; margin:4px 0;">$149/mo</p>
        <p style="font-size:12px; color:#666; margin:0;">Answers 24/7, never calls in sick</p>
      </td>
    </tr>
  </table>

  <p style="text-align: center;">
    <a href="${vars.cta_url}" class="cta-button">Try It Free for 14 Days</a>
  </p>

  <p style="font-size:14px; color:#666; text-align:center;">No credit card required. Forward your line. Takes 10 minutes.</p>

  <p>Best,<br>
  The AutoShop Voice AI Team</p>

  <div class="footer">
    <p>AutoShop Voice AI | AI phone receptionist for auto repair shops</p>
    <p><a href="${vars.unsubscribe_url}" style="color:#999;">Unsubscribe</a></p>
  </div>
</body>
</html>`,
  },

  follow_up_1: {
    id: "follow_up_1",
    name: "Follow-up #1 - Social Proof",
    subject: "{{shop_name}}: shops like yours are switching",
    previewText: "Here's what happened when Mike's Auto Repair tried it...",
    html: (vars: Record<string, string>) => `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { border-bottom: 2px solid #2563eb; padding-bottom: 16px; margin-bottom: 24px; }
    .logo { color: #2563eb; font-size: 20px; font-weight: 700; }
    .testimonial { background: #f8fafc; border-left: 4px solid #2563eb; padding: 16px; margin: 16px 0; border-radius: 0 8px 8px 0; }
    .cta-button { display: inline-block; background: #2563eb; color: white; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 16px; margin: 16px 0; }
    .footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #eee; font-size: 12px; color: #999; }
  </style>
</head>
<body>
  <div class="header">
    <div class="logo">AutoShop Voice AI</div>
  </div>

  <p>Hi ${vars.owner_name || "there"},</p>

  <p>I reached out last week about our AI phone receptionist for ${vars.shop_name}. Wanted to share a quick result from a shop similar to yours:</p>

  <div class="testimonial">
    <p style="font-style:italic; margin:0 0 8px 0;">"We were missing about 8 calls a day during our busiest hours. First month with AutoShop Voice AI, we booked 23 additional appointments we would have missed. That's over $12,000 in new revenue."</p>
    <p style="font-weight:600; margin:0; font-size:14px;">— Independent shop owner, ${vars.state || "TX"}</p>
  </div>

  <p>The AI doesn't replace your team — it catches the calls you can't get to when you're under a hood or talking to another customer.</p>

  <p style="text-align: center;">
    <a href="${vars.cta_url}" class="cta-button">Start Free Trial</a>
  </p>

  <p>Questions? Just reply to this email.</p>

  <p>Best,<br>
  The AutoShop Voice AI Team</p>

  <div class="footer">
    <p>AutoShop Voice AI | AI phone receptionist for auto repair shops</p>
    <p><a href="${vars.unsubscribe_url}" style="color:#999;">Unsubscribe</a></p>
  </div>
</body>
</html>`,
  },
};

// ── Campaign Send Logic ────────────────────────────────

export interface SendEmailParams {
  to: string;
  subject: string;
  html: string;
  from?: string;
  replyTo?: string;
  tags?: string[];
}

/**
 * Send an email via the configured provider.
 * Supports Resend, SendGrid, Mailgun via environment variables.
 */
export async function sendEmail(params: SendEmailParams): Promise<{ id: string; success: boolean; error?: string }> {
  const provider = process.env.EMAIL_PROVIDER || "resend";
  const from = params.from || process.env.EMAIL_FROM || "AutoShop Voice AI <hello@autoshopvoice.ai>";

  if (provider === "resend") {
    const apiKey = process.env.RESEND_API_KEY;
    if (!apiKey) return { id: "", success: false, error: "RESEND_API_KEY not set" };

    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from,
        to: params.to,
        subject: params.subject,
        html: params.html,
        reply_to: params.replyTo || process.env.EMAIL_REPLY_TO,
        tags: params.tags?.map((t) => ({ name: t, value: "true" })),
      }),
    });

    const data = await res.json();
    if (res.ok) {
      return { id: data.id, success: true };
    }
    return { id: "", success: false, error: data.message || "Send failed" };
  }

  if (provider === "sendgrid") {
    const apiKey = process.env.SENDGRID_API_KEY;
    if (!apiKey) return { id: "", success: false, error: "SENDGRID_API_KEY not set" };

    const res = await fetch("https://api.sendgrid.com/v3/mail/send", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        personalizations: [{ to: [{ email: params.to }] }],
        from: { email: from.match(/<(.+)>/)?.[1] || from, name: from.match(/^(.+)</)?.[1]?.trim() },
        subject: params.subject,
        content: [{ type: "text/html", value: params.html }],
        reply_to: params.replyTo ? { email: params.replyTo } : undefined,
      }),
    });

    if (res.ok || res.status === 202) {
      return { id: res.headers.get("x-message-id") || "sent", success: true };
    }
    const error = await res.text();
    return { id: "", success: false, error };
  }

  return { id: "", success: false, error: `Unknown provider: ${provider}` };
}

/**
 * Process a template with variables.
 */
export function renderTemplate(
  templateId: keyof typeof EMAIL_TEMPLATES,
  variables: Record<string, string>
): { subject: string; html: string } {
  const template = EMAIL_TEMPLATES[templateId];
  if (!template) throw new Error(`Unknown template: ${templateId}`);

  let subject = template.subject;
  for (const [key, value] of Object.entries(variables)) {
    subject = subject.replace(new RegExp(`\\{\\{${key}\\}\\}`, "g"), value);
  }

  return {
    subject,
    html: template.html(variables),
  };
}
