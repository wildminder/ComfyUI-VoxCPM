import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { sql } from "drizzle-orm";

/**
 * Campaign management API — admin only.
 * GET: List campaigns with stats
 * POST: Create a new campaign
 */

export async function GET() {
  const session = await getSession();
  if (!session || session.user.role !== "admin") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const campaigns = await db.execute(sql`
    SELECT
      c.*,
      (SELECT count(*) FROM leads l
       WHERE l.status NOT IN ('unsubscribed', 'bounced')
       AND (c.target_states IS NULL OR l.state = ANY(c.target_states))
      ) as available_leads
    FROM campaigns c
    ORDER BY c.created_at DESC
    LIMIT 50
  `);

  // Lead stats
  const [leadStats] = await db.execute(sql`
    SELECT
      count(*) as total,
      count(*) filter (where status = 'new') as new_leads,
      count(*) filter (where status = 'contacted') as contacted,
      count(*) filter (where status = 'opened') as opened,
      count(*) filter (where status = 'clicked') as clicked,
      count(*) filter (where status = 'replied') as replied,
      count(*) filter (where status = 'converted') as converted,
      count(*) filter (where status = 'unsubscribed') as unsubscribed,
      count(*) filter (where status = 'bounced') as bounced
    FROM leads
  `) as any;

  return NextResponse.json({
    campaigns: campaigns.rows || campaigns,
    leadStats: leadStats || {},
  });
}

export async function POST(req: NextRequest) {
  const session = await getSession();
  if (!session || session.user.role !== "admin") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const body = await req.json();

  const { name, subjectLine, subjectLineB, previewText, templateId, targetStates, sendRate } = body;

  if (!name || !subjectLine || !templateId) {
    return NextResponse.json(
      { error: "name, subjectLine, and templateId are required" },
      { status: 400 }
    );
  }

  const [campaign] = await db.execute(sql`
    INSERT INTO campaigns (name, subject_line, subject_line_b, preview_text, template_id, target_states, send_rate)
    VALUES (${name}, ${subjectLine}, ${subjectLineB || null}, ${previewText || ""}, ${templateId}, ${targetStates || null}, ${sendRate || 100})
    RETURNING *
  `) as any;

  return NextResponse.json({ campaign: campaign.rows?.[0] || campaign });
}
