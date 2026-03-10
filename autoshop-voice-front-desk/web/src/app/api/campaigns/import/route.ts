import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { sql } from "drizzle-orm";

/**
 * Import leads from CSV data.
 * POST body: { leads: [{ email, shopName, ownerName?, city?, state?, phone?, tags? }] }
 *
 * Handles deduplication and skips existing emails.
 * Designed to handle bulk imports of 100K+ leads in batches.
 */
export async function POST(req: NextRequest) {
  const session = await getSession();
  if (!session || session.user.role !== "admin") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { leads, source = "import" } = await req.json();

  if (!Array.isArray(leads) || leads.length === 0) {
    return NextResponse.json({ error: "leads array is required" }, { status: 400 });
  }

  // Process in batches of 500
  const BATCH_SIZE = 500;
  let imported = 0;
  let skipped = 0;
  let errors = 0;

  for (let i = 0; i < leads.length; i += BATCH_SIZE) {
    const batch = leads.slice(i, i + BATCH_SIZE);

    for (const lead of batch) {
      if (!lead.email || typeof lead.email !== "string") {
        errors++;
        continue;
      }

      const email = lead.email.toLowerCase().trim();

      // Validate email format
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        errors++;
        continue;
      }

      try {
        await db.execute(sql`
          INSERT INTO leads (email, shop_name, owner_name, phone, city, state, zip, source, tags)
          VALUES (
            ${email},
            ${lead.shopName || null},
            ${lead.ownerName || null},
            ${lead.phone || null},
            ${lead.city || null},
            ${lead.state || null},
            ${lead.zip || null},
            ${source},
            ${lead.tags || []}
          )
          ON CONFLICT (email) DO NOTHING
        `);
        imported++;
      } catch {
        // Duplicate or other error
        skipped++;
      }
    }
  }

  return NextResponse.json({
    imported,
    skipped,
    errors,
    total: leads.length,
  });
}
