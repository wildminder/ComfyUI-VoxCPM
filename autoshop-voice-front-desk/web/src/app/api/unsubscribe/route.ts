import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { sql } from "drizzle-orm";

/**
 * Escape HTML special characters to prevent XSS.
 */
function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * CAN-SPAM compliant unsubscribe endpoint.
 * GET: Shows unsubscribe confirmation page
 * POST: Processes unsubscribe
 */
export async function GET(req: NextRequest) {
  const email = req.nextUrl.searchParams.get("email");

  if (!email) {
    return new NextResponse("Missing email parameter", { status: 400 });
  }

  const safeEmail = escapeHtml(email);

  // Return a simple HTML confirmation page
  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Unsubscribe - AutoShop Voice AI</title>
  <style>
    body { font-family: -apple-system, sans-serif; max-width: 500px; margin: 80px auto; padding: 20px; text-align: center; }
    .card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 40px; }
    .btn { display: inline-block; background: #dc2626; color: white; padding: 12px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; border: none; cursor: pointer; font-size: 16px; }
    .btn:hover { background: #b91c1c; }
    .success { color: #16a34a; }
  </style>
</head>
<body>
  <div class="card" id="confirm">
    <h1>Unsubscribe</h1>
    <p>Click below to unsubscribe <strong>${safeEmail}</strong> from AutoShop Voice AI emails.</p>
    <form method="POST" action="/api/unsubscribe">
      <input type="hidden" name="email" value="${safeEmail}">
      <button type="submit" class="btn">Unsubscribe Me</button>
    </form>
  </div>
</body>
</html>`;

  return new NextResponse(html, {
    headers: { "Content-Type": "text/html" },
  });
}

export async function POST(req: NextRequest) {
  const contentType = req.headers.get("content-type") || "";
  let email: string | null = null;

  if (contentType.includes("application/x-www-form-urlencoded")) {
    const formData = await req.formData();
    email = formData.get("email") as string;
  } else {
    const body = await req.json();
    email = body.email;
  }

  if (!email) {
    return new NextResponse("Missing email", { status: 400 });
  }

  const normalizedEmail = email.toLowerCase().trim();

  // Add to unsubscribe list
  await db.execute(sql`
    INSERT INTO email_unsubscribes (email, reason, unsubscribed_at)
    VALUES (${normalizedEmail}, 'user_request', now())
    ON CONFLICT (email) DO NOTHING
  `);

  // Update lead status
  await db.execute(sql`
    UPDATE leads
    SET status = 'unsubscribed', unsubscribed_at = now(), updated_at = now()
    WHERE email = ${normalizedEmail}
  `);

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Unsubscribed - AutoShop Voice AI</title>
  <style>
    body { font-family: -apple-system, sans-serif; max-width: 500px; margin: 80px auto; padding: 20px; text-align: center; }
    .card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 40px; }
    .success { color: #16a34a; }
  </style>
</head>
<body>
  <div class="card">
    <h1 class="success">Unsubscribed</h1>
    <p>You've been unsubscribed and will no longer receive emails from us.</p>
    <p style="color:#999; font-size:14px;">If this was a mistake, contact us at support@autoshopvoice.ai</p>
  </div>
</body>
</html>`;

  return new NextResponse(html, {
    headers: { "Content-Type": "text/html" },
  });
}
