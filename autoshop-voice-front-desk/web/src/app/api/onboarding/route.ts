import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { shops } from "@/lib/schema";
import { eq } from "drizzle-orm";
import { z } from "zod";
import { dmsIntegrations } from "@/lib/schema";

const VALID_TIMEZONES = [
  "America/New_York", "America/Chicago", "America/Denver",
  "America/Los_Angeles", "America/Phoenix", "America/Anchorage",
  "America/Adak", "Pacific/Honolulu", "America/Indiana/Indianapolis",
  "America/Detroit", "America/Kentucky/Louisville", "America/Boise",
] as const;

const phoneRegex = /^\+?1?\d{10,15}$/;

const step1Schema = z.object({
  step: z.literal("step1"),
  address: z.string().min(1).max(500),
  city: z.string().min(1).max(100),
  state: z.string().min(2).max(50),
  zip: z.string().regex(/^\d{5}(-\d{4})?$/, "Invalid ZIP code format"),
  timezone: z.string().refine(
    (tz) => (VALID_TIMEZONES as readonly string[]).includes(tz),
    "Invalid timezone"
  ),
  transferNumber: z.string().regex(phoneRegex, "Invalid phone number").optional().or(z.literal("")),
});

const step2Schema = z.object({
  step: z.literal("step2"),
  hoursJson: z.record(
    z.object({ open: z.string(), close: z.string() }).nullable()
  ),
  afterHoursEnabled: z.boolean(),
});

const step3Schema = z.object({
  step: z.literal("step3"),
  servicesText: z.string().min(1).max(5000),
  makesServicedText: z.string().min(1).max(2000),
  diagFeeText: z.string().max(1000).optional(),
  towPolicyText: z.string().max(2000).optional(),
});

const stepDmsSchema = z.object({
  step: z.literal("step_dms"),
  dmsProvider: z.enum(["tekmetric", "mitchell1", "shopware", "none"]),
  apiKey: z.string().min(8, "API key too short").max(1000).optional(),
  apiUrl: z.string().url("Must be a valid URL").optional().or(z.literal("")),
  shopExternalId: z.string().min(1).max(255).optional(),
});

export async function POST(req: NextRequest) {
  const session = await getSession();
  if (!session || !session.user.shopId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const shopId = session.user.shopId;

  if (body.step === "step1") {
    const parsed = step1Schema.safeParse(body);
    if (!parsed.success)
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });

    await db
      .update(shops)
      .set({
        address: `${parsed.data.address}, ${parsed.data.city}, ${parsed.data.state} ${parsed.data.zip}`,
        city: parsed.data.city,
        state: parsed.data.state,
        zip: parsed.data.zip,
        timezone: parsed.data.timezone,
        transferNumber: parsed.data.transferNumber || null,
        onboardingStatus: "step1_complete",
        updatedAt: new Date(),
      })
      .where(eq(shops.id, shopId));

    return NextResponse.json({ success: true, next: "/onboarding/step2" });
  }

  if (body.step === "step2") {
    const parsed = step2Schema.safeParse(body);
    if (!parsed.success)
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });

    await db
      .update(shops)
      .set({
        hoursJson: parsed.data.hoursJson,
        afterHoursEnabled: parsed.data.afterHoursEnabled,
        onboardingStatus: "step2_complete",
        updatedAt: new Date(),
      })
      .where(eq(shops.id, shopId));

    return NextResponse.json({ success: true, next: "/onboarding/step3" });
  }

  if (body.step === "step3") {
    const parsed = step3Schema.safeParse(body);
    if (!parsed.success)
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });

    await db
      .update(shops)
      .set({
        servicesText: parsed.data.servicesText,
        makesServicedText: parsed.data.makesServicedText,
        diagFeeText: parsed.data.diagFeeText || null,
        towPolicyText: parsed.data.towPolicyText || null,
        onboardingStatus: "step3_complete",
        updatedAt: new Date(),
      })
      .where(eq(shops.id, shopId));

    return NextResponse.json({ success: true, next: "/onboarding/step-dms" });
  }

  if (body.step === "step_dms") {
    const parsed = stepDmsSchema.safeParse(body);
    if (!parsed.success)
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });

    if (parsed.data.dmsProvider !== "none" && parsed.data.apiKey) {
      await db.insert(dmsIntegrations).values({
        shopId: shopId,
        provider: parsed.data.dmsProvider,
        apiKey: parsed.data.apiKey,
        apiUrl: parsed.data.apiUrl || null,
        shopExternalId: parsed.data.shopExternalId || null,
        enabled: true,
      });
    }

    return NextResponse.json({ success: true, next: "/onboarding/step4" });
  }

  return NextResponse.json({ error: "Unknown step" }, { status: 400 });
}
