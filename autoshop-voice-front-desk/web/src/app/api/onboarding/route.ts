import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";
import { shops } from "@/lib/schema";
import { eq } from "drizzle-orm";
import { z } from "zod";

const step1Schema = z.object({
  step: z.literal("step1"),
  address: z.string().min(1),
  city: z.string().min(1),
  state: z.string().min(1),
  zip: z.string().min(1),
  timezone: z.string().min(1),
  transferNumber: z.string().optional(),
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
  servicesText: z.string().min(1),
  makesServicedText: z.string().min(1),
  diagFeeText: z.string().optional(),
  towPolicyText: z.string().optional(),
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

    return NextResponse.json({ success: true, next: "/onboarding/step4" });
  }

  return NextResponse.json({ error: "Unknown step" }, { status: 400 });
}
