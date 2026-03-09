import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { users, shops } from "@/lib/schema";
import { hashPassword, createSession } from "@/lib/auth";
import { z } from "zod";

const signupSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  name: z.string().min(1),
  shopName: z.string().min(1),
});

export async function POST(req: NextRequest) {
  const body = await req.json();

  const parsed = signupSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: parsed.error.flatten() },
      { status: 400 }
    );
  }

  const { email, password, name, shopName } = parsed.data;

  // Create shop first
  const [shop] = await db
    .insert(shops)
    .values({
      name: shopName,
      slug: shopName
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/(^-|-$)/g, ""),
      onboardingStatus: "pending",
    })
    .returning();

  // Create user
  const passwordHash = await hashPassword(password);
  const [user] = await db
    .insert(users)
    .values({
      email,
      passwordHash,
      name,
      role: "owner",
      shopId: shop.id,
    })
    .returning();

  await createSession(user.id);

  return NextResponse.json({
    user: { id: user.id, email: user.email, name: user.name },
    shop: { id: shop.id, name: shop.name },
    redirect: "/onboarding/step1",
  });
}
