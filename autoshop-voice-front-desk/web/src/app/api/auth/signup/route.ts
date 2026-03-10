import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { users, shops } from "@/lib/schema";
import { hashPassword, createSession } from "@/lib/auth";
import { eq } from "drizzle-orm";
import { z } from "zod";

const signupSchema = z.object({
  email: z.string().email(),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .max(128, "Password is too long")
    .regex(/[A-Z]/, "Password must contain an uppercase letter")
    .regex(/[a-z]/, "Password must contain a lowercase letter")
    .regex(/\d/, "Password must contain a number")
    .regex(/[^A-Za-z0-9]/, "Password must contain a special character"),
  name: z.string().min(1).max(255),
  shopName: z.string().min(1).max(255),
});

// Rate limiter for signup: 5 attempts per IP per 15 minutes
const signupAttempts = new Map<string, { count: number; resetAt: number }>();
const SIGNUP_MAX_ATTEMPTS = 5;
const SIGNUP_WINDOW_MS = 15 * 60 * 1000;

function checkSignupRateLimit(key: string): boolean {
  const now = Date.now();
  const entry = signupAttempts.get(key);

  if (!entry || now > entry.resetAt) {
    signupAttempts.set(key, { count: 1, resetAt: now + SIGNUP_WINDOW_MS });
    return true;
  }

  if (entry.count >= SIGNUP_MAX_ATTEMPTS) {
    return false;
  }

  entry.count++;
  return true;
}

export async function POST(req: NextRequest) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown";

  if (!checkSignupRateLimit(ip)) {
    return NextResponse.json(
      { error: "Too many signup attempts. Try again in 15 minutes." },
      { status: 429 }
    );
  }

  const body = await req.json();

  const parsed = signupSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: parsed.error.flatten() },
      { status: 400 }
    );
  }

  const { email, password, name, shopName } = parsed.data;

  // Check for existing user before insert to provide clear error
  const [existingUser] = await db
    .select({ id: users.id })
    .from(users)
    .where(eq(users.email, email))
    .limit(1);

  if (existingUser) {
    return NextResponse.json(
      { error: "An account with this email already exists" },
      { status: 409 }
    );
  }

  try {
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
  } catch (error) {
    // Handle race condition on unique constraint
    if (
      error instanceof Error &&
      (error.message.includes("unique") || error.message.includes("duplicate"))
    ) {
      return NextResponse.json(
        { error: "An account with this email already exists" },
        { status: 409 }
      );
    }
    console.error("Signup error:", error instanceof Error ? error.message : "Unknown error");
    return NextResponse.json(
      { error: "Registration failed. Please try again." },
      { status: 500 }
    );
  }
}
