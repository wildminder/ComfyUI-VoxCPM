import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { getShopUsageStats } from "@/lib/usage";

export async function GET() {
  const session = await getSession();
  if (!session || !session.user.shopId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const usage = await getShopUsageStats(session.user.shopId);
  if (!usage) {
    return NextResponse.json({ error: "Shop not found" }, { status: 404 });
  }

  return NextResponse.json(usage);
}
