import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth";
import { createDmsAdapter } from "@/lib/dms";
import { z } from "zod";

const schema = z.object({
  provider: z.enum(["tekmetric", "mitchell1", "shopware"]),
  apiKey: z.string().min(1),
  apiUrl: z.string().optional(),
  shopExternalId: z.string().optional(),
});

export async function POST(req: NextRequest) {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ ok: false, error: "Invalid parameters" }, { status: 400 });
  }

  try {
    const adapter = createDmsAdapter({
      provider: parsed.data.provider,
      apiKey: parsed.data.apiKey,
      apiUrl: parsed.data.apiUrl,
      shopExternalId: parsed.data.shopExternalId,
    });

    const result = await adapter.testConnection();
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ ok: false, error: String(e) });
  }
}
