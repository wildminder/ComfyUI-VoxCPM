/**
 * DMS Integration Factory
 *
 * Creates the appropriate DMS adapter based on provider type.
 * Used by both the Next.js API routes and the n8n DMS sync workflow.
 */

export type { DmsAdapter, DmsConfig, DmsProvider, DmsCustomer, DmsVehicle, DmsRepairOrder, DmsAppointment, DmsSyncPayload } from "./types";

import type { DmsAdapter, DmsConfig, DmsProvider } from "./types";
import { TekmetricAdapter } from "./tekmetric";
import { Mitchell1Adapter } from "./mitchell1";
import { ShopWareAdapter } from "./shopware";

export function createDmsAdapter(config: DmsConfig): DmsAdapter {
  switch (config.provider) {
    case "tekmetric":
      return new TekmetricAdapter(config);
    case "mitchell1":
      return new Mitchell1Adapter(config);
    case "shopware":
      return new ShopWareAdapter(config);
    default:
      throw new Error(`Unsupported DMS provider: ${config.provider}`);
  }
}

export const DMS_PROVIDERS: Array<{
  value: DmsProvider;
  label: string;
  description: string;
  authType: "api_key" | "oauth2";
  website: string;
}> = [
  {
    value: "tekmetric",
    label: "Tekmetric",
    description: "Cloud-based shop management with real-time analytics",
    authType: "oauth2",
    website: "https://www.tekmetric.com",
  },
  {
    value: "mitchell1",
    label: "Mitchell 1 Manager SE",
    description: "Industry-standard shop management by Snap-on",
    authType: "api_key",
    website: "https://mitchell1.com",
  },
  {
    value: "shopware",
    label: "Shop-Ware",
    description: "Modern cloud shop management with digital inspections",
    authType: "api_key",
    website: "https://shop-ware.com",
  },
];

/**
 * Parse a caller's full name into first/last for DMS systems.
 */
export function parseCallerName(fullName: string): { firstName: string; lastName: string } {
  const parts = (fullName || "").trim().split(/\s+/);
  if (parts.length === 0 || !parts[0]) return { firstName: "Unknown", lastName: "" };
  if (parts.length === 1) return { firstName: parts[0], lastName: "" };
  return { firstName: parts[0], lastName: parts.slice(1).join(" ") };
}
