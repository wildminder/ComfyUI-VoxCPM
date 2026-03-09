/**
 * DMS (Dealer/Shop Management System) integration types.
 * Unified interface for Tekmetric, Mitchell 1, and Shop-Ware.
 */

export type DmsProvider = "tekmetric" | "mitchell1" | "shopware";

export interface DmsConfig {
  provider: DmsProvider;
  apiKey: string;
  apiUrl?: string;
  shopExternalId?: string;
  oauthToken?: string;
  oauthRefresh?: string;
}

// ── Unified entity types ──────────────────────────────

export interface DmsCustomer {
  externalId: string;
  firstName: string;
  lastName: string;
  phone: string;
  email?: string;
  address?: string;
  notes?: string;
}

export interface DmsVehicle {
  externalId: string;
  customerId: string;
  year?: number;
  make: string;
  model: string;
  vin?: string;
  plate?: string;
  mileage?: number;
}

export interface DmsRepairOrder {
  externalId: string;
  customerId: string;
  vehicleId: string;
  status: string;
  description: string;
  createdAt?: string;
  estimatedCompletion?: string;
  totalEstimate?: number;
}

export interface DmsAppointment {
  externalId: string;
  customerId: string;
  vehicleId?: string;
  scheduledAt: string;
  duration?: number;
  serviceDescription: string;
  status: string;
  notes?: string;
}

// ── Adapter interface ─────────────────────────────────

export interface DmsAdapter {
  readonly provider: DmsProvider;

  testConnection(): Promise<{ ok: boolean; error?: string }>;

  // Customers
  createCustomer(data: Omit<DmsCustomer, "externalId">): Promise<DmsCustomer>;
  findCustomerByPhone(phone: string): Promise<DmsCustomer | null>;
  getCustomer(externalId: string): Promise<DmsCustomer | null>;

  // Vehicles
  createVehicle(data: Omit<DmsVehicle, "externalId">): Promise<DmsVehicle>;
  findVehicleByVin(vin: string): Promise<DmsVehicle | null>;
  getVehicle(externalId: string): Promise<DmsVehicle | null>;

  // Repair Orders
  createRepairOrder(data: Omit<DmsRepairOrder, "externalId">): Promise<DmsRepairOrder>;
  getRepairOrder(externalId: string): Promise<DmsRepairOrder | null>;
  getRepairOrdersByCustomer(customerId: string): Promise<DmsRepairOrder[]>;

  // Appointments
  createAppointment(data: Omit<DmsAppointment, "externalId">): Promise<DmsAppointment>;
  getAppointment(externalId: string): Promise<DmsAppointment | null>;
  cancelAppointment(externalId: string, reason?: string): Promise<boolean>;
}

// ── Sync payload (used by n8n workflow webhook) ───────

export interface DmsSyncPayload {
  shop_id: string;
  action: "sync_customer" | "sync_vehicle" | "sync_repair_order" | "sync_appointment";
  entity_id: string;
  caller_data?: {
    name: string;
    phone: string;
  };
  vehicle_data?: {
    year?: number;
    make: string;
    model: string;
    vin?: string;
  };
  appointment_data?: {
    service_description: string;
    preferred_day: string;
    preferred_time: string;
  };
  issue_summary?: string;
}
