/**
 * Shop-Ware DMS Adapter
 *
 * Shop-Ware API: https://api.shop-ware.com
 * Auth: API key + Secret (Basic auth or Bearer)
 * Endpoints: /customers, /vehicles, /repair_orders, /appointments
 */

import type {
  DmsAdapter,
  DmsConfig,
  DmsCustomer,
  DmsVehicle,
  DmsRepairOrder,
  DmsAppointment,
} from "./types";

const DEFAULT_BASE_URL = "https://api.shop-ware.com/api/v1";

export class ShopWareAdapter implements DmsAdapter {
  readonly provider = "shopware" as const;
  private baseUrl: string;
  private shopId: string;
  private headers: Record<string, string>;

  constructor(config: DmsConfig) {
    this.baseUrl = config.apiUrl || DEFAULT_BASE_URL;
    this.shopId = config.shopExternalId || "";
    this.headers = {
      Authorization: `Bearer ${config.apiKey}`,
      "Content-Type": "application/json",
      Accept: "application/json",
    };
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const res = await fetch(url, {
      ...options,
      headers: { ...this.headers, ...options?.headers },
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Shop-Ware API ${res.status}: ${body}`);
    }

    return res.json();
  }

  async testConnection(): Promise<{ ok: boolean; error?: string }> {
    try {
      await this.request(`/shops/${this.shopId}`);
      return { ok: true };
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  }

  // ── Customers ─────────────────────────────────────────

  async createCustomer(data: Omit<DmsCustomer, "externalId">): Promise<DmsCustomer> {
    const body = {
      shop_id: Number(this.shopId),
      first_name: data.firstName,
      last_name: data.lastName,
      phone_numbers: [{ number: data.phone, type: "mobile" }],
      email_addresses: data.email ? [{ address: data.email }] : [],
      notes: data.notes || "",
    };

    const res = await this.request<{ id: number }>("/customers", {
      method: "POST",
      body: JSON.stringify(body),
    });

    return { ...data, externalId: String(res.id) };
  }

  async findCustomerByPhone(phone: string): Promise<DmsCustomer | null> {
    const cleanPhone = phone.replace(/\D/g, "");
    const res = await this.request<{ results: Array<{ id: number; first_name: string; last_name: string; phone_numbers: Array<{ number: string }>; email_addresses: Array<{ address: string }> }> }>(
      `/customers?shop_id=${this.shopId}&phone=${cleanPhone}&per_page=1`
    );

    if (!res.results?.length) return null;
    const c = res.results[0];
    return {
      externalId: String(c.id),
      firstName: c.first_name || "",
      lastName: c.last_name || "",
      phone: c.phone_numbers?.[0]?.number || phone,
      email: c.email_addresses?.[0]?.address,
    };
  }

  async getCustomer(externalId: string): Promise<DmsCustomer | null> {
    try {
      const c = await this.request<{ id: number; first_name: string; last_name: string; phone_numbers: Array<{ number: string }>; email_addresses: Array<{ address: string }> }>(
        `/customers/${externalId}`
      );
      return {
        externalId: String(c.id),
        firstName: c.first_name || "",
        lastName: c.last_name || "",
        phone: c.phone_numbers?.[0]?.number || "",
        email: c.email_addresses?.[0]?.address,
      };
    } catch {
      return null;
    }
  }

  // ── Vehicles ──────────────────────────────────────────

  async createVehicle(data: Omit<DmsVehicle, "externalId">): Promise<DmsVehicle> {
    const body = {
      customer_id: Number(data.customerId),
      year: data.year,
      make: data.make,
      model: data.model,
      vin: data.vin || "",
      license_plate: data.plate || "",
      odometer: data.mileage,
    };

    const res = await this.request<{ id: number }>("/vehicles", {
      method: "POST",
      body: JSON.stringify(body),
    });

    return { ...data, externalId: String(res.id) };
  }

  async findVehicleByVin(vin: string): Promise<DmsVehicle | null> {
    const res = await this.request<{ results: Array<{ id: number; customer_id: number; year: number; make: string; model: string; vin: string }> }>(
      `/vehicles?vin=${vin}&per_page=1`
    );

    if (!res.results?.length) return null;
    const v = res.results[0];
    return {
      externalId: String(v.id),
      customerId: String(v.customer_id),
      year: v.year,
      make: v.make || "",
      model: v.model || "",
      vin: v.vin,
    };
  }

  async getVehicle(externalId: string): Promise<DmsVehicle | null> {
    try {
      const v = await this.request<{ id: number; customer_id: number; year: number; make: string; model: string; vin: string; license_plate: string }>(
        `/vehicles/${externalId}`
      );
      return {
        externalId: String(v.id),
        customerId: String(v.customer_id),
        year: v.year,
        make: v.make || "",
        model: v.model || "",
        vin: v.vin,
        plate: v.license_plate,
      };
    } catch {
      return null;
    }
  }

  // ── Repair Orders ─────────────────────────────────────

  async createRepairOrder(data: Omit<DmsRepairOrder, "externalId">): Promise<DmsRepairOrder> {
    const body = {
      shop_id: Number(this.shopId),
      customer_id: Number(data.customerId),
      vehicle_id: Number(data.vehicleId),
      customer_concern: data.description,
      state: "estimate",
    };

    const res = await this.request<{ id: number }>("/repair_orders", {
      method: "POST",
      body: JSON.stringify(body),
    });

    return { ...data, externalId: String(res.id) };
  }

  async getRepairOrder(externalId: string): Promise<DmsRepairOrder | null> {
    try {
      const ro = await this.request<{ id: number; customer_id: number; vehicle_id: number; state: string; customer_concern: string; created_at: string }>(
        `/repair_orders/${externalId}`
      );
      return {
        externalId: String(ro.id),
        customerId: String(ro.customer_id),
        vehicleId: String(ro.vehicle_id),
        status: ro.state || "unknown",
        description: ro.customer_concern || "",
        createdAt: ro.created_at,
      };
    } catch {
      return null;
    }
  }

  async getRepairOrdersByCustomer(customerId: string): Promise<DmsRepairOrder[]> {
    const res = await this.request<{ results: Array<{ id: number; customer_id: number; vehicle_id: number; state: string; customer_concern: string }> }>(
      `/repair_orders?customer_id=${customerId}&per_page=10&sort=-created_at`
    );

    return (res.results || []).map((ro) => ({
      externalId: String(ro.id),
      customerId: String(ro.customer_id),
      vehicleId: String(ro.vehicle_id),
      status: ro.state || "unknown",
      description: ro.customer_concern || "",
    }));
  }

  // ── Appointments ──────────────────────────────────────

  async createAppointment(data: Omit<DmsAppointment, "externalId">): Promise<DmsAppointment> {
    const body = {
      shop_id: Number(this.shopId),
      customer_id: Number(data.customerId),
      vehicle_id: data.vehicleId ? Number(data.vehicleId) : undefined,
      start_at: data.scheduledAt,
      duration_minutes: data.duration || 60,
      title: data.serviceDescription,
      notes: data.notes || "",
    };

    const res = await this.request<{ id: number }>("/appointments", {
      method: "POST",
      body: JSON.stringify(body),
    });

    return { ...data, externalId: String(res.id), status: "scheduled" };
  }

  async getAppointment(externalId: string): Promise<DmsAppointment | null> {
    try {
      const a = await this.request<{ id: number; customer_id: number; vehicle_id: number; start_at: string; duration_minutes: number; title: string; status: string }>(
        `/appointments/${externalId}`
      );
      return {
        externalId: String(a.id),
        customerId: String(a.customer_id),
        vehicleId: a.vehicle_id ? String(a.vehicle_id) : undefined,
        scheduledAt: a.start_at,
        duration: a.duration_minutes,
        serviceDescription: a.title || "",
        status: a.status || "scheduled",
      };
    } catch {
      return null;
    }
  }

  async cancelAppointment(externalId: string): Promise<boolean> {
    try {
      await this.request(`/appointments/${externalId}`, {
        method: "PATCH",
        body: JSON.stringify({ status: "cancelled" }),
      });
      return true;
    } catch {
      return false;
    }
  }
}
