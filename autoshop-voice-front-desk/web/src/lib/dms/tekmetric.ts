/**
 * Tekmetric DMS Adapter
 *
 * Tekmetric API: https://api.tekmetric.com
 * Auth: OAuth 2.0 (Bearer token)
 * Endpoints: /customers, /vehicles, /repair-orders, /appointments
 */

import type {
  DmsAdapter,
  DmsConfig,
  DmsCustomer,
  DmsVehicle,
  DmsRepairOrder,
  DmsAppointment,
} from "./types";

const DEFAULT_BASE_URL = "https://shop.tekmetric.com/api/v1";

export class TekmetricAdapter implements DmsAdapter {
  readonly provider = "tekmetric" as const;
  private baseUrl: string;
  private shopId: string;
  private headers: Record<string, string>;

  constructor(config: DmsConfig) {
    this.baseUrl = config.apiUrl || DEFAULT_BASE_URL;
    this.shopId = config.shopExternalId || "";
    this.headers = {
      Authorization: `Bearer ${config.oauthToken || config.apiKey}`,
      "Content-Type": "application/json",
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
      throw new Error(`Tekmetric API ${res.status}: ${body}`);
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
      shopId: Number(this.shopId),
      firstName: data.firstName,
      lastName: data.lastName,
      phone: [{ number: data.phone, type: "Mobile" }],
      email: data.email ? [{ address: data.email, type: "Primary" }] : [],
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
    const res = await this.request<{ content: Array<{ id: number; firstName: string; lastName: string; phone: Array<{ number: string }> }> }>(
      `/customers?shop=${this.shopId}&search=${cleanPhone}&size=1`
    );

    if (!res.content?.length) return null;
    const c = res.content[0];
    return {
      externalId: String(c.id),
      firstName: c.firstName || "",
      lastName: c.lastName || "",
      phone: c.phone?.[0]?.number || phone,
    };
  }

  async getCustomer(externalId: string): Promise<DmsCustomer | null> {
    try {
      const c = await this.request<{ id: number; firstName: string; lastName: string; phone: Array<{ number: string }>; email: Array<{ address: string }> }>(
        `/customers/${externalId}`
      );
      return {
        externalId: String(c.id),
        firstName: c.firstName || "",
        lastName: c.lastName || "",
        phone: c.phone?.[0]?.number || "",
        email: c.email?.[0]?.address,
      };
    } catch {
      return null;
    }
  }

  // ── Vehicles ──────────────────────────────────────────

  async createVehicle(data: Omit<DmsVehicle, "externalId">): Promise<DmsVehicle> {
    const body = {
      customerId: Number(data.customerId),
      year: data.year,
      make: data.make,
      model: data.model,
      vin: data.vin || "",
      licensePlate: data.plate || "",
      mileageIn: data.mileage,
    };

    const res = await this.request<{ id: number }>("/vehicles", {
      method: "POST",
      body: JSON.stringify(body),
    });

    return { ...data, externalId: String(res.id) };
  }

  async findVehicleByVin(vin: string): Promise<DmsVehicle | null> {
    const res = await this.request<{ content: Array<{ id: number; customerId: number; year: number; make: string; model: string; vin: string }> }>(
      `/vehicles?shop=${this.shopId}&search=${vin}&size=1`
    );

    if (!res.content?.length) return null;
    const v = res.content[0];
    return {
      externalId: String(v.id),
      customerId: String(v.customerId),
      year: v.year,
      make: v.make || "",
      model: v.model || "",
      vin: v.vin,
    };
  }

  async getVehicle(externalId: string): Promise<DmsVehicle | null> {
    try {
      const v = await this.request<{ id: number; customerId: number; year: number; make: string; model: string; vin: string; licensePlate: string }>(
        `/vehicles/${externalId}`
      );
      return {
        externalId: String(v.id),
        customerId: String(v.customerId),
        year: v.year,
        make: v.make || "",
        model: v.model || "",
        vin: v.vin,
        plate: v.licensePlate,
      };
    } catch {
      return null;
    }
  }

  // ── Repair Orders ─────────────────────────────────────

  async createRepairOrder(data: Omit<DmsRepairOrder, "externalId">): Promise<DmsRepairOrder> {
    const body = {
      shopId: Number(this.shopId),
      customerId: Number(data.customerId),
      vehicleId: Number(data.vehicleId),
      customerConcern: data.description,
    };

    const res = await this.request<{ id: number }>("/repair-orders", {
      method: "POST",
      body: JSON.stringify(body),
    });

    return { ...data, externalId: String(res.id) };
  }

  async getRepairOrder(externalId: string): Promise<DmsRepairOrder | null> {
    try {
      const ro = await this.request<{ id: number; customerId: number; vehicleId: number; repairOrderStatus: { name: string }; customerConcern: string; createdDate: string }>(
        `/repair-orders/${externalId}`
      );
      return {
        externalId: String(ro.id),
        customerId: String(ro.customerId),
        vehicleId: String(ro.vehicleId),
        status: ro.repairOrderStatus?.name || "unknown",
        description: ro.customerConcern || "",
        createdAt: ro.createdDate,
      };
    } catch {
      return null;
    }
  }

  async getRepairOrdersByCustomer(customerId: string): Promise<DmsRepairOrder[]> {
    const res = await this.request<{ content: Array<{ id: number; customerId: number; vehicleId: number; repairOrderStatus: { name: string }; customerConcern: string }> }>(
      `/repair-orders?shop=${this.shopId}&customerId=${customerId}&size=10`
    );

    return (res.content || []).map((ro) => ({
      externalId: String(ro.id),
      customerId: String(ro.customerId),
      vehicleId: String(ro.vehicleId),
      status: ro.repairOrderStatus?.name || "unknown",
      description: ro.customerConcern || "",
    }));
  }

  // ── Appointments ──────────────────────────────────────

  async createAppointment(data: Omit<DmsAppointment, "externalId">): Promise<DmsAppointment> {
    const body = {
      shopId: Number(this.shopId),
      customerId: Number(data.customerId),
      vehicleId: data.vehicleId ? Number(data.vehicleId) : undefined,
      startTime: data.scheduledAt,
      duration: data.duration || 60,
      note: data.serviceDescription,
    };

    const res = await this.request<{ id: number }>("/appointments", {
      method: "POST",
      body: JSON.stringify(body),
    });

    return { ...data, externalId: String(res.id), status: "scheduled" };
  }

  async getAppointment(externalId: string): Promise<DmsAppointment | null> {
    try {
      const a = await this.request<{ id: number; customerId: number; vehicleId: number; startTime: string; duration: number; note: string; status: string }>(
        `/appointments/${externalId}`
      );
      return {
        externalId: String(a.id),
        customerId: String(a.customerId),
        vehicleId: a.vehicleId ? String(a.vehicleId) : undefined,
        scheduledAt: a.startTime,
        duration: a.duration,
        serviceDescription: a.note || "",
        status: a.status || "scheduled",
      };
    } catch {
      return null;
    }
  }

  async cancelAppointment(externalId: string): Promise<boolean> {
    try {
      await this.request(`/appointments/${externalId}`, {
        method: "DELETE",
      });
      return true;
    } catch {
      return false;
    }
  }
}
