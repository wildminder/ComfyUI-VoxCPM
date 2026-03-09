/**
 * Mitchell 1 RepairCenter DMS Adapter
 *
 * Mitchell 1 RepairCenter Transactional API: https://developer.mitchell.com
 * Auth: apiKey header + shop_id + shop_country headers
 * Protocol: OData/REST, TLS 1.2 required
 * Rate limit: 100 rows per call
 * Endpoints: /customers, /vehicles, /jobs (repair orders), /appointments
 */

import type {
  DmsAdapter,
  DmsConfig,
  DmsCustomer,
  DmsVehicle,
  DmsRepairOrder,
  DmsAppointment,
} from "./types";

const DEFAULT_BASE_URL = "https://api.repaircenter.mitchell.com/api/v2";

export class Mitchell1Adapter implements DmsAdapter {
  readonly provider = "mitchell1" as const;
  private baseUrl: string;
  private shopId: string;
  private headers: Record<string, string>;

  constructor(config: DmsConfig) {
    this.baseUrl = config.apiUrl || DEFAULT_BASE_URL;
    this.shopId = config.shopExternalId || "";
    this.headers = {
      apiKey: config.apiKey,
      shop_id: this.shopId,
      shop_country: "US",
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
      throw new Error(`Mitchell1 API ${res.status}: ${body}`);
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
      shopId: this.shopId,
      firstName: data.firstName,
      lastName: data.lastName,
      phoneNumber: data.phone,
      emailAddress: data.email || "",
      address: data.address || "",
      notes: data.notes || "",
    };

    const res = await this.request<{ customerId: string }>(
      `/shops/${this.shopId}/customers`,
      { method: "POST", body: JSON.stringify(body) }
    );

    return { ...data, externalId: res.customerId };
  }

  async findCustomerByPhone(phone: string): Promise<DmsCustomer | null> {
    const cleanPhone = phone.replace(/\D/g, "");
    const res = await this.request<{ value: Array<{ customerId: string; firstName: string; lastName: string; phoneNumber: string; emailAddress: string }> }>(
      `/shops/${this.shopId}/customers?$filter=contains(phoneNumber,'${cleanPhone}')&$top=1`
    );

    if (!res.value?.length) return null;
    const c = res.value[0];
    return {
      externalId: c.customerId,
      firstName: c.firstName || "",
      lastName: c.lastName || "",
      phone: c.phoneNumber || phone,
      email: c.emailAddress,
    };
  }

  async getCustomer(externalId: string): Promise<DmsCustomer | null> {
    try {
      const c = await this.request<{ customerId: string; firstName: string; lastName: string; phoneNumber: string; emailAddress: string }>(
        `/shops/${this.shopId}/customers/${externalId}`
      );
      return {
        externalId: c.customerId,
        firstName: c.firstName || "",
        lastName: c.lastName || "",
        phone: c.phoneNumber || "",
        email: c.emailAddress,
      };
    } catch {
      return null;
    }
  }

  // ── Vehicles ──────────────────────────────────────────

  async createVehicle(data: Omit<DmsVehicle, "externalId">): Promise<DmsVehicle> {
    const body = {
      customerId: data.customerId,
      year: data.year,
      make: data.make,
      model: data.model,
      vin: data.vin || "",
      licensePlate: data.plate || "",
      odometer: data.mileage,
    };

    const res = await this.request<{ vehicleId: string }>(
      `/shops/${this.shopId}/vehicles`,
      { method: "POST", body: JSON.stringify(body) }
    );

    return { ...data, externalId: res.vehicleId };
  }

  async findVehicleByVin(vin: string): Promise<DmsVehicle | null> {
    const res = await this.request<{ value: Array<{ vehicleId: string; customerId: string; year: number; make: string; model: string; vin: string }> }>(
      `/shops/${this.shopId}/vehicles?$filter=vin eq '${vin}'&$top=1`
    );

    if (!res.value?.length) return null;
    const v = res.value[0];
    return {
      externalId: v.vehicleId,
      customerId: v.customerId,
      year: v.year,
      make: v.make || "",
      model: v.model || "",
      vin: v.vin,
    };
  }

  async getVehicle(externalId: string): Promise<DmsVehicle | null> {
    try {
      const v = await this.request<{ vehicleId: string; customerId: string; year: number; make: string; model: string; vin: string; licensePlate: string }>(
        `/shops/${this.shopId}/vehicles/${externalId}`
      );
      return {
        externalId: v.vehicleId,
        customerId: v.customerId,
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

  // ── Repair Orders (Jobs) ──────────────────────────────

  async createRepairOrder(data: Omit<DmsRepairOrder, "externalId">): Promise<DmsRepairOrder> {
    const body = {
      customerId: data.customerId,
      vehicleId: data.vehicleId,
      concern: data.description,
      status: "Estimate",
    };

    const res = await this.request<{ jobId: string }>(
      `/shops/${this.shopId}/jobs`,
      { method: "POST", body: JSON.stringify(body) }
    );

    return { ...data, externalId: res.jobId };
  }

  async getRepairOrder(externalId: string): Promise<DmsRepairOrder | null> {
    try {
      const ro = await this.request<{ jobId: string; customerId: string; vehicleId: string; status: string; concern: string; createdDate: string }>(
        `/shops/${this.shopId}/jobs/${externalId}`
      );
      return {
        externalId: ro.jobId,
        customerId: ro.customerId,
        vehicleId: ro.vehicleId,
        status: ro.status || "unknown",
        description: ro.concern || "",
        createdAt: ro.createdDate,
      };
    } catch {
      return null;
    }
  }

  async getRepairOrdersByCustomer(customerId: string): Promise<DmsRepairOrder[]> {
    const res = await this.request<{ value: Array<{ jobId: string; customerId: string; vehicleId: string; status: string; concern: string }> }>(
      `/shops/${this.shopId}/jobs?$filter=customerId eq '${customerId}'&$top=10&$orderby=createdDate desc`
    );

    return (res.value || []).map((ro) => ({
      externalId: ro.jobId,
      customerId: ro.customerId,
      vehicleId: ro.vehicleId,
      status: ro.status || "unknown",
      description: ro.concern || "",
    }));
  }

  // ── Appointments ──────────────────────────────────────

  async createAppointment(data: Omit<DmsAppointment, "externalId">): Promise<DmsAppointment> {
    const body = {
      customerId: data.customerId,
      vehicleId: data.vehicleId,
      scheduledDateTime: data.scheduledAt,
      durationMinutes: data.duration || 60,
      serviceDescription: data.serviceDescription,
      notes: data.notes || "",
    };

    const res = await this.request<{ appointmentId: string }>(
      `/shops/${this.shopId}/appointments`,
      { method: "POST", body: JSON.stringify(body) }
    );

    return { ...data, externalId: res.appointmentId, status: "scheduled" };
  }

  async getAppointment(externalId: string): Promise<DmsAppointment | null> {
    try {
      const a = await this.request<{ appointmentId: string; customerId: string; vehicleId: string; scheduledDateTime: string; durationMinutes: number; serviceDescription: string; status: string }>(
        `/shops/${this.shopId}/appointments/${externalId}`
      );
      return {
        externalId: a.appointmentId,
        customerId: a.customerId,
        vehicleId: a.vehicleId,
        scheduledAt: a.scheduledDateTime,
        duration: a.durationMinutes,
        serviceDescription: a.serviceDescription || "",
        status: a.status || "scheduled",
      };
    } catch {
      return null;
    }
  }

  async cancelAppointment(externalId: string, reason?: string): Promise<boolean> {
    try {
      await this.request(
        `/shops/${this.shopId}/appointments/${externalId}`,
        {
          method: "PATCH",
          body: JSON.stringify({ status: "Cancelled", cancellationReason: reason || "" }),
        }
      );
      return true;
    } catch {
      return false;
    }
  }
}
