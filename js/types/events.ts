/**
 * VoxCPM Event Type Definitions
 * 
 * Defines types for server-to-client events used by VoxCPM.
 */

// ============================================================================
// Event Names
// ============================================================================

/**
 * VoxCPM WebSocket event names
 */
export const VOXCPM_EVENTS = {
  /** Status notification event (e.g., missing dependencies) */
  STATUS: "voxcpm.status",
} as const;

export type VoxCPMEventName = (typeof VOXCPM_EVENTS)[keyof typeof VOXCPM_EVENTS];

// ============================================================================
// Status Event Types
// ============================================================================

/**
 * VoxCPM status event detail payload
 * Sent from server to notify frontend about status changes
 */
export interface VoxCPMStatusDetail {
  /** Event type for categorization (optional) */
  type?: "dependency_warning" | "model_loaded" | "error" | "info";

  /** Toast severity level */
  severity: "success" | "info" | "warn" | "error";

  /** Notification title/summary */
  summary: string;

  /** Detailed message (optional) */
  detail?: string;

  /** Auto-dismiss time in milliseconds (optional) */
  life?: number;
}

// ============================================================================
// Event Handler Types
// ============================================================================

/**
 * Generic event handler for VoxCPM events
 */
export type VoxCPMEventHandler<T = unknown> = (detail: T) => void;

/**
 * Status event handler
 */
export type VoxCPMStatusHandler = VoxCPMEventHandler<VoxCPMStatusDetail>;

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if an event detail is a valid VoxCPMStatusDetail
 */
export function isVoxCPMStatusDetail(detail: unknown): detail is VoxCPMStatusDetail {
  if (typeof detail !== "object" || detail === null) {
    return false;
  }

  const d = detail as Record<string, unknown>;
  return (
    typeof d.severity === "string" &&
    ["success", "info", "warn", "error"].includes(d.severity) &&
    typeof d.summary === "string"
  );
}
