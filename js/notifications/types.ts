/**
 * Notification System Types
 * 
 * Type definitions for the VoxCPM notification system.
 */

// ============================================================================
// Severity Types
// ============================================================================

/**
 * Toast notification severity levels
 */
export type NotificationSeverity = "success" | "info" | "warn" | "error";

// ============================================================================
// Notification Options
// ============================================================================

/**
 * Options for showing a notification
 */
export interface NotificationOptions {
  /** Toast severity level */
  severity: NotificationSeverity;

  /** Notification title/summary */
  summary: string;

  /** Detailed message (optional) */
  detail?: string;

  /** Auto-dismiss time in milliseconds (optional) */
  life?: number;

  /** Whether notification is closable (optional) */
  closable?: boolean;
}

// ============================================================================
// Notification Manager Config
// ============================================================================

/**
 * Configuration for NotificationManager
 */
export interface NotificationManagerConfig {
  /** Session storage key for tracking shown notifications */
  sessionKey: string;

  /** Default auto-dismiss time in milliseconds */
  defaultLife: number;
}

// ============================================================================
// Notification Result
// ============================================================================

/**
 * Result of showing a notification
 */
export interface NotificationResult {
  /** Whether the notification was shown */
  shown: boolean;

  /** Reason if not shown */
  reason?: "already_shown" | "toast_unavailable" | "error";
}
