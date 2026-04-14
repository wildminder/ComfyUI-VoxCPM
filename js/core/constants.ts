/**
 * VoxCPM Constants
 * 
 * Central location for all constants used in the VoxCPM frontend extension.
 */

// ============================================================================
// Session Storage Keys
// ============================================================================

/**
 * Session storage keys for persisting state
 */
export const STORAGE_KEYS = {
  /** Key for tracking if notification was shown this session */
  NOTIFICATION_SHOWN: "voxcpm.normalization_notification_shown",

  /** Key for user preferences (future use) */
  USER_PREFERENCES: "voxcpm.user_preferences",
} as const;

// ============================================================================
// Event Names
// ============================================================================

/**
 * WebSocket event names for server-to-client communication
 */
export const EVENT_NAMES = {
  /** Status notification event */
  STATUS: "voxcpm.status",

  /** Model loaded event (future use) */
  MODEL_LOADED: "voxcpm.model_loaded",

  /** Generation progress event (future use) */
  GENERATION_PROGRESS: "voxcpm.generation_progress",
} as const;

// ============================================================================
// Default Values
// ============================================================================

/**
 * Default configuration values
 */
export const DEFAULTS = {
  /** Default toast notification lifetime in milliseconds */
  TOAST_LIFE: 10000,

  /** Default log prefix */
  LOG_PREFIX: "[VoxCPM]",

  /** Extension name for ComfyUI registration */
  EXTENSION_NAME: "voxcpm.frontend",
} as const;

// ============================================================================
// UI Constants
// ============================================================================

/**
 * UI-related constants
 */
export const UI = {
  /** Toast severity levels */
  SEVERITY: {
    SUCCESS: "success" as const,
    INFO: "info" as const,
    WARN: "warn" as const,
    ERROR: "error" as const,
  },

  /** Icon names (PrimeIcons) */
  ICONS: {
    VOLUME: "pi pi-volume-up",
    WARNING: "pi pi-exclamation-triangle",
    INFO: "pi pi-info-circle",
    ERROR: "pi pi-times-circle",
    SUCCESS: "pi pi-check-circle",
  },
} as const;

// ============================================================================
// Type Exports
// ============================================================================

export type StorageKey = (typeof STORAGE_KEYS)[keyof typeof STORAGE_KEYS];
export type EventName = (typeof EVENT_NAMES)[keyof typeof EVENT_NAMES];
