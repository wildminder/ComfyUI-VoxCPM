/**
 * VoxCPM Frontend Extension
 * 
 * Main extension with session-based notification deduplication.
 * Shows notification only once per browser session.
 */

import { app } from "../../scripts/app.js";

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEYS = {
  NOTIFICATION_SHOWN: "voxcpm.normalization_notification_shown",
};

const EVENT_NAMES = {
  STATUS: "voxcpm.status",
};

const DEFAULTS = {
  TOAST_LIFE: 10000,
  LOG_PREFIX: "[VoxCPM]",
  EXTENSION_NAME: "voxcpm.frontend",
};

// ============================================================================
// Logger
// ============================================================================

const logger = {
  log: (...args) => console.log(DEFAULTS.LOG_PREFIX, ...args),
  info: (...args) => console.info(DEFAULTS.LOG_PREFIX, ...args),
  warn: (...args) => console.warn(DEFAULTS.LOG_PREFIX, ...args),
  error: (...args) => console.error(DEFAULTS.LOG_PREFIX, ...args),
};

// ============================================================================
// Session State Helpers
// ============================================================================

function isSessionFlagSet(key) {
  try {
    return sessionStorage.getItem(key) !== null;
  } catch {
    return false;
  }
}

function setSessionFlag(key) {
  try {
    sessionStorage.setItem(key, "true");
    return true;
  } catch {
    return false;
  }
}

// ============================================================================
// Notification Manager
// ============================================================================

const notificationManager = {
  sessionKey: STORAGE_KEYS.NOTIFICATION_SHOWN,
  defaultLife: DEFAULTS.TOAST_LIFE,

  /**
   * Show a notification if not already shown this session
   */
  show(options) {
    // Check if already shown this session
    if (isSessionFlagSet(this.sessionKey)) {
      logger.log("Notification already shown this session, skipping");
      return { shown: false, reason: "already_shown" };
    }

    // Show toast notification
    try {
      app.extensionManager.toast.add({
        severity: options.severity,
        summary: options.summary,
        detail: options.detail,
        life: options.life ?? this.defaultLife,
      });

      // Mark as shown for this session
      setSessionFlag(this.sessionKey);
      logger.log("Toast notification displayed successfully");

      return { shown: true };
    } catch (error) {
      logger.warn("Toast not available:", error);

      // Fallback to console
      logger.log(`${options.summary}: ${options.detail ?? ""}`);

      // Still mark as shown to prevent repeated fallback logging
      setSessionFlag(this.sessionKey);

      return { shown: false, reason: "toast_unavailable" };
    }
  },
};

// ============================================================================
// Extension Definition
// ============================================================================

app.registerExtension({
  name: DEFAULTS.EXTENSION_NAME,

  /**
   * Setup extension (after app is fully loaded)
   */
  async setup() {
    logger.log("Frontend extension loaded");

    // Listen for status events from server
    app.api.addEventListener(EVENT_NAMES.STATUS, (event) => {
      logger.log("Received voxcpm.status event:", event.detail);
      this.handleStatusEvent(event.detail);
    });
  },

  /**
   * Handle status event from server
   */
  handleStatusEvent(detail) {
    // Validate event data
    if (!detail || !detail.severity || !detail.summary) {
      logger.warn("Invalid status event: missing severity or summary", detail);
      return;
    }

    // Show notification with session deduplication
    notificationManager.show({
      severity: detail.severity,
      summary: detail.summary,
      detail: detail.detail,
      life: detail.life,
    });
  },
});
