/**
 * Notification Manager
 * 
 * Handles notification display with session-based deduplication.
 * Ensures notifications are shown only once per browser session.
 */

import { app } from "../../scripts/app.js";
import { isSessionFlagSet, setSessionFlag } from "../core/session";
import { logger } from "../core/logger";
import type { NotificationOptions, NotificationManagerConfig, NotificationResult } from "./types";

// ============================================================================
// NotificationManager Class
// ============================================================================

/**
 * Manages notification display with session-based deduplication
 */
export class NotificationManager {
  private sessionKey: string;
  private defaultLife: number;

  /**
   * Create a new NotificationManager
   * 
   * @param config - Manager configuration
   */
  constructor(config: NotificationManagerConfig) {
    this.sessionKey = config.sessionKey;
    this.defaultLife = config.defaultLife;
  }

  /**
   * Show a notification if not already shown this session
   * 
   * @param options - Notification options
   * @returns Result indicating if notification was shown
   */
  show(options: NotificationOptions): NotificationResult {
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
        closable: options.closable ?? true,
      });

      // Mark as shown for this session
      setSessionFlag(this.sessionKey);
      logger.log("Toast notification displayed:", options.summary);

      return { shown: true };
    } catch (error) {
      logger.warn("Toast not available:", error);

      // Fallback to console
      logger.log(`${options.summary}: ${options.detail ?? ""}`);

      // Still mark as shown to prevent repeated fallback logging
      setSessionFlag(this.sessionKey);

      return { shown: false, reason: "toast_unavailable" };
    }
  }

  /**
   * Show a notification without session deduplication
   * 
   * @param options - Notification options
   * @returns Result indicating if notification was shown
   */
  showAlways(options: NotificationOptions): NotificationResult {
    try {
      app.extensionManager.toast.add({
        severity: options.severity,
        summary: options.summary,
        detail: options.detail,
        life: options.life ?? this.defaultLife,
        closable: options.closable ?? true,
      });

      logger.log("Toast notification displayed:", options.summary);
      return { shown: true };
    } catch (error) {
      logger.warn("Toast not available:", error);
      logger.log(`${options.summary}: ${options.detail ?? ""}`);
      return { shown: false, reason: "toast_unavailable" };
    }
  }

  /**
   * Reset notification state (for testing)
   */
  reset(): void {
    setSessionFlag(this.sessionKey);
  }

  /**
   * Check if notification was already shown this session
   */
  wasShown(): boolean {
    return isSessionFlagSet(this.sessionKey);
  }
}

// ============================================================================
// Default Instance
// ============================================================================

/**
 * Default notification manager instance for missing dependency warnings
 */
export const notificationManager = new NotificationManager({
  sessionKey: "voxcpm.normalization_notification_shown",
  defaultLife: 10000,
});

export default NotificationManager;
