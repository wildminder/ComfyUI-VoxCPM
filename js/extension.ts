/**
 * VoxCPM Frontend Extension
 * 
 * Main extension class with lifecycle hooks for ComfyUI integration.
 * Handles server-to-client events and notification management.
 */

import { app } from "../../scripts/app.js";
import { logger } from "./core/logger";
import { notificationManager } from "./notifications";
import { EVENT_NAMES, DEFAULTS } from "./core/constants";
import type { VoxCPMStatusDetail } from "./types/events";

// ============================================================================
// Extension Definition
// ============================================================================

/**
 * VoxCPM Frontend Extension
 * 
 * Provides:
 * - Session-based notification deduplication
 * - Event handling for server-to-client communication
 * - Foundation for future UI features
 */
export const VoxCPMExtension = {
  /** Extension name for ComfyUI registration */
  name: DEFAULTS.EXTENSION_NAME,

  /**
   * Initialize extension (before nodes are registered)
   * 
   * Called after canvas is created, before nodes are loaded.
   * Use for modifying core behavior or adding global listeners.
   */
  async init(): Promise<void> {
    logger.log("Frontend extension initialized");
  },

  /**
   * Setup extension (after app is fully loaded)
   * 
   * Called after the app is fully loaded and ready.
   * Use for registering event listeners and UI components.
   */
  async setup(): Promise<void> {
    // Register WebSocket event listeners
    this.registerEventListeners();

    logger.log("Frontend extension loaded");
  },

  /**
   * Register WebSocket event listeners
   * 
   * Sets up listeners for server-to-client events.
   */
  registerEventListeners(): void {
    // Listen for status events from server
    app.api.addEventListener(
      EVENT_NAMES.STATUS,
      ((event: CustomEvent<VoxCPMStatusDetail>) => {
        this.handleStatusEvent(event.detail);
      }) as EventListener
    );

    logger.log("Event listeners registered");
  },

  /**
   * Handle status event from server
   * 
   * Processes status events and shows notifications if needed.
   * 
   * @param detail - Event detail payload
   */
  handleStatusEvent(detail: VoxCPMStatusDetail): void {
    logger.log("Received status event:", detail);

    // Validate event data
    if (!detail.severity || !detail.summary) {
      logger.warn("Invalid status event: missing severity or summary");
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
};

// ============================================================================
// Export
// ============================================================================

export default VoxCPMExtension;
