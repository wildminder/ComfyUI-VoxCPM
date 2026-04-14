/**
 * VoxCPM Frontend Extension
 *
 * Main extension with session-based notification deduplication.
 * Shows notification only once per browser session.
 * Disables normalize_text widget when packages are not installed.
 * 
 * Uses WebSocket events for config - no HTTP endpoint needed.
 */

import { app } from "../../scripts/app.js";

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEYS = {
    NOTIFICATION_SHOWN: "voxcpm.normalization_notification_shown",
    DEBUG_ENABLED: "voxcpm.debug_enabled",
    NORMALIZATION_AVAILABLE: "voxcpm.normalization_available", // Persist config in localStorage
};

const EVENT_NAMES = {
    STATUS: "voxcpm.status",
    CONFIG: "voxcpm.config",
};

const DEFAULTS = {
    TOAST_LIFE: 10000,
    LOG_PREFIX: "[VoxCPM]",
    EXTENSION_NAME: "voxcpm.frontend",
    NODE_CLASS: "VoxCPM_TTS",
};

// ============================================================================
// Logger with Levels
// ============================================================================

const LogLevel = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3,
    NONE: 4,
};

// Check if debug mode is enabled (set via sessionStorage or URL param)
const isDebugEnabled = () => {
    try {
        // Check URL parameter first
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has("voxcpm_debug")) {
            return true;
        }
        // Check sessionStorage
        return sessionStorage.getItem(STORAGE_KEYS.DEBUG_ENABLED) === "true";
    } catch {
        return false;
    }
};

const logger = {
    _level: isDebugEnabled() ? LogLevel.DEBUG : LogLevel.WARN,

    debug(...args) {
        if (this._level <= LogLevel.DEBUG) {
            console.log(DEFAULTS.LOG_PREFIX, "[DEBUG]", ...args);
        }
    },

    log(...args) {
        if (this._level <= LogLevel.DEBUG) {
            console.log(DEFAULTS.LOG_PREFIX, ...args);
        }
    },

    info(...args) {
        if (this._level <= LogLevel.INFO) {
            console.info(DEFAULTS.LOG_PREFIX, ...args);
        }
    },

    warn(...args) {
        if (this._level <= LogLevel.WARN) {
            console.warn(DEFAULTS.LOG_PREFIX, ...args);
        }
    },

    error(...args) {
        if (this._level <= LogLevel.ERROR) {
            console.error(DEFAULTS.LOG_PREFIX, ...args);
        }
    },

    /**
     * Enable debug mode
     */
    enableDebug() {
        this._level = LogLevel.DEBUG;
        try {
            sessionStorage.setItem(STORAGE_KEYS.DEBUG_ENABLED, "true");
        } catch {}
        console.log(DEFAULTS.LOG_PREFIX, "Debug mode enabled");
    },

    /**
     * Disable debug mode
     */
    disableDebug() {
        this._level = LogLevel.WARN;
        try {
            sessionStorage.removeItem(STORAGE_KEYS.DEBUG_ENABLED);
        } catch {}
    },
};

// Expose logger to window for debugging
if (typeof window !== "undefined") {
    window.VoxCPMLogger = logger;
}

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
            logger.debug("Notification already shown this session, skipping");
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
            logger.debug("Toast notification displayed successfully");

            return { shown: true };
        } catch (error) {
            logger.warn("Toast not available:", error);

            // Fallback to console
            logger.info(`${options.summary}: ${options.detail ?? ""}`);

            // Still mark as shown to prevent repeated fallback logging
            setSessionFlag(this.sessionKey);

            return { shown: false, reason: "toast_unavailable" };
        }
    },
};

// ============================================================================
// Widget Manager
// ============================================================================

/**
 * Get stored config from localStorage
 * @returns {boolean|null} Stored normalization_available value, or null if not stored
 */
function getStoredConfig() {
    try {
        const stored = localStorage.getItem(STORAGE_KEYS.NORMALIZATION_AVAILABLE);
        if (stored !== null) {
            return stored === "true";
        }
    } catch {}
    return null;
}

/**
 * Store config to localStorage
 * @param {boolean} value normalization_available value
 */
function storeConfig(value) {
    try {
        localStorage.setItem(STORAGE_KEYS.NORMALIZATION_AVAILABLE, String(value));
        logger.debug("Config stored to localStorage:", value);
    } catch (e) {
        logger.warn("Failed to store config to localStorage:", e);
    }
}

const widgetManager = {
    normalizationAvailable: true, // Default to true, will be updated by config event or localStorage
    configReceived: false, // Track if we received config from server in this session
    lastConfigValue: null, // Track last received config value
    initialized: false, // Track if we've initialized from localStorage

    /**
     * Initialize from localStorage (call immediately on module load)
     */
    init() {
        if (this.initialized) {
            return;
        }
        this.initialized = true;

        const stored = getStoredConfig();
        if (stored !== null) {
            this.normalizationAvailable = stored;
            logger.debug("Loaded config from localStorage:", stored);
        }
    },

    /**
     * Update the normalize_text widget state based on config
     */
    updateNormalizationWidget(node) {
        // Initialize if not already done
        if (!this.initialized) {
            this.init();
        }

        // Find the normalize_text widget
        const normalizeWidget = node.widgets?.find(
            (w) => w.name === "normalize_text"
        );

        if (!normalizeWidget) {
            return false;
        }

        if (this.normalizationAvailable) {
            // Re-enable the widget if it was disabled
            if (normalizeWidget.disabled) {
                logger.debug("Re-enabling normalize_text widget for node:", node.id);
                normalizeWidget.disabled = false;
                // Set value to true (normalize mode) when re-enabling
                normalizeWidget.value = true;
                // Clear the tooltip
                if (normalizeWidget.options) {
                    normalizeWidget.options.tooltip = undefined;
                }
            }
            return false;
        } else {
            // Only disable and log if not already disabled
            if (!normalizeWidget.disabled) {
                logger.debug("Disabling normalize_text widget for node:", node.id);
                normalizeWidget.disabled = true;
                normalizeWidget.value = false;
                // Update tooltip to explain why it's disabled
                normalizeWidget.options = normalizeWidget.options || {};
                normalizeWidget.options.tooltip =
                    "Text normalization disabled: 'inflect' and 'wetext' packages not installed. Install with: pip install inflect wetext";
            }
            return true;
        }
    },

    /**
     * Update all existing VoxCPM nodes in the graph
     */
    updateAllNodes() {
        // Initialize if not already done
        if (!this.initialized) {
            this.init();
        }

        const graph = app.graph || app.rootGraph;
        if (graph && graph._nodes) {
            for (const node of graph._nodes) {
                if (node.comfyClass === DEFAULTS.NODE_CLASS) {
                    this.updateNormalizationWidget(node);
                }
            }
        }
    },
};

// Initialize immediately on module load (before any nodes are created)
widgetManager.init();

// ============================================================================
// Extension Definition
// ============================================================================

app.registerExtension({
    name: DEFAULTS.EXTENSION_NAME,

    /**
     * Setup extension (after app is fully loaded)
     */
    async setup() {
        logger.debug("Frontend extension loaded");

        // Listen for status events from server (notification requests)
        app.api.addEventListener(EVENT_NAMES.STATUS, (event) => {
            logger.debug("Received voxcpm.status event:", event.detail);
            this.handleStatusEvent(event.detail);
        });

        // Listen for config events from server (for widget state)
        app.api.addEventListener(EVENT_NAMES.CONFIG, (event) => {
            logger.debug("Received voxcpm.config event:", event.detail);
            this.handleConfigEvent(event.detail);
        });

        // Listen for reconnected event (browser refresh)
        // The server will send config when a new client connects
        app.api.addEventListener("reconnected", () => {
            logger.debug("WebSocket reconnected, waiting for config...");
            // Reset config state so we accept the new config
            widgetManager.configReceived = false;
            widgetManager.lastConfigValue = null;
        });

        // Listen for status event which indicates WebSocket is connected
        // This is a fallback in case config event was missed
        app.api.addEventListener("status", (event) => {
            // If we haven't received config yet, we might need to wait
            // The backend sends config periodically
            if (!widgetManager.configReceived) {
                logger.debug("Status received, waiting for config...");
            }
        });
    },

    /**
     * Called after a node is created
     */
    nodeCreated(node) {
        // Only process VoxCPM TTS nodes
        if (node.comfyClass !== DEFAULTS.NODE_CLASS) {
            return;
        }

        logger.debug("VoxCPM node created:", node.id);

        // Update the widget based on current config
        widgetManager.updateNormalizationWidget(node);
    },

    /**
     * Called when a node is loaded from a saved workflow
     */
    loadedGraphNode(node) {
        // Only process VoxCPM TTS nodes
        if (node.comfyClass !== DEFAULTS.NODE_CLASS) {
            return;
        }

        logger.debug("VoxCPM node loaded from workflow:", node.id);

        // Update the widget based on current config
        widgetManager.updateNormalizationWidget(node);
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

    /**
     * Handle config event from server (for widget state)
     */
    handleConfigEvent(detail) {
        if (!detail) {
            return;
        }

        // Update normalization availability
        if (typeof detail.normalization_available === "boolean") {
            // Check if the value changed
            const valueChanged = widgetManager.lastConfigValue !== detail.normalization_available;

            // Skip if we already processed this exact config value
            // This prevents redundant processing of repeated broadcasts
            if (!valueChanged && widgetManager.configReceived) {
                logger.debug("Config already processed, skipping");
                return;
            }

            widgetManager.normalizationAvailable = detail.normalization_available;
            widgetManager.configReceived = true;
            widgetManager.lastConfigValue = detail.normalization_available;

            // Store to localStorage for browser refresh scenario
            storeConfig(detail.normalization_available);

            logger.debug(
                "Text normalization available:",
                widgetManager.normalizationAvailable
            );

            // Update all existing nodes in the graph (for config changes)
            // Use a small delay to ensure graph is fully loaded
            setTimeout(() => {
                widgetManager.updateAllNodes();
            }, 100);
        }
    },
});
