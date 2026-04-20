import { app as app$1 } from "../../scripts/app.js";
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const STORAGE_KEYS = {
  /** Key for tracking if notification was shown this session */
  NOTIFICATION_SHOWN: "voxcpm.normalization_notification_shown",
  /** Key for user preferences (future use) */
  USER_PREFERENCES: "voxcpm.user_preferences",
  /** Key for settings session cache */
  SETTINGS: "voxcpm.settings",
  /** Key for first run dialog shown flag */
  FIRST_RUN_SHOWN: "voxcpm.first_run_shown",
  /** Key for normalization available config (localStorage) */
  NORMALIZATION_AVAILABLE: "voxcpm.normalization_available"
};
const EVENT_NAMES = {
  /** Status notification event */
  STATUS: "voxcpm.status",
  /** Config event from server */
  CONFIG: "voxcpm.config"
};
const DEFAULTS = {
  /** Default log prefix */
  LOG_PREFIX: "[VoxCPM]",
  /** Extension name for ComfyUI registration */
  EXTENSION_NAME: "voxcpm.frontend",
  /** Node class name for VoxCPM TTS nodes */
  NODE_CLASS: "VoxCPM_TTS"
};
const UI = {
  /** Toast severity levels */
  SEVERITY: {
    SUCCESS: "success",
    INFO: "info",
    WARN: "warn",
    ERROR: "error"
  },
  /** Icon names (PrimeIcons) */
  ICONS: {
    VOLUME: "pi pi-volume-up",
    WARNING: "pi pi-exclamation-triangle",
    INFO: "pi pi-info-circle",
    ERROR: "pi pi-times-circle",
    SUCCESS: "pi pi-check-circle"
  },
  /** Model selector widget constants */
  MODEL_SELECTOR: {
    /** Widget name for the DOM widget */
    WIDGET_NAME: "model_selector",
    /** Widget type for addDOMWidget */
    WIDGET_TYPE: "custom",
    /** Minimum height of the model selector widget in pixels */
    MIN_HEIGHT: 36,
    /** Default model icon (standard path) */
    DEFAULT_ICON: "📁",
    /** Custom path model icon */
    CUSTOM_ICON: "📂",
    /** Dropdown arrow character */
    ARROW: "▼",
    /** Placeholder text when no model is selected */
    PLACEHOLDER: "Select model...",
    /** CSS class prefix for all model selector styles */
    CSS_PREFIX: "voxcpm-model-"
  }
};
const logger = {
  /**
   * Log informational message
   */
  log: (...args) => {
    console.log(DEFAULTS.LOG_PREFIX, ...args);
  },
  /**
   * Log informational message (alias for log)
   */
  info: (...args) => {
    console.info(DEFAULTS.LOG_PREFIX, ...args);
  },
  /**
   * Log warning message
   */
  warn: (...args) => {
    console.warn(DEFAULTS.LOG_PREFIX, ...args);
  },
  /**
   * Log error message
   */
  error: (...args) => {
    console.error(DEFAULTS.LOG_PREFIX, ...args);
  },
  /**
   * Log debug message (only in development)
   */
  debug: (...args) => {
    const isDev = typeof window !== "undefined" && window.__VOXCPM_DEBUG__;
    if (isDev) {
      console.debug(DEFAULTS.LOG_PREFIX, "[DEBUG]", ...args);
    }
  },
  /**
   * Log group start
   */
  group: (label) => {
    console.group(`${DEFAULTS.LOG_PREFIX} ${label}`);
  },
  /**
   * Log group end
   */
  groupEnd: () => {
    console.groupEnd();
  },
  /**
   * Log table data
   */
  table: (data) => {
    console.log(DEFAULTS.LOG_PREFIX);
    console.table(data);
  }
};
function getSessionItem(key) {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null;
  }
}
function setSessionItem(key, value) {
  try {
    sessionStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}
function isSessionFlagSet$1(key) {
  return getSessionItem(key) !== null;
}
function setSessionFlag$1(key) {
  return setSessionItem(key, "true");
}
function getSessionJSON(key) {
  const value = getSessionItem(key);
  if (value === null) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}
function setSessionJSON(key, value) {
  try {
    return setSessionItem(key, JSON.stringify(value));
  } catch {
    return false;
  }
}
class NotificationManager {
  /**
   * Create a new NotificationManager
   * 
   * @param config - Manager configuration
   */
  constructor(config) {
    this.sessionKey = config.sessionKey;
    this.defaultLife = config.defaultLife;
  }
  /**
   * Show a notification if not already shown this session
   * 
   * @param options - Notification options
   * @returns Result indicating if notification was shown
   */
  show(options) {
    if (isSessionFlagSet$1(this.sessionKey)) {
      logger.log("Notification already shown this session, skipping");
      return { shown: false, reason: "already_shown" };
    }
    try {
      app.extensionManager.toast.add({
        severity: options.severity,
        summary: options.summary,
        detail: options.detail,
        life: options.life ?? this.defaultLife,
        closable: options.closable ?? true
      });
      setSessionFlag$1(this.sessionKey);
      logger.log("Toast notification displayed:", options.summary);
      return { shown: true };
    } catch (error) {
      logger.warn("Toast not available:", error);
      logger.log(`${options.summary}: ${options.detail ?? ""}`);
      setSessionFlag$1(this.sessionKey);
      return { shown: false, reason: "toast_unavailable" };
    }
  }
  /**
   * Show a notification without session deduplication
   * 
   * @param options - Notification options
   * @returns Result indicating if notification was shown
   */
  showAlways(options) {
    try {
      app.extensionManager.toast.add({
        severity: options.severity,
        summary: options.summary,
        detail: options.detail,
        life: options.life ?? this.defaultLife,
        closable: options.closable ?? true
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
  reset() {
    setSessionFlag$1(this.sessionKey);
  }
  /**
   * Check if notification was already shown this session
   */
  wasShown() {
    return isSessionFlagSet$1(this.sessionKey);
  }
}
const notificationManager = new NotificationManager({
  sessionKey: "voxcpm.normalization_notification_shown",
  defaultLife: 1e4
});
const SETTINGS_SESSION_KEY = "voxcpm.settings";
class SettingsManager {
  constructor() {
    this.settings = null;
    this.initialized = false;
  }
  /**
   * Initialize settings from server config
   */
  async initialize(configDetail) {
    this.settings = configDetail.settings;
    this.initialized = true;
    setSessionJSON(SETTINGS_SESSION_KEY, this.settings);
    logger.log("Settings initialized:", this.settings);
  }
  /**
   * Get current settings
   */
  getSettings() {
    if (this.settings) {
      return this.settings;
    }
    const cached = getSessionJSON(SETTINGS_SESSION_KEY);
    if (cached) {
      this.settings = cached;
      return this.settings;
    }
    return null;
  }
  /**
   * Check if this is the first run
   */
  isFirstRun() {
    const settings = this.getSettings();
    return settings?.first_run ?? true;
  }
  /**
   * Check if using custom path
   */
  isUsingCustomPath() {
    const settings = this.getSettings();
    return settings?.use_custom_path ?? false;
  }
  /**
   * Get effective model path
   */
  getEffectivePath() {
    const settings = this.getSettings();
    return settings?.effective_path ?? null;
  }
  /**
   * Update settings via server
   */
  async updateSettings(updates) {
    try {
      for (const [key, value] of Object.entries(updates)) {
        await app.api.storeSetting(`voxcpm.${key}`, value);
      }
      if (this.settings) {
        Object.assign(this.settings, updates);
        setSessionJSON(SETTINGS_SESSION_KEY, this.settings);
      }
      logger.log("Settings updated:", updates);
      return true;
    } catch (error) {
      logger.warn("Failed to update settings:", error);
      return false;
    }
  }
  /**
   * Mark first run as complete
   */
  async markFirstRunComplete() {
    return this.updateSettings({ first_run: false });
  }
  /**
   * Set custom path preference
   */
  async setCustomPath(path) {
    return this.updateSettings({
      use_custom_path: true,
      custom_model_path: path
    });
  }
  /**
   * Set default path preference
   */
  async setDefaultPath() {
    return this.updateSettings({
      use_custom_path: false,
      custom_model_path: null
    });
  }
  /**
   * Check if settings are initialized
   */
  isInitialized() {
    return this.initialized;
  }
}
const settingsManager = new SettingsManager();
let activeDialog = null;
function showWelcomeDialog() {
  return new Promise((resolve) => {
    closeActiveDialog();
    const overlay = document.createElement("div");
    overlay.className = "voxcpm-dialog-overlay";
    overlay.innerHTML = `
      <div class="voxcpm-dialog">
        <div class="voxcpm-dialog-header">
          <span class="voxcpm-dialog-icon">🎙️</span>
          <h2 class="voxcpm-dialog-title">VoxCPM Model Setup</h2>
        </div>
        <div class="voxcpm-dialog-body">
          <p class="voxcpm-dialog-description">
            Welcome to VoxCPM! How would you like to configure your model path?
          </p>
          <div class="voxcpm-options">
            <button class="voxcpm-option-card" data-option="default">
              <div class="voxcpm-option-icon">📁</div>
              <div class="voxcpm-option-content">
                <h3>Use Default Path</h3>
                <p class="voxcpm-option-path">models/tts/VoxCPM/</p>
                <p class="voxcpm-option-description">
                  Official models will be downloaded automatically.
                </p>
              </div>
            </button>
            <button class="voxcpm-option-card" data-option="custom">
              <div class="voxcpm-option-icon">📂</div>
              <div class="voxcpm-option-content">
                <h3>Use Custom Path</h3>
                <p class="voxcpm-option-description">
                  Select a custom directory where your VoxCPM models are located.
                  Useful for existing model collections.
                </p>
              </div>
            </button>
          </div>
        </div>
        <div class="voxcpm-dialog-footer">
          <button class="voxcpm-btn voxcpm-btn-secondary" id="voxcpm-cancel">Cancel</button>
          <button class="voxcpm-btn voxcpm-btn-primary" id="voxcpm-continue" disabled>
            Continue
          </button>
        </div>
      </div>
    `;
    let selectedOption = null;
    const optionCards = overlay.querySelectorAll(".voxcpm-option-card");
    optionCards.forEach((card) => {
      card.addEventListener("click", () => {
        optionCards.forEach((c) => c.classList.remove("voxcpm-option-selected"));
        card.classList.add("voxcpm-option-selected");
        selectedOption = card.getAttribute("data-option");
        const continueBtn2 = overlay.querySelector("#voxcpm-continue");
        if (continueBtn2) {
          continueBtn2.disabled = false;
        }
      });
    });
    const cancelBtn = overlay.querySelector("#voxcpm-cancel");
    cancelBtn?.addEventListener("click", () => {
      closeActiveDialog();
      resolve(null);
    });
    const continueBtn = overlay.querySelector("#voxcpm-continue");
    continueBtn?.addEventListener("click", () => {
      if (selectedOption) {
        closeActiveDialog();
        resolve(selectedOption);
      }
    });
    document.body.appendChild(overlay);
    activeDialog = overlay;
    injectDialogStyles();
  });
}
function showPathSelectionDialog(initialPath) {
  return new Promise((resolve) => {
    closeActiveDialog();
    const overlay = document.createElement("div");
    overlay.className = "voxcpm-dialog-overlay";
    overlay.innerHTML = `
      <div class="voxcpm-dialog voxcpm-dialog-large">
        <div class="voxcpm-dialog-header">
          <h2 class="voxcpm-dialog-title">Select Model Directory</h2>
        </div>
        <div class="voxcpm-dialog-body">
          <div class="voxcpm-path-input-group">
            <input
              type="text"
              class="voxcpm-path-input"
              placeholder="Enter model directory path"
              value="${initialPath || ""}"
              id="voxcpm-path-input"
            />
            <button class="voxcpm-btn voxcpm-btn-secondary" id="voxcpm-browse">
              Browse...
            </button>
          </div>
          <div class="voxcpm-models-preview" id="voxcpm-models-preview" style="display: none;">
            <h4>Found Models (<span id="voxcpm-model-count">0</span>)</h4>
            <ul class="voxcpm-models-list" id="voxcpm-models-list"></ul>
          </div>
          <div class="voxcpm-validation-message" id="voxcpm-validation-message"></div>
        </div>
        <div class="voxcpm-dialog-footer">
          <button class="voxcpm-btn voxcpm-btn-secondary" id="voxcpm-cancel">Cancel</button>
          <button class="voxcpm-btn voxcpm-btn-primary" id="voxcpm-confirm" disabled>
            Use This Path
          </button>
        </div>
      </div>
    `;
    let isValid = false;
    let currentPath = initialPath || "";
    const pathInput = overlay.querySelector("#voxcpm-path-input");
    const modelsPreview = overlay.querySelector("#voxcpm-models-preview");
    const modelsList = overlay.querySelector("#voxcpm-models-list");
    const modelCount = overlay.querySelector("#voxcpm-model-count");
    const validationMessage = overlay.querySelector("#voxcpm-validation-message");
    const confirmBtn = overlay.querySelector("#voxcpm-confirm");
    async function validatePath(path) {
      if (!path) {
        modelsPreview.style.display = "none";
        validationMessage.textContent = "";
        confirmBtn.disabled = true;
        isValid = false;
        return;
      }
      try {
        const { app: app2 } = await import("../../scripts/app.js");
        const response = await app2.api.fetchApi(
          `/voxcpm/validate_path?path=${encodeURIComponent(path)}`
        );
        const result = await response.json();
        isValid = result.valid;
        if (result.valid && result.models?.length > 0) {
          modelsPreview.style.display = "block";
          modelCount.textContent = result.models.length.toString();
          modelsList.innerHTML = result.models.map((m) => `
              <li>
                <span class="voxcpm-model-icon">📁</span>
                <span class="voxcpm-model-name">${m.name}</span>
              </li>
            `).join("");
          validationMessage.innerHTML = `<span class="voxcpm-validation-success">Found ${result.models.length} VoxCPM-compatible models</span>`;
          confirmBtn.disabled = false;
        } else {
          modelsPreview.style.display = "none";
          validationMessage.innerHTML = `<span class="voxcpm-validation-error">${result.error || "Invalid path or no models found"}</span>`;
          confirmBtn.disabled = true;
        }
      } catch (error) {
        logger.warn("Path validation failed:", error);
        modelsPreview.style.display = "none";
        validationMessage.innerHTML = `<span class="voxcpm-validation-error">Failed to validate path</span>`;
        confirmBtn.disabled = true;
        isValid = false;
      }
    }
    pathInput?.addEventListener("input", () => {
      currentPath = pathInput.value;
      validatePath(currentPath);
    });
    const browseBtn = overlay.querySelector("#voxcpm-browse");
    browseBtn?.addEventListener("click", () => {
      alert("Please enter the path manually. Native folder picker requires server support.");
    });
    const cancelBtn = overlay.querySelector("#voxcpm-cancel");
    cancelBtn?.addEventListener("click", () => {
      closeActiveDialog();
      resolve(null);
    });
    confirmBtn?.addEventListener("click", () => {
      if (isValid && currentPath) {
        closeActiveDialog();
        resolve(currentPath);
      }
    });
    document.body.appendChild(overlay);
    activeDialog = overlay;
    injectDialogStyles();
    if (initialPath) {
      validatePath(initialPath);
    }
  });
}
function closeActiveDialog() {
  if (activeDialog) {
    activeDialog.remove();
    activeDialog = null;
  }
}
function injectDialogStyles() {
  const styleId = "voxcpm-dialog-styles";
  if (document.getElementById(styleId)) {
    return;
  }
  const css = `
/* VoxCPM Dialog Styles */

/* Overlay */
.voxcpm-dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
}

/* Dialog Container */
.voxcpm-dialog {
  background: var(--bg-color, #1e1e1e);
  border-radius: 8px;
  max-width: 500px;
  width: 90%;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}

.voxcpm-dialog-large {
  max-width: 600px;
}

/* Dialog Header */
.voxcpm-dialog-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-color, #333);
}

.voxcpm-dialog-icon {
  font-size: 24px;
}

.voxcpm-dialog-title {
  margin: 0;
  font-size: 18px;
  color: var(--fg-color, #fff);
}

/* Dialog Body */
.voxcpm-dialog-body {
  padding: 24px;
}

.voxcpm-dialog-description {
  margin: 0 0 20px;
  color: var(--fg-color, #ccc);
  line-height: 1.5;
}

/* Options Container */
.voxcpm-options {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* Option Card */
.voxcpm-option-card {
  display: flex;
  gap: 16px;
  padding: 16px;
  background: var(--input-bg, #2a2a2a);
  border: 2px solid var(--border-color, #333);
  border-radius: 8px;
  cursor: pointer;
  text-align: left;
  transition: all 0.2s;
}

.voxcpm-option-card:hover {
  border-color: var(--primary-color, #4a9eff);
}

.voxcpm-option-selected {
  border-color: var(--primary-color, #4a9eff);
  background: var(--input-bg-hover, #333);
}

.voxcpm-option-icon {
  font-size: 32px;
  flex-shrink: 0;
}

.voxcpm-option-content h3 {
  margin: 0 0 4px;
  color: var(--fg-color, #fff);
  font-size: 14px;
}

.voxcpm-option-path {
  margin: 0 0 8px;
  font-family: monospace;
  font-size: 12px;
  color: var(--muted-color, #888);
}

.voxcpm-option-description {
  margin: 0;
  font-size: 12px;
  color: var(--muted-color, #aaa);
}

/* Dialog Footer */
.voxcpm-dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid var(--border-color, #333);
}

/* Buttons */
.voxcpm-btn {
  padding: 8px 16px;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.voxcpm-btn-secondary {
  background: transparent;
  border: 1px solid var(--border-color, #333);
  color: var(--fg-color, #ccc);
}

.voxcpm-btn-secondary:hover {
  background: var(--input-bg, #2a2a2a);
}

.voxcpm-btn-primary {
  background: var(--primary-color, #4a9eff);
  border: none;
  color: #fff;
}

.voxcpm-btn-primary:hover:not(:disabled) {
  background: var(--primary-color-hover, #3a8eef);
}

.voxcpm-btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Path Input Group */
.voxcpm-path-input-group {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}

.voxcpm-path-input {
  flex: 1;
  padding: 8px 12px;
  background: var(--input-bg, #2a2a2a);
  border: 1px solid var(--border-color, #333);
  border-radius: 4px;
  color: var(--fg-color, #fff);
  font-size: 14px;
}

.voxcpm-path-input:focus {
  outline: none;
  border-color: var(--primary-color, #4a9eff);
}

/* Models Preview */
.voxcpm-models-preview {
  background: var(--input-bg, #2a2a2a);
  border-radius: 4px;
  padding: 12px;
  margin-bottom: 16px;
}

.voxcpm-models-preview h4 {
  margin: 0 0 8px;
  font-size: 12px;
  color: var(--muted-color, #888);
}

.voxcpm-models-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.voxcpm-models-list li {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
}

.voxcpm-model-icon {
  font-size: 14px;
}

.voxcpm-model-name {
  font-size: 13px;
  color: var(--fg-color, #ccc);
}

/* Validation Message */
.voxcpm-validation-message {
  font-size: 12px;
}

.voxcpm-validation-success {
  color: var(--success-color, #4caf50);
}

.voxcpm-validation-error {
  color: var(--error-color, #f44336);
}

/* Path Indicator Widget */
.voxcpm-path-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  background: var(--input-bg, #2a2a2a);
  border-radius: 4px;
  font-size: 12px;
}

.voxcpm-path-text {
  color: var(--muted-color, #888);
  font-family: monospace;
}

.voxcpm-path-change-btn {
  background: transparent;
  border: none;
  cursor: pointer;
  font-size: 14px;
  padding: 2px;
}

.voxcpm-path-change-btn:hover {
  opacity: 0.8;
}
`;
  const style = document.createElement("style");
  style.id = styleId;
  style.textContent = css;
  document.head.appendChild(style);
}
const MODEL_SELECTOR_STYLES = `
/* ============================================================================
 * VoxCPM Model Selector Widget
 * ============================================================================ */

.voxcpm-model-selector {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 0;
  width: 100%;
  box-sizing: border-box;
}

/* Model display area - clickable to open dropdown */
.voxcpm-model-display {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  padding: 5px 8px;
  background: var(--comfy-input-bg, rgba(255, 255, 255, 0.05));
  border: 1px solid var(--border-color, rgba(255, 255, 255, 0.15));
  border-radius: 4px;
  cursor: pointer;
  min-height: 26px;
  color: var(--fg-color, #ddd);
  font-family: var(--comfy-font-family, Arial, sans-serif);
  font-size: 12px;
  transition: border-color 0.15s ease, background 0.15s ease;
  user-select: none;
  overflow: hidden;
}

.voxcpm-model-display:hover {
  border-color: var(--comfy-input-border-hover, rgba(255, 255, 255, 0.3));
  background: var(--comfy-input-bg-hover, rgba(255, 255, 255, 0.08));
}

.voxcpm-model-display:active {
  background: var(--comfy-input-bg-active, rgba(255, 255, 255, 0.1));
}

/* Folder icon indicating model source */
.voxcpm-model-icon {
  font-size: 13px;
  opacity: 0.8;
  flex-shrink: 0;
  line-height: 1;
}

/* Model name text */
.voxcpm-model-text {
  flex: 1;
  color: var(--fg-color, #ddd);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
}

/* Dropdown arrow */
.voxcpm-model-arrow {
  font-size: 9px;
  opacity: 0.5;
  margin-left: auto;
  flex-shrink: 0;
  line-height: 1;
  transition: opacity 0.15s ease;
}

.voxcpm-model-display:hover .voxcpm-model-arrow {
  opacity: 0.8;
}

/* Browse button */
.voxcpm-browse-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 5px 8px !important;
  min-width: 30px;
  height: 26px;
  font-size: 13px;
  border-radius: 4px !important;
  cursor: pointer;
  flex-shrink: 0;
  border: 1px solid var(--border-color, rgba(255, 255, 255, 0.15)) !important;
  background: var(--comfy-input-bg, rgba(255, 255, 255, 0.05)) !important;
  color: var(--fg-color, #ddd) !important;
  transition: border-color 0.15s ease, background 0.15s ease;
  line-height: 1;
}

.voxcpm-browse-btn:hover {
  border-color: var(--comfy-input-border-hover, rgba(255, 255, 255, 0.3)) !important;
  background: var(--comfy-input-bg-hover, rgba(255, 255, 255, 0.08)) !important;
}

.voxcpm-browse-btn:active {
  background: var(--comfy-input-bg-active, rgba(255, 255, 255, 0.1)) !important;
}

/* Custom path indicator (shown below selector when custom path is active) */
.voxcpm-model-custom-path {
  font-size: 10px;
  color: var(--fg-color, rgba(255, 255, 255, 0.4));
  padding: 1px 0 0 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  width: 100%;
  box-sizing: border-box;
  display: none;
}

.voxcpm-model-custom-path--visible {
  display: block;
}

/* Container wrapper for selector + path indicator */
.voxcpm-model-selector-wrapper {
  display: flex;
  flex-direction: column;
  gap: 0;
  width: 100%;
  padding: 2px 0;
  box-sizing: border-box;
}
`.trim();
const STYLE_ELEMENT_ID = "voxcpm-model-selector-styles";
let stylesInjected = false;
function injectModelSelectorStyles() {
  if (stylesInjected) {
    return;
  }
  if (document.getElementById(STYLE_ELEMENT_ID)) {
    stylesInjected = true;
    return;
  }
  const style = document.createElement("style");
  style.id = STYLE_ELEMENT_ID;
  style.textContent = MODEL_SELECTOR_STYLES;
  document.head.appendChild(style);
  stylesInjected = true;
}
const selectorRefsMap = /* @__PURE__ */ new Map();
function createModelSelectorWidget(node) {
  injectModelSelectorStyles();
  const widgets = node.widgets;
  if (!widgets) {
    logger.warn("No widgets found on node:", node.id);
    return;
  }
  const modelWidget = widgets.find((w) => w.name === "model_name");
  if (!modelWidget) {
    logger.warn("model_name widget not found on node:", node.id);
    return;
  }
  hideDefaultComboWidget(modelWidget);
  const wrapper = document.createElement("div");
  wrapper.className = "voxcpm-model-selector-wrapper";
  const container = document.createElement("div");
  container.className = "voxcpm-model-selector";
  const modelDisplay = document.createElement("div");
  modelDisplay.className = "voxcpm-model-display";
  modelDisplay.title = "Click to select a model";
  const folderIcon = document.createElement("span");
  folderIcon.className = "voxcpm-model-icon";
  folderIcon.textContent = UI.MODEL_SELECTOR.DEFAULT_ICON;
  const modelText = document.createElement("span");
  modelText.className = "voxcpm-model-text";
  modelText.textContent = modelWidget.value || UI.MODEL_SELECTOR.PLACEHOLDER;
  const arrow = document.createElement("span");
  arrow.className = "voxcpm-model-arrow";
  arrow.textContent = UI.MODEL_SELECTOR.ARROW;
  const browseBtn = document.createElement("button");
  browseBtn.className = "voxcpm-browse-btn";
  browseBtn.textContent = UI.MODEL_SELECTOR.CUSTOM_ICON;
  browseBtn.title = "Browse for custom model directory";
  const pathIndicator = document.createElement("div");
  pathIndicator.className = "voxcpm-model-custom-path";
  const savedCustomPath = node.properties?.["custom_model_path"];
  if (savedCustomPath) {
    folderIcon.textContent = UI.MODEL_SELECTOR.CUSTOM_ICON;
    pathIndicator.textContent = savedCustomPath;
    pathIndicator.classList.add("voxcpm-model-custom-path--visible");
  }
  modelDisplay.append(folderIcon, modelText, arrow);
  container.append(modelDisplay, browseBtn);
  wrapper.append(container, pathIndicator);
  wrapper.addEventListener("mousedown", (e) => e.stopPropagation());
  wrapper.addEventListener("mouseup", (e) => e.stopPropagation());
  wrapper.addEventListener("click", (e) => e.stopPropagation());
  modelDisplay.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    openModelDropdown(
      e,
      modelWidget,
      modelText,
      folderIcon,
      pathIndicator,
      node
    );
  });
  browseBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    await handleBrowseClick(
      node,
      modelWidget,
      modelText,
      folderIcon,
      pathIndicator
    );
  });
  const domWidget = node.addDOMWidget("model_selector", "custom", wrapper, {
    serialize: false,
    hideOnZoom: false,
    selectOn: ["click"],
    getValue: () => modelWidget.value,
    setValue: (v) => {
      modelWidget.value = v;
      modelText.textContent = v;
    }
  });
  const allWidgets = node.widgets;
  if (allWidgets && allWidgets.length > 1) {
    const widgetIndex = allWidgets.indexOf(domWidget);
    if (widgetIndex > 0) {
      allWidgets.splice(widgetIndex, 1);
      allWidgets.unshift(domWidget);
      logger.debug("Moved model selector widget to position 0 for node:", node.id);
    }
  }
  selectorRefsMap.set(node.id, {
    modelText,
    folderIcon,
    pathIndicator,
    modelWidget
  });
  logger.debug("Created model selector widget for node:", node.id);
}
function hideDefaultComboWidget(widget) {
  widget.type = "converted-widget";
  widget.computeSize = () => [0, -4];
  widget.draw = () => {
  };
  logger.debug("Hidden default combo widget:", widget.name);
}
function openModelDropdown(event, modelWidget, modelText, folderIcon, pathIndicator, node) {
  const models = modelWidget.options?.values || [];
  if (models.length === 0) {
    logger.warn("No models available for dropdown");
    notificationManager.show({
      severity: "warn",
      summary: "No Models Available",
      detail: "No models found. Try browsing for a custom model directory.",
      life: 4e3
    });
    return;
  }
  const canvas = app.canvas;
  const scale = Math.max(1, canvas.ds?.scale || 1);
  try {
    new LiteGraph.ContextMenu(models, {
      event,
      title: "Select Model",
      className: "dark",
      scale,
      callback: (selectedModel) => {
        modelWidget.value = selectedModel;
        modelText.textContent = selectedModel;
        if (modelWidget.callback) {
          modelWidget.callback(selectedModel);
        }
        if (node.setDirtyCanvas) {
          node.setDirtyCanvas(true);
        }
        logger.debug("Model selected:", selectedModel, "for node:", node.id);
      }
    });
  } catch (error) {
    logger.error("Failed to open model dropdown:", error);
  }
}
async function handleBrowseClick(node, modelWidget, modelText, folderIcon, pathIndicator) {
  logger.log("Browse button clicked for node:", node.id);
  const path = await showPathSelectionDialog();
  if (!path) {
    logger.debug("Browse cancelled by user");
    return;
  }
  logger.log("User selected custom path:", path);
  node.properties = node.properties || {};
  node.properties["custom_model_path"] = path;
  try {
    const response = await api.fetchApi("/voxcpm/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    });
    if (response.ok) {
      const data = await response.json();
      logger.log("Models found at custom path:", data.models);
      updateModelOptions(modelWidget, data.models, node);
      folderIcon.textContent = UI.MODEL_SELECTOR.CUSTOM_ICON;
      pathIndicator.textContent = path;
      pathIndicator.classList.add("voxcpm-model-custom-path--visible");
      if (data.models.length > 0) {
        modelText.textContent = modelWidget.value;
      } else {
        modelText.textContent = "No models found";
      }
      notificationManager.show({
        severity: "success",
        summary: "Custom Model Path Set",
        detail: `Found ${data.models.length} model(s) in: ${path}`,
        life: 5e3
      });
    } else {
      const errorText = response.statusText || "Unknown error";
      logger.error("Failed to get models:", errorText);
      notificationManager.show({
        severity: "error",
        summary: "Error",
        detail: `Failed to scan model directory: ${errorText}`,
        life: 5e3
      });
    }
  } catch (error) {
    logger.error("Error fetching models:", error);
    notificationManager.show({
      severity: "error",
      summary: "Error",
      detail: "Failed to connect to server",
      life: 5e3
    });
  }
}
function updateModelOptions(modelWidget, models, node) {
  if (modelWidget.options) {
    modelWidget.options.values = models;
  } else {
    modelWidget.options = { values: models };
  }
  if (models.length > 0 && !models.includes(modelWidget.value)) {
    modelWidget.value = models[0];
    if (modelWidget.callback) {
      modelWidget.callback(models[0]);
    }
  }
  if (node.setDirtyCanvas) {
    node.setDirtyCanvas(true);
  }
  logger.debug("Updated model options:", models);
}
function syncModelSelectorDisplay(node, value) {
  const refs = selectorRefsMap.get(node.id);
  if (!refs) {
    logger.debug("No selector refs found for node:", node.id);
    return;
  }
  const displayValue = value || refs.modelWidget.value;
  refs.modelText.textContent = displayValue;
  const customPath = node.properties?.["custom_model_path"];
  if (customPath) {
    refs.folderIcon.textContent = UI.MODEL_SELECTOR.CUSTOM_ICON;
    refs.pathIndicator.textContent = customPath;
    refs.pathIndicator.classList.add("voxcpm-model-custom-path--visible");
  } else {
    refs.folderIcon.textContent = UI.MODEL_SELECTOR.DEFAULT_ICON;
    refs.pathIndicator.classList.remove("voxcpm-model-custom-path--visible");
  }
  logger.debug("Synced model selector display for node:", node.id, "value:", displayValue);
}
function isSessionFlagSet(key) {
  try {
    const stored = sessionStorage.getItem(key);
    if (stored !== null) {
      return stored;
    }
  } catch {
  }
  return null;
}
function setSessionFlag(key, value) {
  try {
    sessionStorage.setItem(key, value);
  } catch (e) {
    logger.warn(`Failed to set session flag ${key}:`, e);
  }
}
function getStoredConfig() {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.NORMALIZATION_AVAILABLE);
    if (stored !== null) {
      return stored === "true";
    }
  } catch {
  }
  return null;
}
function storeConfig(value) {
  try {
    localStorage.setItem(STORAGE_KEYS.NORMALIZATION_AVAILABLE, String(value));
    logger.debug("Config stored to localStorage:", value);
  } catch (e) {
    logger.warn("Failed to store config to localStorage:", e);
  }
}
const widgetManager = {
  normalizationAvailable: true,
  configReceived: false,
  lastConfigValue: null,
  initialized: false,
  /**
   * Initialize from localStorage
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
    if (!this.initialized) {
      this.init();
    }
    const widgets = node.widgets;
    const normalizeWidget = widgets?.find((w) => w.name === "normalize_text");
    if (!normalizeWidget) {
      return false;
    }
    if (this.normalizationAvailable) {
      if (normalizeWidget.disabled) {
        logger.debug("Re-enabling normalize_text widget for node:", node.id);
        normalizeWidget.disabled = false;
        normalizeWidget.value = true;
        if (normalizeWidget.options) {
          normalizeWidget.options.tooltip = void 0;
        }
      }
      return false;
    } else {
      if (!normalizeWidget.disabled) {
        logger.debug("Disabling normalize_text widget for node:", node.id);
        normalizeWidget.disabled = true;
        normalizeWidget.value = false;
        normalizeWidget.options = normalizeWidget.options || {};
        normalizeWidget.options.tooltip = "Text normalization disabled: 'inflect' and 'wetext' packages not installed. Install with: pip install inflect wetext";
      }
      return true;
    }
  },
  /**
   * Update all existing VoxCPM nodes in the graph
   */
  updateAllNodes() {
    if (!this.initialized) {
      this.init();
    }
    const graph = app$1.graph || app$1.rootGraph;
    if (graph && graph._nodes) {
      for (const node of graph._nodes) {
        if (node.comfyClass === DEFAULTS.NODE_CLASS) {
          this.updateNormalizationWidget(node);
        }
      }
    }
  }
};
widgetManager.init();
const VoxCPMExtension = {
  /** Extension name for ComfyUI registration */
  name: DEFAULTS.EXTENSION_NAME,
  /**
   * Register settings with ComfyUI
   *
   * These settings appear in ComfyUI's settings panel.
   */
  settings: [
    {
      id: "voxcpm.use_custom_path",
      category: ["VoxCPM", "Model Path"],
      name: "Use Custom Model Path",
      type: "boolean",
      defaultValue: false,
      tooltip: "Enable to use a custom directory for VoxCPM models"
    },
    {
      id: "voxcpm.custom_model_path",
      category: ["VoxCPM", "Model Path"],
      name: "Custom Model Path",
      type: "text",
      defaultValue: "",
      tooltip: "Path to custom VoxCPM models directory"
    }
  ],
  /**
   * Initialize extension (before nodes are registered)
   *
   * Called after canvas is created, before nodes are loaded.
   */
  async init() {
    logger.log("Frontend extension initialized");
  },
  /**
   * Setup extension (after app is fully loaded)
   *
   * Called after the app is fully loaded and ready.
   * Use for registering event listeners and UI components.
   */
  async setup() {
    logger.debug("Frontend extension loaded");
    app$1.api.addEventListener(EVENT_NAMES.STATUS, (event) => {
      this.handleStatusEvent(event.detail);
    });
    app$1.api.addEventListener(EVENT_NAMES.CONFIG, (event) => {
      this.handleConfigEvent(event.detail);
    });
    app$1.api.addEventListener("reconnected", () => {
      logger.debug("WebSocket reconnected, waiting for config...");
      widgetManager.configReceived = false;
      widgetManager.lastConfigValue = null;
    });
    app$1.api.addEventListener("status", () => {
      if (!widgetManager.configReceived) {
        logger.debug("Status received, waiting for config...");
      }
    });
    logger.log("Event listeners registered");
  },
  /**
   * Called after a node is created
   */
  nodeCreated(node) {
    if (node.comfyClass !== DEFAULTS.NODE_CLASS) {
      return;
    }
    logger.debug("VoxCPM node created:", node.id);
    widgetManager.updateNormalizationWidget(node);
    createModelSelectorWidget(node);
  },
  /**
   * Called when a node is loaded from a saved workflow
   */
  loadedGraphNode(node) {
    if (node.comfyClass !== DEFAULTS.NODE_CLASS) {
      return;
    }
    logger.debug("VoxCPM node loaded from workflow:", node.id);
    widgetManager.updateNormalizationWidget(node);
    setTimeout(() => {
      syncModelSelectorDisplay(node);
    }, 50);
  },
  /**
   * Handle status event from server
   */
  handleStatusEvent(detail) {
    if (!detail || !detail.severity || !detail.summary) {
      logger.warn("Invalid status event: missing severity or summary", detail);
      return;
    }
    notificationManager.show({
      severity: detail.severity,
      summary: detail.summary,
      detail: detail.detail,
      life: detail.life
    });
  },
  /**
  * Handle config event from server
  */
  async handleConfigEvent(detail) {
    logger.log("Received config event:", detail);
    if (!detail) {
      logger.warn("Config event has no detail");
      return;
    }
    if (typeof detail.normalization_available === "boolean") {
      const valueChanged = widgetManager.lastConfigValue !== detail.normalization_available;
      if (!valueChanged && widgetManager.configReceived) {
        logger.debug("Config already processed, skipping");
        return;
      }
      widgetManager.normalizationAvailable = detail.normalization_available;
      widgetManager.configReceived = true;
      widgetManager.lastConfigValue = detail.normalization_available;
      storeConfig(detail.normalization_available);
      logger.debug("Text normalization available:", widgetManager.normalizationAvailable);
      setTimeout(() => {
        widgetManager.updateAllNodes();
      }, 100);
    }
    if (detail.settings) {
      logger.log("Initializing settings manager with:", detail.settings);
      await settingsManager.initialize(detail);
      const isFirst = settingsManager.isFirstRun();
      logger.log("Is first run:", isFirst);
      if (isFirst) {
        logger.log("Calling showFirstRunDialog()");
        this.showFirstRunDialog();
      }
    } else {
      logger.warn("No settings in config event");
    }
  },
  /**
   * Show first run dialog
   */
  async showFirstRunDialog() {
    const shownKey = STORAGE_KEYS.FIRST_RUN_SHOWN;
    if (isSessionFlagSet(shownKey)) {
      logger.log("First run dialog already shown this session");
      return;
    }
    setSessionFlag(shownKey, "true");
    logger.log("First run detected - showing welcome dialog");
    const selection = await showWelcomeDialog();
    if (selection === "default") {
      await settingsManager.setDefaultPath();
      await settingsManager.markFirstRunComplete();
      logger.log("User selected default path");
    } else if (selection === "custom") {
      const path = await showPathSelectionDialog();
      if (path) {
        await settingsManager.setCustomPath(path);
        await settingsManager.markFirstRunComplete();
        logger.log("User selected custom path:", path);
      }
    } else {
      logger.log("User cancelled first run dialog");
    }
  }
};
app$1.registerExtension(VoxCPMExtension);
export {
  VoxCPMExtension,
  VoxCPMExtension as default
};
