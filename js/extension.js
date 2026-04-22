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
    /** Height of the model selector widget in pixels (includes bottom spacing) */
    MIN_HEIGHT: 44,
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
const DESIGN_TOKEN_STYLES = `
/* ============================================================================
 * VoxCPM Design Tokens
 * ============================================================================
 * Semantic tokens that reference ComfyUI's native CSS variables.
 * Fallback values ensure compatibility with all ComfyUI themes.
 * ============================================================================ */

:root {
  /* ── Spacing ── */
  --voxcpm-space-2xs: 2px;
  --voxcpm-space-xs: 4px;
  --voxcpm-space-s: 6px;
  --voxcpm-space-m: 8px;
  --voxcpm-space-l: 12px;
  --voxcpm-space-xl: 16px;
  --voxcpm-space-2xl: 24px;

  /* ── Typography ── */
  --voxcpm-font-size-2xs: 10px;
  --voxcpm-font-size-xs: 12px;
  --voxcpm-font-size-s: 13px;
  --voxcpm-font-size-m: 14px;
  --voxcpm-font-size-l: 18px;
  --voxcpm-font-size-xl: 24px;
  --voxcpm-font-family: var(--comfy-font-family, Arial, sans-serif);
  --voxcpm-font-family-mono: monospace;

  /* ── Surfaces ── */
  --voxcpm-bg-input: var(--comfy-input-bg, rgba(255, 255, 255, 0.05));
  --voxcpm-bg-input-hover: var(--comfy-input-bg-hover, rgba(255, 255, 255, 0.08));
  --voxcpm-bg-input-active: var(--comfy-input-bg-active, rgba(255, 255, 255, 0.1));
  --voxcpm-bg-surface: var(--bg-color, #1e1e1e);
  --voxcpm-bg-elevated: var(--input-bg, #2a2a2a);

  /* ── Borders ── */
  --voxcpm-border-color: var(--border-color, rgba(255, 255, 255, 0.15));
  --voxcpm-border-color-hover: var(--comfy-input-border-hover, rgba(255, 255, 255, 0.3));
  --voxcpm-border-radius-xs: 4px;
  --voxcpm-border-radius-s: 6px;
  --voxcpm-border-radius-m: 8px;

  /* ── Text ── */
  --voxcpm-text-primary: var(--fg-color, #ddd);
  --voxcpm-text-secondary: var(--fg-color, rgba(255, 255, 255, 0.7));
  --voxcpm-text-muted: var(--fg-color, rgba(255, 255, 255, 0.4));
  --voxcpm-text-on-primary: #fff;

  /* ── Accent ── */
  --voxcpm-accent: var(--primary-color, #4a9eff);
  --voxcpm-accent-hover: var(--primary-color-hover, #3a8eef);

  /* ── Feedback ── */
  --voxcpm-color-success: var(--success-color, #4caf50);
  --voxcpm-color-error: var(--error-color, #f44336);

  /* ── Motion ── */
  --voxcpm-duration-fast: 150ms;
  --voxcpm-duration-normal: 200ms;
  --voxcpm-easing: ease;

  /* ── Widget ── */
  --voxcpm-widget-height: 44px;
  --voxcpm-input-height: 26px;
  --voxcpm-browse-width: 30px;

  /* ── Focus ── */
  --voxcpm-focus-ring-color: var(--voxcpm-accent);
  --voxcpm-focus-ring-width: 2px;
  --voxcpm-focus-ring-offset: 1px;
}
`.trim();
const STYLE_ELEMENT_ID$1 = "voxcpm-design-tokens";
let tokensInjected = false;
function injectDesignTokens() {
  if (tokensInjected) {
    return;
  }
  if (document.getElementById(STYLE_ELEMENT_ID$1)) {
    tokensInjected = true;
    return;
  }
  const style = document.createElement("style");
  style.id = STYLE_ELEMENT_ID$1;
  style.textContent = DESIGN_TOKEN_STYLES;
  document.head.appendChild(style);
  tokensInjected = true;
}
const MODEL_SELECTOR_CLASSES = {
  /** Block — the widget container (applied to WidgetDOM.vue parent div) */
  BLOCK: "voxcpm-model-selector",
  /** Element — horizontal row containing display + browse button */
  ROW: "voxcpm-model-selector__row",
  /** Element — clickable model display area (opens dropdown) */
  DISPLAY: "voxcpm-model-selector__display",
  /** Element — folder icon in the display area */
  ICON: "voxcpm-model-selector__icon",
  /** Element — model name text in the display area */
  TEXT: "voxcpm-model-selector__text",
  /** Element — dropdown arrow in the display area */
  ARROW: "voxcpm-model-selector__arrow",
  /** Element — browse button for custom path selection */
  BROWSE: "voxcpm-model-selector__browse",
  /** Element — custom path indicator shown below the row */
  PATH: "voxcpm-model-selector__path",
  /** Modifier — path indicator visible state */
  PATH_VISIBLE: "voxcpm-model-selector__path--visible"
};
const MODEL_SELECTOR_STYLES = `
/* ============================================================================
 * VoxCPM Model Selector Widget (BEM)
 * Block: .voxcpm-model-selector
 * ============================================================================ */

/* ── Block ── */
.${MODEL_SELECTOR_CLASSES.BLOCK} {
  display: flex;
  flex-direction: column;
  gap: 0;
  width: 100%;
  padding: var(--voxcpm-space-2xs) 0;
  box-sizing: border-box;
  /* Prevent vertical stretching in Node 2.0 mode.
   *
   * WidgetDOM.vue wraps DOM widgets with class="flex flex-col *:flex-1",
   * which applies flex:1 to ALL direct children via the universal selector.
   * This causes DOM widgets to stretch vertically like text areas.
   *
   * Two-part fix:
   * 1. flex:none !important — overrides the *:flex-1 rule to prevent
   *    flex growth/shrink. The !important is required because the parent's
   *    *:flex-1 selector has equal specificity (0,1,0) but we need to win.
   * 2. align-self:flex-start !important — prevents the flex container from
   *    stretching this item along the cross axis (vertical in flex-col).
   *    Without this, even with flex:none, the parent's align-items:stretch
   *    (the default) would still make the widget fill the full column height.
   *
 * Additionally, --comfy-widget-min-height and --comfy-widget-max-height
 * CSS custom properties are set equal as a safety net. These are read
 * by DOMWidgetImpl.computeLayoutSize() as a fallback, but since we
 * intentionally do NOT set computeLayoutSize() on the widget (to avoid
 * triggering 'auto' grid row sizing in Node 2.0), these properties
 * serve as documentation of the intended height rather than an active
 * layout mechanism. */
  flex: none !important;
  align-self: flex-start !important;
  --comfy-widget-min-height: var(--voxcpm-widget-height);
  --comfy-widget-max-height: var(--voxcpm-widget-height);
  --comfy-widget-height: var(--voxcpm-widget-height);
}

/* ── Row: display + browse (horizontal layout) ── */
.${MODEL_SELECTOR_CLASSES.ROW} {
  display: flex;
  align-items: center;
  gap: var(--voxcpm-space-xs);
  width: 100%;
  box-sizing: border-box;
}

/* ── Display: clickable area to open dropdown ── */
.${MODEL_SELECTOR_CLASSES.DISPLAY} {
  display: flex;
  align-items: center;
  gap: var(--voxcpm-space-s);
  flex: 1;
  padding: var(--voxcpm-space-xs) var(--voxcpm-space-m);
  background: var(--voxcpm-bg-input);
  border: 1px solid var(--voxcpm-border-color);
  border-radius: var(--voxcpm-border-radius-xs);
  cursor: pointer;
  min-height: var(--voxcpm-input-height);
  color: var(--voxcpm-text-primary);
  font-family: var(--voxcpm-font-family);
  font-size: var(--voxcpm-font-size-xs);
  transition: border-color var(--voxcpm-duration-fast) var(--voxcpm-easing),
              background var(--voxcpm-duration-fast) var(--voxcpm-easing);
  user-select: none;
  overflow: hidden;
}

.${MODEL_SELECTOR_CLASSES.DISPLAY}:hover {
  border-color: var(--voxcpm-border-color-hover);
  background: var(--voxcpm-bg-input-hover);
}

.${MODEL_SELECTOR_CLASSES.DISPLAY}:active {
  background: var(--voxcpm-bg-input-active);
}

/* ── Icon: folder icon indicating model source ── */
.${MODEL_SELECTOR_CLASSES.ICON} {
  font-size: var(--voxcpm-font-size-s);
  opacity: 0.8;
  flex-shrink: 0;
  line-height: 1;
}

/* ── Text: model name ── */
.${MODEL_SELECTOR_CLASSES.TEXT} {
  flex: 1;
  color: var(--voxcpm-text-primary);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
}

/* ── Arrow: dropdown indicator ── */
.${MODEL_SELECTOR_CLASSES.ARROW} {
  font-size: 9px;
  opacity: 0.5;
  margin-left: auto;
  flex-shrink: 0;
  line-height: 1;
  transition: opacity var(--voxcpm-duration-fast) var(--voxcpm-easing);
}

.${MODEL_SELECTOR_CLASSES.DISPLAY}:hover .${MODEL_SELECTOR_CLASSES.ARROW} {
  opacity: 0.8;
}

/* ── Browse: custom path button ── */
.${MODEL_SELECTOR_CLASSES.BROWSE} {
  /* Reset ComfyUI's global button styles */
  all: initial;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--voxcpm-space-xs) var(--voxcpm-space-m);
  min-width: var(--voxcpm-browse-width);
  height: var(--voxcpm-input-height);
  font-size: var(--voxcpm-font-size-s);
  border-radius: var(--voxcpm-border-radius-xs);
  cursor: pointer;
  flex-shrink: 0;
  border: 1px solid var(--voxcpm-border-color);
  background: var(--voxcpm-bg-input);
  color: var(--voxcpm-text-primary);
  font-family: var(--voxcpm-font-family);
  transition: border-color var(--voxcpm-duration-fast) var(--voxcpm-easing),
              background var(--voxcpm-duration-fast) var(--voxcpm-easing);
  line-height: 1;
  box-sizing: border-box;
}

.${MODEL_SELECTOR_CLASSES.BROWSE}:hover {
  border-color: var(--voxcpm-border-color-hover);
  background: var(--voxcpm-bg-input-hover);
}

.${MODEL_SELECTOR_CLASSES.BROWSE}:active {
  background: var(--voxcpm-bg-input-active);
}

/* ── Path: custom path indicator (below row, hidden by default) ── */
.${MODEL_SELECTOR_CLASSES.PATH} {
  font-size: var(--voxcpm-font-size-2xs);
  color: var(--voxcpm-text-muted);
  padding: var(--voxcpm-space-2xs) 0 0 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  width: 100%;
  box-sizing: border-box;
  display: none;
}

.${MODEL_SELECTOR_CLASSES.PATH_VISIBLE} {
  display: block;
}

/* ── Focus: keyboard accessibility ── */
.${MODEL_SELECTOR_CLASSES.DISPLAY}:focus-visible,
.${MODEL_SELECTOR_CLASSES.BROWSE}:focus-visible {
  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-focus-ring-color);
  outline-offset: var(--voxcpm-focus-ring-offset);
}

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {
  .${MODEL_SELECTOR_CLASSES.BLOCK} *,
  .${MODEL_SELECTOR_CLASSES.BLOCK} *::before,
  .${MODEL_SELECTOR_CLASSES.BLOCK} *::after {
    transition-duration: 0ms !important;
  }
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
let allStylesInjected = false;
function injectAllStyles() {
  if (allStylesInjected) {
    return;
  }
  injectDesignTokens();
  injectModelSelectorStyles();
  allStylesInjected = true;
}
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
    overlay.className = "voxcpm-dialog__overlay";
    overlay.innerHTML = `
    <div class="voxcpm-dialog">
    <div class="voxcpm-dialog__header">
    <span class="voxcpm-dialog__icon">🎙️</span>
    <h2 class="voxcpm-dialog__title">VoxCPM Model Setup</h2>
    </div>
    <div class="voxcpm-dialog__body">
    <p class="voxcpm-dialog__description">
    Welcome to VoxCPM! How would you like to configure your model path?
    </p>
    <div class="voxcpm-dialog__options">
    <button class="voxcpm-dialog__option" data-option="default">
    <div class="voxcpm-dialog__option-icon">📁</div>
    <div class="voxcpm-dialog__option-content">
    <h3>Use Default Path</h3>
    <p class="voxcpm-dialog__option-path">models/tts/VoxCPM/</p>
    <p class="voxcpm-dialog__option-description">
    Official models will be downloaded automatically.
    </p>
    </div>
    </button>
    <button class="voxcpm-dialog__option" data-option="custom">
    <div class="voxcpm-dialog__option-icon">📂</div>
    <div class="voxcpm-dialog__option-content">
    <h3>Use Custom Path</h3>
    <p class="voxcpm-dialog__option-description">
    Select a custom directory where your VoxCPM models are located.
    Useful for existing model collections.
    </p>
    </div>
    </button>
    </div>
    </div>
    <div class="voxcpm-dialog__footer">
    <button class="voxcpm-dialog__btn voxcpm-dialog__btn--secondary" id="voxcpm-cancel">Cancel</button>
    <button class="voxcpm-dialog__btn voxcpm-dialog__btn--primary" id="voxcpm-continue" disabled>
    Continue
    </button>
    </div>
    </div>
    `;
    let selectedOption = null;
    const optionCards = overlay.querySelectorAll(".voxcpm-dialog__option");
    optionCards.forEach((card) => {
      card.addEventListener("click", () => {
        optionCards.forEach((c) => c.classList.remove("voxcpm-dialog__option--selected"));
        card.classList.add("voxcpm-dialog__option--selected");
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
    overlay.className = "voxcpm-dialog__overlay";
    overlay.innerHTML = `
    <div class="voxcpm-dialog voxcpm-dialog--large">
    <div class="voxcpm-dialog__header">
    <h2 class="voxcpm-dialog__title">Select Model Directory</h2>
    </div>
    <div class="voxcpm-dialog__body">
    <div class="voxcpm-dialog__input-group">
    <input
    type="text"
    class="voxcpm-dialog__input"
    placeholder="Enter model directory path"
    value="${initialPath || ""}"
    id="voxcpm-path-input"
    />
    <button class="voxcpm-dialog__btn voxcpm-dialog__btn--secondary" id="voxcpm-browse">
    Browse...
    </button>
    </div>
    <div class="voxcpm-dialog__preview" id="voxcpm-models-preview" style="display: none;">
    <h4>Found Models (<span id="voxcpm-model-count">0</span>)</h4>
    <ul class="voxcpm-dialog__models-list" id="voxcpm-models-list"></ul>
    </div>
    <div class="voxcpm-dialog__validation" id="voxcpm-validation-message"></div>
    </div>
    <div class="voxcpm-dialog__footer">
    <button class="voxcpm-dialog__btn voxcpm-dialog__btn--secondary" id="voxcpm-cancel">Cancel</button>
    <button class="voxcpm-dialog__btn voxcpm-dialog__btn--primary" id="voxcpm-confirm" disabled>
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
          <span class="voxcpm-dialog__model-icon">📁</span>
          <span class="voxcpm-dialog__model-name">${m.name}</span>
          </li>
          `).join("");
          validationMessage.innerHTML = `<span class="voxcpm-dialog__validation--success">Found ${result.models.length} VoxCPM-compatible models</span>`;
          confirmBtn.disabled = false;
        } else {
          modelsPreview.style.display = "none";
          validationMessage.innerHTML = `<span class="voxcpm-dialog__validation--error">${result.error || "Invalid path or no models found"}</span>`;
          confirmBtn.disabled = true;
        }
      } catch (error) {
        logger.warn("Path validation failed:", error);
        modelsPreview.style.display = "none";
        validationMessage.innerHTML = `<span class="voxcpm-dialog__validation--error">Failed to validate path</span>`;
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
/* ============================================================================
 * VoxCPM Dialog Styles (BEM)
 * Block: .voxcpm-dialog
 * ============================================================================ */

/* ── Overlay ── */
.voxcpm-dialog__overlay {
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

/* ── Block: Dialog Container ── */
.voxcpm-dialog {
  background: var(--voxcpm-bg-surface);
  border-radius: var(--voxcpm-border-radius-m);
  max-width: 500px;
  width: 90%;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}

.voxcpm-dialog--large {
  max-width: 600px;
}

/* ── Header ── */
.voxcpm-dialog__header {
  display: flex;
  align-items: center;
  gap: var(--voxcpm-space-l);
  padding: var(--voxcpm-space-2xl);
  border-bottom: 1px solid var(--voxcpm-border-color);
}

.voxcpm-dialog__icon {
  font-size: var(--voxcpm-font-size-xl);
}

.voxcpm-dialog__title {
  margin: 0;
  font-size: var(--voxcpm-font-size-l);
  color: var(--voxcpm-text-primary);
}

/* ── Body ── */
.voxcpm-dialog__body {
  padding: var(--voxcpm-space-2xl);
}

.voxcpm-dialog__description {
  margin: 0 0 20px;
  color: var(--voxcpm-text-secondary);
  line-height: 1.5;
}

/* ── Options Container ── */
.voxcpm-dialog__options {
  display: flex;
  flex-direction: column;
  gap: var(--voxcpm-space-l);
}

/* ── Option Card ── */
.voxcpm-dialog__option {
  display: flex;
  gap: var(--voxcpm-space-xl);
  padding: var(--voxcpm-space-xl);
  background: var(--voxcpm-bg-elevated);
  border: 2px solid var(--voxcpm-border-color);
  border-radius: var(--voxcpm-border-radius-m);
  cursor: pointer;
  text-align: left;
  transition: border-color var(--voxcpm-duration-normal) var(--voxcpm-easing),
              background var(--voxcpm-duration-normal) var(--voxcpm-easing);
}

.voxcpm-dialog__option:hover {
  border-color: var(--voxcpm-accent);
}

.voxcpm-dialog__option--selected {
  border-color: var(--voxcpm-accent);
  background: var(--voxcpm-bg-input-hover);
}

.voxcpm-dialog__option-icon {
  font-size: 32px;
  flex-shrink: 0;
}

.voxcpm-dialog__option-content h3 {
  margin: 0 0 var(--voxcpm-space-xs);
  color: var(--voxcpm-text-primary);
  font-size: var(--voxcpm-font-size-m);
}

.voxcpm-dialog__option-path {
  margin: 0 0 var(--voxcpm-space-m);
  font-family: var(--voxcpm-font-family-mono);
  font-size: var(--voxcpm-font-size-xs);
  color: var(--voxcpm-text-muted);
}

.voxcpm-dialog__option-description {
  margin: 0;
  font-size: var(--voxcpm-font-size-xs);
  color: var(--voxcpm-text-muted);
}

/* ── Footer ── */
.voxcpm-dialog__footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--voxcpm-space-l);
  padding: var(--voxcpm-space-xl) var(--voxcpm-space-2xl);
  border-top: 1px solid var(--voxcpm-border-color);
}

/* ── Buttons ── */
.voxcpm-dialog__btn {
  padding: var(--voxcpm-space-m) var(--voxcpm-space-xl);
  border-radius: var(--voxcpm-border-radius-xs);
  font-size: var(--voxcpm-font-size-m);
  cursor: pointer;
  transition: background var(--voxcpm-duration-normal) var(--voxcpm-easing),
              border-color var(--voxcpm-duration-normal) var(--voxcpm-easing);
}

.voxcpm-dialog__btn--secondary {
  background: transparent;
  border: 1px solid var(--voxcpm-border-color);
  color: var(--voxcpm-text-secondary);
}

.voxcpm-dialog__btn--secondary:hover {
  background: var(--voxcpm-bg-elevated);
}

.voxcpm-dialog__btn--primary {
  background: var(--voxcpm-accent);
  border: none;
  color: var(--voxcpm-text-on-primary);
}

.voxcpm-dialog__btn--primary:hover:not(:disabled) {
  background: var(--voxcpm-accent-hover);
}

.voxcpm-dialog__btn--primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ── Input Group ── */
.voxcpm-dialog__input-group {
  display: flex;
  gap: var(--voxcpm-space-m);
  margin-bottom: var(--voxcpm-space-xl);
}

.voxcpm-dialog__input {
  flex: 1;
  padding: var(--voxcpm-space-m) var(--voxcpm-space-l);
  background: var(--voxcpm-bg-elevated);
  border: 1px solid var(--voxcpm-border-color);
  border-radius: var(--voxcpm-border-radius-xs);
  color: var(--voxcpm-text-primary);
  font-size: var(--voxcpm-font-size-m);
  font-family: var(--voxcpm-font-family);
}

.voxcpm-dialog__input:focus {
  outline: none;
  border-color: var(--voxcpm-accent);
}

/* ── Models Preview ── */
.voxcpm-dialog__preview {
  background: var(--voxcpm-bg-elevated);
  border-radius: var(--voxcpm-border-radius-xs);
  padding: var(--voxcpm-space-l);
  margin-bottom: var(--voxcpm-space-xl);
}

.voxcpm-dialog__preview h4 {
  margin: 0 0 var(--voxcpm-space-m);
  font-size: var(--voxcpm-font-size-xs);
  color: var(--voxcpm-text-muted);
}

.voxcpm-dialog__models-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.voxcpm-dialog__models-list li {
  display: flex;
  align-items: center;
  gap: var(--voxcpm-space-m);
  padding: var(--voxcpm-space-xs) 0;
}

.voxcpm-dialog__model-icon {
  font-size: var(--voxcpm-font-size-m);
}

.voxcpm-dialog__model-name {
  font-size: var(--voxcpm-font-size-s);
  color: var(--voxcpm-text-secondary);
}

/* ── Validation ── */
.voxcpm-dialog__validation {
  font-size: var(--voxcpm-font-size-xs);
}

.voxcpm-dialog__validation--success {
  color: var(--voxcpm-color-success);
}

.voxcpm-dialog__validation--error {
  color: var(--voxcpm-color-error);
}

/* ── Focus: keyboard accessibility ── */
.voxcpm-dialog__option:focus-visible,
.voxcpm-dialog__btn:focus-visible,
.voxcpm-dialog__input:focus-visible {
  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-focus-ring-color);
  outline-offset: var(--voxcpm-focus-ring-offset);
}

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {
  .voxcpm-dialog *,
  .voxcpm-dialog *::before,
  .voxcpm-dialog *::after {
    transition-duration: 0ms !important;
  }
}

/* ============================================================================
 * VoxCPM Path Indicator Widget (BEM)
 * Block: .voxcpm-path-indicator
 * ============================================================================ */

.voxcpm-path-indicator {
  display: flex;
  align-items: center;
  gap: var(--voxcpm-space-m);
  padding: var(--voxcpm-space-xs) var(--voxcpm-space-m);
  background: var(--voxcpm-bg-elevated);
  border-radius: var(--voxcpm-border-radius-xs);
  font-size: var(--voxcpm-font-size-xs);
}

.voxcpm-path-indicator__text {
  color: var(--voxcpm-text-muted);
  font-family: var(--voxcpm-font-family-mono);
}

.voxcpm-path-indicator__change-btn {
  background: transparent;
  border: none;
  cursor: pointer;
  font-size: var(--voxcpm-font-size-m);
  padding: var(--voxcpm-space-2xs);
  color: var(--voxcpm-text-primary);
  transition: opacity var(--voxcpm-duration-fast) var(--voxcpm-easing);
}

.voxcpm-path-indicator__change-btn:hover {
  opacity: 0.8;
}

.voxcpm-path-indicator__change-btn:focus-visible {
  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-focus-ring-color);
  outline-offset: var(--voxcpm-focus-ring-offset);
}
`;
  const style = document.createElement("style");
  style.id = styleId;
  style.textContent = css;
  document.head.appendChild(style);
}
const selectorRefsMap = /* @__PURE__ */ new Map();
function createModelSelectorWidget(node) {
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
  const fragment = document.createDocumentFragment();
  const modelDisplay = document.createElement("div");
  modelDisplay.className = MODEL_SELECTOR_CLASSES.DISPLAY;
  modelDisplay.title = "Click to select a model";
  modelDisplay.tabIndex = 0;
  modelDisplay.setAttribute("role", "combobox");
  modelDisplay.setAttribute("aria-expanded", "false");
  modelDisplay.setAttribute("aria-haspopup", "listbox");
  const folderIcon = document.createElement("span");
  folderIcon.className = MODEL_SELECTOR_CLASSES.ICON;
  folderIcon.textContent = UI.MODEL_SELECTOR.DEFAULT_ICON;
  const modelText = document.createElement("span");
  modelText.className = MODEL_SELECTOR_CLASSES.TEXT;
  modelText.textContent = modelWidget.value || UI.MODEL_SELECTOR.PLACEHOLDER;
  const arrow = document.createElement("span");
  arrow.className = MODEL_SELECTOR_CLASSES.ARROW;
  arrow.textContent = UI.MODEL_SELECTOR.ARROW;
  const browseBtn = document.createElement("button");
  browseBtn.className = MODEL_SELECTOR_CLASSES.BROWSE;
  browseBtn.textContent = UI.MODEL_SELECTOR.CUSTOM_ICON;
  browseBtn.title = "Browse for custom model directory";
  browseBtn.setAttribute("role", "button");
  const pathIndicator = document.createElement("div");
  pathIndicator.className = MODEL_SELECTOR_CLASSES.PATH;
  const savedCustomPath = node.properties?.["custom_model_path"];
  if (savedCustomPath) {
    folderIcon.textContent = UI.MODEL_SELECTOR.CUSTOM_ICON;
    pathIndicator.textContent = savedCustomPath;
    pathIndicator.classList.add(MODEL_SELECTOR_CLASSES.PATH_VISIBLE);
  }
  modelDisplay.append(folderIcon, modelText, arrow);
  const row = document.createElement("div");
  row.className = MODEL_SELECTOR_CLASSES.ROW;
  row.append(modelDisplay, browseBtn);
  fragment.append(row, pathIndicator);
  const widgetChildren = [row, pathIndicator];
  for (const child of widgetChildren) {
    child.addEventListener("mousedown", (e) => e.stopPropagation());
    child.addEventListener("mouseup", (e) => e.stopPropagation());
    child.addEventListener("click", (e) => e.stopPropagation());
  }
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
  modelDisplay.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      modelDisplay.setAttribute("aria-expanded", "true");
      openModelDropdown(
        e,
        modelWidget,
        modelText,
        folderIcon,
        pathIndicator,
        node
      );
    }
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
  const domWidget = node.addDOMWidget("model_selector", "custom", fragment, {
    serialize: false,
    hideOnZoom: false,
    selectOn: ["click"],
    getValue: () => modelWidget.value,
    setValue: (v) => {
      modelWidget.value = v;
      modelText.textContent = v;
    }
  });
  const parentObserver = new MutationObserver(() => {
    const rowParent = row.parentElement;
    if (rowParent) {
      rowParent.classList.add(MODEL_SELECTOR_CLASSES.BLOCK);
      rowParent.setAttribute("node-id", String(node.id));
      parentObserver.disconnect();
      logger.debug("Added", MODEL_SELECTOR_CLASSES.BLOCK, "class + node-id to parent div for node:", node.id);
    }
  });
  parentObserver.observe(document.body, { childList: true, subtree: true });
  domWidget.tooltip = "Select the VoxCPM model to use";
  domWidget.computeSize = () => [0, UI.MODEL_SELECTOR.MIN_HEIGHT];
  domWidget.computeLayoutSize = void 0;
  const allWidgets = node.widgets;
  if (allWidgets && allWidgets.length > 1) {
    const widgetIndex = allWidgets.indexOf(domWidget);
    if (widgetIndex > 0) {
      allWidgets.splice(widgetIndex, 1);
      allWidgets.unshift(domWidget);
      logger.debug("Moved model selector widget to position 0 for node:", node.id);
    }
  }
  const modelInput = node.inputs?.find(
    (inp) => inp.name === "model_name"
  );
  if (modelInput) {
    const existingWidgetName = modelInput.widget?.name;
    modelInput.widget = {
      name: "model_selector",
      // Store original for reference/debugging
      _originalWidget: existingWidgetName
    };
    logger.debug(
      "Bound model_name input to model_selector widget for node:",
      node.id
    );
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
  if (!widget.options) widget.options = {};
  widget.options.canvasOnly = true;
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
      pathIndicator.classList.add(MODEL_SELECTOR_CLASSES.PATH_VISIBLE);
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
    refs.pathIndicator.classList.add(MODEL_SELECTOR_CLASSES.PATH_VISIBLE);
  } else {
    refs.folderIcon.textContent = UI.MODEL_SELECTOR.DEFAULT_ICON;
    refs.pathIndicator.classList.remove(MODEL_SELECTOR_CLASSES.PATH_VISIBLE);
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
    injectAllStyles();
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
