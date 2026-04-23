import { app } from "../../scripts/app.js";
import { app as app$1 } from "../scripts/app.js";
import { api } from "../scripts/api.js";
//#region core/constants.ts
/**
* VoxCPM Constants
* 
* Central location for all constants used in the VoxCPM frontend extension.
*/
/**
* Session storage keys for persisting state
*/
var STORAGE_KEYS = {
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
/**
* WebSocket event names for server-to-client communication
*/
var EVENT_NAMES = {
	/** Status notification event */
	STATUS: "voxcpm.status",
	/** Model loaded event (future use) */
	MODEL_LOADED: "voxcpm.model_loaded",
	/** Generation progress event (future use) */
	GENERATION_PROGRESS: "voxcpm.generation_progress",
	/** Config event from server */
	CONFIG: "voxcpm.config",
	/** Settings update request */
	SETTINGS_UPDATE: "voxcpm.settings_update"
};
/**
* Default configuration values
*/
var DEFAULTS = {
	/** Default toast notification lifetime in milliseconds */
	TOAST_LIFE: 1e4,
	/** Default log prefix */
	LOG_PREFIX: "[VoxCPM]",
	/** Extension name for ComfyUI registration */
	EXTENSION_NAME: "voxcpm.frontend",
	/** Node class name for VoxCPM TTS nodes */
	NODE_CLASS: "VoxCPM_TTS"
};
/**
* UI-related constants
*/
var UI = {
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
		/** Default model icon (standard path) — deprecated, use MODEL_ICONS.FOLDER */
		DEFAULT_ICON: "📁",
		/** Custom path model icon — deprecated, use MODEL_ICONS.FOLDER_OPEN */
		CUSTOM_ICON: "📂",
		/** Dropdown arrow character */
		ARROW: "▼",
		/** Placeholder text when no model is selected */
		PLACEHOLDER: "Select model...",
		/** CSS class prefix for all model selector styles */
		CSS_PREFIX: "voxcpm-model-"
	},
	/** Model dropdown constants (custom cyber-themed dropdown panel) */
	MODEL_DROPDOWN: {
		/** BEM block class name */
		BLOCK_CLASS: "voxcpm-model-dropdown",
		/** Header text displayed in the dropdown panel */
		HEADER_TEXT: "SELECT MODEL",
		/** Maximum visible items before scrolling */
		MAX_VISIBLE_ITEMS: 8,
		/** Item height in pixels (for scroll calculation) */
		ITEM_HEIGHT: 36,
		/** Gap between anchor and dropdown in pixels */
		ANCHOR_GAP: 4,
		/** Animation duration in ms */
		ANIMATION_DURATION: 150,
		/** Z-index for the dropdown panel */
		Z_INDEX: 1e3
	}
};
//#endregion
//#region core/logger.ts
/**
* VoxCPM Logger Utility
* 
* Provides consistent logging with prefix for all VoxCPM frontend modules.
*/
/**
* Logger instance with prefixed console output
*/
var logger = {
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
		if (typeof window !== "undefined" && window.__VOXCPM_DEBUG__) console.debug(DEFAULTS.LOG_PREFIX, "[DEBUG]", ...args);
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
//#endregion
//#region styles/tokens.ts
/**
* VoxCPM Design Tokens
*
* CSS custom properties (design tokens) for the VoxCPM frontend extension.
* These tokens provide a semantic layer on top of ComfyUI's native CSS variables,
* with fallback values for themes that don't define the ComfyUI variables.
*
* Token architecture:
* ```
* :root (ComfyUI-provided)
* └── --comfy-input-bg, --fg-color, --border-color, etc.
*
* :root (VoxCPM design tokens)
* └── --voxcpm-* (semantic tokens referencing ComfyUI tokens)
*
* Component (BEM block)
* └── Uses --voxcpm-* tokens (no raw fallbacks needed)
* ```
*
* @module voxcpm/styles/tokens
*/
/**
* CSS custom property definitions for all VoxCPM design tokens.
*
* Injected into `:root` so they're available globally.
* Each token references a ComfyUI variable with a fallback value
* for themes that don't define the ComfyUI variable.
*/
var DESIGN_TOKEN_STYLES = `
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

  /* ── Cyber Theme ── */
  --voxcpm-cyber-font: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
  --voxcpm-cyber-accent: #00ff9d;
  --voxcpm-cyber-accent-dim: rgba(0, 255, 157, 0.15);
  --voxcpm-cyber-border: rgba(0, 255, 157, 0.3);
  --voxcpm-cyber-glow: 0 0 8px rgba(0, 255, 157, 0.4);
  --voxcpm-cyber-text: #e0e0e0;
  --voxcpm-cyber-text-dim: rgba(224, 224, 224, 0.5);
  --voxcpm-cyber-bg: rgba(10, 15, 20, 0.95);
  --voxcpm-cyber-bg-hover: rgba(0, 255, 157, 0.08);
  --voxcpm-cyber-bg-active: rgba(0, 255, 157, 0.12);
  --voxcpm-cyber-tag-bg: rgba(0, 255, 157, 0.15);
  --voxcpm-cyber-tag-border: rgba(0, 255, 157, 0.4);
  --voxcpm-cyber-tag-text: #00ff9d;
  --voxcpm-cyber-scanline: rgba(0, 255, 157, 0.03);
}
`.trim();
/** ID for the injected design token style element */
var STYLE_ELEMENT_ID$2 = "voxcpm-design-tokens";
/** Whether design tokens have been injected */
var tokensInjected = false;
/**
* Inject VoxCPM design token CSS custom properties into the document head.
*
* This function is idempotent — calling it multiple times has no effect
* after the first injection. Tokens are injected once per page load
* into `:root` so they're available globally to all components.
*
* Should be called early in the extension lifecycle (e.g., `init()` hook)
* before any component styles reference the `--voxcpm-*` tokens.
*/
function injectDesignTokens() {
	if (tokensInjected) return;
	if (document.getElementById(STYLE_ELEMENT_ID$2)) {
		tokensInjected = true;
		return;
	}
	const style = document.createElement("style");
	style.id = STYLE_ELEMENT_ID$2;
	style.textContent = DESIGN_TOKEN_STYLES;
	document.head.appendChild(style);
	tokensInjected = true;
}
//#endregion
//#region widgets/modelSelectorStyles.ts
/**
* Model Selector Widget Styles (BEM)
*
* CSS styles for the custom model selector DOM widget.
* Uses BEM naming convention: `.voxcpm-model-selector__element--modifier`
* Uses VoxCPM design tokens: `var(--voxcpm-*)` for all values.
*
* @module voxcpm/widgets/model-selector-styles
*/
/**
* BEM class names for the model selector widget.
*
* Centralized here so both CSS and TypeScript use the same names.
* If a class name changes, update it here and the CSS will follow.
*/
var MODEL_SELECTOR_CLASSES = {
	/** Block — the widget container (applied to the container div, and also to WidgetDOM.vue parent div in Node 2.0 mode) */
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
	/** Modifier — arrow highlighted (dropdown open) */
	ARROW_ACTIVE: "voxcpm-model-selector__arrow--active",
	/** Element — browse button for custom path selection */
	BROWSE: "voxcpm-model-selector__browse",
	/** Element — custom path indicator shown below the row */
	PATH: "voxcpm-model-selector__path",
	/** Modifier — path indicator visible state */
	PATH_VISIBLE: "voxcpm-model-selector__path--visible"
};
/** CSS styles for the model selector widget */
var MODEL_SELECTOR_STYLES = `
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
  background: var(--voxcpm-cyber-bg);
  border: 1px solid var(--voxcpm-cyber-border);
  border-radius: 2px;
  cursor: pointer;
  min-height: var(--voxcpm-input-height);
  color: var(--voxcpm-cyber-text);
  font-family: var(--voxcpm-cyber-font);
  font-size: var(--voxcpm-font-size-xs);
  transition: border-color var(--voxcpm-duration-fast) var(--voxcpm-easing),
  background var(--voxcpm-duration-fast) var(--voxcpm-easing),
  box-shadow var(--voxcpm-duration-fast) var(--voxcpm-easing);
  user-select: none;
  overflow: hidden;
}

.${MODEL_SELECTOR_CLASSES.DISPLAY}:hover {
  border-color: var(--voxcpm-cyber-accent);
  box-shadow: var(--voxcpm-cyber-glow);
}

.${MODEL_SELECTOR_CLASSES.DISPLAY}:active {
  background: var(--voxcpm-cyber-bg-active);
}

/* ── Icon: folder icon indicating model source ── */
.${MODEL_SELECTOR_CLASSES.ICON} {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  color: var(--voxcpm-cyber-accent);
}

.${MODEL_SELECTOR_CLASSES.ICON} svg {
  width: 16px;
  height: 16px;
}

/* ── Text: model name ── */
.${MODEL_SELECTOR_CLASSES.TEXT} {
  flex: 1;
  color: var(--voxcpm-cyber-text);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
}

/* ── Arrow: dropdown indicator ── */
.${MODEL_SELECTOR_CLASSES.ARROW} {
  font-size: 9px;
  color: var(--voxcpm-cyber-accent);
  margin-left: auto;
  flex-shrink: 0;
  line-height: 1;
  opacity: 0.3;
  transition: opacity var(--voxcpm-duration-fast) var(--voxcpm-easing);
}

/* Arrow highlighted on display hover */
.${MODEL_SELECTOR_CLASSES.DISPLAY}:hover .${MODEL_SELECTOR_CLASSES.ARROW},
/* Arrow highlighted when dropdown is open */
.${MODEL_SELECTOR_CLASSES.ARROW_ACTIVE} {
  opacity: 1;
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
  border-radius: 2px;
  cursor: pointer;
  flex-shrink: 0;
  border: 1px solid var(--voxcpm-cyber-border);
  background: var(--voxcpm-cyber-bg);
  color: var(--voxcpm-cyber-text);
  font-family: var(--voxcpm-cyber-font);
  transition: border-color var(--voxcpm-duration-fast) var(--voxcpm-easing),
  background var(--voxcpm-duration-fast) var(--voxcpm-easing),
  box-shadow var(--voxcpm-duration-fast) var(--voxcpm-easing);
  line-height: 1;
  box-sizing: border-box;
}

.${MODEL_SELECTOR_CLASSES.BROWSE}:hover {
  border-color: var(--voxcpm-cyber-accent);
  box-shadow: var(--voxcpm-cyber-glow);
}

.${MODEL_SELECTOR_CLASSES.BROWSE}:active {
  background: var(--voxcpm-cyber-bg-active);
}

/* ── Path: custom path indicator (below row, hidden by default) ── */
.${MODEL_SELECTOR_CLASSES.PATH} {
  font-size: var(--voxcpm-font-size-2xs);
  color: var(--voxcpm-cyber-text-dim);
  padding: var(--voxcpm-space-2xs) 0 0 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  width: 100%;
  box-sizing: border-box;
  display: none;
  font-family: var(--voxcpm-cyber-font);
}

.${MODEL_SELECTOR_CLASSES.PATH_VISIBLE} {
  display: block;
}

/* ── Focus: keyboard accessibility ── */
.${MODEL_SELECTOR_CLASSES.DISPLAY}:focus-visible,
.${MODEL_SELECTOR_CLASSES.BROWSE}:focus-visible {
  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-cyber-accent);
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
/** ID for the injected style element */
var STYLE_ELEMENT_ID$1 = "voxcpm-model-selector-styles";
/** Whether styles have been injected */
var stylesInjected$1 = false;
/**
* Inject model selector CSS styles into the document head.
*
* This function is idempotent — calling it multiple times has no effect
* after the first injection. Styles are injected once per page load.
*
* Requires design tokens to be injected first (via `injectDesignTokens()`).
*/
function injectModelSelectorStyles() {
	if (stylesInjected$1) return;
	if (document.getElementById(STYLE_ELEMENT_ID$1)) {
		stylesInjected$1 = true;
		return;
	}
	const style = document.createElement("style");
	style.id = STYLE_ELEMENT_ID$1;
	style.textContent = MODEL_SELECTOR_STYLES;
	document.head.appendChild(style);
	stylesInjected$1 = true;
}
//#endregion
//#region widgets/modelDropdownStyles.ts
/**
* Model Dropdown Styles (BEM + Cyber Theme)
*
* CSS styles for the custom model selector dropdown panel.
* Uses BEM naming convention: `.voxcpm-model-dropdown__element--modifier`
* Uses cyber-theme design tokens: `var(--voxcpm-cyber-*)` for all values.
*
* @module voxcpm/widgets/model-dropdown-styles
*/
/**
* BEM class names for the model dropdown panel.
*
* Centralized here so both CSS and TypeScript use the same names.
* If a class name changes, update it here and the CSS will follow.
*/
var MODEL_DROPDOWN_CLASSES = {
	/** Block — the dropdown panel container */
	BLOCK: "voxcpm-model-dropdown",
	/** Element — header bar with shield icon and title */
	HEADER: "voxcpm-model-dropdown__header",
	/** Element — scrollable item list */
	LIST: "voxcpm-model-dropdown__list",
	/** Element — model item row */
	ITEM: "voxcpm-model-dropdown__item",
	/** Modifier — keyboard-selected item */
	ITEM_SELECTED: "voxcpm-model-dropdown__item--selected",
	/** Element — status icon container */
	ICON: "voxcpm-model-dropdown__icon",
	/** Modifier — cloud icon (model needs download) */
	ICON_CLOUD: "voxcpm-model-dropdown__icon--cloud",
	/** Modifier — check icon (model available locally) */
	ICON_CHECK: "voxcpm-model-dropdown__icon--check",
	/** Element — model name text */
	NAME: "voxcpm-model-dropdown__name",
	/** Element — "DEFAULT" tag for official models */
	TAG: "voxcpm-model-dropdown__tag",
	/** Element — size/architecture metadata */
	META: "voxcpm-model-dropdown__meta"
};
/** CSS styles for the model dropdown panel */
var MODEL_DROPDOWN_STYLES = `
/* ============================================================================
 * VoxCPM Model Dropdown (BEM + Cyber Theme)
 * Block: .voxcpm-model-dropdown
 * ============================================================================
 * Custom dropdown panel replacing LiteGraph.ContextMenu.
 * Cyber-security aesthetic: monospace font, neon green accent, scanline overlay.
 * ============================================================================ */

/* ── Block: Panel ── */
.${MODEL_DROPDOWN_CLASSES.BLOCK} {
  position: fixed;
  z-index: 1000;
  min-width: 280px;
  max-width: 400px;
  max-height: 320px;
  background: var(--voxcpm-cyber-bg);
  border: 1px solid var(--voxcpm-cyber-border);
  border-radius: 2px;
  box-shadow: var(--voxcpm-cyber-glow), 0 8px 32px rgba(0, 0, 0, 0.5);
  font-family: var(--voxcpm-cyber-font);
  font-size: var(--voxcpm-font-size-xs);
  color: var(--voxcpm-cyber-text);
  overflow: hidden;
  animation: voxcpm-dropdown-in 150ms ease-out;
  outline: none;
}

/* Scanline overlay — subtle cyber texture */
.${MODEL_DROPDOWN_CLASSES.BLOCK}::after {
  content: '';
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    var(--voxcpm-cyber-scanline) 2px,
    var(--voxcpm-cyber-scanline) 4px
  );
  pointer-events: none;
  z-index: 1;
}

/* ── Element: Header ── */
.${MODEL_DROPDOWN_CLASSES.HEADER} {
  padding: 8px 12px;
  border-bottom: 1px solid var(--voxcpm-cyber-border);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--voxcpm-cyber-accent);
  display: flex;
  align-items: center;
  gap: 6px;
  position: relative;
  z-index: 2;
  user-select: none;
}

.${MODEL_DROPDOWN_CLASSES.HEADER} svg {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
}

/* ── Element: List ── */
.${MODEL_DROPDOWN_CLASSES.LIST} {
  overflow-y: auto;
  overflow-x: hidden;
  max-height: 280px;
  position: relative;
  z-index: 2;
}

/* Custom scrollbar — cyber style */
.${MODEL_DROPDOWN_CLASSES.LIST}::-webkit-scrollbar {
  width: 4px;
}

.${MODEL_DROPDOWN_CLASSES.LIST}::-webkit-scrollbar-track {
  background: transparent;
}

.${MODEL_DROPDOWN_CLASSES.LIST}::-webkit-scrollbar-thumb {
  background: var(--voxcpm-cyber-border);
  border-radius: 0;
}

.${MODEL_DROPDOWN_CLASSES.LIST}::-webkit-scrollbar-thumb:hover {
  background: var(--voxcpm-cyber-accent);
}

/* ── Element: Item ── */
.${MODEL_DROPDOWN_CLASSES.ITEM} {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  transition: background 100ms ease, border-left-color 100ms ease;
  border-left: 2px solid transparent;
  position: relative;
  z-index: 2;
  user-select: none;
}

.${MODEL_DROPDOWN_CLASSES.ITEM}:hover,
.${MODEL_DROPDOWN_CLASSES.ITEM_SELECTED} {
  background: var(--voxcpm-cyber-bg-hover);
  border-left-color: var(--voxcpm-cyber-accent);
}

.${MODEL_DROPDOWN_CLASSES.ITEM}:active {
  background: var(--voxcpm-cyber-bg-active);
}

/* ── Element: Icon ── */
.${MODEL_DROPDOWN_CLASSES.ICON} {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.${MODEL_DROPDOWN_CLASSES.ICON} svg {
  width: 16px;
  height: 16px;
}

/* Modifier: Cloud icon (needs download) — blue accent */
.${MODEL_DROPDOWN_CLASSES.ICON_CLOUD} {
  color: var(--voxcpm-accent);
}

/* Modifier: Check icon (available locally) — green success */
.${MODEL_DROPDOWN_CLASSES.ICON_CHECK} {
  color: var(--voxcpm-color-success);
}

/* ── Element: Name ── */
.${MODEL_DROPDOWN_CLASSES.NAME} {
  flex: 0 1 auto;
  color: var(--voxcpm-cyber-text);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
  min-width: 0;
}

/* ── Element: Meta (size info, after name) ── */
.${MODEL_DROPDOWN_CLASSES.META} {
  font-size: 10px;
  color: var(--voxcpm-cyber-text-dim);
  font-family: var(--voxcpm-cyber-font);
  flex-shrink: 0;
  margin-left: 6px;
}

/* ── Element: Tag ("DEFAULT", right-aligned) ── */
.${MODEL_DROPDOWN_CLASSES.TAG} {
  font-size: 9px;
  letter-spacing: 1px;
  text-transform: uppercase;
  padding: 1px 6px;
  border: 1px solid var(--voxcpm-cyber-tag-border);
  background: var(--voxcpm-cyber-tag-bg);
  color: var(--voxcpm-cyber-tag-text);
  border-radius: 0;
  font-family: var(--voxcpm-cyber-font);
  line-height: 1.4;
  flex-shrink: 0;
  margin-left: auto;
}

/* ── Animation: Entrance ── */
/* NOTE: The dropdown uses transform:scale() for zoom matching.
   The entrance animation must preserve the scale value by using
   a CSS variable (--voxcpm-dropdown-scale, default 1) so that
   the scale is maintained throughout the animation. */
@keyframes voxcpm-dropdown-in {
  from {
    opacity: 0;
    transform: scale(var(--voxcpm-dropdown-scale, 1)) translateY(-4px);
  }
  to {
    opacity: 1;
    transform: scale(var(--voxcpm-dropdown-scale, 1)) translateY(0);
  }
}

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {
  .${MODEL_DROPDOWN_CLASSES.BLOCK} {
    animation: none;
  }
}
`.trim();
/** ID for the injected style element */
var STYLE_ELEMENT_ID = "voxcpm-model-dropdown-styles";
/** Whether styles have been injected */
var stylesInjected = false;
/**
* Inject model dropdown CSS styles into the document head.
*
* This function is idempotent — calling it multiple times has no effect
* after the first injection. Styles are injected once per page load.
*
* Requires design tokens to be injected first (via `injectDesignTokens()`).
*/
function injectModelDropdownStyles() {
	if (stylesInjected) return;
	if (document.getElementById(STYLE_ELEMENT_ID)) {
		stylesInjected = true;
		return;
	}
	const style = document.createElement("style");
	style.id = STYLE_ELEMENT_ID;
	style.textContent = MODEL_DROPDOWN_STYLES;
	document.head.appendChild(style);
	stylesInjected = true;
}
//#endregion
//#region styles/index.ts
/**
* VoxCPM Styles — Central Injection Entry Point
*
* Provides a single `injectAllStyles()` function that injects all VoxCPM
* CSS styles in the correct order:
*
* 1. Design tokens (`:root` custom properties) — must be first
* 2. Component styles (model selector, dialog, path indicator)
*
* This module ensures styles are injected exactly once per page load,
* regardless of how many times `injectAllStyles()` is called.
*
* @module voxcpm/styles
*/
/** Whether all styles have been injected */
var allStylesInjected = false;
/**
* Inject all VoxCPM CSS styles into the document head.
*
* Injection order matters:
* 1. **Design tokens** — `:root` custom properties that other styles reference
* 2. **Component styles** — BEM-block styles that use the design tokens
*
* This function is idempotent — calling it multiple times has no effect
* after the first successful injection.
*
* Should be called once during extension initialization (e.g., `init()` hook)
* before any widgets or dialogs are created.
*/
function injectAllStyles() {
	if (allStylesInjected) return;
	injectDesignTokens();
	injectModelSelectorStyles();
	injectModelDropdownStyles();
	allStylesInjected = true;
}
//#endregion
//#region core/session.ts
/**
* VoxCPM Session State Management
* 
* Provides utilities for managing session-scoped state using sessionStorage.
* Session state persists until the browser tab/window is closed.
*/
/**
* Get a value from session storage
* 
* @param key - Storage key
* @returns Stored value or null if not found
*/
function getSessionItem(key) {
	try {
		return sessionStorage.getItem(key);
	} catch {
		return null;
	}
}
/**
* Set a value in session storage
* 
* @param key - Storage key
* @param value - Value to store
* @returns True if successful, false if storage not available
*/
function setSessionItem(key, value) {
	try {
		sessionStorage.setItem(key, value);
		return true;
	} catch {
		return false;
	}
}
/**
* Remove a value from session storage
* 
* @param key - Storage key
* @returns True if successful, false if storage not available
*/
function removeSessionItem(key) {
	try {
		sessionStorage.removeItem(key);
		return true;
	} catch {
		return false;
	}
}
/**
* Check if a session flag is set
* 
* @param key - Storage key for the flag
* @returns True if flag is set
*/
function isSessionFlagSet$1(key) {
	return getSessionItem(key) !== null;
}
/**
* Set a session flag
* 
* @param key - Storage key for the flag
* @returns True if successful
*/
function setSessionFlag$1(key) {
	return setSessionItem(key, "true");
}
/**
* Clear a session flag
* 
* @param key - Storage key for the flag
* @returns True if successful
*/
function clearSessionFlag(key) {
	return removeSessionItem(key);
}
/**
* Get and parse a JSON value from session storage
* 
* @param key - Storage key
* @returns Parsed value or null if not found/invalid
*/
function getSessionJSON(key) {
	const value = getSessionItem(key);
	if (value === null) return null;
	try {
		return JSON.parse(value);
	} catch {
		return null;
	}
}
/**
* Stringify and set a JSON value in session storage
* 
* @param key - Storage key
* @param value - Value to store
* @returns True if successful
*/
function setSessionJSON(key, value) {
	try {
		return setSessionItem(key, JSON.stringify(value));
	} catch {
		return false;
	}
}
//#endregion
//#region notifications/NotificationManager.ts
/**
* Notification Manager
* 
* Handles notification display with session-based deduplication.
* Ensures notifications are shown only once per browser session.
*/
/**
* Manages notification display with session-based deduplication
*/
var NotificationManager = class {
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
			return {
				shown: false,
				reason: "already_shown"
			};
		}
		try {
			app$1.extensionManager.toast.add({
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
			return {
				shown: false,
				reason: "toast_unavailable"
			};
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
			app$1.extensionManager.toast.add({
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
			return {
				shown: false,
				reason: "toast_unavailable"
			};
		}
	}
	/**
	* Reset notification state (for testing)
	*/
	reset() {
		clearSessionFlag(this.sessionKey);
	}
	/**
	* Check if notification was already shown this session
	*/
	wasShown() {
		return isSessionFlagSet$1(this.sessionKey);
	}
};
/**
* Default notification manager instance for missing dependency warnings
*/
var notificationManager = new NotificationManager({
	sessionKey: "voxcpm.normalization_notification_shown",
	defaultLife: 1e4
});
//#endregion
//#region settings/SettingsManager.ts
/**
* Settings Manager
*
* Manages VoxCPM user settings with ComfyUI integration.
* Uses both session storage and ComfyUI's settings system.
*/
/** Session storage key for settings */
var SETTINGS_SESSION_KEY = "voxcpm.settings";
/**
* Manages VoxCPM user settings
*/
var SettingsManager = class {
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
		if (this.settings) return this.settings;
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
		return this.getSettings()?.first_run ?? true;
	}
	/**
	* Check if using custom path
	*/
	isUsingCustomPath() {
		return this.getSettings()?.use_custom_path ?? false;
	}
	/**
	* Get effective model path
	*/
	getEffectivePath() {
		return this.getSettings()?.effective_path ?? null;
	}
	/**
	* Update settings via server
	*/
	async updateSettings(updates) {
		try {
			for (const [key, value] of Object.entries(updates)) await app$1.api.storeSetting(`voxcpm.${key}`, value);
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
};
/**
* Default settings manager instance
*/
var settingsManager = new SettingsManager();
//#endregion
//#region dialogs/index.ts
/**
* Dialog Components Module Exports
*
* Exports all Vue dialog components.
*/
/** Active dialog instance */
var activeDialog = null;
/**
* Show the welcome dialog
*
* @returns Promise that resolves with 'default' or 'custom' selection
*/
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
				const continueBtn = overlay.querySelector("#voxcpm-continue");
				if (continueBtn) continueBtn.disabled = false;
			});
		});
		overlay.querySelector("#voxcpm-cancel")?.addEventListener("click", () => {
			closeActiveDialog();
			resolve(null);
		});
		overlay.querySelector("#voxcpm-continue")?.addEventListener("click", () => {
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
/**
* Show the path selection dialog
*
* @param initialPath Optional initial path value
* @returns Promise that resolves with the selected path or null
*/
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
				const { app } = await import("../scripts/app.js");
				const result = await (await app.api.fetchApi(`/voxcpm/validate_path?path=${encodeURIComponent(path)}`)).json();
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
		overlay.querySelector("#voxcpm-browse")?.addEventListener("click", () => {
			alert("Please enter the path manually. Native folder picker requires server support.");
		});
		overlay.querySelector("#voxcpm-cancel")?.addEventListener("click", () => {
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
		if (initialPath) validatePath(initialPath);
	});
}
/**
* Close the active dialog
*/
function closeActiveDialog() {
	if (activeDialog) {
		activeDialog.remove();
		activeDialog = null;
	}
}
/**
* Inject dialog styles if not already present
*/
function injectDialogStyles() {
	const styleId = "voxcpm-dialog-styles";
	if (document.getElementById(styleId)) return;
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
//#endregion
//#region widgets/modelIcons.ts
/**
* Model Icons — SVG Icon Constants
*
* Inline SVG strings for model status icons used in the model selector
* dropdown and display widget. All icons use `currentColor` for stroke/fill
* so CSS can control the color via the parent element's `color` property.
*
* @module voxcpm/widgets/model-icons
*/
/**
* SVG icon strings for the model selector UI.
*
* Each icon is a self-contained `<svg>` element string that can be
* assigned to `element.innerHTML`. Icons are sized 16x16px by default
* with a 24x24 viewBox for scalability.
*
* Color is controlled via CSS `color` on the parent element, since
* all icons use `stroke="currentColor"` or `fill="currentColor"`.
*/
var MODEL_ICONS = {
	/** Cloud-download icon — model needs to be downloaded from remote */
	CLOUD: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><polyline points="12 12 12 22"/><path d="m8 18 4 4 4-4"/></svg>`,
	/** Checkmark icon — model exists locally on disk */
	CHECK: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
	/** Shield icon — dropdown header decoration (cyber-security aesthetic) */
	SHIELD: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`,
	/** Folder icon — model display widget (replaces emoji 📁) */
	FOLDER: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`,
	/** Folder-open icon — custom path model (replaces emoji 📂) */
	FOLDER_OPEN: `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 19a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4l2 2h4a2 2 0 0 1 2 2v1"/><path d="M6 12h14l-2.5 7H8.5L6 12z"/></svg>`
};
//#endregion
//#region widgets/ModelDropdown.ts
/**
* Model Dropdown — Custom Cyber-Security Themed Dropdown Panel
*
* Replaces `LiteGraph.ContextMenu` with a custom dropdown that supports:
* - Rich model items with status icons (cloud/check), DEFAULT tags, and metadata
* - Keyboard navigation (ArrowUp/Down, Enter, Escape)
* - Click-outside dismissal
* - Viewport-aware positioning (flips above anchor if near bottom)
* - Cyber-security aesthetic (monospace font, neon green accent, scanline overlay)
*
* @module voxcpm/widgets/model-dropdown
*/
/** Cached model info from the backend API (null = not fetched yet) */
var cachedModelInfo = null;
/**
* Fetch model info from the backend API, with session-level caching.
*
* On first call, fetches from `GET /voxcpm/model_info`. On subsequent
* calls, returns the cached data. Cache is invalidated on WebSocket
* reconnect (server restart) or by calling `invalidateModelInfoCache()`.
*
* @param fetchApi - The `api.fetchApi` function from ComfyUI's API module
* @returns Array of model info items (may be empty on error)
*/
async function fetchModelInfo(fetchApi) {
	if (cachedModelInfo !== null) return cachedModelInfo;
	try {
		const response = await fetchApi("/voxcpm/model_info");
		if (response.ok) {
			cachedModelInfo = (await response.json()).models || [];
			return cachedModelInfo;
		}
	} catch (error) {
		console.warn("[VoxCPM] Failed to fetch model info:", error);
	}
	return [];
}
/**
* Invalidate the model info cache.
*
* Call this when the server reconnects or when model directories
* may have changed (e.g., after a model download).
*/
function invalidateModelInfoCache() {
	cachedModelInfo = null;
}
/**
* Custom dropdown panel for model selection with cyber-security styling.
*
* Usage:
* ```typescript
* const dropdown = new ModelDropdown(modelInfo, anchorRect, scale);
* const selectedModel = await dropdown.show();
* if (selectedModel) { ... }
* ```
*
* The dropdown is positioned below the anchor element. If there isn't
* enough viewport space below, it flips above the anchor. Keyboard
* navigation (ArrowUp/Down, Enter, Escape) and click-outside dismissal
* are built in.
*/
var ModelDropdown = class ModelDropdown {
	constructor(modelInfo, anchorRect, scale, widgetElement) {
		this.selectedIndex = -1;
		this.resolveSelection = null;
		this.cleanups = [];
		this.isOpen = false;
		this.modelInfo = modelInfo;
		this.anchorRect = anchorRect;
		this.scale = scale;
		this.widgetElement = widgetElement || null;
		this.panel = document.createElement("div");
		this.panel.className = MODEL_DROPDOWN_CLASSES.BLOCK;
		this.panel.setAttribute("role", "listbox");
		this.panel.setAttribute("aria-label", "Select model");
		this.header = document.createElement("div");
		this.header.className = MODEL_DROPDOWN_CLASSES.HEADER;
		this.header.innerHTML = `${MODEL_ICONS.SHIELD} <span>${UI.MODEL_DROPDOWN.HEADER_TEXT}</span>`;
		this.list = document.createElement("div");
		this.list.className = MODEL_DROPDOWN_CLASSES.LIST;
		this.panel.appendChild(this.header);
		this.panel.appendChild(this.list);
		this.renderItems();
	}
	/**
	* Show the dropdown and return a promise that resolves with the selection.
	*
	* If another dropdown is already open, it is closed first (only one
	* dropdown can be open at a time).
	*
	* @returns The selected model name, or `null` if dismissed without selection
	*/
	show() {
		return new Promise((resolve) => {
			this.resolveSelection = resolve;
			if (ModelDropdown.currentDropdown && ModelDropdown.currentDropdown !== this) ModelDropdown.currentDropdown.close(null);
			this.positionPanel();
			document.body.appendChild(this.panel);
			this.isOpen = true;
			ModelDropdown.currentDropdown = this;
			this.setupClickOutside();
			this.setupKeyboardNav();
			this.setupZoomChangeListener();
			this.panel.tabIndex = -1;
			this.panel.focus();
		});
	}
	/**
	* Close the dropdown and clean up all event listeners.
	*
	* @param selectedModel - The model name to resolve with, or null
	*/
	close(selectedModel = null) {
		if (!this.isOpen) return;
		this.isOpen = false;
		if (ModelDropdown.currentDropdown === this) ModelDropdown.currentDropdown = null;
		for (const cleanup of this.cleanups) cleanup();
		this.cleanups = [];
		if (this.panel.parentElement) this.panel.parentElement.removeChild(this.panel);
		if (this.resolveSelection) {
			this.resolveSelection(selectedModel);
			this.resolveSelection = null;
		}
	}
	/**
	* Render all model items into the list element.
	*/
	renderItems() {
		this.list.innerHTML = "";
		for (let i = 0; i < this.modelInfo.length; i++) {
			const model = this.modelInfo[i];
			const item = this.createElementItem(model, i);
			this.list.appendChild(item);
		}
	}
	/**
	* Create a single model item element.
	*/
	createElementItem(model, index) {
		const item = document.createElement("div");
		item.className = MODEL_DROPDOWN_CLASSES.ITEM;
		item.setAttribute("role", "option");
		item.setAttribute("data-index", String(index));
		item.setAttribute("data-model", model.name);
		const icon = document.createElement("span");
		const iconModifier = model.is_downloaded ? MODEL_DROPDOWN_CLASSES.ICON_CHECK : MODEL_DROPDOWN_CLASSES.ICON_CLOUD;
		icon.className = `${MODEL_DROPDOWN_CLASSES.ICON} ${iconModifier}`;
		icon.innerHTML = model.is_downloaded ? MODEL_ICONS.CHECK : MODEL_ICONS.CLOUD;
		item.appendChild(icon);
		const name = document.createElement("span");
		name.className = MODEL_DROPDOWN_CLASSES.NAME;
		name.textContent = model.name;
		item.appendChild(name);
		if (model.size_gb > 0) {
			const meta = document.createElement("span");
			meta.className = MODEL_DROPDOWN_CLASSES.META;
			meta.textContent = `${model.size_gb} GB`;
			item.appendChild(meta);
		}
		if (model.type === "official") {
			const tag = document.createElement("span");
			tag.className = MODEL_DROPDOWN_CLASSES.TAG;
			tag.textContent = "DEFAULT";
			item.appendChild(tag);
		}
		item.addEventListener("click", (e) => {
			e.stopPropagation();
			this.close(model.name);
		});
		item.addEventListener("mouseenter", () => {
			this.setSelectedIndex(index);
		});
		return item;
	}
	/**
	* Position the panel relative to the anchor element.
	*
	* Positions below the anchor by default. If there isn't enough
	* viewport space below, flips above the anchor. Also adjusts
	* horizontal position if the panel would overflow the viewport right edge.
	*
	* When the canvas is zoomed (scale > 1), the dropdown is scaled up
	* to match the visual size of the widget. The CSS transform origin
	* is set to "top left" so that the top-left corner stays at the
	* calculated position. The visual size of the panel is
	* `estimatedSize * scale`, so all viewport boundary checks use
	* the scaled dimensions.
	*/
	positionPanel() {
		const scale = this.scale;
		const gap = UI.MODEL_DROPDOWN.ANCHOR_GAP;
		const viewportHeight = window.innerHeight;
		const viewportWidth = window.innerWidth;
		const itemHeight = UI.MODEL_DROPDOWN.ITEM_HEIGHT;
		const headerHeight = 32;
		const maxVisibleItems = UI.MODEL_DROPDOWN.MAX_VISIBLE_ITEMS;
		const scaledHeight = (headerHeight + Math.min(this.modelInfo.length, maxVisibleItems) * itemHeight) * scale;
		const scaledWidth = 300 * scale;
		let top;
		const spaceBelow = viewportHeight - this.anchorRect.bottom;
		const spaceAbove = this.anchorRect.top;
		if (spaceBelow >= scaledHeight + gap) top = this.anchorRect.bottom + gap;
		else if (spaceAbove >= scaledHeight + gap) top = this.anchorRect.top - scaledHeight - gap;
		else top = this.anchorRect.bottom + gap;
		let left = this.anchorRect.left;
		if (left + scaledWidth > viewportWidth) left = viewportWidth - scaledWidth - 8;
		left = Math.max(8, left);
		this.panel.style.position = "fixed";
		this.panel.style.top = `${top}px`;
		this.panel.style.left = `${left}px`;
		this.panel.style.zIndex = String(UI.MODEL_DROPDOWN.Z_INDEX);
		this.panel.style.transformOrigin = "top left";
		this.panel.style.transform = `scale(${scale})`;
		this.panel.style.setProperty("--voxcpm-dropdown-scale", String(scale));
	}
	/**
	* Set up click-outside listener to close the dropdown.
	*/
	setupClickOutside() {
		const handler = (e) => {
			if (!this.isOpen) return;
			const target = e.target;
			if (this.panel.contains(target)) return;
			if (this.widgetElement && this.widgetElement.contains(target)) return;
			this.close(null);
		};
		document.addEventListener("pointerdown", handler, true);
		this.cleanups.push(() => {
			document.removeEventListener("pointerdown", handler, true);
		});
	}
	/**
	* Set up keyboard navigation.
	*
	* - ArrowDown: Move to next item (wrap to first)
	* - ArrowUp: Move to previous item (wrap to last)
	* - Enter: Confirm selection
	* - Escape: Close without selection
	* - Tab: Close without selection
	*/
	setupKeyboardNav() {
		const handler = (e) => {
			if (!this.isOpen) return;
			switch (e.key) {
				case "ArrowDown": {
					e.preventDefault();
					e.stopPropagation();
					const next = this.selectedIndex < this.modelInfo.length - 1 ? this.selectedIndex + 1 : 0;
					this.setSelectedIndex(next);
					break;
				}
				case "ArrowUp": {
					e.preventDefault();
					e.stopPropagation();
					const prev = this.selectedIndex > 0 ? this.selectedIndex - 1 : this.modelInfo.length - 1;
					this.setSelectedIndex(prev);
					break;
				}
				case "Enter":
					e.preventDefault();
					e.stopPropagation();
					if (this.selectedIndex >= 0 && this.selectedIndex < this.modelInfo.length) this.close(this.modelInfo[this.selectedIndex].name);
					else this.close(null);
					break;
				case "Escape":
					e.preventDefault();
					e.stopPropagation();
					this.close(null);
					break;
				case "Tab":
					e.preventDefault();
					this.close(null);
					break;
			}
		};
		document.addEventListener("keydown", handler, true);
		this.cleanups.push(() => {
			document.removeEventListener("keydown", handler, true);
		});
	}
	/**
	* Set up a listener for canvas zoom/pan changes.
	*
	* When the user zooms or pans the canvas while the dropdown is open,
	* the dropdown position becomes stale (it's positioned in fixed screen
	* coordinates but the widget has moved). Rather than trying to
	* reposition, we simply close the dropdown — this matches the behavior
	* of native context menus in most UI frameworks.
	*
	* Uses `canvas.ds.onChanged` — the callback that LiteGraph's
	* DragAndScale manager calls whenever scale or offset changes.
	* We chain onto the existing handler (if any) to avoid breaking
	* ComfyUI's internal zoom tracking (e.g., canvasStore scale sync).
	*/
	setupZoomChangeListener() {
		const canvas = app$1.canvas;
		if (!canvas?.ds) return;
		const originalOnChanged = canvas.ds.onChanged;
		canvas.ds.onChanged = (scale, offset) => {
			if (this.isOpen) this.close(null);
			originalOnChanged?.(scale, offset);
		};
		this.cleanups.push(() => {
			if (canvas.ds) canvas.ds.onChanged = originalOnChanged;
		});
	}
	/**
	* Update the selected index and apply visual highlight.
	*/
	setSelectedIndex(index) {
		const items = this.list.querySelectorAll(`.${MODEL_DROPDOWN_CLASSES.ITEM}`);
		items.forEach((item) => {
			item.classList.remove(MODEL_DROPDOWN_CLASSES.ITEM_SELECTED);
		});
		this.selectedIndex = index;
		if (index >= 0 && index < items.length) {
			const selectedItem = items[index];
			selectedItem.classList.add(MODEL_DROPDOWN_CLASSES.ITEM_SELECTED);
			selectedItem.scrollIntoView({ block: "nearest" });
		}
	}
};
ModelDropdown.currentDropdown = null;
//#endregion
//#region widgets/ModelSelectorWidget.ts
/**
* Model Selector Widget
*
* Custom DOM widget that replaces the default model_name combo widget
* with a professional-looking selector featuring:
* - Cyber-themed model display with dropdown arrow
* - Click-to-open custom dropdown (ModelDropdown) with icons and tags
* - Browse button for custom model path selection
* - Custom path indicator when a non-default path is active
* - Full data flow sync with the hidden model_name widget
*
* @module voxcpm/widgets/model-selector
*/
/**
* Stores DOM element references for model selector widgets keyed by node ID.
*
* Using a WeakMap instead of node.properties avoids circular reference errors
* during JSON.stringify (LiteGraph serialization). DOM elements and widget
* objects have circular references back to the node, which would cause
* "Maximum call stack size exceeded" if stored in serializable properties.
*/
var selectorRefsMap = /* @__PURE__ */ new Map();
/**
* Create and attach a custom model selector DOM widget to a VoxCPM node.
*
* This function:
* 1. Injects CSS styles (idempotent)
* 2. Hides the default model_name combo widget
* 3. Creates a DOM widget with model display + browse button
* 4. Sets up event handlers for dropdown and browse
* 5. Syncs state between DOM widget and hidden combo widget
*
* @param node - The VoxCPM TTS node to add the widget to
*/
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
	const container = document.createElement("div");
	container.className = MODEL_SELECTOR_CLASSES.BLOCK;
	const modelDisplay = document.createElement("div");
	modelDisplay.className = MODEL_SELECTOR_CLASSES.DISPLAY;
	modelDisplay.title = "Click to select a model";
	modelDisplay.tabIndex = 0;
	modelDisplay.setAttribute("role", "combobox");
	modelDisplay.setAttribute("aria-expanded", "false");
	modelDisplay.setAttribute("aria-haspopup", "listbox");
	const folderIcon = document.createElement("span");
	folderIcon.className = MODEL_SELECTOR_CLASSES.ICON;
	folderIcon.innerHTML = MODEL_ICONS.FOLDER;
	const modelText = document.createElement("span");
	modelText.className = MODEL_SELECTOR_CLASSES.TEXT;
	modelText.textContent = modelWidget.value || UI.MODEL_SELECTOR.PLACEHOLDER;
	const arrow = document.createElement("span");
	arrow.className = MODEL_SELECTOR_CLASSES.ARROW;
	arrow.textContent = UI.MODEL_SELECTOR.ARROW;
	const browseBtn = document.createElement("button");
	browseBtn.className = MODEL_SELECTOR_CLASSES.BROWSE;
	browseBtn.innerHTML = MODEL_ICONS.FOLDER_OPEN;
	browseBtn.title = "Browse for custom model directory";
	browseBtn.setAttribute("role", "button");
	const pathIndicator = document.createElement("div");
	pathIndicator.className = MODEL_SELECTOR_CLASSES.PATH;
	const savedCustomPath = node.properties?.["custom_model_path"];
	if (savedCustomPath) {
		folderIcon.innerHTML = MODEL_ICONS.FOLDER_OPEN;
		pathIndicator.textContent = savedCustomPath;
		pathIndicator.classList.add(MODEL_SELECTOR_CLASSES.PATH_VISIBLE);
	}
	modelDisplay.append(folderIcon, modelText, arrow);
	const row = document.createElement("div");
	row.className = MODEL_SELECTOR_CLASSES.ROW;
	row.append(modelDisplay, browseBtn);
	container.append(row, pathIndicator);
	container.addEventListener("mouseup", (e) => e.stopPropagation());
	container.addEventListener("click", (e) => e.stopPropagation());
	modelDisplay.addEventListener("click", (e) => {
		e.preventDefault();
		e.stopPropagation();
		if (isDropdownOpen()) {
			closeCurrentDropdown();
			return;
		}
		openModelDropdown(e, modelWidget, modelText, folderIcon, pathIndicator, node, modelDisplay, arrow);
	});
	modelDisplay.addEventListener("keydown", (e) => {
		if (e.key === "Enter" || e.key === " ") {
			e.preventDefault();
			e.stopPropagation();
			if (isDropdownOpen()) {
				closeCurrentDropdown();
				return;
			}
			openModelDropdown(e, modelWidget, modelText, folderIcon, pathIndicator, node, modelDisplay, arrow);
		}
	});
	browseBtn.addEventListener("click", async (e) => {
		e.preventDefault();
		e.stopPropagation();
		await handleBrowseClick(node, modelWidget, modelText, folderIcon, pathIndicator);
	});
	const domWidget = node.addDOMWidget("model_selector", "custom", container, {
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
			const grandparent = rowParent.parentElement;
			if (grandparent && grandparent !== document.body) {
				grandparent.classList.add(MODEL_SELECTOR_CLASSES.BLOCK);
				grandparent.setAttribute("node-id", String(node.id));
				logger.debug("Added", MODEL_SELECTOR_CLASSES.BLOCK, "class + node-id to WidgetDOM parent div for node:", node.id);
			} else logger.debug("No WidgetDOM grandparent found (LiteGraph mode) for node:", node.id);
			parentObserver.disconnect();
		}
	});
	parentObserver.observe(document.body, {
		childList: true,
		subtree: true
	});
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
	const modelInput = node.inputs?.find((inp) => inp.name === "model_name");
	if (modelInput) {
		modelInput.widget = {
			name: "model_selector",
			_originalWidget: modelInput.widget?.name
		};
		logger.debug("Bound model_name input to model_selector widget for node:", node.id);
	}
	selectorRefsMap.set(node.id, {
		modelText,
		folderIcon,
		pathIndicator,
		modelWidget
	});
	logger.debug("Created model selector widget for node:", node.id);
}
/**
* Hide the default LiteGraph combo widget by converting it to an invisible
* "converted-widget" type. The widget remains in the node's widget list
* for data flow (serialization, API calls) but is not rendered.
*
* @param widget - The combo widget to hide
*/
function hideDefaultComboWidget(widget) {
	widget.type = "converted-widget";
	widget.computeSize = () => [0, -4];
	widget.draw = () => {};
	if (!widget.options) widget.options = {};
	widget.options.canvasOnly = true;
	logger.debug("Hidden default combo widget:", widget.name);
}
/**
* Check if a ModelDropdown is currently open.
*/
function isDropdownOpen() {
	return ModelDropdown.currentDropdown !== null;
}
/**
* Close the currently open ModelDropdown (if any).
*/
function closeCurrentDropdown() {
	if (ModelDropdown.currentDropdown) ModelDropdown.currentDropdown.close(null);
}
/**
* Open a custom ModelDropdown for model selection.
*
* Fetches model metadata from the backend API (with session-level caching),
* then shows a cyber-themed dropdown panel with status icons, DEFAULT tags,
* and size metadata. Falls back to widget values if the API call fails.
*
* @param event - The triggering mouse or keyboard event
* @param modelWidget - The hidden model_name combo widget
* @param modelText - The DOM element displaying the model name
* @param folderIcon - The folder icon DOM element
* @param pathIndicator - The custom path indicator DOM element
* @param node - The parent node
*/
async function openModelDropdown(event, modelWidget, modelText, folderIcon, pathIndicator, node, displayEl, arrowEl) {
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
	let modelInfo = [];
	try {
		modelInfo = await fetchModelInfo(api.fetchApi.bind(api));
	} catch (error) {
		logger.error("Failed to fetch model info:", error);
	}
	if (modelInfo.length === 0) modelInfo = models.map((name) => ({
		name,
		type: "local",
		architecture: "unknown",
		sample_rate: 0,
		size_gb: 0,
		is_downloaded: true
	}));
	const anchorRect = displayEl.getBoundingClientRect();
	const scale = app$1.canvas.ds?.scale || 1;
	arrowEl.classList.add(MODEL_SELECTOR_CLASSES.ARROW_ACTIVE);
	displayEl.setAttribute("aria-expanded", "true");
	try {
		const selectedModel = await new ModelDropdown(modelInfo, anchorRect, scale, displayEl).show();
		if (selectedModel) {
			modelWidget.value = selectedModel;
			modelText.textContent = selectedModel;
			const selectedInfo = modelInfo.find((m) => m.name === selectedModel);
			if (selectedInfo) folderIcon.innerHTML = selectedInfo.is_downloaded ? MODEL_ICONS.CHECK : MODEL_ICONS.CLOUD;
			if (modelWidget.callback) modelWidget.callback(selectedModel);
			if (node.setDirtyCanvas) node.setDirtyCanvas(true);
			logger.debug("Model selected:", selectedModel, "for node:", node.id);
		}
	} catch (error) {
		logger.error("Failed to open model dropdown:", error);
	} finally {
		arrowEl.classList.remove(MODEL_SELECTOR_CLASSES.ARROW_ACTIVE);
		displayEl.setAttribute("aria-expanded", "false");
	}
}
/**
* Handle the browse button click.
*
* Opens the path selection dialog, fetches models from the selected path,
* updates the model_name widget options, and refreshes the DOM display.
*
* @param node - The parent node
* @param modelWidget - The hidden model_name combo widget
* @param modelText - The DOM element displaying the model name
* @param folderIcon - The folder icon DOM element
* @param pathIndicator - The custom path indicator DOM element
*/
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
			folderIcon.innerHTML = MODEL_ICONS.FOLDER_OPEN;
			pathIndicator.textContent = path;
			pathIndicator.classList.add(MODEL_SELECTOR_CLASSES.PATH_VISIBLE);
			if (data.models.length > 0) modelText.textContent = modelWidget.value;
			else modelText.textContent = "No models found";
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
/**
* Update the model_name combo widget's options list.
*
* This updates the hidden widget's values array and sets the current
* value to the first model if the current selection is not in the new list.
*
* @param modelWidget - The hidden model_name combo widget
* @param models - The new list of available model names
* @param node - The parent node (for dirty canvas notification)
*/
function updateModelOptions(modelWidget, models, node) {
	if (modelWidget.options) modelWidget.options.values = models;
	else modelWidget.options = { values: models };
	if (models.length > 0 && !models.includes(modelWidget.value)) {
		modelWidget.value = models[0];
		if (modelWidget.callback) modelWidget.callback(models[0]);
	}
	if (node.setDirtyCanvas) node.setDirtyCanvas(true);
	logger.debug("Updated model options:", models);
}
/**
* Update the model selector DOM widget display from an external source.
*
* Use this when the hidden model_name widget value changes externally
* (e.g., workflow load, API callback) and the DOM display needs to be
* synchronized.
*
* @param node - The node whose selector should be updated
* @param value - The new model name to display (optional, reads from widget if not provided)
*/
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
		refs.folderIcon.innerHTML = MODEL_ICONS.FOLDER_OPEN;
		refs.pathIndicator.textContent = customPath;
		refs.pathIndicator.classList.add(MODEL_SELECTOR_CLASSES.PATH_VISIBLE);
	} else {
		refs.folderIcon.innerHTML = MODEL_ICONS.FOLDER;
		refs.pathIndicator.classList.remove(MODEL_SELECTOR_CLASSES.PATH_VISIBLE);
	}
	logger.debug("Synced model selector display for node:", node.id, "value:", displayValue);
}
//#endregion
//#region extension.ts
/**
* VoxCPM Frontend Extension
*
* Main extension file that integrates all VoxCPM frontend functionality.
* This file is written in TypeScript and compiled to extension.js for ComfyUI.
*
* Features:
* - Session-based notification deduplication
* - Event handling for server-to-client communication
* - Settings management with ComfyUI integration
* - First-run dialog for model path configuration
* - Widget state management (normalize_text widget)
* - Per-node custom model path selection
*
* Build: npm run build:extension
*/
/**
* Check if a session flag is set
* @param key - Storage key to check
* @returns The stored value or null if not set
*/
function isSessionFlagSet(key) {
	try {
		const stored = sessionStorage.getItem(key);
		if (stored !== null) return stored;
	} catch {}
	return null;
}
/**
* Set a session flag
* @param key - Storage key to set
* @param value - Value to store
*/
function setSessionFlag(key, value) {
	try {
		sessionStorage.setItem(key, value);
	} catch (e) {
		logger.warn(`Failed to set session flag ${key}:`, e);
	}
}
/**
* Get stored normalization config from localStorage
* @returns The stored config value or null if not set
*/
function getStoredConfig() {
	try {
		const stored = localStorage.getItem(STORAGE_KEYS.NORMALIZATION_AVAILABLE);
		if (stored !== null) return stored === "true";
	} catch {}
	return null;
}
/**
* Store config to localStorage
* @param value - normalization_available value
*/
function storeConfig(value) {
	try {
		localStorage.setItem(STORAGE_KEYS.NORMALIZATION_AVAILABLE, String(value));
		logger.debug("Config stored to localStorage:", value);
	} catch (e) {
		logger.warn("Failed to store config to localStorage:", e);
	}
}
/**
* Manages widget state based on server configuration
*/
var widgetManager = {
	normalizationAvailable: true,
	configReceived: false,
	lastConfigValue: null,
	initialized: false,
	/**
	* Initialize from localStorage
	*/
	init() {
		if (this.initialized) return;
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
		if (!this.initialized) this.init();
		const normalizeWidget = node.widgets?.find((w) => w.name === "normalize_text");
		if (!normalizeWidget) return false;
		if (this.normalizationAvailable) {
			if (normalizeWidget.disabled) {
				logger.debug("Re-enabling normalize_text widget for node:", node.id);
				normalizeWidget.disabled = false;
				normalizeWidget.value = true;
				if (normalizeWidget.options) normalizeWidget.options.tooltip = void 0;
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
		if (!this.initialized) this.init();
		const graph = app.graph || app.rootGraph;
		if (graph && graph._nodes) {
			for (const node of graph._nodes) if (node.comfyClass === DEFAULTS.NODE_CLASS) this.updateNormalizationWidget(node);
		}
	}
};
widgetManager.init();
/**
* VoxCPM Frontend Extension
*
* Provides:
* - Session-based notification deduplication
* - Event handling for server-to-client communication
* - Settings management with ComfyUI integration
* - First-run dialog for model path configuration
* - Widget state management
*/
var VoxCPMExtension = {
	/** Extension name for ComfyUI registration */
	name: DEFAULTS.EXTENSION_NAME,
	/**
	* Register settings with ComfyUI
	*
	* These settings appear in ComfyUI's settings panel.
	*/
	settings: [{
		id: "voxcpm.use_custom_path",
		category: ["VoxCPM", "Model Path"],
		name: "Use Custom Model Path",
		type: "boolean",
		defaultValue: false,
		tooltip: "Enable to use a custom directory for VoxCPM models"
	}, {
		id: "voxcpm.custom_model_path",
		category: ["VoxCPM", "Model Path"],
		name: "Custom Model Path",
		type: "text",
		defaultValue: "",
		tooltip: "Path to custom VoxCPM models directory"
	}],
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
		app.api.addEventListener(EVENT_NAMES.STATUS, ((event) => {
			this.handleStatusEvent(event.detail);
		}));
		app.api.addEventListener(EVENT_NAMES.CONFIG, ((event) => {
			this.handleConfigEvent(event.detail);
		}));
		app.api.addEventListener("reconnected", () => {
			logger.debug("WebSocket reconnected, waiting for config...");
			widgetManager.configReceived = false;
			invalidateModelInfoCache();
			widgetManager.lastConfigValue = null;
		});
		app.api.addEventListener("status", () => {
			if (!widgetManager.configReceived) logger.debug("Status received, waiting for config...");
		});
		logger.log("Event listeners registered");
	},
	/**
	* Called after a node is created
	*/
	nodeCreated(node) {
		if (node.comfyClass !== DEFAULTS.NODE_CLASS) return;
		logger.debug("VoxCPM node created:", node.id);
		widgetManager.updateNormalizationWidget(node);
		createModelSelectorWidget(node);
	},
	/**
	* Called when a node is loaded from a saved workflow
	*/
	loadedGraphNode(node) {
		if (node.comfyClass !== DEFAULTS.NODE_CLASS) return;
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
			if (!(widgetManager.lastConfigValue !== detail.normalization_available) && widgetManager.configReceived) {
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
		} else logger.warn("No settings in config event");
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
		} else logger.log("User cancelled first run dialog");
	}
};
app.registerExtension(VoxCPMExtension);
//#endregion
export { VoxCPMExtension, VoxCPMExtension as default };
