import { app as e } from "/scripts/app.js";
import { app as t } from "/scripts/app.js";
import { api as n } from "/scripts/api.js";

var r = {
	NOTIFICATION_SHOWN: "voxcpm.normalization_notification_shown",
	USER_PREFERENCES: "voxcpm.user_preferences",
	SETTINGS: "voxcpm.settings",
	NORMALIZATION_AVAILABLE: "voxcpm.normalization_available"
}, i = {
	STATUS: "voxcpm.status",
	MODEL_LOADED: "voxcpm.model_loaded",
	GENERATION_PROGRESS: "voxcpm.generation_progress",
	CONFIG: "voxcpm.config",
	SETTINGS_UPDATE: "voxcpm.settings_update",
	DOWNLOAD_PROGRESS: "voxcpm.download_progress"
}, a = {
	TOAST_LIFE: 1e4,
	LOG_PREFIX: "[VoxCPM]",
	EXTENSION_NAME: "voxcpm.frontend",
	NODE_CLASS: "VoxCPM_TTS"
}, o = new Set([
	"VoxCPM_TTS",
	"VoxCPM_VoiceCloning",
	"VoxCPM_AdvancedParams"
]), s = {
	SEVERITY: {
		SUCCESS: "success",
		INFO: "info",
		WARN: "warn",
		ERROR: "error"
	},
	ICONS: {
		VOLUME: "pi pi-volume-up",
		WARNING: "pi pi-exclamation-triangle",
		INFO: "pi pi-info-circle",
		ERROR: "pi pi-times-circle",
		SUCCESS: "pi pi-check-circle",
		DOWNLOAD: "pi pi-download",
		CANCEL: "pi pi-times"
	},
	MODEL_SELECTOR: {
		WIDGET_NAME: "model_selector",
		WIDGET_TYPE: "custom",
		MIN_HEIGHT: 44,
		DEFAULT_ICON: "📁",
		CUSTOM_ICON: "📂",
		ARROW: "▼",
		PLACEHOLDER: "Select model...",
		CSS_PREFIX: "voxcpm-model-"
	},
	MODEL_DROPDOWN: {
		BLOCK_CLASS: "voxcpm-model-dropdown",
		HEADER_TEXT: "SELECT MODEL",
		MAX_VISIBLE_ITEMS: 8,
		ITEM_HEIGHT: 36,
		ANCHOR_GAP: 4,
		ANIMATION_DURATION: 150,
		Z_INDEX: 1e3
	},
	DOWNLOAD_PROGRESS: {
		BLOCK_CLASS: "voxcpm-download-progress",
		WIDGET_NAME: "download_progress",
		WIDGET_TYPE: "custom",
		MIN_HEIGHT: 48,
		ANIMATION_DURATION: 200,
		CANCEL_LABEL: "Cancel",
		CANCEL_ENDPOINT: "/voxcpm/cancel_download",
		STATUS_ENDPOINT: "/voxcpm/download_status"
	}
}, c = {
	log: (...e) => {
		console.log(a.LOG_PREFIX, ...e);
	},
	info: (...e) => {
		console.info(a.LOG_PREFIX, ...e);
	},
	warn: (...e) => {
		console.warn(a.LOG_PREFIX, ...e);
	},
	error: (...e) => {
		console.error(a.LOG_PREFIX, ...e);
	},
	debug: (...e) => {
		typeof window < "u" && window.__VOXCPM_DEBUG__ && console.debug(a.LOG_PREFIX, "[DEBUG]", ...e);
	},
	group: (e) => {
		console.group(`${a.LOG_PREFIX} ${e}`);
	},
	groupEnd: () => {
		console.groupEnd();
	},
	table: (e) => {
		console.log(a.LOG_PREFIX), console.table(e);
	}
}, l = "\n\n:root {\n  \n  --voxcpm-space-2xs: 2px;\n  --voxcpm-space-xs: 4px;\n  --voxcpm-space-s: 6px;\n  --voxcpm-space-m: 8px;\n  --voxcpm-space-l: 12px;\n  --voxcpm-space-xl: 16px;\n  --voxcpm-space-2xl: 24px;\n\n  \n  --voxcpm-font-size-2xs: 10px;\n  --voxcpm-font-size-xs: 12px;\n  --voxcpm-font-size-s: 13px;\n  --voxcpm-font-size-m: 14px;\n  --voxcpm-font-size-l: 18px;\n  --voxcpm-font-size-xl: 24px;\n  --voxcpm-font-family: var(--comfy-font-family, Arial, sans-serif);\n  --voxcpm-font-family-mono: monospace;\n\n  \n  --voxcpm-bg-input: var(--comfy-input-bg, rgba(255, 255, 255, 0.05));\n  --voxcpm-bg-input-hover: var(--comfy-input-bg-hover, rgba(255, 255, 255, 0.08));\n  --voxcpm-bg-input-active: var(--comfy-input-bg-active, rgba(255, 255, 255, 0.1));\n  --voxcpm-bg-surface: var(--bg-color, #1e1e1e);\n  --voxcpm-bg-elevated: var(--input-bg, #2a2a2a);\n\n  \n  --voxcpm-border-color: var(--border-color, rgba(255, 255, 255, 0.15));\n  --voxcpm-border-color-hover: var(--comfy-input-border-hover, rgba(255, 255, 255, 0.3));\n  --voxcpm-border-radius-xs: 4px;\n  --voxcpm-border-radius-s: 6px;\n  --voxcpm-border-radius-m: 8px;\n\n  \n  --voxcpm-text-primary: var(--fg-color, #ddd);\n  --voxcpm-text-secondary: var(--fg-color, rgba(255, 255, 255, 0.7));\n  --voxcpm-text-muted: var(--fg-color, rgba(255, 255, 255, 0.4));\n  --voxcpm-text-on-primary: #fff;\n\n  \n  --voxcpm-accent: var(--primary-color, #4a9eff);\n  --voxcpm-accent-hover: var(--primary-color-hover, #3a8eef);\n\n  \n  --voxcpm-color-success: var(--success-color, #4caf50);\n  --voxcpm-color-error: var(--error-color, #f44336);\n\n  \n  --voxcpm-duration-fast: 150ms;\n  --voxcpm-duration-normal: 200ms;\n  --voxcpm-easing: ease;\n\n  \n  --voxcpm-widget-height: 44px;\n  --voxcpm-input-height: 26px;\n  --voxcpm-browse-width: 30px;\n\n  \n  --voxcpm-focus-ring-color: var(--voxcpm-accent);\n  --voxcpm-focus-ring-width: 2px;\n  --voxcpm-focus-ring-offset: 1px;\n\n  \n  --voxcpm-cyber-font: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;\n  --voxcpm-cyber-accent: #00ff9d;\n  --voxcpm-cyber-accent-dim: rgba(0, 255, 157, 0.15);\n  --voxcpm-cyber-border: rgba(0, 255, 157, 0.3);\n  --voxcpm-cyber-glow: 0 0 8px rgba(0, 255, 157, 0.4);\n  --voxcpm-cyber-text: #e0e0e0;\n  --voxcpm-cyber-text-dim: rgba(224, 224, 224, 0.5);\n  --voxcpm-cyber-bg: rgba(10, 15, 20, 0.95);\n  --voxcpm-cyber-bg-hover: rgba(0, 255, 157, 0.08);\n  --voxcpm-cyber-bg-active: rgba(0, 255, 157, 0.12);\n  --voxcpm-cyber-tag-bg: rgba(0, 255, 157, 0.15);\n  --voxcpm-cyber-tag-border: rgba(0, 255, 157, 0.4);\n  --voxcpm-cyber-tag-text: #00ff9d;\n  --voxcpm-cyber-scanline: rgba(0, 255, 157, 0.03);\n}", u = "voxcpm-design-tokens", d = !1;
function f() {
	if (d) return;
	if (document.getElementById(u)) {
		d = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = u, e.textContent = l, document.head.appendChild(e), d = !0;
}

var p = {
	BLOCK: "voxcpm-model-selector",
	ROW: "voxcpm-model-selector__row",
	DISPLAY: "voxcpm-model-selector__display",
	ICON: "voxcpm-model-selector__icon",
	TEXT: "voxcpm-model-selector__text",
	ARROW: "voxcpm-model-selector__arrow",
	ARROW_ACTIVE: "voxcpm-model-selector__arrow--active",
	BROWSE: "voxcpm-model-selector__browse",
	PATH: "voxcpm-model-selector__path",
	PATH_VISIBLE: "voxcpm-model-selector__path--visible"
}, m = `

.${p.BLOCK} {
  display: flex;
  flex-direction: column;
  gap: 0;
  width: 100%;
  padding: var(--voxcpm-space-2xs) 0;
  box-sizing: border-box;
  
  flex: none !important;
  align-self: flex-start !important;
  --comfy-widget-min-height: var(--voxcpm-widget-height);
  --comfy-widget-max-height: var(--voxcpm-widget-height);
  --comfy-widget-height: var(--voxcpm-widget-height);
}

.${p.ROW} {
  display: flex;
  align-items: center;
  gap: var(--voxcpm-space-xs);
  width: 100%;
  box-sizing: border-box;
}

.${p.DISPLAY} {
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

.${p.DISPLAY}:hover {
  border-color: var(--voxcpm-cyber-accent);
  box-shadow: var(--voxcpm-cyber-glow);
}

.${p.DISPLAY}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${p.ICON} {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  color: var(--voxcpm-cyber-accent);
}

.${p.ICON} svg {
  width: 16px;
  height: 16px;
}

.${p.TEXT} {
  flex: 1;
  color: var(--voxcpm-cyber-text);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
}

.${p.ARROW} {
  font-size: 9px;
  color: var(--voxcpm-cyber-accent);
  margin-left: auto;
  flex-shrink: 0;
  line-height: 1;
  opacity: 0.3;
  transition: opacity var(--voxcpm-duration-fast) var(--voxcpm-easing);
}

.${p.DISPLAY}:hover .${p.ARROW},

.${p.ARROW_ACTIVE} {
  opacity: 1;
}

.${p.BROWSE} {
  
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

.${p.BROWSE}:hover {
  border-color: var(--voxcpm-cyber-accent);
  box-shadow: var(--voxcpm-cyber-glow);
}

.${p.BROWSE}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${p.PATH} {
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

.${p.PATH_VISIBLE} {
  display: block;
}

.${p.DISPLAY}:focus-visible,
.${p.BROWSE}:focus-visible {
  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-cyber-accent);
  outline-offset: var(--voxcpm-focus-ring-offset);
}

@media (prefers-reduced-motion: reduce) {
  .${p.BLOCK} *,
  .${p.BLOCK} *::before,
  .${p.BLOCK} *::after {
    transition-duration: 0ms !important;
  }
}
`.trim(), h = "voxcpm-model-selector-styles", g = !1;
function _() {
	if (g) return;
	if (document.getElementById(h)) {
		g = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = h, e.textContent = m, document.head.appendChild(e), g = !0;
}

var v = {
	BLOCK: "voxcpm-model-dropdown",
	HEADER: "voxcpm-model-dropdown__header",
	LIST: "voxcpm-model-dropdown__list",
	ITEM: "voxcpm-model-dropdown__item",
	ITEM_SELECTED: "voxcpm-model-dropdown__item--selected",
	ICON: "voxcpm-model-dropdown__icon",
	ICON_CLOUD: "voxcpm-model-dropdown__icon--cloud",
	ICON_CHECK: "voxcpm-model-dropdown__icon--check",
	NAME: "voxcpm-model-dropdown__name",
	TAG: "voxcpm-model-dropdown__tag",
	META: "voxcpm-model-dropdown__meta"
}, ee = `

.${v.BLOCK} {
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

.${v.BLOCK}::after {
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

.${v.HEADER} {
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

.${v.HEADER} svg {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
}

.${v.LIST} {
  overflow-y: auto;
  overflow-x: hidden;
  max-height: 280px;
  position: relative;
  z-index: 2;
}

.${v.LIST}::-webkit-scrollbar {
  width: 4px;
}

.${v.LIST}::-webkit-scrollbar-track {
  background: transparent;
}

.${v.LIST}::-webkit-scrollbar-thumb {
  background: var(--voxcpm-cyber-border);
  border-radius: 0;
}

.${v.LIST}::-webkit-scrollbar-thumb:hover {
  background: var(--voxcpm-cyber-accent);
}

.${v.ITEM} {
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

.${v.ITEM}:hover,
.${v.ITEM_SELECTED} {
  background: var(--voxcpm-cyber-bg-hover);
  border-left-color: var(--voxcpm-cyber-accent);
}

.${v.ITEM}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${v.ICON} {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.${v.ICON} svg {
  width: 16px;
  height: 16px;
}

.${v.ICON_CLOUD} {
  color: var(--voxcpm-accent);
}

.${v.ICON_CHECK} {
  color: var(--voxcpm-color-success);
}

.${v.NAME} {
  flex: 0 1 auto;
  color: var(--voxcpm-cyber-text);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
  min-width: 0;
}

.${v.META} {
  font-size: 10px;
  color: var(--voxcpm-cyber-text-dim);
  font-family: var(--voxcpm-cyber-font);
  flex-shrink: 0;
  margin-left: 6px;
}

.${v.TAG} {
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

@media (prefers-reduced-motion: reduce) {
  .${v.BLOCK} {
    animation: none;
  }
}
`.trim(), y = "voxcpm-model-dropdown-styles", b = !1;
function te() {
	if (b) return;
	if (document.getElementById(y)) {
		b = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = y, e.textContent = ee, document.head.appendChild(e), b = !0;
}

var x = !1;
function ne() {
	x || (f(), _(), te(), x = !0);
}

function S(e) {
	try {
		return sessionStorage.getItem(e);
	} catch {
		return null;
	}
}
function C(e, t) {
	try {
		return sessionStorage.setItem(e, t), !0;
	} catch {
		return !1;
	}
}
function re(e) {
	try {
		return sessionStorage.removeItem(e), !0;
	} catch {
		return !1;
	}
}
function w(e) {
	return S(e) !== null;
}
function T(e) {
	return C(e, "true");
}
function ie(e) {
	return re(e);
}
function ae(e) {
	let t = S(e);
	if (t === null) return null;
	try {
		return JSON.parse(t);
	} catch {
		return null;
	}
}
function E(e, t) {
	try {
		return C(e, JSON.stringify(t));
	} catch {
		return !1;
	}
}
var D = new class {
	constructor(e) {
		this.sessionKey = e.sessionKey, this.defaultLife = e.defaultLife;
	}
	show(e) {
		if (w(this.sessionKey)) return c.log("Notification already shown this session, skipping"), {
			shown: !1,
			reason: "already_shown"
		};
		try {
			return t.extensionManager.toast.add({
				severity: e.severity,
				summary: e.summary,
				detail: e.detail,
				life: e.life ?? this.defaultLife,
				closable: e.closable ?? !0
			}), T(this.sessionKey), c.log("Toast notification displayed:", e.summary), { shown: !0 };
		} catch (t) {
			return c.warn("Toast not available:", t), c.log(`${e.summary}: ${e.detail ?? ""}`), T(this.sessionKey), {
				shown: !1,
				reason: "toast_unavailable"
			};
		}
	}
	showAlways(e) {
		try {
			return t.extensionManager.toast.add({
				severity: e.severity,
				summary: e.summary,
				detail: e.detail,
				life: e.life ?? this.defaultLife,
				closable: e.closable ?? !0
			}), c.log("Toast notification displayed:", e.summary), { shown: !0 };
		} catch (t) {
			return c.warn("Toast not available:", t), c.log(`${e.summary}: ${e.detail ?? ""}`), {
				shown: !1,
				reason: "toast_unavailable"
			};
		}
	}
	reset() {
		ie(this.sessionKey);
	}
	wasShown() {
		return w(this.sessionKey);
	}
}({
	sessionKey: "voxcpm.normalization_notification_shown",
	defaultLife: 1e4
}), O = "voxcpm.settings", k = new class {
	constructor() {
		this.settings = null, this.initialized = !1;
	}
	async initialize(e) {
		this.settings = e.settings, this.initialized = !0, E(O, this.settings), c.log("Settings initialized:", this.settings);
	}
	getSettings() {
		if (this.settings) return this.settings;
		let e = ae(O);
		return e ? (this.settings = e, this.settings) : null;
	}
	isUsingCustomPath() {
		return this.getSettings()?.use_custom_path ?? !1;
	}
	getEffectivePath() {
		return this.getSettings()?.effective_path ?? null;
	}
	async updateSettings(e) {
		try {
			for (let [n, r] of Object.entries(e)) await t.api.storeSetting(`voxcpm.${n}`, r);
			return this.settings && (Object.assign(this.settings, e), E(O, this.settings)), c.log("Settings updated:", e), !0;
		} catch (e) {
			return c.warn("Failed to update settings:", e), !1;
		}
	}
	isInitialized() {
		return this.initialized;
	}
}(), A = !1;
function j() {
	if (A) return;
	A = !0;
	let e = document.createElement("style");
	e.id = "voxcpm-mdd-styles", e.textContent = M, document.head.appendChild(e);
}
var M = "\n\n\n\n.voxcpm-mdd__overlay {\n	position: fixed;\n	top: 0;\n	left: 0;\n	right: 0;\n	bottom: 0;\n	background: rgba(0, 0, 0, 0.75);\n	display: flex;\n	align-items: center;\n	justify-content: center;\n	z-index: 10000;\n	animation: voxcpm-mdd-fade-in 150ms ease-out;\n}\n\n@keyframes voxcpm-mdd-fade-in {\n	from { opacity: 0; }\n	to { opacity: 1; }\n}\n\n\n.voxcpm-mdd {\n	background: var(--voxcpm-cyber-bg, rgba(10, 15, 20, 0.95));\n	border: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	border-radius: 8px;\n	max-width: 640px;\n	width: 90%;\n	max-height: 85vh;\n	display: flex;\n	flex-direction: column;\n	box-shadow: var(--voxcpm-cyber-glow, 0 0 8px rgba(0, 255, 157, 0.4)),\n		0 8px 32px rgba(0, 0, 0, 0.6);\n	font-family: var(--voxcpm-cyber-font, 'JetBrains Mono', monospace);\n	color: var(--voxcpm-cyber-text, #e0e0e0);\n	overflow: hidden;\n}\n\n\n.voxcpm-mdd__header {\n	display: flex;\n	align-items: center;\n	gap: 12px;\n	padding: 16px 20px;\n	border-bottom: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	background: var(--voxcpm-cyber-scanline, rgba(0, 255, 157, 0.03));\n}\n\n.voxcpm-mdd__icon {\n	font-size: 18px;\n}\n\n.voxcpm-mdd__title {\n	margin: 0;\n	font-size: 16px;\n	letter-spacing: 0.15em;\n	color: var(--voxcpm-cyber-accent, #00ff9d);\n	text-transform: uppercase;\n}\n\n\n.voxcpm-mdd__body {\n	padding: 20px;\n	overflow-y: auto;\n	flex: 1;\n}\n\n\n.voxcpm-mdd__section {\n	margin-bottom: 16px;\n}\n\n.voxcpm-mdd__section:last-child {\n	margin-bottom: 0;\n}\n\n.voxcpm-mdd__section-title {\n	margin: 0 0 10px;\n	font-size: 10px;\n	letter-spacing: 0.2em;\n	color: var(--voxcpm-cyber-accent, #00ff9d);\n	text-transform: uppercase;\n}\n\n\n.voxcpm-mdd__divider {\n	height: 1px;\n	background: var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	margin: 16px 0;\n}\n\n\n.voxcpm-mdd__paths-list {\n	display: flex;\n	flex-direction: column;\n	gap: 8px;\n	max-height: 200px;\n	overflow-y: auto;\n}\n\n.voxcpm-mdd__empty {\n	padding: 12px;\n	text-align: center;\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n	font-size: 12px;\n}\n\n\n.voxcpm-mdd__path-item {\n	display: block;\n	width: 100%;\n	padding: 10px 14px;\n	background: transparent;\n	border: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	border-radius: 6px;\n	cursor: pointer;\n	text-align: left;\n	transition: border-color 150ms ease, background 150ms ease;\n	font-family: inherit;\n	color: inherit;\n}\n\n.voxcpm-mdd__path-item:hover {\n	border-color: var(--voxcpm-cyber-accent, #00ff9d);\n	background: var(--voxcpm-cyber-bg-hover, rgba(0, 255, 157, 0.08));\n}\n\n.voxcpm-mdd__path-item--selected {\n	border-color: var(--voxcpm-cyber-accent, #00ff9d);\n	background: var(--voxcpm-cyber-bg-active, rgba(0, 255, 157, 0.12));\n	box-shadow: inset 0 0 12px var(--voxcpm-cyber-accent-dim, rgba(0, 255, 157, 0.15));\n}\n\n\n.voxcpm-mdd__path-info {\n	display: flex;\n	align-items: center;\n	gap: 8px;\n	margin-bottom: 6px;\n}\n\n.voxcpm-mdd__path-text {\n	font-size: 12px;\n	color: var(--voxcpm-cyber-text, #e0e0e0);\n	word-break: break-all;\n}\n\n.voxcpm-mdd__path-badge {\n	font-size: 9px;\n	letter-spacing: 0.1em;\n	padding: 1px 6px;\n	border: 1px solid var(--voxcpm-cyber-accent, #00ff9d);\n	color: var(--voxcpm-cyber-accent, #00ff9d);\n	border-radius: 2px;\n	text-transform: uppercase;\n	flex-shrink: 0;\n}\n\n\n.voxcpm-mdd__path-models {\n	display: flex;\n	flex-wrap: wrap;\n	gap: 4px;\n}\n\n.voxcpm-mdd__path-models--empty {\n	font-size: 11px;\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n}\n\n\n.voxcpm-mdd__model-tag {\n	font-size: 10px;\n	padding: 2px 8px;\n	border: 1px solid var(--voxcpm-cyber-tag-border, rgba(0, 255, 157, 0.4));\n	background: var(--voxcpm-cyber-tag-bg, rgba(0, 255, 157, 0.15));\n	color: var(--voxcpm-cyber-tag-text, #00ff9d);\n	border-radius: 0;\n	font-family: var(--voxcpm-cyber-font, 'JetBrains Mono', monospace);\n	line-height: 1.4;\n}\n\n\n.voxcpm-mdd__input-group {\n	display: flex;\n	gap: 8px;\n	margin-bottom: 8px;\n}\n\n.voxcpm-mdd__input {\n	flex: 1;\n	padding: 8px 12px;\n	background: var(--voxcpm-cyber-bg, rgba(10, 15, 20, 0.95));\n	border: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	border-radius: 4px;\n	color: var(--voxcpm-cyber-text, #e0e0e0);\n	font-family: var(--voxcpm-cyber-font, 'JetBrains Mono', monospace);\n	font-size: 12px;\n}\n\n.voxcpm-mdd__input:focus {\n	outline: none;\n	border-color: var(--voxcpm-cyber-accent, #00ff9d);\n	box-shadow: 0 0 6px var(--voxcpm-cyber-accent-dim, rgba(0, 255, 157, 0.15));\n}\n\n.voxcpm-mdd__input::placeholder {\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n}\n\n\n.voxcpm-mdd__validation {\n	font-size: 11px;\n	min-height: 16px;\n	margin-bottom: 8px;\n}\n\n.voxcpm-mdd__validation--success {\n	color: var(--voxcpm-cyber-accent, #00ff9d);\n}\n\n.voxcpm-mdd__validation--warning {\n	color: #ffaa00;\n}\n\n.voxcpm-mdd__validation--error {\n	color: #ff4444;\n}\n\n.voxcpm-mdd__validation--loading {\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n}\n\n\n.voxcpm-mdd__models-preview {\n	background: var(--voxcpm-cyber-bg-hover, rgba(0, 255, 157, 0.08));\n	border: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	border-radius: 4px;\n	padding: 10px 14px;\n	margin-bottom: 8px;\n}\n\n.voxcpm-mdd__preview-title {\n	margin: 0 0 8px;\n	font-size: 10px;\n	letter-spacing: 0.15em;\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n	text-transform: uppercase;\n}\n\n.voxcpm-mdd__models-list {\n	display: flex;\n	flex-wrap: wrap;\n	gap: 4px;\n}\n\n\n.voxcpm-mdd__footer {\n	display: flex;\n	justify-content: flex-end;\n	gap: 10px;\n	padding: 14px 20px;\n	border-top: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n}\n\n\n.voxcpm-mdd__btn {\n	padding: 8px 16px;\n	border-radius: 4px;\n	font-size: 12px;\n	cursor: pointer;\n	transition: background 150ms ease, border-color 150ms ease,\n		box-shadow 150ms ease;\n	font-family: var(--voxcpm-cyber-font, 'JetBrains Mono', monospace);\n	text-transform: uppercase;\n	letter-spacing: 0.08em;\n}\n\n.voxcpm-mdd__btn--secondary {\n	background: transparent;\n	border: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n}\n\n.voxcpm-mdd__btn--secondary:hover {\n	border-color: var(--voxcpm-cyber-accent, #00ff9d);\n	color: var(--voxcpm-cyber-text, #e0e0e0);\n}\n\n.voxcpm-mdd__btn--primary {\n	background: var(--voxcpm-cyber-accent, #00ff9d);\n	border: none;\n	color: #0a0f14;\n	font-weight: 600;\n}\n\n.voxcpm-mdd__btn--primary:hover:not(:disabled) {\n	box-shadow: 0 0 12px var(--voxcpm-cyber-accent-dim, rgba(0, 255, 157, 0.15));\n}\n\n.voxcpm-mdd__btn--primary:disabled {\n	opacity: 0.4;\n	cursor: not-allowed;\n}\n\n.voxcpm-mdd__btn--register {\n	background: transparent;\n	border: 1px solid var(--voxcpm-cyber-accent, #00ff9d);\n	color: var(--voxcpm-cyber-accent, #00ff9d);\n	flex-shrink: 0;\n}\n\n.voxcpm-mdd__btn--register:hover:not(:disabled) {\n	background: var(--voxcpm-cyber-bg-hover, rgba(0, 255, 157, 0.08));\n}\n\n.voxcpm-mdd__btn--register:disabled {\n	opacity: 0.4;\n	cursor: not-allowed;\n}\n\n.voxcpm-mdd__btn:focus-visible {\n	outline: 2px solid var(--voxcpm-cyber-accent, #00ff9d);\n	outline-offset: 2px;\n}\n\n\n.voxcpm-mdd__paths-list::-webkit-scrollbar,\n.voxcpm-mdd__body::-webkit-scrollbar {\n	width: 6px;\n}\n\n.voxcpm-mdd__paths-list::-webkit-scrollbar-track,\n.voxcpm-mdd__body::-webkit-scrollbar-track {\n	background: transparent;\n}\n\n.voxcpm-mdd__paths-list::-webkit-scrollbar-thumb,\n.voxcpm-mdd__body::-webkit-scrollbar-thumb {\n	background: var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	border-radius: 0;\n}\n\n.voxcpm-mdd__paths-list::-webkit-scrollbar-thumb:hover,\n.voxcpm-mdd__body::-webkit-scrollbar-thumb:hover {\n	background: var(--voxcpm-cyber-accent, #00ff9d);\n}\n", N = null;
async function oe() {
	try {
		let e = await t.api.fetchApi("/voxcpm/tts_search_paths");
		return e.ok ? (await e.json()).paths || [] : (c.warn("Failed to fetch tts search paths:", e.statusText), []);
	} catch (e) {
		return c.warn("Error fetching tts search paths:", e), [];
	}
}
async function se(e) {
	try {
		let n = await t.api.fetchApi(`/voxcpm/validate_directory?path=${encodeURIComponent(e)}`);
		return n.ok ? await n.json() : null;
	} catch (e) {
		return c.warn("Error validating directory:", e), null;
	}
}
async function ce(e) {
	try {
		let n = await t.api.fetchApi("/voxcpm/register_model_path", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ path: e })
		});
		return n.ok ? await n.json() : null;
	} catch (e) {
		return c.warn("Error registering model path:", e), null;
	}
}
function le(e) {
	return new Promise(async (t) => {
		P(), j();
		let n = await oe(), r = document.createElement("div");
		r.className = "voxcpm-mdd__overlay", N = r, r.innerHTML = `
			<div class="voxcpm-mdd">
				<div class="voxcpm-mdd__header">
					<span class="voxcpm-mdd__icon">📂</span>
					<h2 class="voxcpm-mdd__title">SELECT MODEL DIRECTORY</h2>
				</div>
				<div class="voxcpm-mdd__body">
					<div class="voxcpm-mdd__section">
						<h3 class="voxcpm-mdd__section-title">REGISTERED PATHS</h3>
						<div class="voxcpm-mdd__paths-list" id="voxcpm-mdd-paths">
							${n.length === 0 ? "<div class=\"voxcpm-mdd__empty\">No registered tts paths found</div>" : n.map((e) => `
									<button class="voxcpm-mdd__path-item" data-path="${I(e.path)}">
										<div class="voxcpm-mdd__path-info">
											<span class="voxcpm-mdd__path-text">${F(e.path)}</span>
											${e.is_default ? "<span class=\"voxcpm-mdd__path-badge\">DEFAULT</span>" : ""}
										</div>
										${e.models.length > 0 ? `<div class="voxcpm-mdd__path-models">${e.models.map((e) => `<span class="voxcpm-mdd__model-tag">${F(e)}</span>`).join("")}</div>` : "<div class=\"voxcpm-mdd__path-models voxcpm-mdd__path-models--empty\">No models found</div>"}
									</button>
								`).join("")}
						</div>
					</div>
					<div class="voxcpm-mdd__divider"></div>
					<div class="voxcpm-mdd__section">
						<h3 class="voxcpm-mdd__section-title">CUSTOM PATH</h3>
						<div class="voxcpm-mdd__input-group">
							<input
								type="text"
								class="voxcpm-mdd__input"
								placeholder="Enter absolute path to model directory"
								value="${I(e || "")}"
								id="voxcpm-mdd-path-input"
							/>
							<button class="voxcpm-mdd__btn voxcpm-mdd__btn--register" id="voxcpm-mdd-register" disabled>
								REGISTER
							</button>
						</div>
						<div class="voxcpm-mdd__validation" id="voxcpm-mdd-validation"></div>
						<div class="voxcpm-mdd__models-preview" id="voxcpm-mdd-models-preview" style="display: none;">
							<h4 class="voxcpm-mdd__preview-title">FOUND MODELS (<span id="voxcpm-mdd-model-count">0</span>)</h4>
							<div class="voxcpm-mdd__models-list" id="voxcpm-mdd-models-list"></div>
						</div>
					</div>
				</div>
				<div class="voxcpm-mdd__footer">
					<button class="voxcpm-mdd__btn voxcpm-mdd__btn--secondary" id="voxcpm-mdd-cancel">CANCEL</button>
					<button class="voxcpm-mdd__btn voxcpm-mdd__btn--primary" id="voxcpm-mdd-confirm" disabled>
						SELECT PATH
					</button>
				</div>
			</div>
		`;
		let i = null, a = !1, o = !1, s = null, c = r.querySelector("#voxcpm-mdd-path-input"), l = r.querySelector("#voxcpm-mdd-register"), u = r.querySelector("#voxcpm-mdd-validation"), d = r.querySelector("#voxcpm-mdd-models-preview"), f = r.querySelector("#voxcpm-mdd-model-count"), p = r.querySelector("#voxcpm-mdd-models-list"), m = r.querySelector("#voxcpm-mdd-confirm"), h = r.querySelector("#voxcpm-mdd-cancel"), g = r.querySelectorAll(".voxcpm-mdd__path-item");
		g.forEach((e) => {
			e.addEventListener("click", () => {
				g.forEach((e) => e.classList.remove("voxcpm-mdd__path-item--selected")), e.classList.add("voxcpm-mdd__path-item--selected"), i = e.dataset.path || null, a = !0, o = !1, c && i && (c.value = i), m && (m.disabled = !1), l && (l.disabled = !0);
				let t = n.find((e) => e.path === i);
				t && t.models.length > 0 ? (d.style.display = "block", f.textContent = t.models.length.toString(), p.innerHTML = t.models.map((e) => `<span class="voxcpm-mdd__model-tag">${F(e)}</span>`).join("")) : d.style.display = "none", u.innerHTML = "";
			});
		}), c?.addEventListener("input", () => {
			let e = c.value.trim();
			g.forEach((e) => e.classList.remove("voxcpm-mdd__path-item--selected")), s && clearTimeout(s), s = setTimeout(async () => {
				await _(e);
			}, 300);
		});
		async function _(e) {
			if (!e) {
				u.innerHTML = "", d.style.display = "none", m.disabled = !0, l.disabled = !0, a = !1, i = null;
				return;
			}
			u.innerHTML = "<span class=\"voxcpm-mdd__validation--loading\">Validating...</span>";
			let t = await se(e);
			if (!t) {
				u.innerHTML = "<span class=\"voxcpm-mdd__validation--error\">Validation request failed</span>", m.disabled = !0, l.disabled = !0, a = !1, i = null;
				return;
			}
			t.valid ? (a = !0, i = e, o = !1, m.disabled = !1, l.disabled = !0, t.models.length > 0 ? (d.style.display = "block", f.textContent = t.models.length.toString(), p.innerHTML = t.models.map((e) => `<span class="voxcpm-mdd__model-tag">${F(e)}</span>`).join(""), u.innerHTML = `<span class="voxcpm-mdd__validation--success">✓ Valid — ${t.models.length} model(s) found</span>`) : (d.style.display = "none", u.innerHTML = "<span class=\"voxcpm-mdd__validation--success\">✓ Valid directory (no VoxCPM models found yet)</span>")) : t.exists && !t.is_registered ? (a = !1, i = null, m.disabled = !0, l.disabled = !1, d.style.display = "none", u.innerHTML = "<span class=\"voxcpm-mdd__validation--warning\">⚠ Directory not in registered tts paths — click REGISTER to add it</span>") : (a = !1, i = null, m.disabled = !0, l.disabled = !0, d.style.display = "none", u.innerHTML = `<span class="voxcpm-mdd__validation--error">✗ ${t.error || "Invalid path"}</span>`);
		}
		l?.addEventListener("click", async () => {
			let e = c.value.trim();
			if (!e) return;
			l.disabled = !0, l.textContent = "REGISTERING...";
			let t = await ce(e);
			t?.success ? (o = !t.already_registered, i = e, a = !0, m.disabled = !1, l.textContent = "REGISTERED ✓", u.innerHTML = "<span class=\"voxcpm-mdd__validation--success\">✓ Path registered successfully</span>", await _(e)) : (l.disabled = !1, l.textContent = "REGISTER", u.innerHTML = `<span class="voxcpm-mdd__validation--error">✗ ${t?.error || "Registration failed"}</span>`);
		}), h?.addEventListener("click", () => {
			P(), t(null);
		}), m?.addEventListener("click", () => {
			a && i && (P(), t({
				path: i,
				newlyRegistered: o
			}));
		});
		let v = (e) => {
			e.key === "Escape" && (P(), t(null), document.removeEventListener("keydown", v));
		};
		document.addEventListener("keydown", v), r.addEventListener("click", (e) => {
			e.target === r && (P(), t(null), document.removeEventListener("keydown", v));
		}), document.body.appendChild(r), e && await _(e);
	});
}
function P() {
	N && (N.remove(), N = null);
}
function F(e) {
	let t = document.createElement("div");
	return t.textContent = e, t.innerHTML;
}
function I(e) {
	return e.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

var L = {
	CLOUD: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z\"/><polyline points=\"12 12 12 22\"/><path d=\"m8 18 4 4 4-4\"/></svg>",
	CHECK: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><polyline points=\"20 6 9 17 4 12\"/></svg>",
	SHIELD: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z\"/></svg>",
	FOLDER: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z\"/></svg>",
	FOLDER_OPEN: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M5 19a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4l2 2h4a2 2 0 0 1 2 2v1\"/><path d=\"M6 12h14l-2.5 7H8.5L6 12z\"/></svg>"
}, R = null;
async function z(e) {
	if (R !== null) return R;
	try {
		let t = await e("/voxcpm/model_info");
		if (t.ok) return R = (await t.json()).models || [], R;
	} catch (e) {
		console.warn("[VoxCPM] Failed to fetch model info:", e);
	}
	return [];
}
function B() {
	R = null;
}
var V = class e {
	constructor(e, t, n, r) {
		this.selectedIndex = -1, this.resolveSelection = null, this.cleanups = [], this.isOpen = !1, this.modelInfo = e, this.anchorRect = t, this.scale = n, this.widgetElement = r || null, this.panel = document.createElement("div"), this.panel.className = v.BLOCK, this.panel.setAttribute("role", "listbox"), this.panel.setAttribute("aria-label", "Select model"), this.header = document.createElement("div"), this.header.className = v.HEADER, this.header.innerHTML = `${L.SHIELD} <span>${s.MODEL_DROPDOWN.HEADER_TEXT}</span>`, this.list = document.createElement("div"), this.list.className = v.LIST, this.panel.appendChild(this.header), this.panel.appendChild(this.list), this.renderItems();
	}
	show() {
		return new Promise((t) => {
			this.resolveSelection = t, e.currentDropdown && e.currentDropdown !== this && e.currentDropdown.close(null), this.positionPanel(), document.body.appendChild(this.panel), this.isOpen = !0, e.currentDropdown = this, this.setupClickOutside(), this.setupKeyboardNav(), this.setupZoomChangeListener(), this.panel.tabIndex = -1, this.panel.focus();
		});
	}
	close(t = null) {
		if (this.isOpen) {
			this.isOpen = !1, e.currentDropdown === this && (e.currentDropdown = null);
			for (let e of this.cleanups) e();
			this.cleanups = [], this.panel.parentElement && this.panel.parentElement.removeChild(this.panel), this.resolveSelection && (this.resolveSelection(t), this.resolveSelection = null);
		}
	}
	renderItems() {
		this.list.innerHTML = "";
		for (let e = 0; e < this.modelInfo.length; e++) {
			let t = this.modelInfo[e], n = this.createElementItem(t, e);
			this.list.appendChild(n);
		}
	}
	createElementItem(e, t) {
		let n = document.createElement("div");
		n.className = v.ITEM, n.setAttribute("role", "option"), n.setAttribute("data-index", String(t)), n.setAttribute("data-model", e.name);
		let r = document.createElement("span"), i = e.is_downloaded ? v.ICON_CHECK : v.ICON_CLOUD;
		r.className = `${v.ICON} ${i}`, r.innerHTML = e.is_downloaded ? L.CHECK : L.CLOUD, n.appendChild(r);
		let a = document.createElement("span");
		if (a.className = v.NAME, a.textContent = e.name, n.appendChild(a), e.size_gb > 0) {
			let t = document.createElement("span");
			t.className = v.META, t.textContent = `${e.size_gb} GB`, n.appendChild(t);
		}
		if (e.type === "official") {
			let e = document.createElement("span");
			e.className = v.TAG, e.textContent = "DEFAULT", n.appendChild(e);
		}
		return n.addEventListener("click", (t) => {
			t.stopPropagation(), this.close(e.name);
		}), n.addEventListener("mouseenter", () => {
			this.setSelectedIndex(t);
		}), n;
	}
	positionPanel() {
		let e = this.scale, t = s.MODEL_DROPDOWN.ANCHOR_GAP, n = window.innerHeight, r = window.innerWidth, i = s.MODEL_DROPDOWN.ITEM_HEIGHT, a = s.MODEL_DROPDOWN.MAX_VISIBLE_ITEMS, o = (32 + Math.min(this.modelInfo.length, a) * i) * e, c = 300 * e, l, u = n - this.anchorRect.bottom, d = this.anchorRect.top;
		l = u >= o + t ? this.anchorRect.bottom + t : d >= o + t ? this.anchorRect.top - o - t : this.anchorRect.bottom + t;
		let f = this.anchorRect.left;
		f + c > r && (f = r - c - 8), f = Math.max(8, f), this.panel.style.position = "fixed", this.panel.style.top = `${l}px`, this.panel.style.left = `${f}px`, this.panel.style.zIndex = String(s.MODEL_DROPDOWN.Z_INDEX), this.panel.style.transformOrigin = "top left", this.panel.style.transform = `scale(${e})`, this.panel.style.setProperty("--voxcpm-dropdown-scale", String(e));
	}
	setupClickOutside() {
		let e = (e) => {
			if (!this.isOpen) return;
			let t = e.target;
			this.panel.contains(t) || this.widgetElement && this.widgetElement.contains(t) || this.close(null);
		};
		document.addEventListener("pointerdown", e, !0), this.cleanups.push(() => {
			document.removeEventListener("pointerdown", e, !0);
		});
	}
	setupKeyboardNav() {
		let e = (e) => {
			if (this.isOpen) switch (e.key) {
				case "ArrowDown": {
					e.preventDefault(), e.stopPropagation();
					let t = this.selectedIndex < this.modelInfo.length - 1 ? this.selectedIndex + 1 : 0;
					this.setSelectedIndex(t);
					break;
				}
				case "ArrowUp": {
					e.preventDefault(), e.stopPropagation();
					let t = this.selectedIndex > 0 ? this.selectedIndex - 1 : this.modelInfo.length - 1;
					this.setSelectedIndex(t);
					break;
				}
				case "Enter":
					e.preventDefault(), e.stopPropagation(), this.selectedIndex >= 0 && this.selectedIndex < this.modelInfo.length ? this.close(this.modelInfo[this.selectedIndex].name) : this.close(null);
					break;
				case "Escape":
					e.preventDefault(), e.stopPropagation(), this.close(null);
					break;
				case "Tab":
					e.preventDefault(), this.close(null);
					break;
			}
		};
		document.addEventListener("keydown", e, !0), this.cleanups.push(() => {
			document.removeEventListener("keydown", e, !0);
		});
	}
	setupZoomChangeListener() {
		let e = t.canvas;
		if (!e?.ds) return;
		let n = e.ds.onChanged;
		e.ds.onChanged = (e, t) => {
			this.isOpen && this.close(null), n?.(e, t);
		}, this.cleanups.push(() => {
			e.ds && (e.ds.onChanged = n);
		});
	}
	setSelectedIndex(e) {
		let t = this.list.querySelectorAll(`.${v.ITEM}`);
		if (t.forEach((e) => {
			e.classList.remove(v.ITEM_SELECTED);
		}), this.selectedIndex = e, e >= 0 && e < t.length) {
			let n = t[e];
			n.classList.add(v.ITEM_SELECTED), n.scrollIntoView({ block: "nearest" });
		}
	}
};
V.currentDropdown = null;

var H =  new Map();
function U(e) {
	let t = e.widgets;
	if (!t) {
		c.warn("No widgets found on node:", e.id);
		return;
	}
	let n = t.find((e) => e.name === "model_name");
	if (!n) {
		c.warn("model_name widget not found on node:", e.id);
		return;
	}
	ue(n);
	let r = document.createElement("div");
	r.className = p.BLOCK;
	let i = document.createElement("div");
	i.className = p.DISPLAY, i.title = "Click to select a model", i.tabIndex = 0, i.setAttribute("role", "combobox"), i.setAttribute("aria-expanded", "false"), i.setAttribute("aria-haspopup", "listbox");
	let a = document.createElement("span");
	a.className = p.ICON, a.innerHTML = L.FOLDER;
	let o = document.createElement("span");
	o.className = p.TEXT, o.textContent = n.value || s.MODEL_SELECTOR.PLACEHOLDER;
	let l = document.createElement("span");
	l.className = p.ARROW, l.textContent = s.MODEL_SELECTOR.ARROW;
	let u = document.createElement("button");
	u.className = p.BROWSE, u.innerHTML = L.FOLDER_OPEN, u.title = "Browse for custom model directory", u.setAttribute("role", "button");
	let d = document.createElement("div");
	d.className = p.PATH;
	let f = e.properties?.custom_model_path;
	f && (a.innerHTML = L.FOLDER_OPEN, d.textContent = f, d.classList.add(p.PATH_VISIBLE)), i.append(a, o, l);
	let m = document.createElement("div");
	m.className = p.ROW, m.append(i, u), r.append(m, d), r.addEventListener("mouseup", (e) => e.stopPropagation()), r.addEventListener("click", (e) => e.stopPropagation()), i.addEventListener("click", (t) => {
		if (t.preventDefault(), t.stopPropagation(), W()) {
			G();
			return;
		}
		K(t, n, o, a, d, e, i, l);
	}), i.addEventListener("keydown", (t) => {
		if (t.key === "Enter" || t.key === " ") {
			if (t.preventDefault(), t.stopPropagation(), W()) {
				G();
				return;
			}
			K(t, n, o, a, d, e, i, l);
		}
	}), u.addEventListener("click", async (t) => {
		t.preventDefault(), t.stopPropagation(), await de(e, n, o, a, d);
	});
	let h = e.addDOMWidget("model_selector", "custom", r, {
		serialize: !1,
		hideOnZoom: !1,
		selectOn: ["click"],
		getValue: () => n.value,
		setValue: (e) => {
			n.value = e, o.textContent = e;
		}
	}), g = new MutationObserver(() => {
		let t = m.parentElement;
		if (t) {
			let n = t.parentElement;
			n && n !== document.body ? (n.classList.add(p.BLOCK), n.setAttribute("node-id", String(e.id)), c.debug("Added", p.BLOCK, "class + node-id to WidgetDOM parent div for node:", e.id)) : c.debug("No WidgetDOM grandparent found (LiteGraph mode) for node:", e.id), g.disconnect();
		}
	});
	g.observe(document.body, {
		childList: !0,
		subtree: !0
	}), h.tooltip = "Select the VoxCPM model to use", h.computeSize = () => [0, s.MODEL_SELECTOR.MIN_HEIGHT], h.computeLayoutSize = void 0;
	let _ = e.widgets;
	if (_ && _.length > 1) {
		let t = _.indexOf(h);
		t > 0 && (_.splice(t, 1), _.unshift(h), c.debug("Moved model selector widget to position 0 for node:", e.id));
	}
	let v = e.inputs?.find((e) => e.name === "model_name");
	v && (v.widget = {
		name: "model_selector",
		_originalWidget: v.widget?.name
	}, c.debug("Bound model_name input to model_selector widget for node:", e.id)), H.set(e.id, {
		modelText: o,
		folderIcon: a,
		pathIndicator: d,
		modelWidget: n
	}), c.debug("Created model selector widget for node:", e.id);
}
function ue(e) {
	e.type = "converted-widget", e.computeSize = () => [0, -4], e.draw = () => {}, e.options || (e.options = {}), e.options.canvasOnly = !0, c.debug("Hidden default combo widget:", e.name);
}
function W() {
	return V.currentDropdown !== null;
}
function G() {
	V.currentDropdown && V.currentDropdown.close(null);
}
async function K(e, r, i, a, o, s, l, u) {
	let d = r.options?.values || [];
	if (d.length === 0) {
		c.warn("No models available for dropdown"), D.show({
			severity: "warn",
			summary: "No Models Available",
			detail: "No models found. Try browsing for a custom model directory.",
			life: 4e3
		});
		return;
	}
	let f = [];
	try {
		f = await z(n.fetchApi.bind(n));
	} catch (e) {
		c.error("Failed to fetch model info:", e);
	}
	f.length === 0 && (f = d.map((e) => ({
		name: e,
		type: "local",
		architecture: "unknown",
		sample_rate: 0,
		size_gb: 0,
		is_downloaded: !0
	})));
	let m = l.getBoundingClientRect(), h = t.canvas.ds?.scale || 1;
	u.classList.add(p.ARROW_ACTIVE), l.setAttribute("aria-expanded", "true");
	try {
		let e = await new V(f, m, h, l).show();
		if (e) {
			r.value = e, i.textContent = e;
			let t = f.find((t) => t.name === e);
			t && (a.innerHTML = t.is_downloaded ? L.CHECK : L.CLOUD), r.callback && r.callback(e), s.setDirtyCanvas && s.setDirtyCanvas(!0), c.debug("Model selected:", e, "for node:", s.id);
		}
	} catch (e) {
		c.error("Failed to open model dropdown:", e);
	} finally {
		u.classList.remove(p.ARROW_ACTIVE), l.setAttribute("aria-expanded", "false");
	}
}
async function de(e, t, r, i, a) {
	c.log("Browse button clicked for node:", e.id);
	let o = await le(k.getSettings()?.effective_path || void 0);
	if (!o?.path) {
		c.debug("Browse cancelled by user");
		return;
	}
	c.log("User selected model directory:", o.path, "(newly registered:", o.newlyRegistered, ")"), e.properties = e.properties || {}, e.properties.custom_model_path = o.path, o.newlyRegistered && await k.updateSettings({
		use_custom_path: !0,
		custom_model_path: o.path
	});
	try {
		let s = await n.fetchApi("/voxcpm/models", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ path: o.path })
		});
		if (s.ok) {
			let n = await s.json();
			c.log("Models found at custom path:", n.models), fe(t, n.models, e), i.innerHTML = L.FOLDER_OPEN, a.textContent = o.path, a.classList.add(p.PATH_VISIBLE), n.models.length > 0 ? r.textContent = t.value : r.textContent = "No models found", D.show({
				severity: "success",
				summary: "Model Directory Set",
				detail: `Found ${n.models.length} model(s) in: ${o.path}`,
				life: 5e3
			});
		} else {
			let e = s.statusText || "Unknown error";
			c.error("Failed to get models:", e), D.show({
				severity: "error",
				summary: "Error",
				detail: `Failed to scan model directory: ${e}`,
				life: 5e3
			});
		}
	} catch (e) {
		c.error("Error fetching models:", e), D.show({
			severity: "error",
			summary: "Error",
			detail: "Failed to connect to server",
			life: 5e3
		});
	}
}
function fe(e, t, n) {
	e.options ? e.options.values = t : e.options = { values: t }, t.length > 0 && !t.includes(e.value) && (e.value = t[0], e.callback && e.callback(t[0])), n.setDirtyCanvas && n.setDirtyCanvas(!0), c.debug("Updated model options:", t);
}
function pe(e, t) {
	let n = H.get(e.id);
	if (!n) {
		c.debug("No selector refs found for node:", e.id);
		return;
	}
	let r = t || n.modelWidget.value;
	n.modelText.textContent = r;
	let i = e.properties?.custom_model_path;
	i ? (n.folderIcon.innerHTML = L.FOLDER_OPEN, n.pathIndicator.textContent = i, n.pathIndicator.classList.add(p.PATH_VISIBLE)) : (n.folderIcon.innerHTML = L.FOLDER, n.pathIndicator.classList.remove(p.PATH_VISIBLE)), c.debug("Synced model selector display for node:", e.id, "value:", r);
}
var q = new class {
	constructor() {
		this.downloads =  new Map(), this.listeners =  new Map(), this.globalListeners =  new Set(), this._bindWebSocketEvents();
	}
	getState(e) {
		return this.downloads.get(e);
	}
	getAllStates() {
		return new Map(this.downloads);
	}
	isDownloading(e) {
		return this.downloads.get(e)?.status === "downloading";
	}
	async cancelDownload(e) {
		try {
			let t = await (await fetch(s.DOWNLOAD_PROGRESS.CANCEL_ENDPOINT, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model_name: e })
			})).json();
			return t.success ? (c.info(`Download cancelled: ${e}`), !0) : (c.warn(`Cancel failed: ${t.message}`), !1);
		} catch (t) {
			return c.error(`Failed to cancel download for '${e}':`, t), !1;
		}
	}
	onModelUpdate(e, t) {
		return this.listeners.has(e) || this.listeners.set(e,  new Set()), this.listeners.get(e).add(t), () => {
			this.listeners.get(e)?.delete(t);
		};
	}
	onAnyUpdate(e) {
		return this.globalListeners.add(e), () => {
			this.globalListeners.delete(e);
		};
	}
	_bindWebSocketEvents() {
		try {
			t.api.addEventListener(i.DOWNLOAD_PROGRESS, ((e) => {
				this._handleProgressEvent(e.detail);
			})), c.debug("Download progress WebSocket listener registered");
		} catch (e) {
			c.warn("Failed to bind download progress WebSocket listener:", e);
		}
	}
	_handleProgressEvent(e) {
		let t = e.model_name || e.transfer_id;
		if (!t) {
			c.warn("Download progress event missing model_name:", e);
			return;
		}
		let n = e.event_type, r = this.downloads.get(t);
		switch (r || (r = this._createInitialState(t), this.downloads.set(t, r)), n) {
			case "start":
				r.status = "downloading", r.isXet = e.is_xet ?? !1;
				break;
			case "progress":
				r.status = "downloading", r.percentage = e.percentage ?? 0, r.currentFile = e.filename ?? "", r.speed = e.speed ?? 0, r.bytesCompleted = e.bytes_completed ?? 0, r.totalBytes = e.total_bytes ?? 0, r.fileIndex = e.file_index ?? 0, r.totalFiles = e.total_files ?? 1, e.transfer_bytes_completed !== void 0 && (r.transferBytesCompleted = e.transfer_bytes_completed, r.transferBytesTotal = e.transfer_bytes_total ?? 0, r.transferSpeed = e.transfer_speed ?? 0, r.dedupSavedBytes = e.dedup_saved_bytes ?? 0, r.isXet = !0);
				break;
			case "complete":
				r.status = "complete", r.percentage = 100;
				break;
			case "error":
				e.error?.includes("cancelled") ? r.status = "cancelled" : r.status = "error", r.errorMessage = e.error ?? "Unknown error";
				break;
		}
		r.lastUpdated = Date.now(), this._notifyListeners(n, t, r), (n === "complete" || n === "error") && setTimeout(() => {
			this.downloads.delete(t);
		}, 5e3);
	}
	_createInitialState(e) {
		return {
			modelName: e,
			status: "idle",
			percentage: 0,
			currentFile: "",
			speed: 0,
			bytesCompleted: 0,
			totalBytes: 0,
			fileIndex: 0,
			totalFiles: 1,
			isXet: !1,
			transferBytesCompleted: 0,
			transferBytesTotal: 0,
			transferSpeed: 0,
			dedupSavedBytes: 0,
			errorMessage: null,
			lastUpdated: Date.now()
		};
	}
	_notifyListeners(e, t, n) {
		let r = this.listeners.get(t);
		if (r) for (let e of r) try {
			e(n);
		} catch (e) {
			c.error(`Download listener error for '${t}':`, e);
		}
		for (let t of this.globalListeners) try {
			t(e, n);
		} catch (e) {
			c.error("Global download listener error:", e);
		}
	}
}(), J = s.DOWNLOAD_PROGRESS.BLOCK_CLASS;
function me() {
	let e = `${J}-styles`;
	if (document.getElementById(e)) return;
	let t = document.createElement("style");
	t.id = e, t.textContent = he, document.head.appendChild(t);
}
var he = `

.${J} {
	display: flex;
	flex-direction: column;
	gap: 4px;
	padding: 6px 8px;
	background: var(--voxcpm-bg-surface, rgba(30, 30, 40, 0.9));
	border: 1px solid var(--voxcpm-border-default, rgba(100, 100, 140, 0.3));
	border-radius: var(--voxcpm-radius-sm, 4px);
	font-family: var(--voxcpm-font-mono, 'Consolas', 'Monaco', monospace);
	font-size: 11px;
	color: var(--voxcpm-text-primary, #e0e0e0);
	min-height: ${s.DOWNLOAD_PROGRESS.MIN_HEIGHT}px;
	transition: opacity var(--voxcpm-duration-fast, 150ms) ease;
}

.${J}--hidden {
	display: none;
}

.${J}--complete {
	border-color: var(--voxcpm-color-success, #4caf50);
	opacity: 0.7;
}

.${J}--error {
	border-color: var(--voxcpm-color-error, #f44336);
}

.${J}--cancelled {
	border-color: var(--voxcpm-color-warning, #ff9800);
	opacity: 0.7;
}

.${J}__header {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 8px;
}

.${J}__title {
	display: flex;
	align-items: center;
	gap: 4px;
	font-size: 11px;
	font-weight: 600;
	color: var(--voxcpm-text-secondary, #a0a0b0);
	white-space: nowrap;
	overflow: hidden;
	text-overflow: ellipsis;
}

.${J}__title-icon {
	font-size: 12px;
	color: var(--voxcpm-color-accent, #7c8cf8);
}

.${J}__cancel-btn {
	display: flex;
	align-items: center;
	justify-content: center;
	padding: 2px 8px;
	border: 1px solid var(--voxcpm-color-error, #f44336);
	border-radius: var(--voxcpm-radius-sm, 3px);
	background: transparent;
	color: var(--voxcpm-color-error, #f44336);
	font-family: inherit;
	font-size: 10px;
	font-weight: 600;
	cursor: pointer;
	transition: all var(--voxcpm-duration-fast, 150ms) ease;
	white-space: nowrap;
	flex-shrink: 0;
}

.${J}__cancel-btn:hover {
	background: var(--voxcpm-color-error, #f44336);
	color: #fff;
}

.${J}__cancel-btn:active {
	transform: scale(0.95);
}

.${J}__cancel-btn:disabled {
	opacity: 0.4;
	cursor: not-allowed;
}

.${J}__bar-container {
	position: relative;
	height: 6px;
	background: var(--voxcpm-bg-inset, rgba(0, 0, 0, 0.3));
	border-radius: var(--voxcpm-radius-sm, 3px);
	overflow: hidden;
}

.${J}__bar-fill {
	height: 100%;
	background: linear-gradient(
		90deg,
		var(--voxcpm-color-accent, #7c8cf8),
		var(--voxcpm-color-accent-light, #a0b0ff)
	);
	border-radius: var(--voxcpm-radius-sm, 3px);
	transition: width ${s.DOWNLOAD_PROGRESS.ANIMATION_DURATION}ms ease-out;
	min-width: 0;
}

.${J}--complete .${J}__bar-fill {
	background: linear-gradient(
		90deg,
		var(--voxcpm-color-success, #4caf50),
		var(--voxcpm-color-success-light, #81c784)
	);
}

.${J}--error .${J}__bar-fill {
	background: var(--voxcpm-color-error, #f44336);
}

.${J}--cancelled .${J}__bar-fill {
	background: var(--voxcpm-color-warning, #ff9800);
}

.${J}__stats {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 8px;
	font-size: 10px;
	color: var(--voxcpm-text-tertiary, #707080);
}

.${J}__percentage {
	font-weight: 700;
	color: var(--voxcpm-text-primary, #e0e0e0);
	min-width: 36px;
}

.${J}__speed {
	color: var(--voxcpm-text-tertiary, #707080);
}

.${J}__file-info {
	overflow: hidden;
	text-overflow: ellipsis;
	white-space: nowrap;
	flex: 1;
	text-align: right;
}

.${J}__xet-badge {
	display: inline-flex;
	align-items: center;
	padding: 0 4px;
	border-radius: 2px;
	background: var(--voxcpm-color-accent, #7c8cf8);
	color: #fff;
	font-size: 9px;
	font-weight: 700;
	letter-spacing: 0.5px;
	margin-left: 4px;
}

.${J}__status {
	font-size: 10px;
	color: var(--voxcpm-text-tertiary, #707080);
	text-align: center;
	padding: 2px 0;
}

.${J}--error .${J}__status {
	color: var(--voxcpm-color-error, #f44336);
}

.${J}--cancelled .${J}__status {
	color: var(--voxcpm-color-warning, #ff9800);
}

.${J}--complete .${J}__status {
	color: var(--voxcpm-color-success, #4caf50);
}
`, Y = s.DOWNLOAD_PROGRESS.BLOCK_CLASS, ge = s.DOWNLOAD_PROGRESS.WIDGET_NAME;
function X(e) {
	if (e === 0) return "0 B";
	let t = [
		"B",
		"KB",
		"MB",
		"GB"
	], n = Math.floor(Math.log(e) / Math.log(1024));
	return `${(e / 1024 ** n).toFixed(+(n > 0))} ${t[n]}`;
}
function _e(e) {
	return e === 0 ? "--" : `${X(e)}/s`;
}
var ve = class {
	constructor() {
		this.modelName = "", this.unsubscribe = null, this.hideTimeout = null, me(), this.element = document.createElement("div"), this.element.className = `${Y} ${Y}--hidden`, this.element.innerHTML = this._buildHTML(), this.container = this.element, this.barFill = this.element.querySelector(`.${Y}__bar-fill`), this.percentageEl = this.element.querySelector(`.${Y}__percentage`), this.speedEl = this.element.querySelector(`.${Y}__speed`), this.fileInfoEl = this.element.querySelector(`.${Y}__file-info`), this.cancelBtn = this.element.querySelector(`.${Y}__cancel-btn`), this.statusEl = this.element.querySelector(`.${Y}__status`), this.xetBadge = this.element.querySelector(`.${Y}__xet-badge`), this.cancelBtn.addEventListener("click", (e) => {
			e.stopPropagation(), this._onCancel();
		}), this.element.addEventListener("mousedown", (e) => e.stopPropagation()), this.element.addEventListener("wheel", (e) => e.stopPropagation());
	}
	setModelName(e) {
		if (this.unsubscribe && (this.unsubscribe(), this.unsubscribe = null), this.modelName = e, e) {
			this.unsubscribe = q.onModelUpdate(e, (e) => {
				this._updateFromState(e);
			});
			let t = q.getState(e);
			t && t.status === "downloading" && this._updateFromState(t);
		}
	}
	destroy() {
		this.unsubscribe && (this.unsubscribe(), this.unsubscribe = null), this.hideTimeout && (clearTimeout(this.hideTimeout), this.hideTimeout = null);
	}
	_buildHTML() {
		return `
			<div class="${Y}__header">
				<div class="${Y}__title">
					<span class="${Y}__title-icon ${s.ICONS.DOWNLOAD}"></span>
					<span>Downloading</span>
					<span class="${Y}__xet-badge" style="display:none">XET</span>
				</div>
				<button class="${Y}__cancel-btn" title="Cancel download">
					<span class="${s.ICONS.CANCEL}" style="font-size:10px;margin-right:3px"></span>
					${s.DOWNLOAD_PROGRESS.CANCEL_LABEL}
				</button>
			</div>
			<div class="${Y}__bar-container">
				<div class="${Y}__bar-fill" style="width: 0%"></div>
			</div>
			<div class="${Y}__stats">
				<span class="${Y}__percentage">0%</span>
				<span class="${Y}__speed">--</span>
				<span class="${Y}__file-info"></span>
			</div>
			<div class="${Y}__status" style="display:none"></div>
		`;
	}
	_updateFromState(e) {
		switch (this.hideTimeout && (clearTimeout(this.hideTimeout), this.hideTimeout = null), e.status) {
			case "downloading":
				this._showDownloading(e);
				break;
			case "complete":
				this._showComplete(e);
				break;
			case "error":
				this._showError(e);
				break;
			case "cancelled":
				this._showCancelled(e);
				break;
			default:
				this._hide();
				break;
		}
	}
	_showDownloading(e) {
		this.container.className = Y, this.barFill.style.width = `${Math.min(e.percentage, 100)}%`, this.percentageEl.textContent = `${Math.round(e.percentage)}%`, this.speedEl.textContent = _e(e.speed);
		let t = e.totalFiles > 1 ? `file ${e.fileIndex + 1}/${e.totalFiles} • ${e.currentFile}` : e.currentFile;
		this.fileInfoEl.textContent = t, this.xetBadge.style.display = e.isXet ? "inline-flex" : "none", this.cancelBtn.disabled = !1, this.statusEl.style.display = "none";
	}
	_showComplete(e) {
		this.container.className = `${Y} ${Y}--complete`, this.barFill.style.width = "100%", this.percentageEl.textContent = "100%", this.speedEl.textContent = "", this.fileInfoEl.textContent = "", this.cancelBtn.disabled = !0, this.statusEl.style.display = "block", this.statusEl.textContent = "Download complete ✓", this._scheduleHide(3e3);
	}
	_showError(e) {
		this.container.className = `${Y} ${Y}--error`, this.cancelBtn.disabled = !0, this.statusEl.style.display = "block", this.statusEl.textContent = e.errorMessage || "Download failed", this._scheduleHide(5e3);
	}
	_showCancelled(e) {
		this.container.className = `${Y} ${Y}--cancelled`, this.cancelBtn.disabled = !0, this.statusEl.style.display = "block", this.statusEl.textContent = "Download cancelled", this._scheduleHide(3e3);
	}
	_hide() {
		this.container.className = `${Y} ${Y}--hidden`;
	}
	_scheduleHide(e) {
		this.hideTimeout = setTimeout(() => {
			this._hide(), this.hideTimeout = null;
		}, e);
	}
	async _onCancel() {
		this.modelName && (this.cancelBtn.disabled = !0, this.statusEl.style.display = "block", this.statusEl.textContent = "Cancelling...", await q.cancelDownload(this.modelName) || (this.cancelBtn.disabled = !1, this.statusEl.textContent = "Cancel failed"));
	}
};
function ye(e) {
	let t = new ve(), n = e.addDOMWidget(ge, s.DOWNLOAD_PROGRESS.WIDGET_TYPE, t.element, {
		getValue: () => "",
		setValue: () => {},
		minHeight: s.DOWNLOAD_PROGRESS.MIN_HEIGHT
	});
	return n.serialize = !1, e._voxcpmDownloadProgress || (e._voxcpmDownloadProgress = t), {
		widget: n,
		progressWidget: t
	};
}
function Z(e, t) {
	let n = e._voxcpmDownloadProgress;
	n && n.setModelName(t);
}

ne();
function be() {
	try {
		let e = localStorage.getItem(r.NORMALIZATION_AVAILABLE);
		if (e !== null) return e === "true";
	} catch {}
	return null;
}
function xe(e) {
	try {
		localStorage.setItem(r.NORMALIZATION_AVAILABLE, String(e)), c.debug("Config stored to localStorage:", e);
	} catch (e) {
		c.warn("Failed to store config to localStorage:", e);
	}
}
var Q = {
	normalizationAvailable: !0,
	configReceived: !1,
	lastConfigValue: null,
	initialized: !1,
	init() {
		if (this.initialized) return;
		this.initialized = !0;
		let e = be();
		e !== null && (this.normalizationAvailable = e, c.debug("Loaded config from localStorage:", e));
	},
	updateNormalizationWidget(e) {
		this.initialized || this.init();
		let t = e.widgets?.find((e) => e.name === "normalize_text");
		return t ? this.normalizationAvailable ? (t.disabled && (c.debug("Re-enabling normalize_text widget for node:", e.id), t.disabled = !1, t.value = !0, t.options && (t.options.tooltip = void 0)), !1) : (t.disabled || (c.debug("Disabling normalize_text widget for node:", e.id), t.disabled = !0, t.value = !1, t.options = t.options || {}, t.options.tooltip = "Text normalization disabled: 'inflect' and 'wetext' packages not installed. Install with: pip install inflect wetext"), !0) : !1;
	},
	updateAllNodes() {
		this.initialized || this.init();
		let t = e.graph || e.rootGraph;
		if (t && t._nodes) for (let e of t._nodes) o.has(e.comfyClass) && this.updateNormalizationWidget(e);
	}
};
Q.init();
var $ = {
	name: "voxcpm.heavyExtension",
	async setup() {
		c.debug("Heavy extension loaded"), e.api.addEventListener(i.STATUS, ((e) => {
			this.handleStatusEvent(e.detail);
		})), e.api.addEventListener(i.CONFIG, ((e) => {
			this.handleConfigEvent(e.detail);
		})), e.api.addEventListener("reconnected", () => {
			c.debug("WebSocket reconnected, waiting for config..."), Q.configReceived = !1, B(), Q.lastConfigValue = null;
		}), e.api.addEventListener("status", () => {
			Q.configReceived || c.debug("Status received, waiting for config...");
		}), c.log("Event listeners registered");
		try {
			let t = await e.api.fetchApi("/voxcpm/model_info");
			if (t.ok) {
				let e = await t.json();
				e && typeof e.normalization_available == "boolean" && (Q.configReceived || (c.debug("Fetched config from API fallback:", e), this.handleConfigEvent(e)));
			}
		} catch (e) {
			c.debug("Config API fallback failed (non-critical):", e);
		}
	},
	nodeCreated(e) {
		if (!o.has(e.comfyClass)) return;
		c.debug("VoxCPM node created:", e.id), Q.updateNormalizationWidget(e), U(e), ye(e);
		let t = e.widgets;
		if (t) {
			let n = t.find((e) => e.name === "model_name");
			if (n) {
				let t = n.callback;
				n.callback = (r) => {
					Z(e, String(r || "")), t && t.call(n, r);
				}, n.value && Z(e, String(n.value));
			}
		}
	},
	loadedGraphNode(e) {
		o.has(e.comfyClass) && (c.debug("VoxCPM node loaded from workflow:", e.id), Q.updateNormalizationWidget(e), setTimeout(() => {
			pe(e);
			let t = e.widgets;
			if (t) {
				let n = t.find((e) => e.name === "model_name");
				n?.value && Z(e, String(n.value));
			}
		}, 60));
	},
	handleStatusEvent(e) {
		if (!e || !e.severity || !e.summary) {
			c.warn("Invalid status event: missing severity or summary", e);
			return;
		}
		D.show({
			severity: e.severity,
			summary: e.summary,
			detail: e.detail,
			life: e.life
		});
	},
	async handleConfigEvent(e) {
		if (c.log("Received config event:", e), !e) {
			c.warn("Config event has no detail");
			return;
		}
		if (typeof e.normalization_available == "boolean") {
			if (Q.lastConfigValue === e.normalization_available && Q.configReceived) {
				c.debug("Config already processed, skipping");
				return;
			}
			Q.normalizationAvailable = e.normalization_available, Q.configReceived = !0, Q.lastConfigValue = e.normalization_available, xe(e.normalization_available), c.debug("Text normalization available:", Q.normalizationAvailable), setTimeout(() => {
				Q.updateAllNodes();
			}, 100);
		}
		e.settings ? (c.log("Initializing settings manager with:", e.settings), await k.initialize(e)) : c.warn("No settings in config event");
	}
};
e.registerExtension($);

export { $ as VoxCPMHeavyExtension, $ as default };
