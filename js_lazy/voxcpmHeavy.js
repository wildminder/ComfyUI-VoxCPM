import { app as e } from "/scripts/app.js";
import { app as t } from "/scripts/app.js";
import { api as n } from "/scripts/api.js";

var r = {
	NOTIFICATION_SHOWN: "voxcpm.normalization_notification_shown",
	USER_PREFERENCES: "voxcpm.user_preferences",
	SETTINGS: "voxcpm.settings",
	FIRST_RUN_SHOWN: "voxcpm.first_run_shown",
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
	VoxCPM2: 8,
	"VoxCPM1.5": 1.6,
	"VoxCPM-0.5B": 1
};
function c(e) {
	return s[e] ?? 2;
}
var l = {
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
}, u = {
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
}, d = "\n\n:root {\n  \n  --voxcpm-space-2xs: 2px;\n  --voxcpm-space-xs: 4px;\n  --voxcpm-space-s: 6px;\n  --voxcpm-space-m: 8px;\n  --voxcpm-space-l: 12px;\n  --voxcpm-space-xl: 16px;\n  --voxcpm-space-2xl: 24px;\n\n  \n  --voxcpm-font-size-2xs: 10px;\n  --voxcpm-font-size-xs: 12px;\n  --voxcpm-font-size-s: 13px;\n  --voxcpm-font-size-m: 14px;\n  --voxcpm-font-size-l: 18px;\n  --voxcpm-font-size-xl: 24px;\n  --voxcpm-font-family: var(--comfy-font-family, Arial, sans-serif);\n  --voxcpm-font-family-mono: monospace;\n\n  \n  --voxcpm-bg-input: var(--comfy-input-bg, rgba(255, 255, 255, 0.05));\n  --voxcpm-bg-input-hover: var(--comfy-input-bg-hover, rgba(255, 255, 255, 0.08));\n  --voxcpm-bg-input-active: var(--comfy-input-bg-active, rgba(255, 255, 255, 0.1));\n  --voxcpm-bg-surface: var(--bg-color, #1e1e1e);\n  --voxcpm-bg-elevated: var(--input-bg, #2a2a2a);\n\n  \n  --voxcpm-border-color: var(--border-color, rgba(255, 255, 255, 0.15));\n  --voxcpm-border-color-hover: var(--comfy-input-border-hover, rgba(255, 255, 255, 0.3));\n  --voxcpm-border-radius-xs: 4px;\n  --voxcpm-border-radius-s: 6px;\n  --voxcpm-border-radius-m: 8px;\n\n  \n  --voxcpm-text-primary: var(--fg-color, #ddd);\n  --voxcpm-text-secondary: var(--fg-color, rgba(255, 255, 255, 0.7));\n  --voxcpm-text-muted: var(--fg-color, rgba(255, 255, 255, 0.4));\n  --voxcpm-text-on-primary: #fff;\n\n  \n  --voxcpm-accent: var(--primary-color, #4a9eff);\n  --voxcpm-accent-hover: var(--primary-color-hover, #3a8eef);\n\n  \n  --voxcpm-color-success: var(--success-color, #4caf50);\n  --voxcpm-color-error: var(--error-color, #f44336);\n\n  \n  --voxcpm-duration-fast: 150ms;\n  --voxcpm-duration-normal: 200ms;\n  --voxcpm-easing: ease;\n\n  \n  --voxcpm-widget-height: 44px;\n  --voxcpm-input-height: 26px;\n  --voxcpm-browse-width: 30px;\n\n  \n  --voxcpm-focus-ring-color: var(--voxcpm-accent);\n  --voxcpm-focus-ring-width: 2px;\n  --voxcpm-focus-ring-offset: 1px;\n\n  \n  --voxcpm-cyber-font: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;\n  --voxcpm-cyber-accent: #00ff9d;\n  --voxcpm-cyber-accent-dim: rgba(0, 255, 157, 0.15);\n  --voxcpm-cyber-border: rgba(0, 255, 157, 0.3);\n  --voxcpm-cyber-glow: 0 0 8px rgba(0, 255, 157, 0.4);\n  --voxcpm-cyber-text: #e0e0e0;\n  --voxcpm-cyber-text-dim: rgba(224, 224, 224, 0.5);\n  --voxcpm-cyber-bg: rgba(10, 15, 20, 0.95);\n  --voxcpm-cyber-bg-hover: rgba(0, 255, 157, 0.08);\n  --voxcpm-cyber-bg-active: rgba(0, 255, 157, 0.12);\n  --voxcpm-cyber-tag-bg: rgba(0, 255, 157, 0.15);\n  --voxcpm-cyber-tag-border: rgba(0, 255, 157, 0.4);\n  --voxcpm-cyber-tag-text: #00ff9d;\n  --voxcpm-cyber-scanline: rgba(0, 255, 157, 0.03);\n}", f = "voxcpm-design-tokens", p = !1;
function m() {
	if (p) return;
	if (document.getElementById(f)) {
		p = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = f, e.textContent = d, document.head.appendChild(e), p = !0;
}

var h = {
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
}, g = `

.${h.BLOCK} {
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

.${h.ROW} {
  display: flex;
  align-items: center;
  gap: var(--voxcpm-space-xs);
  width: 100%;
  box-sizing: border-box;
}

.${h.DISPLAY} {
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

.${h.DISPLAY}:hover {
  border-color: var(--voxcpm-cyber-accent);
  box-shadow: var(--voxcpm-cyber-glow);
}

.${h.DISPLAY}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${h.ICON} {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  color: var(--voxcpm-cyber-accent);
}

.${h.ICON} svg {
  width: 16px;
  height: 16px;
}

.${h.TEXT} {
  flex: 1;
  color: var(--voxcpm-cyber-text);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
}

.${h.ARROW} {
  font-size: 9px;
  color: var(--voxcpm-cyber-accent);
  margin-left: auto;
  flex-shrink: 0;
  line-height: 1;
  opacity: 0.3;
  transition: opacity var(--voxcpm-duration-fast) var(--voxcpm-easing);
}

.${h.DISPLAY}:hover .${h.ARROW},

.${h.ARROW_ACTIVE} {
  opacity: 1;
}

.${h.BROWSE} {
  
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

.${h.BROWSE}:hover {
  border-color: var(--voxcpm-cyber-accent);
  box-shadow: var(--voxcpm-cyber-glow);
}

.${h.BROWSE}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${h.PATH} {
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

.${h.PATH_VISIBLE} {
  display: block;
}

.${h.DISPLAY}:focus-visible,
.${h.BROWSE}:focus-visible {
  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-cyber-accent);
  outline-offset: var(--voxcpm-focus-ring-offset);
}

@media (prefers-reduced-motion: reduce) {
  .${h.BLOCK} *,
  .${h.BLOCK} *::before,
  .${h.BLOCK} *::after {
    transition-duration: 0ms !important;
  }
}
`.trim(), _ = "voxcpm-model-selector-styles", v = !1;
function ee() {
	if (v) return;
	if (document.getElementById(_)) {
		v = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = _, e.textContent = g, document.head.appendChild(e), v = !0;
}

var y = {
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
}, te = `

.${y.BLOCK} {
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

.${y.BLOCK}::after {
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

.${y.HEADER} {
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

.${y.HEADER} svg {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
}

.${y.LIST} {
  overflow-y: auto;
  overflow-x: hidden;
  max-height: 280px;
  position: relative;
  z-index: 2;
}

.${y.LIST}::-webkit-scrollbar {
  width: 4px;
}

.${y.LIST}::-webkit-scrollbar-track {
  background: transparent;
}

.${y.LIST}::-webkit-scrollbar-thumb {
  background: var(--voxcpm-cyber-border);
  border-radius: 0;
}

.${y.LIST}::-webkit-scrollbar-thumb:hover {
  background: var(--voxcpm-cyber-accent);
}

.${y.ITEM} {
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

.${y.ITEM}:hover,
.${y.ITEM_SELECTED} {
  background: var(--voxcpm-cyber-bg-hover);
  border-left-color: var(--voxcpm-cyber-accent);
}

.${y.ITEM}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${y.ICON} {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.${y.ICON} svg {
  width: 16px;
  height: 16px;
}

.${y.ICON_CLOUD} {
  color: var(--voxcpm-accent);
}

.${y.ICON_CHECK} {
  color: var(--voxcpm-color-success);
}

.${y.NAME} {
  flex: 0 1 auto;
  color: var(--voxcpm-cyber-text);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
  min-width: 0;
}

.${y.META} {
  font-size: 10px;
  color: var(--voxcpm-cyber-text-dim);
  font-family: var(--voxcpm-cyber-font);
  flex-shrink: 0;
  margin-left: 6px;
}

.${y.TAG} {
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
  .${y.BLOCK} {
    animation: none;
  }
}
`.trim(), b = "voxcpm-model-dropdown-styles", x = !1;
function ne() {
	if (x) return;
	if (document.getElementById(b)) {
		x = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = b, e.textContent = te, document.head.appendChild(e), x = !0;
}

var S = !1;
function re() {
	S || (m(), ee(), ne(), S = !0);
}

function C(e) {
	try {
		return sessionStorage.getItem(e);
	} catch {
		return null;
	}
}
function w(e, t) {
	try {
		return sessionStorage.setItem(e, t), !0;
	} catch {
		return !1;
	}
}
function ie(e) {
	try {
		return sessionStorage.removeItem(e), !0;
	} catch {
		return !1;
	}
}
function T(e) {
	return C(e) !== null;
}
function E(e) {
	return w(e, "true");
}
function ae(e) {
	return ie(e);
}
function D(e) {
	let t = C(e);
	if (t === null) return null;
	try {
		return JSON.parse(t);
	} catch {
		return null;
	}
}
function O(e, t) {
	try {
		return w(e, JSON.stringify(t));
	} catch {
		return !1;
	}
}
var k = new class {
	constructor(e) {
		this.sessionKey = e.sessionKey, this.defaultLife = e.defaultLife;
	}
	show(e) {
		if (T(this.sessionKey)) return u.log("Notification already shown this session, skipping"), {
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
			}), E(this.sessionKey), u.log("Toast notification displayed:", e.summary), { shown: !0 };
		} catch (t) {
			return u.warn("Toast not available:", t), u.log(`${e.summary}: ${e.detail ?? ""}`), E(this.sessionKey), {
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
			}), u.log("Toast notification displayed:", e.summary), { shown: !0 };
		} catch (t) {
			return u.warn("Toast not available:", t), u.log(`${e.summary}: ${e.detail ?? ""}`), {
				shown: !1,
				reason: "toast_unavailable"
			};
		}
	}
	reset() {
		ae(this.sessionKey);
	}
	wasShown() {
		return T(this.sessionKey);
	}
}({
	sessionKey: "voxcpm.normalization_notification_shown",
	defaultLife: 1e4
}), A = "voxcpm.settings", j = new class {
	constructor() {
		this.settings = null, this.initialized = !1;
	}
	async initialize(e) {
		this.settings = e.settings, this.initialized = !0, O(A, this.settings), u.log("Settings initialized:", this.settings);
	}
	getSettings() {
		if (this.settings) return this.settings;
		let e = D(A);
		return e ? (this.settings = e, this.settings) : null;
	}
	isFirstRun() {
		return this.getSettings()?.first_run ?? !0;
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
			return this.settings && (Object.assign(this.settings, e), O(A, this.settings)), u.log("Settings updated:", e), !0;
		} catch (e) {
			return u.warn("Failed to update settings:", e), !1;
		}
	}
	async markFirstRunComplete() {
		return this.updateSettings({ first_run: !1 });
	}
	async setCustomPath(e) {
		return this.updateSettings({
			use_custom_path: !0,
			custom_model_path: e
		});
	}
	async setDefaultPath() {
		return this.updateSettings({
			use_custom_path: !1,
			custom_model_path: null
		});
	}
	isInitialized() {
		return this.initialized;
	}
	async validateModelPath(e) {
		return await (await t.api.fetchApi(`/voxcpm/validate_model_path?model_name=${encodeURIComponent(e)}`)).json();
	}
}(), M = "voxcpm-download-path-dialog-styles", N = !1;
function oe() {
	if (N) return;
	if (document.getElementById(M)) {
		N = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = M, e.textContent = "\n\n\n\n.voxcpm-dpd__overlay {\n	position: fixed;\n	top: 0;\n	left: 0;\n	right: 0;\n	bottom: 0;\n	background: rgba(0, 0, 0, 0.75);\n	display: flex;\n	align-items: center;\n	justify-content: center;\n	z-index: 10000;\n	backdrop-filter: blur(4px);\n}\n\n\n.voxcpm-dpd {\n	background: var(--voxcpm-cyber-bg, rgba(10, 15, 20, 0.95));\n	border: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	border-radius: 8px;\n	max-width: 520px;\n	width: 90%;\n	box-shadow: var(--voxcpm-cyber-glow, 0 0 8px rgba(0, 255, 157, 0.4)),\n				0 8px 32px rgba(0, 0, 0, 0.6);\n	font-family: var(--voxcpm-cyber-font, 'JetBrains Mono', monospace);\n	color: var(--voxcpm-cyber-text, #e0e0e0);\n	overflow: hidden;\n}\n\n\n.voxcpm-dpd__header {\n	display: flex;\n	align-items: center;\n	gap: 12px;\n	padding: 16px 20px;\n	border-bottom: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	background: var(--voxcpm-cyber-scanline, rgba(0, 255, 157, 0.03));\n}\n\n.voxcpm-dpd__header-icon {\n	font-size: 18px;\n	color: var(--voxcpm-cyber-accent, #00ff9d);\n}\n\n.voxcpm-dpd__header-title {\n	font-size: 14px;\n	font-weight: 700;\n	letter-spacing: 0.15em;\n	color: var(--voxcpm-cyber-accent, #00ff9d);\n	text-transform: uppercase;\n}\n\n\n.voxcpm-dpd__body {\n	padding: 20px;\n}\n\n\n.voxcpm-dpd__model-info {\n	font-size: 13px;\n	line-height: 1.6;\n	margin-bottom: 20px;\n	color: var(--voxcpm-cyber-text, #e0e0e0);\n}\n\n.voxcpm-dpd__model-name {\n	color: var(--voxcpm-cyber-accent, #00ff9d);\n	font-weight: 600;\n}\n\n.voxcpm-dpd__model-size {\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n}\n\n\n.voxcpm-dpd__options {\n	display: flex;\n	flex-direction: column;\n	gap: 10px;\n}\n\n\n.voxcpm-dpd__option {\n	display: flex;\n	gap: 14px;\n	padding: 14px 16px;\n	background: transparent;\n	border: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	border-radius: 6px;\n	cursor: pointer;\n	text-align: left;\n	transition: border-color 150ms ease,\n				background 150ms ease,\n				box-shadow 150ms ease;\n	width: 100%;\n	font-family: inherit;\n	color: inherit;\n}\n\n.voxcpm-dpd__option:hover {\n	border-color: var(--voxcpm-cyber-accent, #00ff9d);\n	background: var(--voxcpm-cyber-bg-hover, rgba(0, 255, 157, 0.08));\n}\n\n.voxcpm-dpd__option--selected {\n	border-color: var(--voxcpm-cyber-accent, #00ff9d);\n	background: var(--voxcpm-cyber-bg-active, rgba(0, 255, 157, 0.12));\n	box-shadow: inset 0 0 12px var(--voxcpm-cyber-accent-dim, rgba(0, 255, 157, 0.15));\n}\n\n\n.voxcpm-dpd__option-icon {\n	font-size: 24px;\n	flex-shrink: 0;\n	line-height: 1;\n}\n\n\n.voxcpm-dpd__option-content {\n	flex: 1;\n	min-width: 0;\n}\n\n.voxcpm-dpd__option-title {\n	font-size: 12px;\n	font-weight: 700;\n	letter-spacing: 0.12em;\n	color: var(--voxcpm-cyber-text, #e0e0e0);\n	margin-bottom: 4px;\n	text-transform: uppercase;\n}\n\n.voxcpm-dpd__option-path {\n	font-size: 11px;\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n	margin-bottom: 4px;\n	word-break: break-all;\n	font-family: var(--voxcpm-cyber-font, 'JetBrains Mono', monospace);\n}\n\n.voxcpm-dpd__option-desc {\n	font-size: 11px;\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n	line-height: 1.4;\n}\n\n\n.voxcpm-dpd__footer {\n	display: flex;\n	justify-content: flex-end;\n	gap: 10px;\n	padding: 14px 20px;\n	border-top: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n}\n\n\n.voxcpm-dpd__btn {\n	padding: 8px 20px;\n	border-radius: 4px;\n	font-size: 12px;\n	font-weight: 600;\n	letter-spacing: 0.08em;\n	cursor: pointer;\n	transition: background 150ms ease,\n				border-color 150ms ease,\n				box-shadow 150ms ease;\n	font-family: var(--voxcpm-cyber-font, 'JetBrains Mono', monospace);\n	text-transform: uppercase;\n}\n\n.voxcpm-dpd__btn--secondary {\n	background: transparent;\n	border: 1px solid var(--voxcpm-cyber-border, rgba(0, 255, 157, 0.3));\n	color: var(--voxcpm-cyber-text-dim, rgba(224, 224, 224, 0.5));\n}\n\n.voxcpm-dpd__btn--secondary:hover {\n	border-color: var(--voxcpm-cyber-accent, #00ff9d);\n	color: var(--voxcpm-cyber-text, #e0e0e0);\n}\n\n.voxcpm-dpd__btn--primary {\n	background: var(--voxcpm-cyber-accent, #00ff9d);\n	border: none;\n	color: #0a0f14;\n}\n\n.voxcpm-dpd__btn--primary:hover:not(:disabled) {\n	box-shadow: 0 0 12px var(--voxcpm-cyber-accent-dim, rgba(0, 255, 157, 0.15));\n}\n\n.voxcpm-dpd__btn--primary:disabled {\n	opacity: 0.4;\n	cursor: not-allowed;\n}\n\n\n.voxcpm-dpd__option:focus-visible,\n.voxcpm-dpd__btn:focus-visible {\n	outline: 2px solid var(--voxcpm-cyber-accent, #00ff9d);\n	outline-offset: 2px;\n}\n\n\n@media (prefers-reduced-motion: reduce) {\n	.voxcpm-dpd *,\n	.voxcpm-dpd *::before,\n	.voxcpm-dpd *::after {\n		transition-duration: 0ms !important;\n	}\n}\n", document.head.appendChild(e), N = !0;
}

var P = null;
function se(e, t, n) {
	return new Promise((r) => {
		F(), oe();
		let i = document.createElement("div");
		i.className = "voxcpm-dpd__overlay";
		let a = document.createElement("div");
		a.className = "voxcpm-dpd";
		let o = document.createElement("div");
		o.className = "voxcpm-dpd__header", o.innerHTML = "\n			<span class=\"voxcpm-dpd__header-icon\">⬇</span>\n			<span class=\"voxcpm-dpd__header-title\">MODEL REQUIRED</span>\n		";
		let s = document.createElement("div");
		s.className = "voxcpm-dpd__body";
		let c = document.createElement("div");
		c.className = "voxcpm-dpd__model-info", c.innerHTML = `
			Model <span class="voxcpm-dpd__model-name">"${e}"</span>
			<span class="voxcpm-dpd__model-size">(${t.toFixed(1)} GB)</span>
			needs to be downloaded.
		`;
		let l = document.createElement("div");
		l.className = "voxcpm-dpd__options";
		let d = null, f = document.createElement("button");
		f.className = "voxcpm-dpd__option", f.setAttribute("data-option", "download_default"), f.innerHTML = `
			<div class="voxcpm-dpd__option-icon">📁</div>
			<div class="voxcpm-dpd__option-content">
				<div class="voxcpm-dpd__option-title">DEFAULT PATH</div>
				<div class="voxcpm-dpd__option-path">${n}</div>
				<div class="voxcpm-dpd__option-desc">Download to ComfyUI's default model directory</div>
			</div>
		`;
		let p = document.createElement("button");
		p.className = "voxcpm-dpd__option", p.setAttribute("data-option", "download_custom"), p.innerHTML = "\n			<div class=\"voxcpm-dpd__option-icon\">📂</div>\n			<div class=\"voxcpm-dpd__option-content\">\n				<div class=\"voxcpm-dpd__option-title\">CUSTOM PATH</div>\n				<div class=\"voxcpm-dpd__option-desc\">Choose an existing model directory or download to a different location</div>\n			</div>\n		";
		let m = [f, p];
		m.forEach((e) => {
			e.addEventListener("click", () => {
				m.forEach((e) => e.classList.remove("voxcpm-dpd__option--selected")), e.classList.add("voxcpm-dpd__option--selected"), d = e.getAttribute("data-option"), _.disabled = !1;
			});
		}), l.appendChild(f), l.appendChild(p), s.appendChild(c), s.appendChild(l);
		let h = document.createElement("div");
		h.className = "voxcpm-dpd__footer";
		let g = document.createElement("button");
		g.className = "voxcpm-dpd__btn voxcpm-dpd__btn--secondary", g.textContent = "Cancel";
		let _ = document.createElement("button");
		_.className = "voxcpm-dpd__btn voxcpm-dpd__btn--primary", _.textContent = "Continue", _.disabled = !0, g.addEventListener("click", () => {
			F(), r({ action: "cancel" });
		}), _.addEventListener("click", async () => {
			if (d) {
				if (d === "download_default") F(), r({ action: "download_default" });
				else if (d === "download_custom") {
					let e = await L();
					e && (F(), r({
						action: "download_custom",
						customPath: e
					}));
				}
			}
		}), i.addEventListener("click", (e) => {
			e.target === i && (F(), r({ action: "cancel" }));
		});
		let v = (e) => {
			e.key === "Escape" && (document.removeEventListener("keydown", v), F(), r({ action: "cancel" }));
		};
		document.addEventListener("keydown", v), h.appendChild(g), h.appendChild(_), a.appendChild(o), a.appendChild(s), a.appendChild(h), i.appendChild(a), document.body.appendChild(i), P = i, u.log("Download path dialog shown for model:", e);
	});
}
function F() {
	P && (P.remove(), P = null);
}

var I = null;
function L(e) {
	return new Promise((t) => {
		R();
		let n = document.createElement("div");
		n.className = "voxcpm-dialog__overlay", n.innerHTML = `
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
    value="${e || ""}"
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
		let r = !1, i = e || "", a = n.querySelector("#voxcpm-path-input"), o = n.querySelector("#voxcpm-models-preview"), s = n.querySelector("#voxcpm-models-list"), c = n.querySelector("#voxcpm-model-count"), l = n.querySelector("#voxcpm-validation-message"), d = n.querySelector("#voxcpm-confirm");
		async function f(e) {
			if (!e) {
				o.style.display = "none", l.textContent = "", d.disabled = !0, r = !1;
				return;
			}
			try {
				let { app: t } = await import("/scripts/app.js"), n = await (await t.api.fetchApi(`/voxcpm/validate_path?path=${encodeURIComponent(e)}`)).json();
				r = n.valid, n.valid && n.models?.length > 0 ? (o.style.display = "block", c.textContent = n.models.length.toString(), s.innerHTML = n.models.map((e) => `
          <li>
          <span class="voxcpm-dialog__model-icon">📁</span>
          <span class="voxcpm-dialog__model-name">${e.name}</span>
          </li>
          `).join(""), l.innerHTML = `<span class="voxcpm-dialog__validation--success">Found ${n.models.length} VoxCPM-compatible models</span>`, d.disabled = !1) : (o.style.display = "none", l.innerHTML = `<span class="voxcpm-dialog__validation--error">${n.error || "Invalid path or no models found"}</span>`, d.disabled = !0);
			} catch (e) {
				u.warn("Path validation failed:", e), o.style.display = "none", l.innerHTML = "<span class=\"voxcpm-dialog__validation--error\">Failed to validate path</span>", d.disabled = !0, r = !1;
			}
		}
		a?.addEventListener("input", () => {
			i = a.value, f(i);
		}), n.querySelector("#voxcpm-browse")?.addEventListener("click", () => {
			alert("Please enter the path manually. Native folder picker requires server support.");
		}), n.querySelector("#voxcpm-cancel")?.addEventListener("click", () => {
			R(), t(null);
		}), d?.addEventListener("click", () => {
			r && i && (R(), t(i));
		}), document.body.appendChild(n), I = n, ce(), e && f(e);
	});
}
function R() {
	I && (I.remove(), I = null);
}
function ce() {
	let e = "voxcpm-dialog-styles";
	if (document.getElementById(e)) return;
	let t = document.createElement("style");
	t.id = e, t.textContent = "\n\n\n\n.voxcpm-dialog__overlay {\n  position: fixed;\n  top: 0;\n  left: 0;\n  right: 0;\n  bottom: 0;\n  background: rgba(0, 0, 0, 0.7);\n  display: flex;\n  align-items: center;\n  justify-content: center;\n  z-index: 10000;\n}\n\n\n.voxcpm-dialog {\n  background: var(--voxcpm-bg-surface);\n  border-radius: var(--voxcpm-border-radius-m);\n  max-width: 500px;\n  width: 90%;\n  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);\n}\n\n.voxcpm-dialog--large {\n  max-width: 600px;\n}\n\n\n.voxcpm-dialog__header {\n  display: flex;\n  align-items: center;\n  gap: var(--voxcpm-space-l);\n  padding: var(--voxcpm-space-2xl);\n  border-bottom: 1px solid var(--voxcpm-border-color);\n}\n\n.voxcpm-dialog__icon {\n  font-size: var(--voxcpm-font-size-xl);\n}\n\n.voxcpm-dialog__title {\n  margin: 0;\n  font-size: var(--voxcpm-font-size-l);\n  color: var(--voxcpm-text-primary);\n}\n\n\n.voxcpm-dialog__body {\n  padding: var(--voxcpm-space-2xl);\n}\n\n.voxcpm-dialog__description {\n  margin: 0 0 20px;\n  color: var(--voxcpm-text-secondary);\n  line-height: 1.5;\n}\n\n\n.voxcpm-dialog__options {\n  display: flex;\n  flex-direction: column;\n  gap: var(--voxcpm-space-l);\n}\n\n\n.voxcpm-dialog__option {\n  display: flex;\n  gap: var(--voxcpm-space-xl);\n  padding: var(--voxcpm-space-xl);\n  background: var(--voxcpm-bg-elevated);\n  border: 2px solid var(--voxcpm-border-color);\n  border-radius: var(--voxcpm-border-radius-m);\n  cursor: pointer;\n  text-align: left;\n  transition: border-color var(--voxcpm-duration-normal) var(--voxcpm-easing),\n              background var(--voxcpm-duration-normal) var(--voxcpm-easing);\n}\n\n.voxcpm-dialog__option:hover {\n  border-color: var(--voxcpm-accent);\n}\n\n.voxcpm-dialog__option--selected {\n  border-color: var(--voxcpm-accent);\n  background: var(--voxcpm-bg-input-hover);\n}\n\n.voxcpm-dialog__option-icon {\n  font-size: 32px;\n  flex-shrink: 0;\n}\n\n.voxcpm-dialog__option-content h3 {\n  margin: 0 0 var(--voxcpm-space-xs);\n  color: var(--voxcpm-text-primary);\n  font-size: var(--voxcpm-font-size-m);\n}\n\n.voxcpm-dialog__option-path {\n  margin: 0 0 var(--voxcpm-space-m);\n  font-family: var(--voxcpm-font-family-mono);\n  font-size: var(--voxcpm-font-size-xs);\n  color: var(--voxcpm-text-muted);\n}\n\n.voxcpm-dialog__option-description {\n  margin: 0;\n  font-size: var(--voxcpm-font-size-xs);\n  color: var(--voxcpm-text-muted);\n}\n\n\n.voxcpm-dialog__footer {\n  display: flex;\n  justify-content: flex-end;\n  gap: var(--voxcpm-space-l);\n  padding: var(--voxcpm-space-xl) var(--voxcpm-space-2xl);\n  border-top: 1px solid var(--voxcpm-border-color);\n}\n\n\n.voxcpm-dialog__btn {\n  padding: var(--voxcpm-space-m) var(--voxcpm-space-xl);\n  border-radius: var(--voxcpm-border-radius-xs);\n  font-size: var(--voxcpm-font-size-m);\n  cursor: pointer;\n  transition: background var(--voxcpm-duration-normal) var(--voxcpm-easing),\n              border-color var(--voxcpm-duration-normal) var(--voxcpm-easing);\n}\n\n.voxcpm-dialog__btn--secondary {\n  background: transparent;\n  border: 1px solid var(--voxcpm-border-color);\n  color: var(--voxcpm-text-secondary);\n}\n\n.voxcpm-dialog__btn--secondary:hover {\n  background: var(--voxcpm-bg-elevated);\n}\n\n.voxcpm-dialog__btn--primary {\n  background: var(--voxcpm-accent);\n  border: none;\n  color: var(--voxcpm-text-on-primary);\n}\n\n.voxcpm-dialog__btn--primary:hover:not(:disabled) {\n  background: var(--voxcpm-accent-hover);\n}\n\n.voxcpm-dialog__btn--primary:disabled {\n  opacity: 0.5;\n  cursor: not-allowed;\n}\n\n\n.voxcpm-dialog__input-group {\n  display: flex;\n  gap: var(--voxcpm-space-m);\n  margin-bottom: var(--voxcpm-space-xl);\n}\n\n.voxcpm-dialog__input {\n  flex: 1;\n  padding: var(--voxcpm-space-m) var(--voxcpm-space-l);\n  background: var(--voxcpm-bg-elevated);\n  border: 1px solid var(--voxcpm-border-color);\n  border-radius: var(--voxcpm-border-radius-xs);\n  color: var(--voxcpm-text-primary);\n  font-size: var(--voxcpm-font-size-m);\n  font-family: var(--voxcpm-font-family);\n}\n\n.voxcpm-dialog__input:focus {\n  outline: none;\n  border-color: var(--voxcpm-accent);\n}\n\n\n.voxcpm-dialog__preview {\n  background: var(--voxcpm-bg-elevated);\n  border-radius: var(--voxcpm-border-radius-xs);\n  padding: var(--voxcpm-space-l);\n  margin-bottom: var(--voxcpm-space-xl);\n}\n\n.voxcpm-dialog__preview h4 {\n  margin: 0 0 var(--voxcpm-space-m);\n  font-size: var(--voxcpm-font-size-xs);\n  color: var(--voxcpm-text-muted);\n}\n\n.voxcpm-dialog__models-list {\n  list-style: none;\n  margin: 0;\n  padding: 0;\n}\n\n.voxcpm-dialog__models-list li {\n  display: flex;\n  align-items: center;\n  gap: var(--voxcpm-space-m);\n  padding: var(--voxcpm-space-xs) 0;\n}\n\n.voxcpm-dialog__model-icon {\n  font-size: var(--voxcpm-font-size-m);\n}\n\n.voxcpm-dialog__model-name {\n  font-size: var(--voxcpm-font-size-s);\n  color: var(--voxcpm-text-secondary);\n}\n\n\n.voxcpm-dialog__validation {\n  font-size: var(--voxcpm-font-size-xs);\n}\n\n.voxcpm-dialog__validation--success {\n  color: var(--voxcpm-color-success);\n}\n\n.voxcpm-dialog__validation--error {\n  color: var(--voxcpm-color-error);\n}\n\n\n.voxcpm-dialog__option:focus-visible,\n.voxcpm-dialog__btn:focus-visible,\n.voxcpm-dialog__input:focus-visible {\n  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-focus-ring-color);\n  outline-offset: var(--voxcpm-focus-ring-offset);\n}\n\n\n@media (prefers-reduced-motion: reduce) {\n  .voxcpm-dialog *,\n  .voxcpm-dialog *::before,\n  .voxcpm-dialog *::after {\n    transition-duration: 0ms !important;\n  }\n}\n\n\n\n.voxcpm-path-indicator {\n  display: flex;\n  align-items: center;\n  gap: var(--voxcpm-space-m);\n  padding: var(--voxcpm-space-xs) var(--voxcpm-space-m);\n  background: var(--voxcpm-bg-elevated);\n  border-radius: var(--voxcpm-border-radius-xs);\n  font-size: var(--voxcpm-font-size-xs);\n}\n\n.voxcpm-path-indicator__text {\n  color: var(--voxcpm-text-muted);\n  font-family: var(--voxcpm-font-family-mono);\n}\n\n.voxcpm-path-indicator__change-btn {\n  background: transparent;\n  border: none;\n  cursor: pointer;\n  font-size: var(--voxcpm-font-size-m);\n  padding: var(--voxcpm-space-2xs);\n  color: var(--voxcpm-text-primary);\n  transition: opacity var(--voxcpm-duration-fast) var(--voxcpm-easing);\n}\n\n.voxcpm-path-indicator__change-btn:hover {\n  opacity: 0.8;\n}\n\n.voxcpm-path-indicator__change-btn:focus-visible {\n  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-focus-ring-color);\n  outline-offset: var(--voxcpm-focus-ring-offset);\n}\n", document.head.appendChild(t);
}

var z = {
	CLOUD: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z\"/><polyline points=\"12 12 12 22\"/><path d=\"m8 18 4 4 4-4\"/></svg>",
	CHECK: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><polyline points=\"20 6 9 17 4 12\"/></svg>",
	SHIELD: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z\"/></svg>",
	FOLDER: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z\"/></svg>",
	FOLDER_OPEN: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M5 19a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4l2 2h4a2 2 0 0 1 2 2v1\"/><path d=\"M6 12h14l-2.5 7H8.5L6 12z\"/></svg>"
}, B = null;
async function le(e) {
	if (B !== null) return B;
	try {
		let t = await e("/voxcpm/model_info");
		if (t.ok) return B = (await t.json()).models || [], B;
	} catch (e) {
		console.warn("[VoxCPM] Failed to fetch model info:", e);
	}
	return [];
}
function ue() {
	B = null;
}
var V = class e {
	constructor(e, t, n, r) {
		this.selectedIndex = -1, this.resolveSelection = null, this.cleanups = [], this.isOpen = !1, this.modelInfo = e, this.anchorRect = t, this.scale = n, this.widgetElement = r || null, this.panel = document.createElement("div"), this.panel.className = y.BLOCK, this.panel.setAttribute("role", "listbox"), this.panel.setAttribute("aria-label", "Select model"), this.header = document.createElement("div"), this.header.className = y.HEADER, this.header.innerHTML = `${z.SHIELD} <span>${l.MODEL_DROPDOWN.HEADER_TEXT}</span>`, this.list = document.createElement("div"), this.list.className = y.LIST, this.panel.appendChild(this.header), this.panel.appendChild(this.list), this.renderItems();
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
		n.className = y.ITEM, n.setAttribute("role", "option"), n.setAttribute("data-index", String(t)), n.setAttribute("data-model", e.name);
		let r = document.createElement("span"), i = e.is_downloaded ? y.ICON_CHECK : y.ICON_CLOUD;
		r.className = `${y.ICON} ${i}`, r.innerHTML = e.is_downloaded ? z.CHECK : z.CLOUD, n.appendChild(r);
		let a = document.createElement("span");
		if (a.className = y.NAME, a.textContent = e.name, n.appendChild(a), e.size_gb > 0) {
			let t = document.createElement("span");
			t.className = y.META, t.textContent = `${e.size_gb} GB`, n.appendChild(t);
		}
		if (e.type === "official") {
			let e = document.createElement("span");
			e.className = y.TAG, e.textContent = "DEFAULT", n.appendChild(e);
		}
		return n.addEventListener("click", (t) => {
			t.stopPropagation(), this.close(e.name);
		}), n.addEventListener("mouseenter", () => {
			this.setSelectedIndex(t);
		}), n;
	}
	positionPanel() {
		let e = this.scale, t = l.MODEL_DROPDOWN.ANCHOR_GAP, n = window.innerHeight, r = window.innerWidth, i = l.MODEL_DROPDOWN.ITEM_HEIGHT, a = l.MODEL_DROPDOWN.MAX_VISIBLE_ITEMS, o = (32 + Math.min(this.modelInfo.length, a) * i) * e, s = 300 * e, c, u = n - this.anchorRect.bottom, d = this.anchorRect.top;
		c = u >= o + t ? this.anchorRect.bottom + t : d >= o + t ? this.anchorRect.top - o - t : this.anchorRect.bottom + t;
		let f = this.anchorRect.left;
		f + s > r && (f = r - s - 8), f = Math.max(8, f), this.panel.style.position = "fixed", this.panel.style.top = `${c}px`, this.panel.style.left = `${f}px`, this.panel.style.zIndex = String(l.MODEL_DROPDOWN.Z_INDEX), this.panel.style.transformOrigin = "top left", this.panel.style.transform = `scale(${e})`, this.panel.style.setProperty("--voxcpm-dropdown-scale", String(e));
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
		let t = this.list.querySelectorAll(`.${y.ITEM}`);
		if (t.forEach((e) => {
			e.classList.remove(y.ITEM_SELECTED);
		}), this.selectedIndex = e, e >= 0 && e < t.length) {
			let n = t[e];
			n.classList.add(y.ITEM_SELECTED), n.scrollIntoView({ block: "nearest" });
		}
	}
};
V.currentDropdown = null;

var H =  new Map();
function de(e) {
	let t = e.widgets;
	if (!t) {
		u.warn("No widgets found on node:", e.id);
		return;
	}
	let n = t.find((e) => e.name === "model_name");
	if (!n) {
		u.warn("model_name widget not found on node:", e.id);
		return;
	}
	fe(n);
	let r = document.createElement("div");
	r.className = h.BLOCK;
	let i = document.createElement("div");
	i.className = h.DISPLAY, i.title = "Click to select a model", i.tabIndex = 0, i.setAttribute("role", "combobox"), i.setAttribute("aria-expanded", "false"), i.setAttribute("aria-haspopup", "listbox");
	let a = document.createElement("span");
	a.className = h.ICON, a.innerHTML = z.FOLDER;
	let o = document.createElement("span");
	o.className = h.TEXT, o.textContent = n.value || l.MODEL_SELECTOR.PLACEHOLDER;
	let s = document.createElement("span");
	s.className = h.ARROW, s.textContent = l.MODEL_SELECTOR.ARROW;
	let c = document.createElement("button");
	c.className = h.BROWSE, c.innerHTML = z.FOLDER_OPEN, c.title = "Browse for custom model directory", c.setAttribute("role", "button");
	let d = document.createElement("div");
	d.className = h.PATH;
	let f = e.properties?.custom_model_path;
	f && (a.innerHTML = z.FOLDER_OPEN, d.textContent = f, d.classList.add(h.PATH_VISIBLE)), i.append(a, o, s);
	let p = document.createElement("div");
	p.className = h.ROW, p.append(i, c), r.append(p, d), r.addEventListener("mouseup", (e) => e.stopPropagation()), r.addEventListener("click", (e) => e.stopPropagation()), i.addEventListener("click", (t) => {
		if (t.preventDefault(), t.stopPropagation(), U()) {
			W();
			return;
		}
		G(t, n, o, a, d, e, i, s);
	}), i.addEventListener("keydown", (t) => {
		if (t.key === "Enter" || t.key === " ") {
			if (t.preventDefault(), t.stopPropagation(), U()) {
				W();
				return;
			}
			G(t, n, o, a, d, e, i, s);
		}
	}), c.addEventListener("click", async (t) => {
		t.preventDefault(), t.stopPropagation(), await pe(e, n, o, a, d);
	});
	let m = e.addDOMWidget("model_selector", "custom", r, {
		serialize: !1,
		hideOnZoom: !1,
		selectOn: ["click"],
		getValue: () => n.value,
		setValue: (e) => {
			n.value = e, o.textContent = e;
		}
	}), g = new MutationObserver(() => {
		let t = p.parentElement;
		if (t) {
			let n = t.parentElement;
			n && n !== document.body ? (n.classList.add(h.BLOCK), n.setAttribute("node-id", String(e.id)), u.debug("Added", h.BLOCK, "class + node-id to WidgetDOM parent div for node:", e.id)) : u.debug("No WidgetDOM grandparent found (LiteGraph mode) for node:", e.id), g.disconnect();
		}
	});
	g.observe(document.body, {
		childList: !0,
		subtree: !0
	}), m.tooltip = "Select the VoxCPM model to use", m.computeSize = () => [0, l.MODEL_SELECTOR.MIN_HEIGHT], m.computeLayoutSize = void 0;
	let _ = e.widgets;
	if (_ && _.length > 1) {
		let t = _.indexOf(m);
		t > 0 && (_.splice(t, 1), _.unshift(m), u.debug("Moved model selector widget to position 0 for node:", e.id));
	}
	let v = e.inputs?.find((e) => e.name === "model_name");
	v && (v.widget = {
		name: "model_selector",
		_originalWidget: v.widget?.name
	}, u.debug("Bound model_name input to model_selector widget for node:", e.id)), H.set(e.id, {
		modelText: o,
		folderIcon: a,
		pathIndicator: d,
		modelWidget: n
	}), u.debug("Created model selector widget for node:", e.id);
}
function fe(e) {
	e.type = "converted-widget", e.computeSize = () => [0, -4], e.draw = () => {}, e.options || (e.options = {}), e.options.canvasOnly = !0, u.debug("Hidden default combo widget:", e.name);
}
function U() {
	return V.currentDropdown !== null;
}
function W() {
	V.currentDropdown && V.currentDropdown.close(null);
}
async function G(e, r, i, a, o, s, c, l) {
	let d = r.options?.values || [];
	if (d.length === 0) {
		u.warn("No models available for dropdown"), k.show({
			severity: "warn",
			summary: "No Models Available",
			detail: "No models found. Try browsing for a custom model directory.",
			life: 4e3
		});
		return;
	}
	let f = [];
	try {
		f = await le(n.fetchApi.bind(n));
	} catch (e) {
		u.error("Failed to fetch model info:", e);
	}
	f.length === 0 && (f = d.map((e) => ({
		name: e,
		type: "local",
		architecture: "unknown",
		sample_rate: 0,
		size_gb: 0,
		is_downloaded: !0
	})));
	let p = c.getBoundingClientRect(), m = t.canvas.ds?.scale || 1;
	l.classList.add(h.ARROW_ACTIVE), c.setAttribute("aria-expanded", "true");
	try {
		let e = await new V(f, p, m, c).show();
		if (e) {
			r.value = e, i.textContent = e;
			let t = f.find((t) => t.name === e);
			t && (a.innerHTML = t.is_downloaded ? z.CHECK : z.CLOUD), r.callback && r.callback(e), s.setDirtyCanvas && s.setDirtyCanvas(!0), u.debug("Model selected:", e, "for node:", s.id);
		}
	} catch (e) {
		u.error("Failed to open model dropdown:", e);
	} finally {
		l.classList.remove(h.ARROW_ACTIVE), c.setAttribute("aria-expanded", "false");
	}
}
async function pe(e, t, r, i, a) {
	u.log("Browse button clicked for node:", e.id);
	let o = await L();
	if (!o) {
		u.debug("Browse cancelled by user");
		return;
	}
	u.log("User selected custom path:", o), e.properties = e.properties || {}, e.properties.custom_model_path = o;
	try {
		let s = await n.fetchApi("/voxcpm/models", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ path: o })
		});
		if (s.ok) {
			let n = await s.json();
			u.log("Models found at custom path:", n.models), me(t, n.models, e), i.innerHTML = z.FOLDER_OPEN, a.textContent = o, a.classList.add(h.PATH_VISIBLE), n.models.length > 0 ? r.textContent = t.value : r.textContent = "No models found", k.show({
				severity: "success",
				summary: "Custom Model Path Set",
				detail: `Found ${n.models.length} model(s) in: ${o}`,
				life: 5e3
			});
		} else {
			let e = s.statusText || "Unknown error";
			u.error("Failed to get models:", e), k.show({
				severity: "error",
				summary: "Error",
				detail: `Failed to scan model directory: ${e}`,
				life: 5e3
			});
		}
	} catch (e) {
		u.error("Error fetching models:", e), k.show({
			severity: "error",
			summary: "Error",
			detail: "Failed to connect to server",
			life: 5e3
		});
	}
}
function me(e, t, n) {
	e.options ? e.options.values = t : e.options = { values: t }, t.length > 0 && !t.includes(e.value) && (e.value = t[0], e.callback && e.callback(t[0])), n.setDirtyCanvas && n.setDirtyCanvas(!0), u.debug("Updated model options:", t);
}
function he(e, t) {
	let n = H.get(e.id);
	if (!n) {
		u.debug("No selector refs found for node:", e.id);
		return;
	}
	let r = t || n.modelWidget.value;
	n.modelText.textContent = r;
	let i = e.properties?.custom_model_path;
	i ? (n.folderIcon.innerHTML = z.FOLDER_OPEN, n.pathIndicator.textContent = i, n.pathIndicator.classList.add(h.PATH_VISIBLE)) : (n.folderIcon.innerHTML = z.FOLDER, n.pathIndicator.classList.remove(h.PATH_VISIBLE)), u.debug("Synced model selector display for node:", e.id, "value:", r);
}
var K = !1;
async function ge() {
	let e = t.graph || t.rootGraph;
	if (!e?._nodes) return null;
	let n = e._nodes.filter((e) => o.has(e.comfyClass));
	if (n.length === 0) return null;
	for (let e of n) {
		let t = e.widgets;
		if (!t) continue;
		let n = t.find((e) => e.name === "model_name");
		if (!n) continue;
		let r = String(n.value || "");
		if (!(!r || r.includes("No models found"))) try {
			let e = await j.validateModelPath(r);
			if (!e.valid) return {
				modelName: r,
				modelSizeGb: c(r),
				effectivePath: e.effective_path,
				error: e.error
			};
		} catch (e) {
			u.warn("Model path validation failed for", r, e);
		}
	}
	return null;
}
function _e() {
	if (K) return;
	K = !0;
	let e = t.queuePrompt;
	t.queuePrompt = async function(...t) {
		try {
			let e = await ge();
			if (e) {
				u.log("Model validation failed, showing download path dialog:", e);
				let t = await se(e.modelName, e.modelSizeGb, e.effectivePath);
				if (t.action === "cancel") {
					u.log("User cancelled download path dialog, aborting queue");
					return;
				}
				t.action === "download_custom" && t.customPath && (await j.setCustomPath(t.customPath), u.log("User selected custom path:", t.customPath)), t.action === "download_default" && (await j.setDefaultPath(), u.log("User selected default path")), await j.markFirstRunComplete();
			}
		} catch (e) {
			u.warn("Queue interception error (non-fatal):", e);
		}
		return e.apply(this, t);
	}, u.debug("Queue prompt intercepted for VoxCPM model validation");
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
			let t = await (await fetch(l.DOWNLOAD_PROGRESS.CANCEL_ENDPOINT, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model_name: e })
			})).json();
			return t.success ? (u.info(`Download cancelled: ${e}`), !0) : (u.warn(`Cancel failed: ${t.message}`), !1);
		} catch (t) {
			return u.error(`Failed to cancel download for '${e}':`, t), !1;
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
			})), u.debug("Download progress WebSocket listener registered");
		} catch (e) {
			u.warn("Failed to bind download progress WebSocket listener:", e);
		}
	}
	_handleProgressEvent(e) {
		let t = e.model_name || e.transfer_id;
		if (!t) {
			u.warn("Download progress event missing model_name:", e);
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
			u.error(`Download listener error for '${t}':`, e);
		}
		for (let t of this.globalListeners) try {
			t(e, n);
		} catch (e) {
			u.error("Global download listener error:", e);
		}
	}
}(), J = l.DOWNLOAD_PROGRESS.BLOCK_CLASS;
function ve() {
	let e = `${J}-styles`;
	if (document.getElementById(e)) return;
	let t = document.createElement("style");
	t.id = e, t.textContent = ye, document.head.appendChild(t);
}
var ye = `

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
	min-height: ${l.DOWNLOAD_PROGRESS.MIN_HEIGHT}px;
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
	transition: width ${l.DOWNLOAD_PROGRESS.ANIMATION_DURATION}ms ease-out;
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
`, Y = l.DOWNLOAD_PROGRESS.BLOCK_CLASS, be = l.DOWNLOAD_PROGRESS.WIDGET_NAME;
function xe(e) {
	if (e === 0) return "0 B";
	let t = [
		"B",
		"KB",
		"MB",
		"GB"
	], n = Math.floor(Math.log(e) / Math.log(1024));
	return `${(e / 1024 ** n).toFixed(+(n > 0))} ${t[n]}`;
}
function Se(e) {
	return e === 0 ? "--" : `${xe(e)}/s`;
}
var Ce = class {
	constructor() {
		this.modelName = "", this.unsubscribe = null, this.hideTimeout = null, ve(), this.element = document.createElement("div"), this.element.className = `${Y} ${Y}--hidden`, this.element.innerHTML = this._buildHTML(), this.container = this.element, this.barFill = this.element.querySelector(`.${Y}__bar-fill`), this.percentageEl = this.element.querySelector(`.${Y}__percentage`), this.speedEl = this.element.querySelector(`.${Y}__speed`), this.fileInfoEl = this.element.querySelector(`.${Y}__file-info`), this.cancelBtn = this.element.querySelector(`.${Y}__cancel-btn`), this.statusEl = this.element.querySelector(`.${Y}__status`), this.xetBadge = this.element.querySelector(`.${Y}__xet-badge`), this.cancelBtn.addEventListener("click", (e) => {
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
					<span class="${Y}__title-icon ${l.ICONS.DOWNLOAD}"></span>
					<span>Downloading</span>
					<span class="${Y}__xet-badge" style="display:none">XET</span>
				</div>
				<button class="${Y}__cancel-btn" title="Cancel download">
					<span class="${l.ICONS.CANCEL}" style="font-size:10px;margin-right:3px"></span>
					${l.DOWNLOAD_PROGRESS.CANCEL_LABEL}
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
		this.container.className = Y, this.barFill.style.width = `${Math.min(e.percentage, 100)}%`, this.percentageEl.textContent = `${Math.round(e.percentage)}%`, this.speedEl.textContent = Se(e.speed);
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
function X(e) {
	let t = new Ce(), n = e.addDOMWidget(be, l.DOWNLOAD_PROGRESS.WIDGET_TYPE, t.element, {
		getValue: () => "",
		setValue: () => {},
		minHeight: l.DOWNLOAD_PROGRESS.MIN_HEIGHT
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

re();
function we() {
	try {
		let e = localStorage.getItem(r.NORMALIZATION_AVAILABLE);
		if (e !== null) return e === "true";
	} catch {}
	return null;
}
function Te(e) {
	try {
		localStorage.setItem(r.NORMALIZATION_AVAILABLE, String(e)), u.debug("Config stored to localStorage:", e);
	} catch (e) {
		u.warn("Failed to store config to localStorage:", e);
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
		let e = we();
		e !== null && (this.normalizationAvailable = e, u.debug("Loaded config from localStorage:", e));
	},
	updateNormalizationWidget(e) {
		this.initialized || this.init();
		let t = e.widgets?.find((e) => e.name === "normalize_text");
		return t ? this.normalizationAvailable ? (t.disabled && (u.debug("Re-enabling normalize_text widget for node:", e.id), t.disabled = !1, t.value = !0, t.options && (t.options.tooltip = void 0)), !1) : (t.disabled || (u.debug("Disabling normalize_text widget for node:", e.id), t.disabled = !0, t.value = !1, t.options = t.options || {}, t.options.tooltip = "Text normalization disabled: 'inflect' and 'wetext' packages not installed. Install with: pip install inflect wetext"), !0) : !1;
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
		u.debug("Heavy extension loaded"), _e(), e.api.addEventListener(i.STATUS, ((e) => {
			this.handleStatusEvent(e.detail);
		})), e.api.addEventListener(i.CONFIG, ((e) => {
			this.handleConfigEvent(e.detail);
		})), e.api.addEventListener("reconnected", () => {
			u.debug("WebSocket reconnected, waiting for config..."), Q.configReceived = !1, ue(), Q.lastConfigValue = null;
		}), e.api.addEventListener("status", () => {
			Q.configReceived || u.debug("Status received, waiting for config...");
		}), u.log("Event listeners registered");
		try {
			let t = await e.api.fetchApi("/voxcpm/model_info");
			if (t.ok) {
				let e = await t.json();
				e && typeof e.normalization_available == "boolean" && (Q.configReceived || (u.debug("Fetched config from API fallback:", e), this.handleConfigEvent(e)));
			}
		} catch (e) {
			u.debug("Config API fallback failed (non-critical):", e);
		}
	},
	nodeCreated(e) {
		if (!o.has(e.comfyClass)) return;
		u.debug("VoxCPM node created:", e.id), Q.updateNormalizationWidget(e), de(e), X(e);
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
		o.has(e.comfyClass) && (u.debug("VoxCPM node loaded from workflow:", e.id), Q.updateNormalizationWidget(e), setTimeout(() => {
			he(e);
			let t = e.widgets;
			if (t) {
				let n = t.find((e) => e.name === "model_name");
				n?.value && Z(e, String(n.value));
			}
		}, 60));
	},
	handleStatusEvent(e) {
		if (!e || !e.severity || !e.summary) {
			u.warn("Invalid status event: missing severity or summary", e);
			return;
		}
		k.show({
			severity: e.severity,
			summary: e.summary,
			detail: e.detail,
			life: e.life
		});
	},
	async handleConfigEvent(e) {
		if (u.log("Received config event:", e), !e) {
			u.warn("Config event has no detail");
			return;
		}
		if (typeof e.normalization_available == "boolean") {
			if (Q.lastConfigValue === e.normalization_available && Q.configReceived) {
				u.debug("Config already processed, skipping");
				return;
			}
			Q.normalizationAvailable = e.normalization_available, Q.configReceived = !0, Q.lastConfigValue = e.normalization_available, Te(e.normalization_available), u.debug("Text normalization available:", Q.normalizationAvailable), setTimeout(() => {
				Q.updateAllNodes();
			}, 100);
		}
		e.settings ? (u.log("Initializing settings manager with:", e.settings), await j.initialize(e)) : u.warn("No settings in config event");
	}
};
e.registerExtension($);

export { $ as VoxCPMHeavyExtension, $ as default };
