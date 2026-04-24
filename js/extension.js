import { app as e } from "../../scripts/app.js";
import { app as t } from "../../scripts/app.js";
import { api as n } from "../../scripts/api.js";

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
	SETTINGS_UPDATE: "voxcpm.settings_update"
}, a = {
	TOAST_LIFE: 1e4,
	LOG_PREFIX: "[VoxCPM]",
	EXTENSION_NAME: "voxcpm.frontend",
	NODE_CLASS: "VoxCPM_TTS"
}, o = {
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
		SUCCESS: "pi pi-check-circle"
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
	}
}, s = {
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
}, c = "\n\n:root {\n  \n  --voxcpm-space-2xs: 2px;\n  --voxcpm-space-xs: 4px;\n  --voxcpm-space-s: 6px;\n  --voxcpm-space-m: 8px;\n  --voxcpm-space-l: 12px;\n  --voxcpm-space-xl: 16px;\n  --voxcpm-space-2xl: 24px;\n\n  \n  --voxcpm-font-size-2xs: 10px;\n  --voxcpm-font-size-xs: 12px;\n  --voxcpm-font-size-s: 13px;\n  --voxcpm-font-size-m: 14px;\n  --voxcpm-font-size-l: 18px;\n  --voxcpm-font-size-xl: 24px;\n  --voxcpm-font-family: var(--comfy-font-family, Arial, sans-serif);\n  --voxcpm-font-family-mono: monospace;\n\n  \n  --voxcpm-bg-input: var(--comfy-input-bg, rgba(255, 255, 255, 0.05));\n  --voxcpm-bg-input-hover: var(--comfy-input-bg-hover, rgba(255, 255, 255, 0.08));\n  --voxcpm-bg-input-active: var(--comfy-input-bg-active, rgba(255, 255, 255, 0.1));\n  --voxcpm-bg-surface: var(--bg-color, #1e1e1e);\n  --voxcpm-bg-elevated: var(--input-bg, #2a2a2a);\n\n  \n  --voxcpm-border-color: var(--border-color, rgba(255, 255, 255, 0.15));\n  --voxcpm-border-color-hover: var(--comfy-input-border-hover, rgba(255, 255, 255, 0.3));\n  --voxcpm-border-radius-xs: 4px;\n  --voxcpm-border-radius-s: 6px;\n  --voxcpm-border-radius-m: 8px;\n\n  \n  --voxcpm-text-primary: var(--fg-color, #ddd);\n  --voxcpm-text-secondary: var(--fg-color, rgba(255, 255, 255, 0.7));\n  --voxcpm-text-muted: var(--fg-color, rgba(255, 255, 255, 0.4));\n  --voxcpm-text-on-primary: #fff;\n\n  \n  --voxcpm-accent: var(--primary-color, #4a9eff);\n  --voxcpm-accent-hover: var(--primary-color-hover, #3a8eef);\n\n  \n  --voxcpm-color-success: var(--success-color, #4caf50);\n  --voxcpm-color-error: var(--error-color, #f44336);\n\n  \n  --voxcpm-duration-fast: 150ms;\n  --voxcpm-duration-normal: 200ms;\n  --voxcpm-easing: ease;\n\n  \n  --voxcpm-widget-height: 44px;\n  --voxcpm-input-height: 26px;\n  --voxcpm-browse-width: 30px;\n\n  \n  --voxcpm-focus-ring-color: var(--voxcpm-accent);\n  --voxcpm-focus-ring-width: 2px;\n  --voxcpm-focus-ring-offset: 1px;\n\n  \n  --voxcpm-cyber-font: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;\n  --voxcpm-cyber-accent: #00ff9d;\n  --voxcpm-cyber-accent-dim: rgba(0, 255, 157, 0.15);\n  --voxcpm-cyber-border: rgba(0, 255, 157, 0.3);\n  --voxcpm-cyber-glow: 0 0 8px rgba(0, 255, 157, 0.4);\n  --voxcpm-cyber-text: #e0e0e0;\n  --voxcpm-cyber-text-dim: rgba(224, 224, 224, 0.5);\n  --voxcpm-cyber-bg: rgba(10, 15, 20, 0.95);\n  --voxcpm-cyber-bg-hover: rgba(0, 255, 157, 0.08);\n  --voxcpm-cyber-bg-active: rgba(0, 255, 157, 0.12);\n  --voxcpm-cyber-tag-bg: rgba(0, 255, 157, 0.15);\n  --voxcpm-cyber-tag-border: rgba(0, 255, 157, 0.4);\n  --voxcpm-cyber-tag-text: #00ff9d;\n  --voxcpm-cyber-scanline: rgba(0, 255, 157, 0.03);\n}", l = "voxcpm-design-tokens", u = !1;
function d() {
	if (u) return;
	if (document.getElementById(l)) {
		u = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = l, e.textContent = c, document.head.appendChild(e), u = !0;
}

var f = {
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
}, p = `

.${f.BLOCK} {
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

.${f.ROW} {
  display: flex;
  align-items: center;
  gap: var(--voxcpm-space-xs);
  width: 100%;
  box-sizing: border-box;
}

.${f.DISPLAY} {
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

.${f.DISPLAY}:hover {
  border-color: var(--voxcpm-cyber-accent);
  box-shadow: var(--voxcpm-cyber-glow);
}

.${f.DISPLAY}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${f.ICON} {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  color: var(--voxcpm-cyber-accent);
}

.${f.ICON} svg {
  width: 16px;
  height: 16px;
}

.${f.TEXT} {
  flex: 1;
  color: var(--voxcpm-cyber-text);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
}

.${f.ARROW} {
  font-size: 9px;
  color: var(--voxcpm-cyber-accent);
  margin-left: auto;
  flex-shrink: 0;
  line-height: 1;
  opacity: 0.3;
  transition: opacity var(--voxcpm-duration-fast) var(--voxcpm-easing);
}

.${f.DISPLAY}:hover .${f.ARROW},

.${f.ARROW_ACTIVE} {
  opacity: 1;
}

.${f.BROWSE} {
  
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

.${f.BROWSE}:hover {
  border-color: var(--voxcpm-cyber-accent);
  box-shadow: var(--voxcpm-cyber-glow);
}

.${f.BROWSE}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${f.PATH} {
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

.${f.PATH_VISIBLE} {
  display: block;
}

.${f.DISPLAY}:focus-visible,
.${f.BROWSE}:focus-visible {
  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-cyber-accent);
  outline-offset: var(--voxcpm-focus-ring-offset);
}

@media (prefers-reduced-motion: reduce) {
  .${f.BLOCK} *,
  .${f.BLOCK} *::before,
  .${f.BLOCK} *::after {
    transition-duration: 0ms !important;
  }
}
`.trim(), m = "voxcpm-model-selector-styles", h = !1;
function g() {
	if (h) return;
	if (document.getElementById(m)) {
		h = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = m, e.textContent = p, document.head.appendChild(e), h = !0;
}

var _ = {
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
}, v = `

.${_.BLOCK} {
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

.${_.BLOCK}::after {
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

.${_.HEADER} {
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

.${_.HEADER} svg {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
}

.${_.LIST} {
  overflow-y: auto;
  overflow-x: hidden;
  max-height: 280px;
  position: relative;
  z-index: 2;
}

.${_.LIST}::-webkit-scrollbar {
  width: 4px;
}

.${_.LIST}::-webkit-scrollbar-track {
  background: transparent;
}

.${_.LIST}::-webkit-scrollbar-thumb {
  background: var(--voxcpm-cyber-border);
  border-radius: 0;
}

.${_.LIST}::-webkit-scrollbar-thumb:hover {
  background: var(--voxcpm-cyber-accent);
}

.${_.ITEM} {
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

.${_.ITEM}:hover,
.${_.ITEM_SELECTED} {
  background: var(--voxcpm-cyber-bg-hover);
  border-left-color: var(--voxcpm-cyber-accent);
}

.${_.ITEM}:active {
  background: var(--voxcpm-cyber-bg-active);
}

.${_.ICON} {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.${_.ICON} svg {
  width: 16px;
  height: 16px;
}

.${_.ICON_CLOUD} {
  color: var(--voxcpm-accent);
}

.${_.ICON_CHECK} {
  color: var(--voxcpm-color-success);
}

.${_.NAME} {
  flex: 0 1 auto;
  color: var(--voxcpm-cyber-text);
  font-size: var(--voxcpm-font-size-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
  min-width: 0;
}

.${_.META} {
  font-size: 10px;
  color: var(--voxcpm-cyber-text-dim);
  font-family: var(--voxcpm-cyber-font);
  flex-shrink: 0;
  margin-left: 6px;
}

.${_.TAG} {
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
  .${_.BLOCK} {
    animation: none;
  }
}
`.trim(), y = "voxcpm-model-dropdown-styles", b = !1;
function x() {
	if (b) return;
	if (document.getElementById(y)) {
		b = !0;
		return;
	}
	let e = document.createElement("style");
	e.id = y, e.textContent = v, document.head.appendChild(e), b = !0;
}

var S = !1;
function C() {
	S || (d(), g(), x(), S = !0);
}

function w(e) {
	try {
		return sessionStorage.getItem(e);
	} catch {
		return null;
	}
}
function T(e, t) {
	try {
		return sessionStorage.setItem(e, t), !0;
	} catch {
		return !1;
	}
}
function E(e) {
	try {
		return sessionStorage.removeItem(e), !0;
	} catch {
		return !1;
	}
}
function D(e) {
	return w(e) !== null;
}
function O(e) {
	return T(e, "true");
}
function ee(e) {
	return E(e);
}
function te(e) {
	let t = w(e);
	if (t === null) return null;
	try {
		return JSON.parse(t);
	} catch {
		return null;
	}
}
function k(e, t) {
	try {
		return T(e, JSON.stringify(t));
	} catch {
		return !1;
	}
}
var A = new class {
	constructor(e) {
		this.sessionKey = e.sessionKey, this.defaultLife = e.defaultLife;
	}
	show(e) {
		if (D(this.sessionKey)) return s.log("Notification already shown this session, skipping"), {
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
			}), O(this.sessionKey), s.log("Toast notification displayed:", e.summary), { shown: !0 };
		} catch (t) {
			return s.warn("Toast not available:", t), s.log(`${e.summary}: ${e.detail ?? ""}`), O(this.sessionKey), {
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
			}), s.log("Toast notification displayed:", e.summary), { shown: !0 };
		} catch (t) {
			return s.warn("Toast not available:", t), s.log(`${e.summary}: ${e.detail ?? ""}`), {
				shown: !1,
				reason: "toast_unavailable"
			};
		}
	}
	reset() {
		ee(this.sessionKey);
	}
	wasShown() {
		return D(this.sessionKey);
	}
}({
	sessionKey: "voxcpm.normalization_notification_shown",
	defaultLife: 1e4
}), j = "voxcpm.settings", M = new class {
	constructor() {
		this.settings = null, this.initialized = !1;
	}
	async initialize(e) {
		this.settings = e.settings, this.initialized = !0, k(j, this.settings), s.log("Settings initialized:", this.settings);
	}
	getSettings() {
		if (this.settings) return this.settings;
		let e = te(j);
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
			return this.settings && (Object.assign(this.settings, e), k(j, this.settings)), s.log("Settings updated:", e), !0;
		} catch (e) {
			return s.warn("Failed to update settings:", e), !1;
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
}(), N = null;
function P() {
	return new Promise((e) => {
		I();
		let t = document.createElement("div");
		t.className = "voxcpm-dialog__overlay", t.innerHTML = "\n    <div class=\"voxcpm-dialog\">\n    <div class=\"voxcpm-dialog__header\">\n    <span class=\"voxcpm-dialog__icon\">🎙️</span>\n    <h2 class=\"voxcpm-dialog__title\">VoxCPM Model Setup</h2>\n    </div>\n    <div class=\"voxcpm-dialog__body\">\n    <p class=\"voxcpm-dialog__description\">\n    Welcome to VoxCPM! How would you like to configure your model path?\n    </p>\n    <div class=\"voxcpm-dialog__options\">\n    <button class=\"voxcpm-dialog__option\" data-option=\"default\">\n    <div class=\"voxcpm-dialog__option-icon\">📁</div>\n    <div class=\"voxcpm-dialog__option-content\">\n    <h3>Use Default Path</h3>\n    <p class=\"voxcpm-dialog__option-path\">models/tts/VoxCPM/</p>\n    <p class=\"voxcpm-dialog__option-description\">\n    Official models will be downloaded automatically.\n    </p>\n    </div>\n    </button>\n    <button class=\"voxcpm-dialog__option\" data-option=\"custom\">\n    <div class=\"voxcpm-dialog__option-icon\">📂</div>\n    <div class=\"voxcpm-dialog__option-content\">\n    <h3>Use Custom Path</h3>\n    <p class=\"voxcpm-dialog__option-description\">\n    Select a custom directory where your VoxCPM models are located.\n    Useful for existing model collections.\n    </p>\n    </div>\n    </button>\n    </div>\n    </div>\n    <div class=\"voxcpm-dialog__footer\">\n    <button class=\"voxcpm-dialog__btn voxcpm-dialog__btn--secondary\" id=\"voxcpm-cancel\">Cancel</button>\n    <button class=\"voxcpm-dialog__btn voxcpm-dialog__btn--primary\" id=\"voxcpm-continue\" disabled>\n    Continue\n    </button>\n    </div>\n    </div>\n    ";
		let n = null, r = t.querySelectorAll(".voxcpm-dialog__option");
		r.forEach((e) => {
			e.addEventListener("click", () => {
				r.forEach((e) => e.classList.remove("voxcpm-dialog__option--selected")), e.classList.add("voxcpm-dialog__option--selected"), n = e.getAttribute("data-option");
				let i = t.querySelector("#voxcpm-continue");
				i && (i.disabled = !1);
			});
		}), t.querySelector("#voxcpm-cancel")?.addEventListener("click", () => {
			I(), e(null);
		}), t.querySelector("#voxcpm-continue")?.addEventListener("click", () => {
			n && (I(), e(n));
		}), document.body.appendChild(t), N = t, L();
	});
}
function F(e) {
	return new Promise((t) => {
		I();
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
		let r = !1, i = e || "", a = n.querySelector("#voxcpm-path-input"), o = n.querySelector("#voxcpm-models-preview"), c = n.querySelector("#voxcpm-models-list"), l = n.querySelector("#voxcpm-model-count"), u = n.querySelector("#voxcpm-validation-message"), d = n.querySelector("#voxcpm-confirm");
		async function f(e) {
			if (!e) {
				o.style.display = "none", u.textContent = "", d.disabled = !0, r = !1;
				return;
			}
			try {
				let { app: t } = await import("../../scripts/app.js"), n = await (await t.api.fetchApi(`/voxcpm/validate_path?path=${encodeURIComponent(e)}`)).json();
				r = n.valid, n.valid && n.models?.length > 0 ? (o.style.display = "block", l.textContent = n.models.length.toString(), c.innerHTML = n.models.map((e) => `
          <li>
          <span class="voxcpm-dialog__model-icon">📁</span>
          <span class="voxcpm-dialog__model-name">${e.name}</span>
          </li>
          `).join(""), u.innerHTML = `<span class="voxcpm-dialog__validation--success">Found ${n.models.length} VoxCPM-compatible models</span>`, d.disabled = !1) : (o.style.display = "none", u.innerHTML = `<span class="voxcpm-dialog__validation--error">${n.error || "Invalid path or no models found"}</span>`, d.disabled = !0);
			} catch (e) {
				s.warn("Path validation failed:", e), o.style.display = "none", u.innerHTML = "<span class=\"voxcpm-dialog__validation--error\">Failed to validate path</span>", d.disabled = !0, r = !1;
			}
		}
		a?.addEventListener("input", () => {
			i = a.value, f(i);
		}), n.querySelector("#voxcpm-browse")?.addEventListener("click", () => {
			alert("Please enter the path manually. Native folder picker requires server support.");
		}), n.querySelector("#voxcpm-cancel")?.addEventListener("click", () => {
			I(), t(null);
		}), d?.addEventListener("click", () => {
			r && i && (I(), t(i));
		}), document.body.appendChild(n), N = n, L(), e && f(e);
	});
}
function I() {
	N && (N.remove(), N = null);
}
function L() {
	let e = "voxcpm-dialog-styles";
	if (document.getElementById(e)) return;
	let t = document.createElement("style");
	t.id = e, t.textContent = "\n\n\n\n.voxcpm-dialog__overlay {\n  position: fixed;\n  top: 0;\n  left: 0;\n  right: 0;\n  bottom: 0;\n  background: rgba(0, 0, 0, 0.7);\n  display: flex;\n  align-items: center;\n  justify-content: center;\n  z-index: 10000;\n}\n\n\n.voxcpm-dialog {\n  background: var(--voxcpm-bg-surface);\n  border-radius: var(--voxcpm-border-radius-m);\n  max-width: 500px;\n  width: 90%;\n  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);\n}\n\n.voxcpm-dialog--large {\n  max-width: 600px;\n}\n\n\n.voxcpm-dialog__header {\n  display: flex;\n  align-items: center;\n  gap: var(--voxcpm-space-l);\n  padding: var(--voxcpm-space-2xl);\n  border-bottom: 1px solid var(--voxcpm-border-color);\n}\n\n.voxcpm-dialog__icon {\n  font-size: var(--voxcpm-font-size-xl);\n}\n\n.voxcpm-dialog__title {\n  margin: 0;\n  font-size: var(--voxcpm-font-size-l);\n  color: var(--voxcpm-text-primary);\n}\n\n\n.voxcpm-dialog__body {\n  padding: var(--voxcpm-space-2xl);\n}\n\n.voxcpm-dialog__description {\n  margin: 0 0 20px;\n  color: var(--voxcpm-text-secondary);\n  line-height: 1.5;\n}\n\n\n.voxcpm-dialog__options {\n  display: flex;\n  flex-direction: column;\n  gap: var(--voxcpm-space-l);\n}\n\n\n.voxcpm-dialog__option {\n  display: flex;\n  gap: var(--voxcpm-space-xl);\n  padding: var(--voxcpm-space-xl);\n  background: var(--voxcpm-bg-elevated);\n  border: 2px solid var(--voxcpm-border-color);\n  border-radius: var(--voxcpm-border-radius-m);\n  cursor: pointer;\n  text-align: left;\n  transition: border-color var(--voxcpm-duration-normal) var(--voxcpm-easing),\n              background var(--voxcpm-duration-normal) var(--voxcpm-easing);\n}\n\n.voxcpm-dialog__option:hover {\n  border-color: var(--voxcpm-accent);\n}\n\n.voxcpm-dialog__option--selected {\n  border-color: var(--voxcpm-accent);\n  background: var(--voxcpm-bg-input-hover);\n}\n\n.voxcpm-dialog__option-icon {\n  font-size: 32px;\n  flex-shrink: 0;\n}\n\n.voxcpm-dialog__option-content h3 {\n  margin: 0 0 var(--voxcpm-space-xs);\n  color: var(--voxcpm-text-primary);\n  font-size: var(--voxcpm-font-size-m);\n}\n\n.voxcpm-dialog__option-path {\n  margin: 0 0 var(--voxcpm-space-m);\n  font-family: var(--voxcpm-font-family-mono);\n  font-size: var(--voxcpm-font-size-xs);\n  color: var(--voxcpm-text-muted);\n}\n\n.voxcpm-dialog__option-description {\n  margin: 0;\n  font-size: var(--voxcpm-font-size-xs);\n  color: var(--voxcpm-text-muted);\n}\n\n\n.voxcpm-dialog__footer {\n  display: flex;\n  justify-content: flex-end;\n  gap: var(--voxcpm-space-l);\n  padding: var(--voxcpm-space-xl) var(--voxcpm-space-2xl);\n  border-top: 1px solid var(--voxcpm-border-color);\n}\n\n\n.voxcpm-dialog__btn {\n  padding: var(--voxcpm-space-m) var(--voxcpm-space-xl);\n  border-radius: var(--voxcpm-border-radius-xs);\n  font-size: var(--voxcpm-font-size-m);\n  cursor: pointer;\n  transition: background var(--voxcpm-duration-normal) var(--voxcpm-easing),\n              border-color var(--voxcpm-duration-normal) var(--voxcpm-easing);\n}\n\n.voxcpm-dialog__btn--secondary {\n  background: transparent;\n  border: 1px solid var(--voxcpm-border-color);\n  color: var(--voxcpm-text-secondary);\n}\n\n.voxcpm-dialog__btn--secondary:hover {\n  background: var(--voxcpm-bg-elevated);\n}\n\n.voxcpm-dialog__btn--primary {\n  background: var(--voxcpm-accent);\n  border: none;\n  color: var(--voxcpm-text-on-primary);\n}\n\n.voxcpm-dialog__btn--primary:hover:not(:disabled) {\n  background: var(--voxcpm-accent-hover);\n}\n\n.voxcpm-dialog__btn--primary:disabled {\n  opacity: 0.5;\n  cursor: not-allowed;\n}\n\n\n.voxcpm-dialog__input-group {\n  display: flex;\n  gap: var(--voxcpm-space-m);\n  margin-bottom: var(--voxcpm-space-xl);\n}\n\n.voxcpm-dialog__input {\n  flex: 1;\n  padding: var(--voxcpm-space-m) var(--voxcpm-space-l);\n  background: var(--voxcpm-bg-elevated);\n  border: 1px solid var(--voxcpm-border-color);\n  border-radius: var(--voxcpm-border-radius-xs);\n  color: var(--voxcpm-text-primary);\n  font-size: var(--voxcpm-font-size-m);\n  font-family: var(--voxcpm-font-family);\n}\n\n.voxcpm-dialog__input:focus {\n  outline: none;\n  border-color: var(--voxcpm-accent);\n}\n\n\n.voxcpm-dialog__preview {\n  background: var(--voxcpm-bg-elevated);\n  border-radius: var(--voxcpm-border-radius-xs);\n  padding: var(--voxcpm-space-l);\n  margin-bottom: var(--voxcpm-space-xl);\n}\n\n.voxcpm-dialog__preview h4 {\n  margin: 0 0 var(--voxcpm-space-m);\n  font-size: var(--voxcpm-font-size-xs);\n  color: var(--voxcpm-text-muted);\n}\n\n.voxcpm-dialog__models-list {\n  list-style: none;\n  margin: 0;\n  padding: 0;\n}\n\n.voxcpm-dialog__models-list li {\n  display: flex;\n  align-items: center;\n  gap: var(--voxcpm-space-m);\n  padding: var(--voxcpm-space-xs) 0;\n}\n\n.voxcpm-dialog__model-icon {\n  font-size: var(--voxcpm-font-size-m);\n}\n\n.voxcpm-dialog__model-name {\n  font-size: var(--voxcpm-font-size-s);\n  color: var(--voxcpm-text-secondary);\n}\n\n\n.voxcpm-dialog__validation {\n  font-size: var(--voxcpm-font-size-xs);\n}\n\n.voxcpm-dialog__validation--success {\n  color: var(--voxcpm-color-success);\n}\n\n.voxcpm-dialog__validation--error {\n  color: var(--voxcpm-color-error);\n}\n\n\n.voxcpm-dialog__option:focus-visible,\n.voxcpm-dialog__btn:focus-visible,\n.voxcpm-dialog__input:focus-visible {\n  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-focus-ring-color);\n  outline-offset: var(--voxcpm-focus-ring-offset);\n}\n\n\n@media (prefers-reduced-motion: reduce) {\n  .voxcpm-dialog *,\n  .voxcpm-dialog *::before,\n  .voxcpm-dialog *::after {\n    transition-duration: 0ms !important;\n  }\n}\n\n\n\n.voxcpm-path-indicator {\n  display: flex;\n  align-items: center;\n  gap: var(--voxcpm-space-m);\n  padding: var(--voxcpm-space-xs) var(--voxcpm-space-m);\n  background: var(--voxcpm-bg-elevated);\n  border-radius: var(--voxcpm-border-radius-xs);\n  font-size: var(--voxcpm-font-size-xs);\n}\n\n.voxcpm-path-indicator__text {\n  color: var(--voxcpm-text-muted);\n  font-family: var(--voxcpm-font-family-mono);\n}\n\n.voxcpm-path-indicator__change-btn {\n  background: transparent;\n  border: none;\n  cursor: pointer;\n  font-size: var(--voxcpm-font-size-m);\n  padding: var(--voxcpm-space-2xs);\n  color: var(--voxcpm-text-primary);\n  transition: opacity var(--voxcpm-duration-fast) var(--voxcpm-easing);\n}\n\n.voxcpm-path-indicator__change-btn:hover {\n  opacity: 0.8;\n}\n\n.voxcpm-path-indicator__change-btn:focus-visible {\n  outline: var(--voxcpm-focus-ring-width) solid var(--voxcpm-focus-ring-color);\n  outline-offset: var(--voxcpm-focus-ring-offset);\n}\n", document.head.appendChild(t);
}

var R = {
	CLOUD: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z\"/><polyline points=\"12 12 12 22\"/><path d=\"m8 18 4 4 4-4\"/></svg>",
	CHECK: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><polyline points=\"20 6 9 17 4 12\"/></svg>",
	SHIELD: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z\"/></svg>",
	FOLDER: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z\"/></svg>",
	FOLDER_OPEN: "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M5 19a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4l2 2h4a2 2 0 0 1 2 2v1\"/><path d=\"M6 12h14l-2.5 7H8.5L6 12z\"/></svg>"
}, z = null;
async function B(e) {
	if (z !== null) return z;
	try {
		let t = await e("/voxcpm/model_info");
		if (t.ok) return z = (await t.json()).models || [], z;
	} catch (e) {
		console.warn("[VoxCPM] Failed to fetch model info:", e);
	}
	return [];
}
function V() {
	z = null;
}
var H = class e {
	constructor(e, t, n, r) {
		this.selectedIndex = -1, this.resolveSelection = null, this.cleanups = [], this.isOpen = !1, this.modelInfo = e, this.anchorRect = t, this.scale = n, this.widgetElement = r || null, this.panel = document.createElement("div"), this.panel.className = _.BLOCK, this.panel.setAttribute("role", "listbox"), this.panel.setAttribute("aria-label", "Select model"), this.header = document.createElement("div"), this.header.className = _.HEADER, this.header.innerHTML = `${R.SHIELD} <span>${o.MODEL_DROPDOWN.HEADER_TEXT}</span>`, this.list = document.createElement("div"), this.list.className = _.LIST, this.panel.appendChild(this.header), this.panel.appendChild(this.list), this.renderItems();
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
		n.className = _.ITEM, n.setAttribute("role", "option"), n.setAttribute("data-index", String(t)), n.setAttribute("data-model", e.name);
		let r = document.createElement("span"), i = e.is_downloaded ? _.ICON_CHECK : _.ICON_CLOUD;
		r.className = `${_.ICON} ${i}`, r.innerHTML = e.is_downloaded ? R.CHECK : R.CLOUD, n.appendChild(r);
		let a = document.createElement("span");
		if (a.className = _.NAME, a.textContent = e.name, n.appendChild(a), e.size_gb > 0) {
			let t = document.createElement("span");
			t.className = _.META, t.textContent = `${e.size_gb} GB`, n.appendChild(t);
		}
		if (e.type === "official") {
			let e = document.createElement("span");
			e.className = _.TAG, e.textContent = "DEFAULT", n.appendChild(e);
		}
		return n.addEventListener("click", (t) => {
			t.stopPropagation(), this.close(e.name);
		}), n.addEventListener("mouseenter", () => {
			this.setSelectedIndex(t);
		}), n;
	}
	positionPanel() {
		let e = this.scale, t = o.MODEL_DROPDOWN.ANCHOR_GAP, n = window.innerHeight, r = window.innerWidth, i = o.MODEL_DROPDOWN.ITEM_HEIGHT, a = o.MODEL_DROPDOWN.MAX_VISIBLE_ITEMS, s = (32 + Math.min(this.modelInfo.length, a) * i) * e, c = 300 * e, l, u = n - this.anchorRect.bottom, d = this.anchorRect.top;
		l = u >= s + t ? this.anchorRect.bottom + t : d >= s + t ? this.anchorRect.top - s - t : this.anchorRect.bottom + t;
		let f = this.anchorRect.left;
		f + c > r && (f = r - c - 8), f = Math.max(8, f), this.panel.style.position = "fixed", this.panel.style.top = `${l}px`, this.panel.style.left = `${f}px`, this.panel.style.zIndex = String(o.MODEL_DROPDOWN.Z_INDEX), this.panel.style.transformOrigin = "top left", this.panel.style.transform = `scale(${e})`, this.panel.style.setProperty("--voxcpm-dropdown-scale", String(e));
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
		let t = this.list.querySelectorAll(`.${_.ITEM}`);
		if (t.forEach((e) => {
			e.classList.remove(_.ITEM_SELECTED);
		}), this.selectedIndex = e, e >= 0 && e < t.length) {
			let n = t[e];
			n.classList.add(_.ITEM_SELECTED), n.scrollIntoView({ block: "nearest" });
		}
	}
};
H.currentDropdown = null;

var U =  new Map();
function W(e) {
	let t = e.widgets;
	if (!t) {
		s.warn("No widgets found on node:", e.id);
		return;
	}
	let n = t.find((e) => e.name === "model_name");
	if (!n) {
		s.warn("model_name widget not found on node:", e.id);
		return;
	}
	G(n);
	let r = document.createElement("div");
	r.className = f.BLOCK;
	let i = document.createElement("div");
	i.className = f.DISPLAY, i.title = "Click to select a model", i.tabIndex = 0, i.setAttribute("role", "combobox"), i.setAttribute("aria-expanded", "false"), i.setAttribute("aria-haspopup", "listbox");
	let a = document.createElement("span");
	a.className = f.ICON, a.innerHTML = R.FOLDER;
	let c = document.createElement("span");
	c.className = f.TEXT, c.textContent = n.value || o.MODEL_SELECTOR.PLACEHOLDER;
	let l = document.createElement("span");
	l.className = f.ARROW, l.textContent = o.MODEL_SELECTOR.ARROW;
	let u = document.createElement("button");
	u.className = f.BROWSE, u.innerHTML = R.FOLDER_OPEN, u.title = "Browse for custom model directory", u.setAttribute("role", "button");
	let d = document.createElement("div");
	d.className = f.PATH;
	let p = e.properties?.custom_model_path;
	p && (a.innerHTML = R.FOLDER_OPEN, d.textContent = p, d.classList.add(f.PATH_VISIBLE)), i.append(a, c, l);
	let m = document.createElement("div");
	m.className = f.ROW, m.append(i, u), r.append(m, d), r.addEventListener("mouseup", (e) => e.stopPropagation()), r.addEventListener("click", (e) => e.stopPropagation()), i.addEventListener("click", (t) => {
		if (t.preventDefault(), t.stopPropagation(), K()) {
			q();
			return;
		}
		J(t, n, c, a, d, e, i, l);
	}), i.addEventListener("keydown", (t) => {
		if (t.key === "Enter" || t.key === " ") {
			if (t.preventDefault(), t.stopPropagation(), K()) {
				q();
				return;
			}
			J(t, n, c, a, d, e, i, l);
		}
	}), u.addEventListener("click", async (t) => {
		t.preventDefault(), t.stopPropagation(), await Y(e, n, c, a, d);
	});
	let h = e.addDOMWidget("model_selector", "custom", r, {
		serialize: !1,
		hideOnZoom: !1,
		selectOn: ["click"],
		getValue: () => n.value,
		setValue: (e) => {
			n.value = e, c.textContent = e;
		}
	}), g = new MutationObserver(() => {
		let t = m.parentElement;
		if (t) {
			let n = t.parentElement;
			n && n !== document.body ? (n.classList.add(f.BLOCK), n.setAttribute("node-id", String(e.id)), s.debug("Added", f.BLOCK, "class + node-id to WidgetDOM parent div for node:", e.id)) : s.debug("No WidgetDOM grandparent found (LiteGraph mode) for node:", e.id), g.disconnect();
		}
	});
	g.observe(document.body, {
		childList: !0,
		subtree: !0
	}), h.tooltip = "Select the VoxCPM model to use", h.computeSize = () => [0, o.MODEL_SELECTOR.MIN_HEIGHT], h.computeLayoutSize = void 0;
	let _ = e.widgets;
	if (_ && _.length > 1) {
		let t = _.indexOf(h);
		t > 0 && (_.splice(t, 1), _.unshift(h), s.debug("Moved model selector widget to position 0 for node:", e.id));
	}
	let v = e.inputs?.find((e) => e.name === "model_name");
	v && (v.widget = {
		name: "model_selector",
		_originalWidget: v.widget?.name
	}, s.debug("Bound model_name input to model_selector widget for node:", e.id)), U.set(e.id, {
		modelText: c,
		folderIcon: a,
		pathIndicator: d,
		modelWidget: n
	}), s.debug("Created model selector widget for node:", e.id);
}
function G(e) {
	e.type = "converted-widget", e.computeSize = () => [0, -4], e.draw = () => {}, e.options || (e.options = {}), e.options.canvasOnly = !0, s.debug("Hidden default combo widget:", e.name);
}
function K() {
	return H.currentDropdown !== null;
}
function q() {
	H.currentDropdown && H.currentDropdown.close(null);
}
async function J(e, r, i, a, o, c, l, u) {
	let d = r.options?.values || [];
	if (d.length === 0) {
		s.warn("No models available for dropdown"), A.show({
			severity: "warn",
			summary: "No Models Available",
			detail: "No models found. Try browsing for a custom model directory.",
			life: 4e3
		});
		return;
	}
	let p = [];
	try {
		p = await B(n.fetchApi.bind(n));
	} catch (e) {
		s.error("Failed to fetch model info:", e);
	}
	p.length === 0 && (p = d.map((e) => ({
		name: e,
		type: "local",
		architecture: "unknown",
		sample_rate: 0,
		size_gb: 0,
		is_downloaded: !0
	})));
	let m = l.getBoundingClientRect(), h = t.canvas.ds?.scale || 1;
	u.classList.add(f.ARROW_ACTIVE), l.setAttribute("aria-expanded", "true");
	try {
		let e = await new H(p, m, h, l).show();
		if (e) {
			r.value = e, i.textContent = e;
			let t = p.find((t) => t.name === e);
			t && (a.innerHTML = t.is_downloaded ? R.CHECK : R.CLOUD), r.callback && r.callback(e), c.setDirtyCanvas && c.setDirtyCanvas(!0), s.debug("Model selected:", e, "for node:", c.id);
		}
	} catch (e) {
		s.error("Failed to open model dropdown:", e);
	} finally {
		u.classList.remove(f.ARROW_ACTIVE), l.setAttribute("aria-expanded", "false");
	}
}
async function Y(e, t, r, i, a) {
	s.log("Browse button clicked for node:", e.id);
	let o = await F();
	if (!o) {
		s.debug("Browse cancelled by user");
		return;
	}
	s.log("User selected custom path:", o), e.properties = e.properties || {}, e.properties.custom_model_path = o;
	try {
		let c = await n.fetchApi("/voxcpm/models", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ path: o })
		});
		if (c.ok) {
			let n = await c.json();
			s.log("Models found at custom path:", n.models), X(t, n.models, e), i.innerHTML = R.FOLDER_OPEN, a.textContent = o, a.classList.add(f.PATH_VISIBLE), n.models.length > 0 ? r.textContent = t.value : r.textContent = "No models found", A.show({
				severity: "success",
				summary: "Custom Model Path Set",
				detail: `Found ${n.models.length} model(s) in: ${o}`,
				life: 5e3
			});
		} else {
			let e = c.statusText || "Unknown error";
			s.error("Failed to get models:", e), A.show({
				severity: "error",
				summary: "Error",
				detail: `Failed to scan model directory: ${e}`,
				life: 5e3
			});
		}
	} catch (e) {
		s.error("Error fetching models:", e), A.show({
			severity: "error",
			summary: "Error",
			detail: "Failed to connect to server",
			life: 5e3
		});
	}
}
function X(e, t, n) {
	e.options ? e.options.values = t : e.options = { values: t }, t.length > 0 && !t.includes(e.value) && (e.value = t[0], e.callback && e.callback(t[0])), n.setDirtyCanvas && n.setDirtyCanvas(!0), s.debug("Updated model options:", t);
}
function Z(e, t) {
	let n = U.get(e.id);
	if (!n) {
		s.debug("No selector refs found for node:", e.id);
		return;
	}
	let r = t || n.modelWidget.value;
	n.modelText.textContent = r;
	let i = e.properties?.custom_model_path;
	i ? (n.folderIcon.innerHTML = R.FOLDER_OPEN, n.pathIndicator.textContent = i, n.pathIndicator.classList.add(f.PATH_VISIBLE)) : (n.folderIcon.innerHTML = R.FOLDER, n.pathIndicator.classList.remove(f.PATH_VISIBLE)), s.debug("Synced model selector display for node:", e.id, "value:", r);
}

function ne(e) {
	try {
		let t = sessionStorage.getItem(e);
		if (t !== null) return t;
	} catch {}
	return null;
}
function re(e, t) {
	try {
		sessionStorage.setItem(e, t);
	} catch (t) {
		s.warn(`Failed to set session flag ${e}:`, t);
	}
}
function ie() {
	try {
		let e = localStorage.getItem(r.NORMALIZATION_AVAILABLE);
		if (e !== null) return e === "true";
	} catch {}
	return null;
}
function ae(e) {
	try {
		localStorage.setItem(r.NORMALIZATION_AVAILABLE, String(e)), s.debug("Config stored to localStorage:", e);
	} catch (e) {
		s.warn("Failed to store config to localStorage:", e);
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
		let e = ie();
		e !== null && (this.normalizationAvailable = e, s.debug("Loaded config from localStorage:", e));
	},
	updateNormalizationWidget(e) {
		this.initialized || this.init();
		let t = e.widgets?.find((e) => e.name === "normalize_text");
		return t ? this.normalizationAvailable ? (t.disabled && (s.debug("Re-enabling normalize_text widget for node:", e.id), t.disabled = !1, t.value = !0, t.options && (t.options.tooltip = void 0)), !1) : (t.disabled || (s.debug("Disabling normalize_text widget for node:", e.id), t.disabled = !0, t.value = !1, t.options = t.options || {}, t.options.tooltip = "Text normalization disabled: 'inflect' and 'wetext' packages not installed. Install with: pip install inflect wetext"), !0) : !1;
	},
	updateAllNodes() {
		this.initialized || this.init();
		let t = e.graph || e.rootGraph;
		if (t && t._nodes) for (let e of t._nodes) e.comfyClass === a.NODE_CLASS && this.updateNormalizationWidget(e);
	}
};
Q.init();
var $ = {
	name: a.EXTENSION_NAME,
	settings: [{
		id: "voxcpm.use_custom_path",
		category: ["VoxCPM", "Model Path"],
		name: "Use Custom Model Path",
		type: "boolean",
		defaultValue: !1,
		tooltip: "Enable to use a custom directory for VoxCPM models"
	}, {
		id: "voxcpm.custom_model_path",
		category: ["VoxCPM", "Model Path"],
		name: "Custom Model Path",
		type: "text",
		defaultValue: "",
		tooltip: "Path to custom VoxCPM models directory"
	}],
	async init() {
		C(), s.log("Frontend extension initialized");
	},
	async setup() {
		s.debug("Frontend extension loaded"), e.api.addEventListener(i.STATUS, ((e) => {
			this.handleStatusEvent(e.detail);
		})), e.api.addEventListener(i.CONFIG, ((e) => {
			this.handleConfigEvent(e.detail);
		})), e.api.addEventListener("reconnected", () => {
			s.debug("WebSocket reconnected, waiting for config..."), Q.configReceived = !1, V(), Q.lastConfigValue = null;
		}), e.api.addEventListener("status", () => {
			Q.configReceived || s.debug("Status received, waiting for config...");
		}), s.log("Event listeners registered");
	},
	nodeCreated(e) {
		e.comfyClass === a.NODE_CLASS && (s.debug("VoxCPM node created:", e.id), Q.updateNormalizationWidget(e), W(e));
	},
	loadedGraphNode(e) {
		e.comfyClass === a.NODE_CLASS && (s.debug("VoxCPM node loaded from workflow:", e.id), Q.updateNormalizationWidget(e), setTimeout(() => {
			Z(e);
		}, 50));
	},
	handleStatusEvent(e) {
		if (!e || !e.severity || !e.summary) {
			s.warn("Invalid status event: missing severity or summary", e);
			return;
		}
		A.show({
			severity: e.severity,
			summary: e.summary,
			detail: e.detail,
			life: e.life
		});
	},
	async handleConfigEvent(e) {
		if (s.log("Received config event:", e), !e) {
			s.warn("Config event has no detail");
			return;
		}
		if (typeof e.normalization_available == "boolean") {
			if (Q.lastConfigValue === e.normalization_available && Q.configReceived) {
				s.debug("Config already processed, skipping");
				return;
			}
			Q.normalizationAvailable = e.normalization_available, Q.configReceived = !0, Q.lastConfigValue = e.normalization_available, ae(e.normalization_available), s.debug("Text normalization available:", Q.normalizationAvailable), setTimeout(() => {
				Q.updateAllNodes();
			}, 100);
		}
		if (e.settings) {
			s.log("Initializing settings manager with:", e.settings), await M.initialize(e);
			let t = M.isFirstRun();
			s.log("Is first run:", t), t && (s.log("Calling showFirstRunDialog()"), this.showFirstRunDialog());
		} else s.warn("No settings in config event");
	},
	async showFirstRunDialog() {
		let e = r.FIRST_RUN_SHOWN;
		if (ne(e)) {
			s.log("First run dialog already shown this session");
			return;
		}
		re(e, "true"), s.log("First run detected - showing welcome dialog");
		let t = await P();
		if (t === "default") await M.setDefaultPath(), await M.markFirstRunComplete(), s.log("User selected default path");
		else if (t === "custom") {
			let e = await F();
			e && (await M.setCustomPath(e), await M.markFirstRunComplete(), s.log("User selected custom path:", e));
		} else s.log("User cancelled first run dialog");
	}
};
e.registerExtension($);

export { $ as VoxCPMExtension, $ as default };
