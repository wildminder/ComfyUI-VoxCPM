
var e = {
	NOTIFICATION_SHOWN: "voxcpm.normalization_notification_shown",
	USER_PREFERENCES: "voxcpm.user_preferences",
	SETTINGS: "voxcpm.settings",
	FIRST_RUN_SHOWN: "voxcpm.first_run_shown",
	NORMALIZATION_AVAILABLE: "voxcpm.normalization_available"
}, t = {
	STATUS: "voxcpm.status",
	MODEL_LOADED: "voxcpm.model_loaded",
	GENERATION_PROGRESS: "voxcpm.generation_progress",
	CONFIG: "voxcpm.config",
	SETTINGS_UPDATE: "voxcpm.settings_update",
	DOWNLOAD_PROGRESS: "voxcpm.download_progress"
}, n = {
	TOAST_LIFE: 1e4,
	LOG_PREFIX: "[VoxCPM]",
	EXTENSION_NAME: "voxcpm.frontend",
	NODE_CLASS: "VoxCPM_TTS"
}, r = new Set([
	"VoxCPM_TTS",
	"VoxCPM_VoiceCloning",
	"VoxCPM_AdvancedParams"
]), i = {
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
};

export { r as a, i, t as n, e as r, n as t };
