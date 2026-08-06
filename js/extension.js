import { app as e } from "../../scripts/app.js";

var t = new Set([
	"VoxCPM_TTS",
	"VoxCPM_VoiceCloning",
	"VoxCPM_AdvancedParams"
]), n = !1, r = null, i = "/voxcpm/heavy_extension/voxcpmHeavy.js";
async function a() {
	return n ? [] : r || (r = (async () => {
		let t = new Set(e.extensions?.map((e) => e.name) ?? []);
		return await import(
			
			i
), n = !0, e.extensions?.filter((e) => !t.has(e.name)) ?? [];
	})(), r);
}
function o(e) {
	return e?.nodes ? e.nodes.some((e) => e.type && t.has(e.type)) : !1;
}
async function s(t) {
	let n = await a();
	for (let r of n) await r.nodeCreated?.(t, e);
}
async function c(t) {
	let n = await a();
	for (let r of n) await r.loadedGraphNode?.(t, e);
}
e.registerExtension({
	name: "voxcpm.lazyLoader",
	settings: [
		{
			id: "voxcpm.use_custom_path",
			category: ["VoxCPM", "Model Path"],
			name: "Use Custom Model Path",
			type: "boolean",
			defaultValue: !1,
			tooltip: "Enable to use a custom directory for VoxCPM models"
		},
		{
			id: "voxcpm.custom_model_path",
			category: ["VoxCPM", "Model Path"],
			name: "Custom Model Path",
			type: "text",
			defaultValue: "",
			tooltip: "Path to custom VoxCPM models directory"
		},
		{
			id: "voxcpm.reset_settings",
			category: ["VoxCPM", "Model Path"],
			name: "Reset Custom Model Path Settings",
			type: "boolean",
			defaultValue: !1,
			tooltip: "Clears the saved custom model path and resets to defaults. The page will reload after clearing.",
			onChange: async (t) => {
				if (t) try {
					await e.api.fetchApi("/voxcpm/settings", { method: "DELETE" }), await e.api.storeSetting("voxcpm.reset_settings", !1), window.location.reload();
				} catch (e) {
					console.error("[VoxCPM] Failed to clear settings:", e);
				}
			}
		}
	],
	async beforeConfigureGraph(e) {
		o(e) && await a();
	},
	async nodeCreated(e) {
		let n = e.constructor?.comfyClass || e.type;
		t.has(n) && await s(e);
	},
	async loadedGraphNode(e) {
		let n = e.constructor?.comfyClass || e.type;
		t.has(n) && await c(e);
	}
});

