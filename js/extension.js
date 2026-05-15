import { a as e } from "/voxcpm/heavy_extension/shared.js";
import { app as t } from "../../scripts/app.js";

var n = !1, r = null, i = "/voxcpm/heavy_extension/voxcpmHeavy.js";
async function a() {
	return n ? [] : r || (r = (async () => {
		let e = new Set(t.extensions?.map((e) => e.name) ?? []);
		return await import(
			
			i
), n = !0, t.extensions?.filter((t) => !e.has(t.name)) ?? [];
	})(), r);
}
function o(t) {
	return t?.nodes ? t.nodes.some((t) => t.type && e.has(t.type)) : !1;
}
async function s(e) {
	let n = await a();
	for (let r of n) await r.nodeCreated?.(e, t);
}
async function c(e) {
	let n = await a();
	for (let r of n) await r.loadedGraphNode?.(e, t);
}
t.registerExtension({
	name: "voxcpm.lazyLoader",
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
	async beforeConfigureGraph(e) {
		o(e) && await a();
	},
	async nodeCreated(t) {
		let n = t.constructor?.comfyClass || t.type;
		e.has(n) && await s(t);
	},
	async loadedGraphNode(t) {
		let n = t.constructor?.comfyClass || t.type;
		e.has(n) && await c(t);
	}
});

