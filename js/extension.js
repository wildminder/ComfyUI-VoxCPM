import { a as e } from "./shared.js";
import { app as t } from "../../scripts/app.js";

var n = !1, r = null;
async function i() {
	return n ? [] : r || (r = (async () => {
		let e = new Set(t.extensions?.map((e) => e.name) ?? []);
		return await import(
			
			"./voxcpmHeavy.js"
), n = !0, t.extensions?.filter((t) => !e.has(t.name)) ?? [];
	})(), r);
}
function a(t) {
	return t?.nodes ? t.nodes.some((t) => t.type && e.has(t.type)) : !1;
}
async function o(e) {
	let n = await i();
	for (let r of n) await r.nodeCreated?.(e, t);
}
async function s(e) {
	let n = await i();
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
		a(e) && await i();
	},
	async nodeCreated(t) {
		let n = t.constructor?.comfyClass || t.type;
		e.has(n) && await o(t);
	},
	async loadedGraphNode(t) {
		let n = t.constructor?.comfyClass || t.type;
		e.has(n) && await s(t);
	}
});

