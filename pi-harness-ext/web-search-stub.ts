import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Same schema as tool-harness.py's web_search, so the check is comparable.
const webSearch = defineTool({
	name: "web_search",
	label: "Web search",
	description: "Search the web.",
	parameters: Type.Object({
		query: Type.String({ description: "The search query." }),
		max_results: Type.Optional(Type.Integer({ description: "Maximum number of results to return." })),
	}),
	async execute(_id, params) {
		return { content: [{ type: "text", text: "No results." }], details: params };
	},
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(webSearch);
}
