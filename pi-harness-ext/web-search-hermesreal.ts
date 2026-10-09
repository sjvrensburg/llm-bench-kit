import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Hermes's production web_search (~/.hermes/hermes-agent/tools/web_tools.py:475),
// not the harness's simplified one. The bound is `limit`. It separates "pi's prompt
// loses the bound" from "pi-web-access's schema loses it".
const webSearch = defineTool({
	name: "web_search",
	label: "Web search",
	description: "Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. The query is passed through to the configured backend, so operators such as site:domain, filetype:pdf, intitle:word, -term, and \"exact phrase\" may work when the backend supports them.",
	parameters: Type.Object({
		query: Type.String({ description: "The search query to look up on the web. You may include backend-supported operators such as site:example.com, filetype:pdf, intitle:word, -term, or \"exact phrase\"." }),
		limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 100, default: 5, description: "Maximum number of results to return. Defaults to 5." })),
	}),
	async execute(_id, params) {
		return { content: [{ type: "text", text: "No results." }], details: params };
	},
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(webSearch);
}
