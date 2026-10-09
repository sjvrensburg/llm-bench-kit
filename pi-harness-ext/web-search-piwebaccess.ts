import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

// The model-facing parameters of pi-web-access 0.31.0's web_search (index.ts:1832),
// minus provider/workflow/proxy, which carry policy text and are not what a caller
// is asked to fill. This is the schema docs/13 Phase C will actually install; the
// sibling web-search-stub.ts is the Hermes harness's schema, for comparability.
const webSearch = defineTool({
	name: "web_search",
	label: "Web Search",
	description: "Search the web and return source-linked results.",
	parameters: Type.Object({
		query: Type.Optional(Type.String({ description: "Single search query. For research tasks, prefer 'queries' with multiple varied angles instead." })),
		queries: Type.Optional(Type.Array(Type.String(), { description: "Multiple queries searched concurrently (up to three at a time)." })),
		numResults: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, description: "Results per query (default: 5, max: 20)" })),
		includeContent: Type.Optional(Type.Boolean({ description: "Fetch full page content (async)" })),
		recencyFilter: Type.Optional(Type.Union([Type.Literal("day"), Type.Literal("week"), Type.Literal("month"), Type.Literal("year")], { description: "Filter by recency" })),
		domainFilter: Type.Optional(Type.Array(Type.String(), { description: "Limit to domains (prefix with - to exclude)" })),
	}),
	async execute(_id, params) {
		return { content: [{ type: "text", text: "No results." }], details: params };
	},
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(webSearch);
}
