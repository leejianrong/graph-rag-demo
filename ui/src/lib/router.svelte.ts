/** A hash router: `#/query` shows the Query view. One entry per view; later cards add a file, not a line here. */
export const routes = [
  { path: "ingest", label: "Ingest" },
  { path: "query", label: "Query" },
  { path: "explorer", label: "Explorer" },
  { path: "keys", label: "API keys" },
] as const;

export type RoutePath = (typeof routes)[number]["path"];

const parse = (): RoutePath => {
  const name = location.hash.replace(/^#\/?/, "");
  return routes.find((r) => r.path === name)?.path ?? "query";
};

export const route = $state({ current: parse() });

window.addEventListener("hashchange", () => {
  route.current = parse();
});
