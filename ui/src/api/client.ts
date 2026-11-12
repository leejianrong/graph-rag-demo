import createClient from "openapi-fetch";
import type { paths } from "./schema";

/** The typed API client. `schema.d.ts` is generated from docs/openapi.json by `npm run gen:api`. */
export function makeClient(baseUrl = "", fetchImpl?: typeof fetch) {
  return createClient<paths>({ baseUrl, fetch: fetchImpl });
}

export const api = makeClient();

/** True when `/health` answers `{"status": "ok"}`; a network error or other answer is false. */
export async function checkHealth(client = api): Promise<boolean> {
  try {
    const { data } = await client.GET("/health");
    return data?.status === "ok";
  } catch {
    return false;
  }
}
