import createClient from "openapi-fetch";
import type { paths } from "./schema";

/** What a client needs to act as the signed-in user; omit it for calls that need no key. */
export interface AuthHooks {
  key: () => string | null;
  onUnauthorized: () => void;
}

/** The typed API client. `schema.d.ts` is generated from docs/openapi.json by `npm run gen:api`. */
export function makeClient(baseUrl = "", fetchImpl?: typeof fetch, auth?: AuthHooks) {
  const client = createClient<paths>({ baseUrl, fetch: fetchImpl });
  if (auth) {
    client.use({
      onRequest({ request }) {
        const key = auth.key();
        // Bearer, as graph_rag/api.py reads it. A call that sets its own header keeps it.
        if (key && !request.headers.has("authorization")) {
          request.headers.set("Authorization", `Bearer ${key}`);
        }
        return request;
      },
      onResponse({ response }) {
        if (response.status === 401) auth.onUnauthorized();
      },
    });
  }
  return client;
}

/** For calls that need no key, such as `/health`. Signed-in views use `api` from `lib/auth.svelte`. */
export const publicApi = makeClient();

/** True when `/health` answers `{"status": "ok"}`; a network error or other answer is false. */
export async function checkHealth(client = publicApi): Promise<boolean> {
  try {
    const { data } = await client.GET("/health");
    return data?.status === "ok";
  } catch {
    return false;
  }
}
