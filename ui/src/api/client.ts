import createClient from "openapi-fetch";
import type { paths } from "./schema";

/** What a client needs to act as the signed-in user; omit it for calls that need no key. */
export interface AuthHooks {
  key: () => string | null;
  onUnauthorized: () => void;
}

/**
 * A `fetch` that acts as the signed-in user: it attaches the key and signs out on a 401. For calls
 * `openapi-fetch` cannot express, such as a multipart upload or a streamed response body.
 */
export function authedFetch(auth: AuthHooks, fetchImpl: typeof fetch = (...a) => fetch(...a)): typeof fetch {
  return async (input, init) => {
    const headers = new Headers(init?.headers);
    const key = auth.key();
    if (key && !headers.has("authorization")) headers.set("Authorization", `Bearer ${key}`);
    const response = await fetchImpl(input, { ...init, headers });
    if (response.status === 401) auth.onUnauthorized();
    return response;
  };
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

/** A call's outcome: the data, or the HTTP status (0 for a network failure) and the API's own message. */
export type Outcome<T> =
  | { ok: true; data: T }
  | { ok: false; status: number; message?: string; code?: string; details?: Record<string, unknown> };

type Client = ReturnType<typeof makeClient>;

async function settle<T>(
  call: () => Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<Outcome<T>> {
  try {
    const { data, error, response } = await call();
    if (data !== undefined && response.ok) return { ok: true, data };
    const body = error as
      | { error?: { code?: string; message?: string; details?: Record<string, unknown> } }
      | undefined;
    return {
      ok: false,
      status: response.status,
      message: body?.error?.message,
      code: body?.error?.code,
      details: body?.error?.details,
    };
  } catch {
    return { ok: false, status: 0 };
  }
}

type SearchQuery = NonNullable<paths["/v1/entities"]["get"]["parameters"]["query"]>;
type DetailQuery = NonNullable<paths["/v1/entities/{entity_id}"]["get"]["parameters"]["query"]>;

/** `GET /v1/entities`: entities whose name or alias contains `q`, newest first, one page at a time. */
export function searchEntities(client: Client, query: SearchQuery) {
  return settle(() => client.GET("/v1/entities", { params: { query } }));
}

/** `GET /v1/entities/{id}`: one entity with its neighbours, aliases and source documents. */
export function entityDetail(client: Client, entityId: string, query: DetailQuery = {}) {
  return settle(() =>
    client.GET("/v1/entities/{entity_id}", { params: { path: { entity_id: entityId }, query } }),
  );
}

/** `POST /v1/entities/{keep_id}/merge`: fold `mergeIds` into `keepId`. A 409's `details.reason` says why not. */
export function mergeEntities(client: Client, keepId: string, mergeIds: string[], namespace?: string) {
  return settle(() =>
    client.POST("/v1/entities/{keep_id}/merge", {
      params: { path: { keep_id: keepId } },
      body: { merge_ids: mergeIds, namespace },
    }),
  );
}

/** `POST /v1/entities/{keep_id}/split`: put `entityIds` of merge `mergeId` back as separate entities. */
export function splitEntity(
  client: Client,
  keepId: string,
  mergeId: string,
  entityIds: string[],
  namespace?: string,
) {
  return settle(() =>
    client.POST("/v1/entities/{keep_id}/split", {
      params: { path: { keep_id: keepId } },
      body: { merge_id: mergeId, entity_ids: entityIds, namespace },
    }),
  );
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
