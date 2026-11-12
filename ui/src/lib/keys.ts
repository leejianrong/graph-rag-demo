import type { components } from "../api/schema";
import { makeClient } from "../api/client";

export type KeyInfo = components["schemas"]["KeyInfo"];
export type CreatedKey = components["schemas"]["CreatedKey"];
type Client = ReturnType<typeof makeClient>;

/** Keys one tenant may have active at once (MAX_KEYS_PER_TENANT, default 10). The API enforces it. */
export const MAX_ACTIVE_KEYS = 10;
export const MAX_LABEL = 64;

/** The error envelope `{"error": {code, message, details?}}`. Not in the OpenAPI schema, so typed here. */
export interface ApiErrorBody {
  error?: {
    code?: string;
    message?: string;
    details?: { reason?: string; field?: string; retry_after_seconds?: number };
  };
}

export type Failure = { ok: false; status: number | null; message: string };
export type Result<T> = { ok: true; value: T } | Failure;

/** A human sentence for a refused key request. Each `details.reason` the API sends has its own. */
export function describeKeyError(
  status: number,
  body: ApiErrorBody | undefined,
  retryAfter?: string | null,
): string {
  const details = body?.error?.details;
  const message = body?.error?.message;
  if (status === 400) {
    switch (details?.reason) {
      case "key_limit":
        return `This tenant already has ${MAX_ACTIVE_KEYS} active keys. Revoke one, then create a new key.`;
      case "own_key":
        return "That is the key you are signed in with. Sign in with another key to revoke it.";
      case "auth_disabled":
        return "This deployment runs without authentication, so there are no keys to manage.";
    }
    if (details?.field === "label") return `The label was not accepted. ${message ?? ""}`.trim();
    return message ?? "The API refused the request.";
  }
  if (status === 401) return "Your key was rejected, so you were signed out. Sign in again.";
  if (status === 404) return "That key no longer exists. The list has been refreshed.";
  if (status === 429) {
    const wait = details?.retry_after_seconds ?? Number(retryAfter);
    return wait > 0
      ? `Too many keys created. Wait ${Math.ceil(wait)} seconds and try again.`
      : "Too many keys created. Wait a moment and try again.";
  }
  return `The API answered ${status}. Try again in a moment.`;
}

const unreachable: Failure = {
  ok: false,
  status: null,
  message: "The API did not answer. Check that the service is running.",
};

async function call<T>(
  run: () => Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<Result<T>> {
  try {
    const { data, error, response } = await run();
    if (data !== undefined && response.ok) return { ok: true, value: data };
    return {
      ok: false,
      status: response.status,
      message: describeKeyError(
        response.status,
        error as ApiErrorBody | undefined,
        response.headers.get("retry-after"),
      ),
    };
  } catch {
    return unreachable;
  }
}

export const listKeys = (client: Client) =>
  call(async () => {
    const r = await client.GET("/v1/keys");
    return { ...r, data: r.data?.keys };
  });

export const createKey = (client: Client, label: string) =>
  call(() => client.POST("/v1/keys", { body: { label } }));

export const revokeKey = (client: Client, keyId: string) =>
  call(() => client.DELETE("/v1/keys/{key_id}", { params: { path: { key_id: keyId } } }));

/** How a key reads in the table and in the revoke prompt: the last four characters, or dots when unknown. */
export const maskedSuffix = (key: Pick<KeyInfo, "key_suffix">): string => `••••${key.key_suffix ?? ""}`;

export const displayLabel = (key: Pick<KeyInfo, "label">): string => key.label || "(no label)";

export const activeCount = (keys: KeyInfo[]): number => keys.filter((k) => k.status === "active").length;

/** Why Create is off right now, or null when it is allowed. */
export function createBlockedReason(keys: KeyInfo[], busy: boolean): string | null {
  if (busy) return "Creating a key…";
  if (activeCount(keys) >= MAX_ACTIVE_KEYS)
    return `This tenant has ${MAX_ACTIVE_KEYS} active keys, the most allowed. Revoke one to create another.`;
  return null;
}

/** Why Revoke is off for a row, or null when it is allowed. */
export function revokeBlockedReason(key: KeyInfo): string | null {
  return key.current ? "The key you are using now" : null;
}

/** `2026-10-02T09:41:00Z` as `2026-10-02 09:41` (UTC, as the API stores it). */
export function formatCreated(iso: string): string {
  const m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/.exec(iso);
  return m ? `${m[1]} ${m[2]}` : iso;
}

/** The one-time secret panel. It holds the secret and nothing else; null means no secret is held. */
export type Reveal = { keyId: string; label: string; secret: string } | null;
export type RevealEvent =
  | { type: "created"; key: CreatedKey }
  | { type: "dismissed" }
  | { type: "signed-out" }
  | { type: "left" };

/** Created replaces any earlier secret; every other event drops it. */
export function nextReveal(_prev: Reveal, event: RevealEvent): Reveal {
  if (event.type === "created")
    return { keyId: event.key.key_id, label: event.key.label, secret: event.key.secret };
  return null;
}

export type CopyOutcome = "copied" | "failed";

/** Copy through the clipboard API; "failed" tells the caller to select the text for a manual copy. */
export async function copySecret(
  secret: string,
  clipboard: Pick<Clipboard, "writeText"> | undefined = globalThis.navigator?.clipboard,
): Promise<CopyOutcome> {
  try {
    if (!clipboard) return "failed";
    await clipboard.writeText(secret);
    return "copied";
  } catch {
    return "failed";
  }
}
