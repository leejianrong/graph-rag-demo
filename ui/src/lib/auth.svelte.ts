import { makeClient } from "../api/client";

/**
 * The signed-in session. The key lives only in this module's memory: a reload or a closed tab
 * signs the user out, and nothing here writes to storage, cookies, the URL or the console.
 */
export const session = $state<{
  key: string | null;
  tenant: string | null;
  namespace: string | null;
  keySuffix: string | null;
  /** Why the user is on the sign-in view, when it was not their own doing. */
  notice: string | null;
}>({ key: null, tenant: null, namespace: null, keySuffix: null, notice: null });

export function signOut(notice: string | null = null) {
  session.key = null;
  session.tenant = null;
  session.namespace = null;
  session.keySuffix = null;
  session.notice = notice;
}

/** The API client for signed-in views: it attaches the key and signs out on a 401. */
export const api = makeClient("", undefined, {
  key: () => session.key,
  onUnauthorized: () => signOut("Your key was rejected, so you were signed out. Sign in again."),
});

export type SignInResult = { ok: true } | { ok: false; message: string };

/** Check `key` against `/v1/me`; on success it becomes the session, on failure nothing is kept. */
export async function signIn(rawKey: string, fetchImpl?: typeof fetch, baseUrl = ""): Promise<SignInResult> {
  const key = rawKey.trim();
  if (!key) return { ok: false, message: "Enter your API key." };
  try {
    const { data, response } = await makeClient(baseUrl, fetchImpl).GET("/v1/me", {
      headers: { Authorization: `Bearer ${key}` },
    });
    if (data) {
      session.key = key;
      session.tenant = data.tenant;
      session.namespace = data.namespace ?? null;
      session.keySuffix = data.key_suffix ?? null;
      session.notice = null;
      return { ok: true };
    }
    if (response.status === 401) {
      return {
        ok: false,
        message:
          "That key was not accepted. Check it for a missing character, or ask your operator for a new one.",
      };
    }
    if (response.status === 429) {
      return {
        ok: false,
        message: "Too many attempts from this address. Wait a minute and try again.",
      };
    }
    return { ok: false, message: `The API answered ${response.status}. Try again in a moment.` };
  } catch {
    return { ok: false, message: "The API did not answer. Check that the service is running." };
  }
}
