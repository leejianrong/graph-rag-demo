import { expect, test, vi } from "vitest";
import { makeClient } from "../api/client";
import {
  copySecret,
  createBlockedReason,
  createKey,
  describeKeyError,
  formatCreated,
  listKeys,
  maskedSuffix,
  nextReveal,
  revokeBlockedReason,
  revokeKey,
  type CreatedKey,
  type KeyInfo,
} from "./keys";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
const info = (n: number, over: Partial<KeyInfo> = {}): KeyInfo => ({
  key_id: `k${n}`,
  label: `key ${n}`,
  created_at: "2026-10-02T09:41:00Z",
  key_suffix: "wxyz",
  status: "active",
  revoked_at: null,
  current: false,
  ...over,
});
const clientWith = (impl: (r: Request) => Response) => {
  const fetchImpl = vi.fn(async (r: Request) => impl(r));
  return { client: makeClient("http://x", fetchImpl as unknown as typeof fetch), fetchImpl };
};
const sent = (f: ReturnType<typeof vi.fn>) => f.mock.calls[0][0] as Request;

test("listKeys unwraps the keys array", async () => {
  const { client, fetchImpl } = clientWith(() => json({ keys: [info(1), info(2)] }));
  const r = await listKeys(client);
  expect(r.ok && r.value.map((k) => k.key_id)).toEqual(["k1", "k2"]);
  expect(new URL(sent(fetchImpl).url).pathname).toBe("/v1/keys");
});

test("createKey posts the label and returns the secret", async () => {
  const secret = `s-${Math.random().toString(36).slice(2)}`;
  const { client, fetchImpl } = clientWith(() => json({ ...info(3), secret }, 201));
  const r = await createKey(client, "ci");
  expect(r.ok && r.value.secret).toBe(secret);
  expect(sent(fetchImpl).method).toBe("POST");
  expect(await sent(fetchImpl).json()).toEqual({ label: "ci" });
});

test("revokeKey sends DELETE for the id", async () => {
  const { client, fetchImpl } = clientWith(() => json(info(1, { status: "revoked" })));
  const r = await revokeKey(client, "k1");
  expect(r.ok && r.value.status).toBe("revoked");
  expect(sent(fetchImpl).method).toBe("DELETE");
  expect(new URL(sent(fetchImpl).url).pathname).toBe("/v1/keys/k1");
});

test("a refusal becomes a failure with a message, and a dead network too", async () => {
  const { client } = clientWith(() =>
    json({ error: { code: "invalid_request", message: "m", details: { reason: "key_limit" } } }, 400),
  );
  const r = await createKey(client, "");
  expect(!r.ok && r.status).toBe(400);
  expect(!r.ok && r.message).toMatch(/10 active keys/);
  const down = makeClient("http://x", (async () => {
    throw new TypeError("x");
  }) as unknown as typeof fetch);
  const d = await listKeys(down);
  expect(!d.ok && d.status).toBeNull();
});

test("each refusal reason has its own message", () => {
  const err = (reason?: string, field?: string) => ({ error: { message: "bad", details: { reason, field } } });
  expect(describeKeyError(400, err("key_limit"))).toMatch(/Revoke one/);
  expect(describeKeyError(400, err("own_key"))).toMatch(/signed in with/);
  expect(describeKeyError(400, err("auth_disabled"))).toMatch(/without authentication/);
  expect(describeKeyError(400, err(undefined, "label"))).toMatch(/label/);
  expect(describeKeyError(404, undefined)).toMatch(/no longer exists/);
  expect(describeKeyError(401, undefined)).toMatch(/signed out/);
  expect(describeKeyError(500, undefined)).toMatch(/500/);
});

test("429 shows the retry time from the body, else the header", () => {
  const body = { error: { details: { retry_after_seconds: 12.2 } } };
  expect(describeKeyError(429, body)).toMatch(/13 seconds/);
  expect(describeKeyError(429, undefined, "7")).toMatch(/7 seconds/);
  expect(describeKeyError(429, undefined, null)).toMatch(/a moment/);
});

test("view-model helpers", () => {
  expect(maskedSuffix({ key_suffix: null })).toBe("••••");
  expect(maskedSuffix({ key_suffix: "7f3a" })).toBe("••••7f3a");
  expect(revokeBlockedReason(info(1, { current: true }))).toMatch(/using now/);
  expect(revokeBlockedReason(info(1))).toBeNull();
  const ten = Array.from({ length: 10 }, (_, i) => info(i));
  expect(createBlockedReason(ten, false)).toMatch(/10 active/);
  expect(createBlockedReason([...ten.slice(1), info(9, { status: "revoked" })], false)).toBeNull();
  expect(createBlockedReason([], true)).toBeTruthy();
  expect(formatCreated("2026-10-02T09:41:07.123Z")).toBe("2026-10-02 09:41");
});

test("the secret is held until dismissed, signed out, left, or replaced", () => {
  const made = (n: number): CreatedKey => ({ ...info(n), secret: `sec-${n}` });
  const held = nextReveal(null, { type: "created", key: made(1) });
  expect(held?.secret).toBe("sec-1");
  expect(nextReveal(held, { type: "created", key: made(2) })?.secret).toBe("sec-2");
  for (const type of ["dismissed", "signed-out", "left"] as const) {
    expect(nextReveal(held, { type })).toBeNull();
  }
});

test("copySecret reports success and failure", async () => {
  const writeText = vi.fn(async () => {});
  expect(await copySecret("abc", { writeText })).toBe("copied");
  expect(writeText).toHaveBeenCalledWith("abc");
  const denied = {
    writeText: async () => {
      throw new Error("denied");
    },
  };
  expect(await copySecret("abc", denied)).toBe("failed");
  expect(await copySecret("abc", undefined)).toBe("failed");
});
