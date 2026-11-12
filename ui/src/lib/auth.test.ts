import { beforeEach, expect, test, vi } from "vitest";
import { api, session, signIn, signOut } from "./auth.svelte";

const KEY = "test-key-abcd";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

beforeEach(() => signOut());

test("a good key signs in and shows the tenant, never the whole key", async () => {
  const fetchImpl = vi.fn(async () =>
    json({ tenant: "acme", namespace: null, key_suffix: KEY.slice(-4) }),
  ) as unknown as typeof fetch;
  expect(await signIn(`  ${KEY} `, fetchImpl, "http://x")).toEqual({ ok: true });
  expect(session.tenant).toBe("acme");
  expect(session.keySuffix).toBe("abcd");
  const request = (fetchImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0] as Request;
  expect(request.headers.get("authorization")).toBe(`Bearer ${KEY}`);
  expect(new URL(request.url).pathname).toBe("/v1/me");
});

test("a bad key shows an error and keeps nothing", async () => {
  const fetchImpl = (async () =>
    json({ error: { code: "unauthorized", message: "no" } }, 401)) as unknown as typeof fetch;
  const result = await signIn(KEY, fetchImpl, "http://x");
  expect(result.ok).toBe(false);
  expect(!result.ok && result.message).toMatch(/not accepted/);
  expect(session.key).toBeNull();
});

test("an empty key is refused without a request", async () => {
  const fetchImpl = vi.fn() as unknown as typeof fetch;
  expect((await signIn("   ", fetchImpl, "http://x")).ok).toBe(false);
  expect(fetchImpl).not.toHaveBeenCalled();
});

test("an unreachable API and a rate limit each get their own message", async () => {
  const down = (async () => {
    throw new TypeError("offline");
  }) as unknown as typeof fetch;
  const limited = (async () => json({ error: {} }, 429)) as unknown as typeof fetch;
  const a = await signIn(KEY, down, "http://x");
  const b = await signIn(KEY, limited, "http://x");
  expect(!a.ok && a.message).toMatch(/did not answer/);
  expect(!b.ok && b.message).toMatch(/Too many/);
  expect(session.key).toBeNull();
});

test("sign out clears the key and the tenant", async () => {
  await signIn(KEY, (async () => json({ tenant: "acme", key_suffix: "abcd" })) as never, "http://x");
  expect(session.key).toBe(KEY);
  signOut();
  expect(session).toMatchObject({ key: null, tenant: null, keySuffix: null, notice: null });
});

test("the signed-in client attaches the key, and a 401 signs out with a notice", async () => {
  await signIn(KEY, (async () => json({ tenant: "acme", key_suffix: "abcd" })) as never, "http://x");
  const seen: string[] = [];
  const spy = (async (input: Request) => {
    seen.push(input.headers.get("authorization") ?? "");
    return json({ error: {} }, 401);
  }) as unknown as typeof fetch;
  const { makeClient } = await import("../api/client");
  const client = makeClient("http://x", spy, {
    key: () => session.key,
    onUnauthorized: () => signOut("rejected"),
  });
  await client.GET("/v1/me");
  expect(seen).toEqual([`Bearer ${KEY}`]);
  expect(session.key).toBeNull();
  expect(session.notice).toBe("rejected");
  expect(api).toBeDefined();
});
