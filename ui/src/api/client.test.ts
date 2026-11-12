import { expect, test } from "vitest";
import { checkHealth, makeClient } from "./client";

const answer = (body: unknown, status = 200): typeof fetch =>
  async () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    });

test("health is true when the API says ok", async () => {
  expect(await checkHealth(makeClient("http://x", answer({ status: "ok" })))).toBe(true);
});

test("health is false on another answer or a failed request", async () => {
  expect(await checkHealth(makeClient("http://x", answer({ status: "down" })))).toBe(false);
  const broken: typeof fetch = async () => {
    throw new TypeError("offline");
  };
  expect(await checkHealth(makeClient("http://x", broken))).toBe(false);
});
