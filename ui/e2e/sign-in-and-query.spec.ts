import { expect, test } from "@playwright/test";

const TEXT = "Ada Lovelace works for Acme Corp. Acme Corp is based in London.";

test("sign in, ingest a text, ask a question, sign out", async ({ page }) => {
  const baseUrl = process.env.E2E_BASE_URL!;
  const key = process.env.E2E_KEY!;

  // No external network: refuse everything but the service, and fail on any attempt or failed request.
  const violations: string[] = [];
  await page.route(/^https?:\/\/(?!127\.0\.0\.1)/, (route) => {
    violations.push(`blocked external request: ${route.request().url()}`);
    return route.abort();
  });
  const fonts: string[] = [];
  page.on("response", (r) => {
    if (/\.woff2?$/.test(r.url()) && r.ok()) fonts.push(r.url());
  });
  page.on("requestfailed", (r) => violations.push(`request failed: ${r.url()}`));
  page.on("console", (m) => {
    if (/content security policy/i.test(m.text())) violations.push(m.text());
  });

  await page.goto(baseUrl);
  await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();

  // A wrong key is refused with a message and stays on the sign-in page.
  await page.getByLabel("API key").fill("grk_nope_nope");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("alert").filter({ hasText: /key/i })).toBeVisible();
  await expect(page.getByTestId("tenant")).toHaveCount(0);

  // The real key shows the tenant it names.
  await page.getByLabel("API key").fill(key);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByTestId("tenant")).toHaveText("e2e-tenant");

  // Ingest pasted text and wait for the job to finish.
  await page.getByRole("link", { name: "Ingest" }).click();
  await page.getByRole("tab", { name: /text/i }).click();
  await page.getByLabel("Text to ingest").fill(TEXT);
  await page.getByLabel("Document name").fill("e2e-note.txt");
  await page.getByRole("button", { name: "Ingest", exact: true }).click();
  await expect(page.locator(".job .chip", { hasText: "Done" })).toBeVisible({ timeout: 45_000 });

  // Ask a question and see an answer.
  await page.getByRole("link", { name: "Query" }).click();
  await page.getByLabel("Question").fill("Where is Acme Corp based?");
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.getByRole("heading", { name: "Answer" })).toBeVisible();
  await expect(page.locator(".answer .big")).not.toHaveText(/No answer/);

  // Signing out returns to the sign-in page.
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();

  // The bundled IBM Plex files were served, not fetched from a font host.
  expect(fonts.length).toBeGreaterThan(0);
  expect(violations).toEqual([]);
});
