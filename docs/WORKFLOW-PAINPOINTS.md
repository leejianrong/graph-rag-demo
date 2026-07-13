# build-plan-product / build-plan-specs — workflow pain points

A shared log of friction, bugs, and broken dependencies hit while running the
`build-plan-product` (and later `build-plan-specs`) process on this project.
The goal is to **consolidate these and edit the skills** so the process runs
cleanly next time.

> **For agents appending here:** add findings under the step you were running,
> keep the existing severity convention (**P0** blocks the step · **P1** real
> friction · **P2** minor), state the concrete symptom + file/skill involved,
> and end each with a `→ Fix:` suggestion. Sign each entry with the step and
> model you were on. Don't delete others' entries.

---

## Step A — grill-with-docs (initial grilling + domain modeling)

### P0 — broken dependency chain (blocks the step)

1. **`grill-with-docs` depends on two uninstalled skills.**
   `~/.claude/skills/grill-with-docs/SKILL.md` is a one-liner:
   > Run a `/grilling` session, using the `/domain-modeling` skill.
   Neither `/grilling` nor `/domain-modeling` exists under `~/.claude/skills/`.
   The skill is therefore non-functional as shipped — it points at nothing.

2. **The documented install never installs those deps.**
   `build-plan-product/assets/process-plan-product.md` install block:
   ```
   npx skills add -g mattpocock/skills --skill grill-with-docs --skill to-prd
   npx skills add -g rjs/shaping-skills --skill shaping --skill breadboarding
   ```
   Installs `grill-with-docs` but **not** `grilling` / `domain-modeling`, so a
   user who follows the instructions exactly ends up with a broken step A.
   → **Fix:** either add the missing skills to the install list, or bundle their
   methodology directly into `grill-with-docs` so it is self-contained.

3. **No graceful fallback when deps are missing.** `grill-with-docs/SKILL.md`
   has no embedded methodology — if the delegated skills are absent there is
   literally nothing to execute. (Recovered the intended behaviour from a prior
   project's transcript instead of failing.)
   → **Fix:** inline the interview + domain-modeling method as a fallback.

### P1 — process handoff friction

4. **Guide-vs-doer context boundary is manual and easy to violate.**
   `build-plan-product` insists each step runs in a FRESH context and that the
   guide window is "guidance only." But the natural user action is to run
   `/grill-with-docs` right there in the guide window (which is what happened).
   Nothing detects or prevents it.
   → **Fix:** either explicitly bless running in-window, or have the guide
   launch the step (subagent) rather than relying on the user to open a new
   chat and reset the model.

5. **Model switching is advisory only.** The guide states Medium/High + fresh
   context per step, but nothing reminds/enforces at execution time, so it's
   easy to run a step on the wrong model. (Ran step A on Opus, not Medium.)

### P2 — minor

6. **REQS.md path assumed to be repo root.** Templates hardcode `REQS.md` at
   root; a prior project kept it at `docs/REQS.md` and every prompt had to be
   hand-edited. → Make the path a parameter, or detect it.

7. **No specified location/format for answers.** The grilling produced a
   `QUESTIONS.md`, but nothing says *where the user writes answers*. This led to
   a parallel `ANSWERS.md` (a hand-copied duplicate of `QUESTIONS.md`), which
   then risks drift from the canonical question list.
   → **Fix:** have the skill state the answer convention (answer inline in
   `QUESTIONS.md`, or generate a dedicated answer sheet with a stable format).

_— observed running Step A on Opus 4.8 (should have been Medium)._

---

## Step B — to-prd (write the PRD)

_(Predictions below now CONFIRMED/CORRECTED by the agent that actually ran
Step B — Opus 4.8 (High), fresh context, driven from `CONTEXT.md` + `REQS.md`
+ `docs/adr/*` per the invocation's two overrides.)_

### P1 — assumptions that clash with the build-plan-product process

8. **[CONFIRMED] `to-prd` assumes same-context synthesis, but the process
   demands a FRESH context.** The skill says *"Do NOT interview the user — just
   synthesize what you already know [from this conversation]."* In this fresh
   window there was **no prior conversation**; synthesis only worked because the
   invocation explicitly redirected me to treat `CONTEXT.md` + `REQS.md` +
   `docs/adr/*.md` as the context (override #2). Without that redirection the
   skill, read literally, has nothing to synthesize and would either stall or
   (worse) start interviewing — the one thing it tells itself not to do.
   → **Fix:** `to-prd` should accept explicit source docs and synthesize from
   them when the conversation is empty (detect empty context → look for
   `CONTEXT.md`/`REQS.md`/`docs/adr/`).

9. **[CONFIRMED] `to-prd` wants to publish to an issue tracker; the process
   wants a file.** The skill's step 3 hardcodes *"publish it to the project
   issue tracker [and] apply the `ready-for-agent` triage label."* The
   invocation had to override this explicitly (override #1) to get output at
   **`docs/PRD.md`**. Left to its defaults the skill would have looked for a
   tracker that doesn't exist for this local demo. The `<prd-template>` itself
   is file-friendly and worked well as-is; only the *destination* clashed.
   → **Fix:** make the output target a parameter (file path vs. tracker);
   default the build-plan-product flow to `docs/PRD.md`.

10. **[PARTIALLY CONFIRMED] hidden dependency on `/setup-matt-pocock-skills`.**
    The skill still references the sibling setup skill (for tracker + triage
    label vocab), same shape as `grill-with-docs`'s missing deps in Step A. I
    could **not** verify whether it's installed because override #1 told me not
    to touch the tracker path — so the dependency was side-stepped, not
    exercised. It remains a latent P1 for anyone who runs `to-prd` *without* the
    file override: they'd hit the same broken-dep chain flagged in Step A.
    → **Fix:** same as Step A — bundle the vocabulary into the skill or drop the
    tracker path entirely for file-output flows.

### P2 — minor (newly observed)

11. **[NEW] Skill step 2 ("check seams with the user") worked well and should
    stay.** Sketching test seams and confirming them via a quick question
    *before* writing the PRD was the smoothest part of the step — it caught a
    real fork (fakes-only vs. fakes + thin real-container layer) that would
    otherwise have been guessed. No friction; noting it as a keep, not a fix.

12. **[NEW] `to-prd` gives no guidance on greenfield "prior art" for tests.**
    The template's Testing Decisions section asks for *"prior art for the tests
    (similar tests in the codebase)."* This repo has **no code yet**, so there
    is no prior art — I had to state that explicitly and reframe the primary
    seam as the prior art future tests will follow. Fine once recognized, but
    the template assumes a brownfield repo.
    → **Fix:** have the template acknowledge the greenfield case (no prior art →
    establish the pattern) so the agent isn't fishing for tests that can't
    exist.

_— observed running Step B on Opus 4.8 (High), fresh context._

---

## Step C — shaping (frame → requirements → shapes → fit → detail)

_(To be filled in by the agent that runs Step C.)_

## Step D — breadboarding

_(To be filled in by the agent that runs Step D.)_

## Step E — grill-with-docs (extract ADRs & final consistency)

_(To be filled in by the agent that runs Step E.)_
