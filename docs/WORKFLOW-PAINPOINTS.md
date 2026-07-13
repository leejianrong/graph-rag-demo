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

_(Started by the agent running Step C — Opus 4.8, fresh context. C1 (Frame)
complete; C2–C7 pending in the same context. Findings so far below.)_

### P0 — broken dependency (blocks later sub-steps C6/C7)

13. **[NEW — investigated & narrowed] `/shaping` references a `/breadboarding`
    skill that does not exist, but the method is (partly) inlined in `/shaping`
    itself.** `~/.claude/skills/breadboarding` is **missing**, and a
    `grep -rli breadboard ~/.claude/skills/` finds the standalone procedure
    **nowhere** — the only hits are a doc-name mention in `write-tech-blog-post`
    and unrelated "vertical slice" matches in `dev-playbook`/
    `project-manager-kanban`. So the process-plan-product install line
    (`rjs/shaping-skills … --skill breadboarding`) is a **dangling reference**:
    there is no such skill to install (confirmed known issue per product owner).
    `/shaping`'s SKILL.md offloads **Detailing** and **Slicing** to it
    ("Use the `/breadboarding` skill…") — the exact hidden-skill / broken-dep
    pattern from Step A (#1–#3). **Mitigation found:** `/shaping` *inlines the
    breadboard output spec* (SKILL.md ~311–326, 443–452) — the three deliverables
    (UI Affordances table, Non-UI Affordances table, Wiring diagram grouped by
    Place), plus "tables are source of truth / diagram renders them," the
    `Wires Out` column, and `CURRENT` as reserved baseline. That is enough to run
    C4/C6 as an **inline fallback**; only the standalone skill's finer procedural
    detail (exact column schemas, slicing recipe) is unrecoverable. So this is
    **not a hard blocker** — downgraded from "blocks C4/C6" to "degraded, inline
    fallback used."
    → **Fix:** either ship the real `breadboarding` skill (the install line
    promises it), or — since its essentials already live in `/shaping` —
    **remove the `/breadboarding` cross-references from `/shaping` and promote the
    inlined spec to the canonical method**, so `/shaping` is self-contained and
    stops pointing at a skill that was never published.
    → **Workaround applied (C4):** built `docs/BREADBOARD.md` directly from the
    breadboard spec inlined in `/shaping` — UI Affordances table + Non-UI
    Affordances table (with a `Wires Out` column) + a Mermaid wiring diagram
    grouped by Place, generated *from* the tables (tables = source of truth).
    Column schema was reconstructed from the `/shaping` examples since the
    standalone skill's canonical schema is unavailable; if the real
    `/breadboarding` ever ships with different columns, this doc may need a
    reformat. No content was blocked — only the exact table shape is
    unverified-against-canonical.

### P2 — minor

14. **[CORRECTED — prediction #6 does NOT fire for `/shaping`.]** Predicted
    painpoint #6 said templates assume `REQS.md`/`CONTEXT.md` at repo root.
    `grep -niE 'reqs|context|prd|repo root|frame\.md'` over
    `~/.claude/skills/shaping/` returns **nothing** — the `/shaping` skill never
    references those files or a repo-root location at all. It is fully
    path-agnostic, so moving the docs under `docs/` caused it **zero** friction.
    Painpoint #6 is a property of the *process-plan-product templates*, not of
    `/shaping`. Override #2 was therefore a no-op for this step (harmless).

15. **[NEW] `/shaping` is silent on *where* to write its documents.** The skill's
    Documents table names the artifacts (Frame, Shaping doc, Slices doc) and
    mandates `shaping: true` frontmatter, but specifies **no filename and no
    directory** for any of them — not repo root, not `docs/`, nothing. Left to
    its defaults the agent must guess a location. Override #1 (write under
    `docs/` as `FRAME.md`, `SHAPING.md`, …) supplied the missing convention and
    was genuinely needed — but as a gap-filler, not a correction. Distinct from
    #6: the problem isn't a *wrong* hardcoded path, it's the *absence* of any
    path guidance.
    → **Fix:** have `/shaping` state a default output location/filenames (e.g.
    `FRAME.md`, `SHAPING.md`, `SLICES.md` beside the other planning docs) and let
    the invocation override it, mirroring how `to-prd` should parameterize its
    output target (#9).

### P1 — process-ordering friction (shaping runs after the shape is decided)

16. **[NEW] Step C (shaping) runs *after* A/B have already locked 9 ADRs, so
    shaping's core engine has little to bite on.** `/shaping` is built to explore
    a solution space: propose mutually-exclusive shapes (A/B/C), play them
    against R in a binary fit check, and *pick one*. But in the
    build-plan-product ordering, the winning shape is already fully determined by
    the Accepted ADRs before shaping starts — there is no live A-vs-B choice left
    to make. Consequently R (C2) reads almost entirely "Must-have + will-be-
    satisfied," and C3's shapes risk being a single foregone shape (essentially
    "the ADRs") with no genuine alternative to compare — making the fit check a
    formality rather than a decision tool. Not blocking (the R set + a single
    detailed shape + breadboard are still useful as a consolidated spec view),
    but it means the skill is being used against its grain.
    → **Fix:** either (a) have build-plan-product tell Step C to *skip shape
    exploration* and go straight to detailing/breadboarding the already-decided
    shape (frame → R → Detail → breadboard → slice), or (b) run shaping *before*
    ADRs are frozen. As-is, document that C3 will be a single "CURRENT-of-record"
    shape derived from the ADRs, not a genuine multi-shape bake-off.

17. **[NEW] Step C (shaping) and Step D (breadboarding) overlap — the `/shaping`
    skill already does D's job.** The build-plan-product flow lists **C = shaping**
    then **D = breadboarding** as *separate* steps (and this very file has a
    distinct `## Step D — breadboarding` heading). But `/shaping` explicitly owns
    breadboarding *and* slicing within itself ("Shaping → Slicing"; the Documents
    table includes Breadboard + Slices doc). Running Step C to completion
    therefore already produced `BREADBOARD.md` **and** `SLICES.md` — leaving Step
    D with nothing distinct to do. The two steps are redundant as scoped.
    → **Fix:** collapse D into C (shaping *is* frame→R→shape→breadboard→slice), or
    re-scope D to mean "refine/validate the breadboard against real code" —
    something genuinely post-shaping — so the steps don't duplicate. Update the
    `## Step D` heading here accordingly.

18. **[NEW — keep, not a fix] No spike was needed, and that's correctly
    signalled.** `/shaping` says a selected shape "should have no flags (all ⚠️
    resolved), or explicit spikes to resolve them." Shape A had **zero** flagged
    unknowns (every mechanism is pinned by an ADR), so no `SPIKE-*.md` was
    created — the flag convention did its job as a gate. Noting it so the absence
    of a spike file isn't later read as a missed step.

_— observed running Step C (C1–C5, full shaping arc) on Opus 4.8, fresh context._

## Step D — breadboarding

_(To be filled in by the agent that runs Step D.)_

## Step E — grill-with-docs (extract ADRs & final consistency)

### P0 — broken dependency (same as Step A #1–#3, still unfixed)

19. **[CONFIRMED — Step A's P0 recurs verbatim] `grill-with-docs` is still a
    dangling one-liner.** `~/.claude/skills/grill-with-docs/SKILL.md` reads, in
    full: *"Run a `/grilling` session, using the `/domain-modeling` skill."* Both
    `/grilling` and `/domain-modeling` are **still missing**
    (`ls ~/.claude/skills/{grilling,domain-modeling}` → MISSING). So Step E, like
    Step A, has **nothing to execute** as shipped. Recovered the intended
    behaviour from `build-plan-product/assets/process-plan-product.md` §E ("E1:
    Ensure consistency and capture decisions" — review shaping files, assess
    CONTEXT/PRD/ADRs, log inconsistencies + questions to QUESTIONS.md) and ran
    that method by hand.
    → **Fix:** same as Step A #1–#3 — either ship `/grilling` + `/domain-modeling`
    or inline the method into `grill-with-docs`. This has now bitten **two**
    separate steps (A and E) that both invoke `grill-with-docs`; the fix is
    high-value.

### P2 — minor

20. **[NEW] The Step E prompt lists shaping files but omits `SLICES.md` and
    `FRAME.md`.** `process-plan-product.md` §E says review "FRAME.md +
    SHAPING.md + BREADBOARD.md + (…spike files)" — it names FRAME and SHAPING but
    **not** `SLICES.md`, even though slicing is part of Step C's output and is
    exactly where a consistency issue surfaced (Q46, the ES-write-timing gap).
    Reviewed it anyway. → **Fix:** add `SLICES.md` to the §E review list.

21. **[NEW — keep] Step E earned its keep: the sweep found 3 real items.** Even
    though shaping was derived straight from the ADRs, the cross-check surfaced a
    genuine REQS↔ADR-0001 conflict (raw-doc write timing, Q46), an unspecified
    node type (NORP, Q47), and a load-bearing decision with no ADR (the port
    seam → proposed ADR-0010, Q48). The "extract ADRs & check consistency" step
    is worth keeping in the flow — it's not a rubber stamp.

_— observed running Step E on Opus 4.8, fresh context (recovered method)._
