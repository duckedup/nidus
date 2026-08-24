## Fit

Feature thought work, before any spec: is this the right fit, does it make sense, would
users use it? Read-only — no branch, no blueprints, no code. The target is an idea (a
description, or an issue number to resolve with `bd show`).

1. **Check it is not already decided.** Search open and closed issues
   (`bd search "<terms>"`, `bd list --all --label decision`) and SPEC §9's
   shipped/deferred/DECIDED entries for prior art — the migration carried the closed
   history across, so a rejected idea is still findable. One returning without new
   evidence gets the old answer, cited.
2. **Gap or feature?** If the product already implies the capability (a doc describes it,
   a surface half-has it, sibling surfaces have it and this one lacks it), it is a **gap**:
   skip fit, file it as a `task`/`bug` with the evidence, done. Fit is for genuinely new
   capability.
3. **Right fit.** Judge against SPEC §1 (the core foundation and thesis) and §2
   (goals/non-goals): does it belong in nidus core, or in the host application, an SDK,
   the docs, or a separate tool? Does the public positioning (development and small-scale
   use, nothing promised beyond what ships) survive it?
4. **Does it make sense — the cost side.**
   - On-disk or wire format change? SPEC §9's rule applies: a format change needs a
     **named caller**; query-path features are judged on their own merits.
   - New dependency? The build budget is a CI-enforced gate; a heavy dep is a design
     change, not an implementation detail.
   - New surface? Every surface owes a load-bearing test in CI (§11), and a new server
     capability owes all three SDKs, the HTTP reference, and MCP consideration — count
     that cost, not just the core diff.
5. **Would users use it — name the caller.** A concrete user or workflow that is blocked
   or degraded today, what they do instead (the workaround is evidence), and what changes
   for them if this ships. "It would be nice" names nobody.
6. **Verdict**, recorded durably, one of:
   - **Pursue**: file the issue (`feature` label, priority argued from the caller), with
     the assessment as the body. Offer to continue into **Spec**.
   - **Defer with a trigger**: file a `decision`-labeled issue naming the condition that
     reopens it ("revisit when a caller asks for X"), the SPEC §9 pattern.
   - **Reject**: file or comment the decision with the reason, so the next person who has
     the idea finds the answer instead of re-deriving it. Format-adjacent rejections also
     earn a DECIDED entry in SPEC §9.

The gate the user sees is the verdict and its reasoning, not a wall of research. Three
sharp paragraphs beat ten pages.

