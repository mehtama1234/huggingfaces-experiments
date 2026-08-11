# InteracVid — reproducible experiments

## E1 — Cause-vs-decoration probe (does the query actually matter?) — **runnable now**
- **Setup:** take ~200 real-query tuples `(𝒞, 𝒬, 𝒴)`. Prompt a small VLM/LLM planner
  to produce the response caption in two conditions: (a) given context 𝒞 + query 𝒬,
  (b) given 𝒞 with the query **shuffled/removed**. Score outputs against the
  ground-truth response with a judge model + a few human raters.
- **Shows:** if (a) beats (b) meaningfully, the data genuinely encodes
  **stimulus→response causality** (not just "keep talking about the same topic").
  This is the cheapest test of the paper's central premise, and needs no video
  generation — just text/caption scoring.
- **Feasibility:** high.

## E2 — Reconstructed-query fidelity audit
- **Setup:** on videos that *do* have real chat, hide it and run the
  reconstructed-query branch. Compare the VLM-reconstructed query to the true chat
  message (embedding similarity + a human "would this have prompted this reply?"
  rating on ~100 pairs).
- **Shows:** whether the 903h reconstructed branch is trustworthy or introduces a
  distribution shift — it's ~10× the real branch, so it could dominate training.
- **Feasibility:** high (needs a VLM + a small labeled set).

## E3 — Tiny interactive-response fine-tune on one domain
- **Setup:** pick one narrow scenario (cooking, unboxing), assemble ~2–5k tuples,
  LoRA-fine-tune a small talking-head / AV generator to produce a short spoken+
  visual reply given context+query. Baseline = same generator, no fine-tune.
  Evaluate lip-sync distance, identity consistency, and human "does the reply fit
  the question and the scene?".
- **Shows:** whether interactive-response supervision improves *appropriateness of
  reaction* (not just AV quality) even at hobby scale.
- **Feasibility:** medium.

**Ethics/licensing note:** this data is scraped from real creators' livestreams.
Any reproduction must respect platform ToS, creator consent, and privacy — treat
the *methodology* as the reusable artifact, not a license to re-scrape.
