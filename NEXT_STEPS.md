# TG-IES → TNNLS Revision Roadmap

**Budget: 50 GPU-hours maximum, 3 seeds per cell (not 5).** This supersedes the earlier
unlimited-GPU draft of this document — that version is preserved in
[§A — Deferred / future work](#a-deferred--future-work-not-in-the-50h-budget) below so
nothing is lost, but everything in §A is explicitly **out of scope** for this pass.
Reviewer item numbers `[#N]` refer to the original TNNLS critique for traceability.

## How to use this document

Work through the priority-ordered list in §1 top to bottom. Each row's "cumulative"
column tracks running GPU-hours spent so the person executing this can see exactly how
much budget remains at any point and stop (or ask before continuing) if something runs
over its estimate. After every numbered item, do the integration checklist in §2 before
moving to the next one — don't let results pile up unintegrated.

Conventions to preserve (same as the existing repo):
- CSV naming `results/<method>_<dataset>_<model>_<optimizer>_seed<k>.csv`, one row/epoch.
- `run_*.sh` driver scripts skip a cell if its CSV already has `>= 201` lines — safe to
  re-invoke repeatedly to resume. `train.py` has no mid-run checkpoint, so a killed run
  restarts that cell from epoch 0 — budget for that when estimating.
- `setup_seed(seed)` is called once per run before model init/data loading so all
  methods sharing a `--seed` are paired (init + data order) — required for the paired
  stats in `app:stats`. Keep this for every new run.
- Wall-clock-sensitive runs (efficiency numbers) must run alone, sequentially. Everything
  else can run concurrently if multiple GPUs are available, which shortens wall-clock
  time but does not change the GPU-hour totals below.

---

## §1. Priority-ordered plan (cumulative ≤ 50h)

Per-run cost basis (measured in this repo unless marked *est.*): CIFAR-10/CIFAR-100 +
ResNet-18, 200 epochs ≈ **0.25h**; DenseNet-121 + CIFAR ≈ **0.75h** *(est., from a prior
session's 8-run/5-6h batch)*; a CIFAR-native ViT-Tiny (see item 9) ≈ **0.5h** *(est.)*.

| # | Item(s) | What | Runs | Cost | Cumulative |
|---|---|---|---|---|---|
| 0 | `[#2,3,4,8,17]` | ✅ **DONE** (2026-08-30) — `run_p2_p3_chain.sh` (LR-schedule sweep + drift-term instrumentation) finished clean, `P2_P3_CHAIN_ALL_DONE`, all exits 0. | — | ~4h *(sunk)* | 0h |
| 1 | `[#10]` | ✅ **DONE** (2026-09-02) — CIFAR-100 half done 2026-08-31 (76.01±0.08% acc / 57.76±0.04% saved). DenseNet-121 half also done: `run_densenet_seeds12.sh` completed (8 new runs, seeds 1-2), 95.28±0.09% acc / 54.53±0.66% saved. Full paper integration for both (table, abstract, intro, `tab:zeroshot`, stats appendix incl. new paired-stats block, full-results appendix, every single-seed prose mention) done, recompiled clean. See `EXPERIMENT_LOG.md`. | 8 + 8 = 16 runs | 7.5h | 7.5h |
| 2 | `[#2,#17]` | ✅ **DONE** (2026-08-31) — wrote `drift_term_analysis.py`, direct-instrumented $D_i^{(t)}$ (item #2) against `thm:curvature`'s decomposition. **Finding: complicates the theorem's "drift dominates" reading** — weak correlation (r=0.23–0.37) between $\eta_tD_i^{(t)}$ and observed $\Delta^2L_i^{(t)}$, and the drift term does NOT dominate the curvature+remainder residual under any schedule (ratio 0.00–0.69×), even in late training. Paper updated (`sec:level3`, `sec:lrmechanism`, abstract, conclusion, `sec:limitations` item 3, new `fig:driftvalidation`). Item #17's `‖g_i‖` vs `\|Δ²L_i\|` check now also done: `drift_term_run.py` extended to save the norm, 3 schedules rerun, `analyze_norms()` added — **confirmatory finding**, corr 0.47–0.49, mastery-check-passing instances have 30–33× smaller gradient norms. New `sec:bridge` paragraph + `fig:bridgenormcheck`. See `EXPERIMENT_LOG.md`. | 3 reruns | 1.5h | 9h |
| 3 | `[#4,#18]` | ✅ **DONE** (2026-08-30) — refined `order_of_difference_analysis.py`: tested each instance exactly at its own calibrated-δ crossing epoch (not a coarse proxy). **N=2 now confirmed** as MSE-minimizing there (2–3 orders of magnitude better than N=1/N=3), resolving the earlier exploratory finding — that one used a naive population average that doesn't isolate `prop:mse`'s SNR band. Also ran the training-phase quartile analysis: empirical $\hat r_i$ rises (0.53→0.76) and its CV falls (1.05→0.75) across training quartiles — `ass:geometric` fits better in the second half of training. Paper updated (`sec:level2`, two new figures `fig:orderdiff`/`fig:phasequartile`) — see `EXPERIMENT_LOG.md`. | post-processing only | $0 | 9h |
| 4 | `[#12,#13]` | ✅ **DONE** (2026-08-31) — extended `equal_compute_analysis.py` to all 3 cells with new-baseline data. **Finding generalizes**: InfoBatch-family leads TG-IES by 2.5–4.3pts at 20–30% budget in every cell, narrows to 0.1–0.8pts by 40%, reverses on CIFAR-100 (TG-IES +0.12pt at 40%). New `sec:equalcompute` + `fig:equalcompute` (3-panel). See `EXPERIMENT_LOG.md`. | post-processing only | $0 | 7.5h |
| 5 | `[#19]` | ✅ **DONE** (verified already present, 2026-08-31) — abstract/`sec:ablation`/`sec:conclusion` already state exactly this framing from an earlier unlogged pass; no edit needed. See `EXPERIMENT_LOG.md`. | text edit only | $0 | 9h |
| 6 | `[#9]` | ✅ **DONE** (2026-08-31, updated 2026-09-03) — `tab:zeroshot` in `sec:selfcalib-scope`, now 11 distinct cells TG-IES has run on (DenseNet-121 and both ViT-Tiny cells added since), same 4 fixed constants throughout, savings range 23–58%. 8/11 seed-averaged (n=3); only the 3 LR-sweep cells remain single-seed, by design (footnoted). See `EXPERIMENT_LOG.md`. | post-processing only | $0 | 9h |
| 7 | `[#20]` | ✅ **DONE** (2026-09-03) — K*-sweep (`K ∈ {1,2,4,8,15,30}`, primary cell, 3 seeds, δ self-calibrated, `--tgies_fix_kstar`) + new trajectory-divergence instrumentation (`--dump_traj_every`/`--traj_ref_dir` added to `train.py`: baseline writes shared parameter-vector snapshots every 20 epochs, tgies runs log L2 distance against them). 21 runs total (3 baseline reruns w/ snapshots + 18 K-sweep). **Findings**: savings fall smoothly 67.0%→44.1% and accuracy rises 94.55%→94.87% as K* goes 1→30, no cliff; absolute trajectory distance from baseline *decreases and plateaus* over training for every K* (the opposite of unbounded compounding drift), with a clean, crossing-free K*-ordering matching `thm:safe`'s predicted direction at all 10 snapshot epochs — empirical evidence against the paper's biggest open theoretical gap's worst-case failure mode, not a proof of a bound. New `sec:kstartraj` (table + 2-panel figure), `rem:shared-traj`/`sec:limitations` future-work item (5) updated from "untested" to "empirically consistent with, formally still open". Recompiled clean. See `EXPERIMENT_LOG.md`. | 21 runs | ~3h | 13.5h |
| 8 | `[#11]` | ✅ **DONE** (2026-09-06) — resumed via scoped `run_items_8_11.sh` (sequential, shared GPU had an unrelated VLLM process holding most of its memory). All 3 seeds completed: $93.31\pm0.11\%$ acc, $66.72\pm0.01\%$ saved — highest savings of any method in the paper, but the largest accuracy cost too ($1.72$pt vs.\ baseline). Added to `app:newbaselines` in the new ICLR-format draft (`paper_iclr2026/`) with a Pareto-extreme framing. See `EXPERIMENT_LOG.md`. | 3 runs | 0.75h | 14.25h |
| 9 | `[#5,#6]` (scoped) | ✅ **DONE** (2026-09-03) — CIFAR-native ViT-Tiny (patch-4, 6 layers/192-dim, 2.69M params, AdamW warmup-cosine), 4 methods × 3 seeds × {CIFAR-10, CIFAR-100} = 24 runs. Baseline validated before the full comparison (84.07±0.35% / 59.25±0.24% acc). **Findings**: savings transfer and grow with difficulty (25.52±0.26% saved CIFAR-10, 26.28±0.14% CIFAR-100, 1.5–2.1× IES (literal)); mechanism is self-calibration (δ rescaled ~186–333× vs.\ the fixed-δ methods' hand-tuned constant, no retuning); absolute savings much lower than CNNs (~26% vs.\ ~54.5% DenseNet-121); small accuracy cost (0.36–0.69pt) and widest seed spread of any method, unlike CNNs. New `sec:vit` with 2 tables, abstract/intro/`tab:zeroshot`/`sec:flops`/`sec:limitations` (future-work item 2 now "partly closed")/stats appendix (2 new paired-stats blocks)/full-results appendix all updated, recompiled clean. See `EXPERIMENT_LOG.md`. | 24 runs | 12h | 26.25h |
| 10 | `[#14,#15]` | ✅ **DONE** (2026-09-03) — new `batch_size_sweep.py` (reuses `measure_efficiency.py`'s nvidia-smi-polling pattern), `{32,64,128,256}` × `{baseline, ies_shipped, tgies}`, short 8-epoch instrumented runs. Completed before the pause (ran first since it needs an uncontended GPU); `results/batch_size_sweep_summary.csv` has all 12 rows. Not yet integrated into the paper. | 12 short runs | 1h | 27.25h |
| 11 | `[#23]` | ✅ **DONE** (2026-09-06) — all 3 seeds completed (200 epochs each, restarted from 0). Probed runs match unprobed TG-IES closely, confirming the instrumentation doesn't perturb training. **Finding: `thm:safe`'s per-instance gradient-floor hypothesis is violated by ~47.4% of the excluded set at every check (range 34–59%), no decreasing trend over training** — median probed gradient norm sits near the calibrated $\varepsilon_0$ but the mean is pulled up by a heavy right tail. Sits in real tension with item 7's aggregate-plateau finding; reconciliation (candidate: `thm:safe`'s bound is an average, $1/N$-scaled quantity) is named but not verified. New appendix section `app:reactprobe` in `paper_iclr2026/`, pointer added from Limitations. See `EXPERIMENT_LOG.md`. | 3 runs (+ side-pass overhead) | 1h | 28.25h |
| 12 | `[#22]` (scoped) | ⏸️ **NOT STARTED** (2026-09-03, paused before this stage) — added `--imbalance_ratio` to `train.py` (exponential per-class subsampling profile, seed-paired via the existing `setup_seed` RNG convention so all 4 methods sharing a seed see the identical subset). Smoke-tested (ratio=10 gave exactly 500/50 min/max class counts). Not yet launched: CIFAR-100, 4 core methods, 3 seeds. | 12 runs | 3.5h | 31.75h |
| 13 | `[#21]` (scoped) | ⏸️ **NOT STARTED** (2026-09-03, paused before this stage) — added `--label_noise` to `train.py` (symmetric label noise, seed-paired). Smoke-tested (rate=0.2 corrupted 9994/50000 ≈ 20% as expected). Not yet launched: CIFAR-10/ResNet-18, 4 core methods, 3 seeds, 2 levels. | 24 runs | 6h | 37.75h |

**Resuming later:** just rerun `bash run_remaining_items.sh` (from the repo root) — its `run()` helper skips any cell whose CSV already has ≥201 lines and resumes any cell with a saved checkpoint from where it left off, so it's safe to invoke repeatedly. It does NOT re-run `batch_size_sweep.py` (already complete); run that separately only if its output needs regenerating, and never concurrently with the throttled batch (it needs a clean GPU for its memory/power measurements).

**Total: ~37.75h, leaving ~12.25h buffer** against runs going over estimate (very likely
for item 9, the ViT experiment, since nothing like it has been tried in this repo). If
buffer runs out, drop item 13 first (cheapest scientific value per hour of the paid
items), then item 12 — both are already scoped down from the full version in §A and can
be extended later without invalidating anything already done.

**Zero-cost, run in parallel by whoever is *not* running the GPU queue** (doesn't touch
the 50h budget, doesn't block it either):
- `[#2]` ✅ **DONE** (2026-09-03) — attempted the multi-step trajectory-divergence bound.
  New `app:multistep` (`prop:multistep`): under `ass:smooth` + `thm:safe`'s own
  gradient-floor hypothesis extended along the real trajectory, a discrete
  Gronwall-style argument gives a genuine bound, but one that's exponential in $t$ and
  vacuous over a full ~200-epoch run — explains *mechanistically* why a Lipschitz-only
  argument can't give a useful multi-step certificate. `rem:contraction-direction`
  names the missing ingredient for a tighter one (local strong-convexity/PL along the
  trajectory, unverified for this paper's architectures) and notes it would give a
  bounded, plateau-shaped limit matching item 7's empirical finding — flagged as the
  next step, not derived. `rem:shared-traj`/`sec:limitations` future-work item (5)/
  `rem:smooth-why`/one intro bullet all updated to point to this. A genuine partial
  result, not a failed attempt with nothing to show — the honest exponential bound
  plus a named path forward beats an unattempted gap either way. Recompiled clean.
  See `EXPERIMENT_LOG.md`.

---

## §2. After every numbered item — integration checklist

- [ ] Regenerate the relevant figure (extend/add a `figures/make_*_figure.py` script,
      save both `.png` and `.pdf`).
- [ ] Update the corresponding `paper/main.tex` section/table/`\includegraphics`.
- [ ] Update `sec:limitations`'s Future Work list if the item closes or narrows a gap.
- [ ] Recompile: `cd paper && tectonic main.tex`.
- [ ] Append an entry to `EXPERIMENT_LOG.md`.
- [ ] Re-run `stats_analysis.py` once a cell reaches `n ≥ 3` seeds.

---

## §A. Deferred / future work (NOT in the 50h budget)

Kept here so scope can be restored later if more GPU time becomes available — do not
start any of this without an explicit new go-ahead, since it dwarfs the budget above.

- **TinyImageNet / ImageNet-100 / ImageNet-1K matrix** `[#5,#6]` — the full version (4–9
  methods × 3–5 models × 5 seeds across three dataset scales) was estimated at
  **~1,350–1,450 GPU-hours** on its own, i.e. 30–40× this entire budget. If revisited,
  do TinyImageNet-only first (~500–600h at 3 seeds) before considering ImageNet-100/1K.
- **DeiT-Tiny, ViT-Small** — deferred with the above; item 9 above covers only a
  CIFAR-native ViT-Tiny as a first, budget-scoped transformer data point.
- **5-seed depth** — using 3 seeds throughout this budget instead; bumping the primary
  cells (already 3-seed) and item-1/9's cells to 5 seeds later would cost roughly
  40% more GPU-hours than their 3-seed cost above.
- **Full noisy-label matrix** (4 levels: 0/10/20/40%) — this budget covers 20%/40% only;
  adding 0% (already have as the unmodified baseline) and 10% costs ~6h more.
- **Full class-imbalance matrix** (3 ratios: balanced/10:1/50:1) — this budget covers
  10:1 only; adding 50:1 costs ~1.5h more (balanced = existing unmodified data).
- **RHO-LOSS beyond the primary cell** — already run at the primary cell per prior
  sessions; VGG-16/CIFAR-100 extension not in this budget.
- **Two-stage algorithm prototype** `[#24]` — explicitly out of scope; needs new theory
  before `\cref{thm:safe}` could apply to it, and the reviewer themselves frames it as
  optional/next-paper material.
- **Multi-step trajectory-divergence bound** `[#2]` sketched form to attempt (zero-cost,
  parallel-track — see §1's zero-cost row, repeated here for the full context):
  $$\|\theta_t^{\mathrm{TG}}-\theta_t^{\mathrm{full}}\| \le \sum_{s=1}^{t}\frac{\eta_s}{N}\sum_{i\in R_s}\|g_i^{(s)}\|$$
  contracted via `ass:smooth`'s β-smoothness toward
  $$\|\theta_t^{\mathrm{TG}}-\theta_t^{\mathrm{full}}\| \le \frac{\eta\epsilon R}{N}\cdot\frac{1-\rho^t}{1-\rho}.$$
  Cross-check any derived bound's predicted growth shape against item 7's empirical
  trajectory-divergence curve once that data exists.

---

## Reviewer-item cross-reference

| Item | Where | Item | Where |
|---|---|---|---|
| #2 | §1 zero-cost row; §A (full attempt) | #14 | §1 item 10 |
| #3 | §1 item 3 (via item 0's data) | #15 | §1 item 10 |
| #4 | §1 item 3 | #17 | §1 item 2 |
| #5 | §1 item 9 (scoped); §A (full) | #18 | §1 item 3 |
| #6 | §1 item 9 (scoped); §A (full) | #19 | §1 item 5 |
| #8 | §1 item 0 | #20 | §1 item 7 |
| #9 | §1 item 6 | #21 | §1 item 13 (scoped); §A (full) |
| #10 | §1 item 1 (3 seeds); §A (5 seeds) | #22 | §1 item 12 (scoped); §A (full) |
| #11 | §1 item 8 | #23 | §1 item 11 |
| #12 | §1 item 4 | #24 | §A (deferred) |
| #13 | §1 item 4 | | |
