# Experiment Log

One entry per completed phase from `NEXT_STEPS.md`. Append, don't rewrite history.
Each entry: date, phase, what ran, key numbers, files produced, paper section touched,
and any deviation from the plan (and why).

---

## Template (copy for each new entry)

### Phase N — <name> — <date>
- **What ran:** commands/scripts used, settings.
- **GPU time:** wall-clock spent (informational only, not a constraint).
- **Key results:** the 2–4 numbers that matter.
- **Files produced:** `results/...`, `figures/...`.
- **Paper section updated:** `\label{...}` touched, and how.
- **Deviations from plan:** anything skipped, changed, or that turned up unexpectedly.

---

## Pre-existing work (before this log started)

### Phase 3 (LR-mechanism + drift instrumentation) — IN PROGRESS as of 2026-08-30
- **What ran:** `run_p2_p3_chain.sh`, launched detached (`setsid`+`nohup`+`disown`) at
  2026-08-30 11:04 to survive session end. Chains `run_lr_sweep.sh` (linear/step/cosine
  LR schedules × 4 methods, ResNet-18/CIFAR-10, seed 0) then three `drift_term_run.py`
  passes (`--schedule` exponential/fixed/Adam).
- **Status:** check `run_p2_p3_driver.log` for `P2_P3_CHAIN_ALL_DONE`; if absent and no
  `train.py`/`drift_term_run.py` process is running, it died and needs relaunching per
  `NEXT_STEPS.md` Phase 3.
- **Files produced so far:** `results/{baseline,ies_shipped}_cifar10_resnet18_SGD_linear_learning_rate_seed0.csv`
  (complete); `ies_alg1` linear-LR in progress at last check.
- **Not yet done:** `drift_term_analysis.py` (Phase 3b) does not exist yet — write once
  the three drift runs finish.

### Phase 3d exploratory pass (order-of-difference) — 2026-08-30
- **What ran:** `order_of_difference_analysis.py` against the existing
  `results/noise_autocorr_losses.npy` (50000 instances × 100 epochs, epochs 101–200,
  ResNet-18/CIFAR-10/SGD-exponential baseline, seed 0). No new GPU run.
- **Key results:** empirical MSE is monotonically increasing in N ({1:4.4e-4, 2:1.3e-3,
  3:4.2e-3, 4:1.4e-2}), i.e. **N=1 beats N=2** in this aggregate test — same ordering
  holds even restricted to "still-decaying" instances (mean loss in [0.01, 0.5)). Does
  not confirm `prop:mse`'s N=2-optimality claim as naively tested; likely because this
  population-average test doesn't isolate the theorem's actual per-instance SNR band.
- **Files produced:** `figures/fig_order_of_difference.{png,pdf}`.
- **Paper section updated:** none yet — this is exploratory; see Phase 3d in
  `NEXT_STEPS.md` for the properly-scoped re-test (per-instance, at the actual
  δ-crossing epoch) to run once Phase 3's drift dump lands.

### Phase 4 exploratory pass (equal-compute) — 2026-08-30
- **What ran:** `equal_compute_analysis.py` against existing primary-cell CSVs
  (ResNet-18/CIFAR-10/SGD-exponential, seed 0, all 7 methods with data). No new GPU run.
- **Key results:** at fixed compute budgets of 20/30/40/50% of the full 200-epoch
  budget, **InfoBatch and BLS-InfoBatch beat TG-IES** (and baseline) through 40% budget;
  TG-IES only leads at its own final budget point (~46%, since it saves ~53.6% of
  backprop). Final-vs-best accuracy gaps: 0.20–0.55 points across methods, consistent
  with what the paper already reports for this check.
- **Files produced:** `figures/fig_equal_compute.{png,pdf}`.
- **Paper section updated:** none yet — needs the Phase 4 framing decision (final-acc as
  primary metric?) before going into the main table; see `NEXT_STEPS.md` Phase 4.

---

### Phase 3, item 0 — LR-sweep + drift runs — COMPLETE 2026-08-30 14:49
- **What ran:** `run_p2_p3_chain.sh` finished clean (`P2_P3_CHAIN_ALL_DONE`, all `exit=0`):
  `run_lr_sweep.sh` (linear/step/cosine × 4 methods) then 3 `drift_term_run.py` passes
  (exponential/fixed/Adam schedules).
- **GPU time:** ran from 11:04 to 14:49 (~3h45m), close to the ~4h estimate.
- **Files produced:** 10 new `results/*cifar10_resnet18_SGD_{linear,step,cosine}*.csv`;
  `results/drift_{exponential,fixed,Adam}_{losses,inner,subset_idx,eta}.npy`.

### NEXT_STEPS item 2 — direct $D_i^{(t)}$ instrumentation `[#2, #17 partial]` — 2026-08-30
- **What ran:** new `drift_term_analysis.py`, using the drift dumps above. Tested
  `thm:curvature`'s decomposition $\Delta^2 L_i^{(t)} = \eta_t D_i^{(t)} + O(\eta_t^2) +
  R_{i,t}$ directly (pairing $\eta_t D_i^{(t)}$ against the observed $\Delta^2 L_i^{(t)}$
  for the 300-instance subset, all valid epochs, and again restricted to epoch≥150).
  Caught and fixed a real epoch-indexing misalignment between the loss/inner arrays and
  the eta trace before trusting the result (verified the fix barely moved the numbers,
  as expected given how slowly these schedules' LR changes per epoch — confirms the
  finding below isn't an artifact of that bug).
- **Key results (the finding, not a confirmation):** $\eta_t D_i^{(t)}$ correlates only
  weakly with $\Delta^2 L_i^{(t)}$ (Pearson $r$: exponential 0.34, fixed-LR 0.37, Adam
  0.23; late-training epoch≥150: 0.24, 0.11, and **−0.05** for exponential — sign
  reversal). The drift term does **not** dominate the curvature+remainder residual under
  any schedule (mean $|\eta_tD_i^{(t)}|$ / mean $|$residual$|$ = 0.69× exponential, 0.32×
  fixed-LR, ~0.00× Adam) — the opposite of `thm:curvature`'s "drift dominates at moderate
  η" framing as literally stated. Read as an asymptotic ($\eta\to0$) vs. empirical
  (actual step sizes up to 0.1) distinction, not a refutation of the theorem itself.
- **Files produced:** `figures/fig_drift_term_validation.{png,pdf}` (copied into
  `paper/figures/` too — note `paper/figures/` is a separate directory from the repo-root
  `figures/`, not a symlink; every new figure needs copying there before compiling).
- **Paper section updated:** `sec:level3` (softened the "drift dominates" remark right
  after `thm:curvature`), new paragraph + `fig:driftvalidation` in `sec:lrmechanism`,
  abstract, conclusion, and `sec:limitations` future-work item (3) — all now say
  "asymptotically dominated" and point to the direct-instrumentation result rather than
  asserting empirical dominance. Recompiled clean with `tectonic main.tex`.
- **Deviations from plan:** item 2 was budgeted at $0 (post-processing only) on the
  assumption the drift dump already had everything needed. It doesn't: the dump only
  saved the scalar dot product $\langle g_i,g_\text{pop}\rangle$, not $\|g_i\|$ itself,
  so item #17's actual ask (the $\|g_i\|$-vs-$|\Delta^2L_i|$ calibration-bound plot) is
  **still not done** and needs a small code change to `drift_term_run.py` (save the norm
  too) plus a ~1.5h rerun of the 3 schedules — added back into the budget (new running
  total 9h, still well inside the 50h cap; see `NEXT_STEPS.md` item 2).

### NEXT_STEPS item 3 — refined order-of-difference + phase quartile analysis `[#4, #18]` — 2026-08-30
- **What ran:** extended `order_of_difference_analysis.py` with `refined_near_threshold_test()`
  and `phase_quartile_analysis()`, both against `results/drift_exponential_losses.npy`
  (full 0–200 epoch trajectory, 50000 instances, from item 0/2's drift dump).
- **Key results:**
  - Refined order test: evaluating each instance's finite difference exactly at the
    epoch its own $|\Delta^2L_i|$ first drops below the paper's calibrated $\delta=0.01597$
    (46,779 instances with enough history), **N=2 wins clearly**: MSE = $7.5\times10^{-5}$
    vs. $2.3\times10^{-2}$ (N=1) and $0.58$ (N=3) — confirms `prop:mse`'s claim at the
    actual operating point, resolving the earlier exploratory (naive population-average)
    finding that N=1 wins in aggregate — that test simply didn't isolate the SNR band
    the theorem's claim is conditional on.
  - Phase quartile analysis: empirical $\hat r_i$ (first-difference ratio) rises from
    0.53 (epochs 1–50) to 0.76 (151–199) while its CV falls from 1.05 to 0.75 — evidence
    `ass:geometric`'s fixed-rate model fits better in the second half of training,
    consistent with the algorithm's epoch-30 recalibration timing already landing in a
    more favorable regime (not derived from this analysis, just consistent with it).
- **Files produced:** `figures/fig_order_of_difference.{png,pdf}` (naive test, unchanged
  from the earlier exploratory pass), `figures/fig_phase_quartile_r.{png,pdf}` (new).
  Both copied into `paper/figures/`.
- **Paper section updated:** two new paragraphs + two new figures (`fig:orderdiff`,
  `fig:phasequartile`) inserted in `sec:level2` right after `prop:mse`'s discussion.
  Recompiled clean with `tectonic main.tex`.
- **Deviations from plan:** none — this one landed as estimated ($0 GPU, post-processing
  only).

### NEXT_STEPS item 4 — equal-compute analysis extended to every cell `[#12,#13]` — 2026-08-31
- **What ran:** extended `equal_compute_analysis.py` from the single primary-cell pass to
  loop over all three cells with new-baseline (InfoBatch/BLS-InfoBatch/AlignPrune) data:
  ResNet-18/CIFAR-10, ResNet-18/CIFAR-100, VGG-16/CIFAR-10 (all SGD-exponential, seed 0).
  Pure post-processing of existing `results/*.csv`, no new GPU run. rho\_loss excluded:
  it never produced a `results/*.csv` (its driver chain was cut before running, per
  `project_tgies_experiment_matrix` memory), not silently missing.
- **Key results:** the InfoBatch-family's low-budget lead generalizes across all three
  cells (not just the primary cell the earlier exploratory pass checked): $2.5$–$4.3$
  points higher best-so-far accuracy than TG-IES at $20$–$30\%$ of the full compute
  budget, in every cell. The gap narrows sharply by $40\%$ ($0.1$–$0.8$ points) and
  reverses on ResNet-18/CIFAR-100, where TG-IES is $0.12$ points ahead of BLS-InfoBatch
  already at $40\%$. Read as a genuine budget-dependent trade-off (InfoBatch-family skips
  both forward+backward on pruned instances, so gets more updates per unit compute early,
  before TG-IES's calibrated $\delta$ starts making confident removals of its own), not a
  clean win for either family.
- **Files produced:** `figures/fig_equal_compute.{png,pdf}` (now a 3-panel figure, one
  per cell), copied to `paper/figures/`.
- **Paper section updated:** new `sec:equalcompute` subsection inserted right after
  `sec:newbaselines-secondary` (before `sec:noise-validation`), with `fig:equalcompute`.
  Recompiled clean with `tectonic main.tex` (only pre-existing underfull-vbox warnings).
- **Deviations from plan:** none — landed at the estimated $0 GPU cost.

### NEXT_STEPS item 5 — ablation reframe `[#19]` — verified already done, 2026-08-31
- **What ran:** nothing new — checked the abstract, `sec:ablation`, and `sec:conclusion`
  against item 5's requested framing ("noise-calibrated $\delta$ provides the principal
  computational gain; stability-derived $K^*$ provides principled control on exclusion
  persistence") and found all three sections already state this precisely (abstract:
  "calibrating the threshold $\delta$ alone accounts for essentially all of TG-IES's
  savings advantage ... while calibrating the patience $K^*$ against an incompatible
  fixed threshold can independently collapse savings to near zero, since the derived
  patience supplies a principled stability criterion rather than additional savings").
  This must have been done in an earlier, unlogged pass; no further edit needed.
- **Deviations from plan:** item was marked pending in `NEXT_STEPS.md` but the paper
  content was already correct — updating `NEXT_STEPS.md` to reflect this rather than
  re-editing already-correct text.

### NEXT_STEPS item 2 remainder — item #17 closed: ||g_i|| vs |Delta^2 L_i| `[#2,#17]` — 2026-08-31
- **What ran:** `drift_term_run.py` already had `grad_norm()`/norms-saving code added in an
  earlier (unlogged) pass; resumed the interrupted rerun chain (`run_p2_item1_chain.sh`,
  relaunched 11:16 after it died mid-fixed-schedule-run at epoch 196/200 — no
  checkpointing, so it restarted that schedule from epoch 0) through completion:
  exponential (done earlier, 10:40), fixed (11:45), Adam (12:15), all exit=0. Wrote
  `analyze_norms()` in `drift_term_analysis.py`: since `thm:bridge`'s exact bound
  $\varepsilon_i(\delta){=}\frac{r_i}{1-r_i}\sqrt{2\Lambda_i\delta}$ needs per-instance
  $r_i,\Lambda_i$ this paper doesn't estimate, tested the bound's qualitative content
  instead — correlation of $\|g_i^{(t)}\|$ with $\sqrt{|\Delta^2L_i^{(t)}|}$, and whether
  instances passing each schedule's own calibrated $\delta$ (read from that schedule's
  `tgies_..._seed0.csv` final-row `delta` column: exponential 0.01597, fixed 0.002468,
  Adam 0.0002569) have systematically smaller gradient norms than failing instances.
- **Key results:** confirms `thm:bridge`'s predicted direction, consistently across all
  three schedules — corr($\|g_i\|$, $\sqrt{|\Delta^2L_i|}$) = 0.47–0.49; mean $\|g_i\|$
  for mastery-check-passing instances is 30–33$\times$ smaller than for failing instances
  (exponential 0.53 vs 17.7, fixed 1.26 vs 39.7, Adam 0.11 vs 3.16); among passing
  instances only 22–39% have above-population-median gradient norm (vs. the 50% a
  check unrelated to $\|g_i\|$ would give). Unlike item 2's original $D_i^{(t)}$-dominance
  finding (a complicating result), this one is confirmatory.
- **Files produced:** `figures/fig_bridge_norm_check.{png,pdf}` (3-panel, one per
  schedule), copied to `paper/figures/`.
- **Paper section updated:** new "Empirical check" paragraph + `fig:bridgenormcheck`
  inserted in `sec:bridge` right after `rem:bridge-scope`; `sec:limitations` future-work
  item (3)'s "still-open gap" sentence rewritten to "now partly closed gap" reporting the
  finding (exact bound still open, needing per-instance $r_i,\Lambda_i$). Recompiled clean
  with `tectonic main.tex`.
- **Deviations from plan:** none beyond the resume — NEXT_STEPS item 2 is now fully done
  (both the $D_i^{(t)}$ direct instrumentation and item #17's norm check).

### NEXT_STEPS item 1 (CIFAR-100 half) — backfilled to 3 seeds `[#10]` — 2026-08-31
- **What ran:** `run_cifar100_seeds12.sh` (new, MAXJOBS=3 throttle pattern), chained after
  the drift-norm rerun via `run_p2_item1_chain.sh`, launched 12:15, finished 13:49 (~1h34m).
  8 new runs (baseline/ies_shipped/ies_alg1/tgies $\times$ seeds 1-2), joining the
  existing seed-0 data for a full 3-seed cell. Regenerated `results/flop_summary.csv`
  (`flop_analysis.py`, pure post-processing) and extended `stats_analysis.py`'s
  `SETTINGS` dict to add the CIFAR-100 cell, then reran it.
- **Key results:** TG-IES 76.01$\pm$0.08\% acc / 57.76$\pm$0.04\% saved (n=3), closely
  matching the earlier single-seed value (75.97/57.81) and now the tightest-variance
  savings figure of any method in the cell. Paired stats: TG-IES's accuracy cost is a
  real, small effect specifically vs.\ IES (shipped) ($-0.86\pm[-1.55,-0.17]$, CI excludes
  zero) but not vs.\ Baseline or IES (literal) (CIs straddle zero despite similar mean
  diffs, due to higher per-seed variance there); backprop/FLOP-saved advantage over every
  method is enormous and unambiguous ($d_z$ up to 1522.5, the largest effect size in the
  paper's stats table).
- **Files produced:** `results/{baseline,ies_shipped,ies_alg1,tgies}_cifar100_resnet18_SGD_exponential_learning_rate_seed{1,2}.csv`
  (8 files), updated `results/flop_summary.csv`, `results/statistical_analysis.csv`.
- **Paper section updated:** `tab:cifar100`/`sec:cifar100` (full seed-averaged table +
  paragraph), `tab:stats`/`app:stats` (new CIFAR-100 block + updated support/non-support
  prose), `tab:fullresults`/`app:experiment` (8 new rows, 56→64 total runs, caption
  fixed), plus every prose mention of CIFAR-100-as-single-seed corrected: abstract, intro
  contributions bullet, `sec:setup`, `sec:generalization` intro, `sec:newbaselines`
  (new-baselines cross-ref reworded so it no longer implies the core comparison is still
  single-seed), conclusion, `sec:limitations` empirical-scope paragraph (also verified,
  not assumed, that CIFAR-100 seed 0 was never affected by the stale recalibration-timing
  bug that hit VGG-16/fixed-LR seed 0 -- delta recalibrates exactly once, at epoch 31,
  in that run's log -- so no correction was needed there, only a backfill). Recompiled
  clean with `tectonic main.tex`.
- **Deviations from plan:** caught and fixed a real bug while building
  `zero_shot_transfer_analysis.py` (item 6, prepped in parallel): naively grouping
  `results/summary.csv` by (dataset,model,optimizer) for method=tgies pulls in one-off
  ablation/sensitivity-sweep runs that reuse seed=0 with a suffixed `run_name`, inflating
  the primary cell's apparent savings std from the true $0.30$ to a spurious $15.95$;
  fixed by filtering to the canonical `run_name` (no suffix) before grouping.

### NEXT_STEPS item 6 — zero-shot-transfer table `[#9]` — 2026-08-31
- **What ran:** `zero_shot_transfer_analysis.py` (written and bug-fixed during item 1's
  CIFAR-100 backfill, see that entry above), run against the now-updated `summary.csv`.
  Assembled one row per distinct (dataset, architecture, optimizer/schedule) TG-IES has
  ever been run on in this paper (9 cells), all sharing the same 4 fixed calibration
  constants (`sec:selfcalib-scope`).
- **Key results:** savings range $23$--$58\%$ across cells (expected -- this is exactly
  the schedule/dataset-dependent variation `sec:lrmechanism`/`sec:densenet` already
  discuss), with no per-cell constant retuning anywhere. 7/9 cells at n=3 (5 already
  seed-averaged, plus CIFAR-100 from item 1's backfill); 2/9 single-seed by design
  (DenseNet-121, pending the same backfill currently running; the 3 LR-sweep cells
  linear/step/cosine were run once, seed 0, as a matched-epoch schedule comparison, not
  meant to be seed-averaged -- table footnote states this explicitly rather than let it
  read as missing data).
- **Files produced:** none new (table lives directly in the paper; script reads
  `results/summary.csv`).
- **Paper section updated:** new `tab:zeroshot` + paragraph inserted in
  `sec:selfcalib-scope`, right after the paragraph whose point (2) it makes checkable.
  Recompiled clean with `tectonic main.tex`.
- **Deviations from plan:** finalized now rather than waiting ~5-6h for the DenseNet-121
  backfill (still running) to also reach 3 seeds, since blocking this $0-cost item on one
  slow cell wasn't worth it; the table already explains DenseNet-121's row is pending. Will
  update that one row (and its `n`) once the backfill lands.

### Figure clarity fix + DenseNet-121 backfill resume — 2026-09-01
- **What ran:** user reported the generated PNGs weren't sharp at 100% zoom. Found
  `drift_term_analysis.py`, `equal_compute_analysis.py`, and `order_of_difference_analysis.py`
  (5 `savefig` calls total: `fig_drift_term_validation`, `fig_bridge_norm_check`,
  `fig_equal_compute`, `fig_order_of_difference`, `fig_phase_quartile_r`) were saving PNGs
  at `dpi=150` with no `bbox_inches="tight"`, inconsistent with every other figure script
  in the repo (`figures/make_*.py`, `noise_autocorrelation_analysis.py`), which all use
  `dpi=300`. Bumped all 5 to `dpi=300, bbox_inches="tight"` and regenerated (post-processing
  only, no GPU) — `fig_equal_compute.png` went from ~2685x669 to 5369x1337px, confirmed
  via PIL. Numbers unchanged (spot-checked against this log's existing entries for items
  2/3/4). Copied all 5 png+pdf pairs into `paper/figures/`, recompiled with `tectonic
  main.tex` (only pre-existing "already defined" warnings, exit 0).
  Also found `run_densenet_seeds12.sh` (item 1's DenseNet-121 half) had died mid-run from
  the prior session — seed1 stuck at epoch 32/200, seed2 never started, GPU idle. Resumed
  it (`bash run_densenet_seeds12.sh`, MAXJOBS=3; note the script's exec bit was gone,
  ran via `bash` explicitly instead of `./`) with a `kill -0`-polling background chain to
  log completion.
- **GPU time:** $0 for the figure fix; DenseNet-121 backfill resumes its original ~5-6h estimate.
- **Files produced:** regenerated `figures/fig_{drift_term_validation,bridge_norm_check,
  equal_compute,order_of_difference,phase_quartile_r}.{png,pdf}` and their `paper/figures/`
  copies.
- **Paper section updated:** none (figure content/numbers unchanged, only resolution).
- **Deviations from plan:** none — user separately asked about adding a ViT/ImageNet
  experiment; clarified this is item 9 (CIFAR-native ViT-Tiny, in-budget) vs. the §A
  deferred full ImageNet matrix (~1350-1450h, out of budget) before proceeding, still
  awaiting the user's choice.

### NEXT_STEPS item 1 — DenseNet-121 backfill, full paper integration — 2026-09-03
- **What ran:** results delivered as a zip (`tgies_densenet121_3seed_2026-09-02.zip`,
  run remotely on a RunPod RTX 3090) with the 8 seed-1/seed-2 runs from the session-crossing
  resume noted in the previous entry, plus updated `train.py`/`common.py` (gained
  checkpoint/resume every 10 epochs, all RNG streams + per-instance state) and a fresh
  `results/summary.csv`/`flop_summary.csv`. Unzipped, placed the 12 canonical
  `results/densenet_cifar10/*.csv` into flat `results/`, merged the zip's summary.csv
  rows into the repo's (replacing 6 stale/incomplete densenet rows), reran
  `flop_analysis.py` (now also covers `vit_tiny_cifar`, see next entry) and
  `stats_analysis.py`/`zero_shot_transfer_analysis.py` after adding `densenet121_exp` to
  both scripts' settings tables.
- **Key results:** TG-IES 95.28±0.09% acc / 54.53±0.66% saved (n=3), vs.\ baseline
  95.36±0.14%/0.03%, IES (shipped) 95.36±0.11%/48.60±1.04%, IES (literal)
  95.24±0.02%/51.60±0.89%. TG-IES's accuracy cost (−0.08pt) is smaller than baseline's
  own seed spread; one seed (seed 2) even has TG-IES beating every other method's
  accuracy for that seed. Paired stats (new `app:stats` block): backprop-saved vs.\
  baseline $54.49\%\pm[52.85,56.13]$, $d_z=82.5$.
- **Files produced:** `results/*densenet121*.csv` (12 files, replacing/adding to the
  prior 9), `logs/*densenet121*seed{1,2}*.log`, updated `results/summary.csv`,
  `results/flop_summary.csv`, `results/statistical_analysis.csv`.
- **Paper section updated:** abstract, intro, `sec:generalization` intro, `sec:densenet`
  (table converted to 3-seed mean±std, prose rewritten), `tab:zeroshot` (row un-asterisked),
  `sec:limitations` (Empirical scope paragraph, no longer flags DenseNet-121 as
  single-seed), `app:experiment`'s `tab:fullresults` (4→12 DenseNet-121 rows, run count
  updated), `app:stats` ("which settings covers" paragraph, new DenseNet-121 paired-stats
  block, "what this does/does not support" paragraphs). Recompiled clean with `tectonic
  main.tex`.
- **Deviations from plan:** none GPU-side. Delivered together with item 9 (ViT-Tiny) in
  the same session; both integrated in one pass rather than sequentially, since the
  zip's `train.py`/`common.py` for item 9 is a strict superset of item 1's (both copied
  from the ViT zip, verified identical to the DenseNet zip's version modulo the ViT
  additions) — see next entry.

### NEXT_STEPS item 9 — ViT-Tiny on CIFAR-10/CIFAR-100, full paper integration `[#5,#6]` — 2026-09-03
- **What ran:** results delivered as a zip (`tgies_vit_tiny_cifar_2026-09-03.zip`, run
  remotely on a RunPod RTX 3090, 4,800 epochs total at MAXJOBS=3). Code changes: new
  `ViT_CIFAR`/`ViTBlock` classes in `common.py` (patch-4, 6 layers, 192-dim, 3 heads,
  2.69M params) and an `AdamW_warmup_cosine` optimizer branch in `train.py` (10-epoch
  linear warmup then cosine decay over 200 epochs); both copied into the repo root
  (superset of item 1's checkpoint/resume changes). Added `run_vit.sh`. Placed the 24
  canonical `results/vit_tiny/*.csv` into flat `results/`, merged summary.csv rows,
  added `vit_tiny_cifar` to `flop_analysis.py`'s `ARCHS` (measured bwd/fwd FLOP ratio
  $2.023$, the first architecture in this repo measurably off the $\sim\!2.00\times$
  convolutional convention), added two `vit_cifar10`/`vit_cifar100` settings to
  `stats_analysis.py` and two rows to `zero_shot_transfer_analysis.py`, reran both.
- **Key results:** savings transfer and grow with task difficulty — TG-IES $25.52\pm0.26\%$
  saved on CIFAR-10 ($1.53\times$ IES (literal)), $26.28\pm0.14\%$ on CIFAR-100
  ($2.10\times$); mechanism is self-calibration, not the fixed-$\delta$ methods' hand-tuned
  constant (Cor. 4.5 rescaled $\delta$ $\sim\!186\times$ on CIFAR-10, $\sim\!333\times$ on
  CIFAR-100, no constant retuned by hand); absolute savings much lower than CNNs ($\sim\!26\%$
  vs.\ $\sim\!54.5\%$ DenseNet-121); small accuracy cost (CIFAR-10 $-0.36$pt, CIFAR-100
  $-0.69$pt) and widest seed spread of any method here, unlike every convolutional cell.
  Caveat carried into the paper: $K^*$ was pinned at the `--max_kstar` ceiling of 15 in
  every ViT run (Cor. 6.4's closed form asked for more), so these numbers reflect the
  theory clamped by configuration.
- **Files produced:** `results/*vit_tiny_cifar*.csv` (24 files), `logs/*vit_tiny_cifar*.log`
  (24 files), updated `results/summary.csv`, `results/flop_summary.csv`,
  `results/statistical_analysis.csv`.
- **Paper section updated:** abstract, intro, `sec:generalization` intro, new
  `sec:vit` subsection (2 tables `tab:vit10`/`tab:vit100`, 4-finding paragraph, caveats
  paragraph) inserted between `sec:densenet` and `sec:adam`, `tab:zeroshot` (+2 rows,
  wrapped in `\resizebox` after the new rows caused an overfull hbox), `sec:flops` (ViT's
  $2.023\times$ ratio noted, "architecture-independent" framing scoped to the
  convolutional architectures), `sec:limitations` (Empirical scope paragraph; Future-work
  item 2 "Architecture family" rewritten from "not attempted" to "now partly closed"),
  `app:experiment`'s `tab:fullresults` (+24 rows, two new settings, run count 64→96),
  `app:stats` ("which settings covers" paragraph now says no cell remains single-seed
  except the by-design LR-sweep cells; two new paired-stats blocks; "what this does/does
  not support" paragraphs extended, removed the stale "nothing here extends to
  DenseNet-121" sentence). Added a `dosovitskiy2021image` bib entry (ViT had no citation
  in this paper's bibliography before). Recompiled clean with `tectonic main.tex` (only
  pre-existing "already defined" xdvipdfmx warnings, unrelated to this change).
- **Deviations from plan:** none against item 9's scoped plan. The zero-shot table needed
  a `\resizebox` fix not anticipated by the original plan — the new ViT row labels pushed
  an already-tight table (the primary-cell row was already close to the single-column
  width) into an overfull hbox; confirmed by a throwaway build with the new rows removed
  that the table compiled clean without them, isolating the cause before fixing it.

### NEXT_STEPS item 7 — K*-sweep + trajectory-divergence instrumentation `[#20]` — 2026-09-03
- **What ran:** added `--dump_traj_every N`/`--traj_ref_dir` to `train.py`: every N epochs,
  `baseline` writes a flattened-parameter snapshot (`torch.cat` over `model.parameters()`,
  CPU float32) keyed by `(dataset,model,optimizer,seed)`; any other method loads the
  matching snapshot and logs `param_l2_dist`/`ref_param_l2_norm`/`rel_dist` to
  `results/traj_dist_<run_name>.csv`. Smoke-tested (2-epoch dummy baseline+tgies pair,
  `/tmp`) before the real batch. New driver `run_kstar_trajectory.sh`: Phase 1 reruns the
  primary-cell baseline (3 seeds, tagged `_traj`, `--dump_traj_every 20`) to completion,
  then Phase 2 sweeps `--tgies_fix_kstar {1,2,4,8,15,30}` x 3 seeds (18 runs, `MAXJOBS=6`,
  same snapshot flag). Launched detached (`setsid`+`nohup`+`disown`), ran ~3h10m
  (12:43-15:52), GPU pinned at 100% throughout (confirmed 3-concurrent already saturates
  this RTX 5090, unlike the remote RTX 3090 sessions' 3-concurrent==100% assumption).
  All 21 runs exited 0.
- **Key results:** `kstar_trajectory_analysis.py` (new). K*-sweep: savings fall smoothly
  from $66.98\pm0.62\%$ (K*=1) to $44.13\pm0.40\%$ (K*=30) while accuracy rises from
  $94.55\pm0.16\%$ to $94.87\pm0.09\%$ (matched-batch baseline: $95.08\pm0.14\%$) — a
  smooth, monotonic, crossing-free trade-off, not a cliff at either extreme. Trajectory
  divergence: *absolute* parameter distance from the shared baseline reference
  ($\|\theta_{\mathrm{TGIES}}-\theta_{\mathrm{baseline}}\|$) **decreases and plateaus**
  over training for every K* (≈54 at epoch 20 → 29–37 by epoch 200, flat over the last
  third) — the opposite of the unbounded-compounding-drift failure mode `rem:shared-traj`
  could not rule out theoretically. The K*-ordering (smaller K* → larger divergence) holds
  at all 10 snapshot epochs with zero crossings, matching `thm:safe`'s per-step bound's
  predicted direction. *Relative* distance (normalized by the baseline's own shrinking
  parameter norm) rises instead, purely because the reference norm itself contracts
  faster (40.7→20.8 over the same window) than the absolute gap does — reported both ways
  rather than only the more favorable-looking one. Sanity check: reference norm at each
  snapshot epoch is identical (to float precision) across all 6 K* values, confirming
  every K* run is being compared against the correct shared baseline.
- **Files produced:** `results/{baseline,tgies}_..._traj*.csv`/`_kstar{1,2,4,8,15,30}.csv`
  (21 files), `results/traj_dist_tgies_..._kstar*.csv` (18 files),
  `figures/fig_kstar_trajectory.png/.pdf` (copied into `paper/figures/` — caught the
  missing-copy step this time before compiling, unlike a close call on the other 4
  figures touched in the same session's figure-clarity pass, see next entry).
- **Paper section updated:** new `sec:kstartraj` subsection (table `tab:kstarsweep` +
  2-panel figure `fig:kstartraj`) inserted after `sec:ablation`, distinguished explicitly
  from `tab:ablation`'s "K* only" row (that one fixes $\delta$ at IES(literal)'s constant;
  this sweep keeps $\delta$ self-calibrated throughout and only fixes K*). `rem:shared-traj`,
  the "central open theoretical question" paragraph, and `sec:limitations` future-work
  item (5) all updated to point to this new empirical (not proof) evidence. Recompiled
  clean with `tectonic main.tex`.
- **Deviations from plan:** original NEXT_STEPS estimate was "~15 new runs, 4.5h,
  K=15 already exists [so skippable]" — actually needed 21 runs (added 3 fresh baseline
  reruns for snapshot references, and reran K=15 too since the existing K=15 data point
  has no trajectory snapshots) at ~3h wall-clock; a reasonable, disclosed deviation from
  the original per-run-count estimate, not a scope cut.

### Figure clarity pass + proof reformatting — 2026-09-03
- **What ran:** user reported four problems: (1) `app:proofs`' derivations were dense,
  many equations chained inline in paragraph text rather than displayed; (2) Figure 7
  (`fig3_tradeoff.png`, main scatter) used a gray/blue/teal/navy palette where 3 of 4
  colors were all in the blue family, hard to tell apart; (3) Figure 9
  (`fig_drift_term_validation.png`) was a 3-panel, 13in-wide figure squeezed into
  `\columnwidth` (single-column), shrinking already-small (7-10pt) fonts to illegibility;
  (4) Figure 11 (`fig_equal_compute.png`) used unstyled default-color thin lines for 7
  overlapping methods with half the x-axis wasted on an already-converged tail.
  Rewrote all 8 proofs in `app:proofs` into `align*`/display-equation blocks (one
  derivation step per line). Changed `COLORS` in `figures/make_figures.py` and
  `figures/make_newbaselines_figure.py` from gray/#3B76B5/#1B9AAA/#0A2C55 to
  gray/#E69F00/#009E73/#0A2C55 (orange/green replacing the two blues), keeping both
  files' first-four-methods palette in sync per their own header comments; also nudged
  `make_newbaselines_figure.py`'s infobatch/bls_infobatch off the new orange to avoid a
  fresh clash. Widened `fig_drift_term_validation` to `figure*`/`0.95\textwidth` (was
  `figure`/`\columnwidth`) and bumped `drift_term_analysis.py`'s figsize/fonts. Gave
  `equal_compute_analysis.py` the same 7-color palette plus per-method linestyles,
  bold/solid/on-top styling for tgies, bigger fonts, and zoomed the x-axis to 0-45% (the
  region where curves actually separate; confirmed via the printed per-budget table that
  nothing informative happens past ~40%).
- **Deviations from plan / a real mistake caught before it shipped:** regenerated 4
  figures into the repo-root `figures/` dir and initially verified them there via the
  Read tool — but `paper/figures/` is a **separate, not-symlinked** directory (a fact
  already in project memory from the 2026-08-30 drift-validation session), so the first
  recompile after this pass silently kept rendering the **old, unfixed** figures; caught
  only because the *next* item (item 7's new `fig_kstar_trajectory.png`) hard-failed
  tectonic with a missing-file error, which prompted checking whether the other 4 were
  stale too (they were, via `cmp`). Copied all 4 (`.png`+`.pdf`) into `paper/figures/`
  before the recompile that actually shipped this fix. Lesson: after regenerating any
  figure in this repo, `cmp` or explicitly `cp` into `paper/figures/` before trusting a
  "clean compile" as evidence the fix landed — a clean compile only proves the files
  tectonic *found* were consistent with each other, not that they were the current ones.

### Zero-cost item [#2] — first attempt at the multi-step trajectory-divergence bound — 2026-09-03
- **What ran:** no GPU — pure derivation, done while the item 8/10/11/12/13 GPU batch
  ran in the background. Set up two trajectories from the same $\theta_0$: full-batch
  GD on the population loss vs.\ Algorithm 1's actual update (the real, evolving active
  set $A_t$, not a shared reference). Split the per-step gradient difference into a
  same-map term (bounded by `ass:smooth`'s $\beta$) and a set-difference term (bounded
  by `thm:safe`'s own gradient-floor hypothesis, $\|g_i(\theta_t)\|<\varepsilon_i$ for
  excluded $i$, now assumed to hold along the real trajectory rather than only a shared
  reference). Triangle inequality gives a linear recursion in the divergence
  $\|e_t\|$; unrolled it by induction (verified base case + inductive step by hand)
  to a closed form, then bounded the per-step floor term by $\varepsilon_{\max}$ and
  summed the resulting geometric series.
- **Key results:** a genuine, correctly-derived bound (`prop:multistep`), but
  $\|e_t\| \le (\varepsilon_{\max}/\beta)[(1+\eta\beta)^t - 1]$ — exponential in $t$,
  vacuous over a ~200-epoch/many-thousand-step run for any realistic $\beta$. This is
  useful precisely because it's a *correct negative-ish* result: it explains
  mechanistically why a smoothness-only (Lipschitz, non-contractive) argument can't
  give a practically useful multi-step certificate, rather than just asserting the gap
  is hard. Did NOT attempt to derive the strong-convexity/PL contraction extension
  rigorously (the standard co-coercivity argument would need adapting from the
  textbook two-point case to this $g^{\mathrm{full}}$-vs-$\hat g_t$ split, and would
  need justifying local strong-convexity/PL along a deep network's real trajectory —
  both nontrivial and not something to risk getting subtly wrong in a theorem-paper
  appendix from memory) — reported instead as a named, precise open direction
  (`rem:contraction-direction`), explicit that it's unverified, not a completed proof.
  Noted explicitly that this direction would predict a bounded, $t$-independent
  plateau — which is qualitatively exactly what item 7's `fig:kstartraj` found
  empirically, a nice (if informal) point of contact between the two results.
- **Files produced:** none (paper text + new appendix section only).
- **Paper section updated:** new `app:multistep` (`prop:multistep` + proof +
  `rem:contraction-direction`), `rem:smooth-why` (no longer says $\beta$-smoothness is
  unused — it's now used), `rem:shared-traj`, the "central open theoretical question"
  paragraph, `sec:limitations` future-work item (5), and one intro bullet, all updated
  to point to this. Recompiled clean with `tectonic main.tex` — verified via pypdf
  text extraction that Proposition B.1 and its numbered equation render correctly.
- **Deviations from plan:** none — this was scoped as "attempt it, a documented
  attempt beats an unattempted gap either way" and that's exactly what shipped: a real
  partial result (the exponential bound) plus a precisely-named next step, not a
  vague "this is hard" gesture.

### Items 8, 10, 11, 12, 13 — implemented and launched — 2026-09-03
- **What ran:** in response to "complete everything" (the remaining NEXT_STEPS.md §1
  items after item 7), implemented four new `train.py` capabilities and one new
  standalone script, each smoke-tested on a 1-2 epoch toy run before committing to the
  real batch:
  - **Item 8** `selective_backprop` method (Jiang et al. 2019): reuses the
    infobatch-family scaffold (forward the full dataset, keep-mask a subset, backprop
    only the kept ones) but with percentile-rank accept probability
    ($p^{\beta_{\mathrm{sb}}-1}$ on the persistent last-observed-loss score, new
    `--sb_beta`, default 3.0) instead of below-mean-with-probability-r, NO gradient
    rescaling (SB is an explicitly biased sampler by design, unlike InfoBatch's
    unbiased correction), and no annealing back to the full dataset.
  - **Item 11** `--reactivation_probe_every`/`--reactivation_probe_n`: tgies-only,
    self-calibrated-mode-only (needs `eps0`, which `--tgies_fix_kstar` skips
    computing — added `eps0 = None` initialized before the epoch loop, guarded on).
    Every N epochs, samples up to `reactivation_probe_n` currently-passive instances
    and computes each one's TRUE gradient norm via an individual batch-size-1
    forward+backward pass (reusing `drift_term_run.py`'s per-instance-gradient
    pattern), checked against the last-calibrated `eps0` — the exact quantity
    `cor:bridgeprob` is about, not the cheap last-layer proxy `eps0` itself was
    calibrated from. Logs to `results/reactivation_<run_name>.csv`.
  - **Items 12/13** `--imbalance_ratio` / `--label_noise`: both applied once, right
    after `get_dataset()` and before `N` is fixed, using `np.random` directly (already
    seeded by `setup_seed(args.seed)` at that point, no separate `RandomState`
    needed) — so all four methods sharing a `--seed` see the identical imbalanced
    subset / identical corrupted labels, preserving the paired-seed design
    `app:stats`'s statistics depend on. Imbalance: exponential per-class subsampling
    profile (a seed-random class order, so "most frequent" isn't tied to label
    indexing) giving exactly the requested ratio between most- and least-kept class.
    Label noise: symmetric (uniform reassignment to a different class).
  - New `batch_size_sweep.py` (item 10), closely modeled on `measure_efficiency.py`'s
    out-of-process nvidia-smi-polling pattern, parameterized by `--batch_size` instead
    of comparing methods at a fixed one.
  - New driver `run_kstar_trajectory.sh`-style `run_remaining_items.sh` (`MAXJOBS=6`),
    chained after `batch_size_sweep.py` (which needs an uncontended GPU for its
    peak-memory/power measurements, so it must run alone, never concurrently with the
    throttled batch) via a single `setsid`+`nohup`+`disown` background shell.
- **Status as of this entry:** user paused the batch (out of time for the day) shortly
  after item 12/13 would have started. Killed cleanly via `kill -TERM` on the whole
  detached process group (`setsid`'s session id doubled as the pgid, so
  `kill -TERM -<sid>` took down the driver and every live `train.py` in one signal;
  verified via `nvidia-smi` and `pgrep` that nothing was left running) rather than
  individual PIDs. Also stopped the background completion-wait task
  (`TaskStop`) so it doesn't poll forever for a sentinel that will now never appear.
  **Item 10 (batch-size sweep) completed in full** before the pause
  (`results/batch_size_sweep_summary.csv`, 12/12 rows) but is not yet integrated into
  the paper. **Item 8** (selective_backprop) got to epoch 10/200 on all 3 seeds before
  the kill -- a checkpoint exists at that point (`--ckpt_every` default 10), so
  resuming is cheap. **Item 11** (reactivation probe) only reached epoch 3/200 on all
  3 seeds, before its first checkpoint -- resuming restarts these 3 from epoch 0,
  which is inexpensive either way. **Items 12/13 never launched.** NEXT_STEPS.md
  updated with per-item status (⏸️ PAUSED / NOT STARTED) and a one-line "how to
  resume" note (`bash run_remaining_items.sh`, safe to invoke repeatedly; do NOT
  concurrently run `batch_size_sweep.py`, which needs an uncontended GPU).
- **Files produced so far:** `train.py` (4 new capabilities), `batch_size_sweep.py`,
  `run_remaining_items.sh`, `results/batch_size_sweep_summary.csv` (complete),
  partial/incomplete CSVs for items 8 and 11 (safely resumable, see above).
- **Paper section updated:** none yet — pending complete results for all 5 items.
- **Deviations from plan:** scope unchanged from NEXT_STEPS.md's versions of items
  8/10/11/12/13; only the *timing* changed (user-requested pause, not a scope cut).

### Items 8 and 11 — completed, analyzed, integrated into an ICLR 2026 draft — 2026-09-06
- **What ran:** resumed only items 8 (selective_backprop, 3 seeds, had reached epoch
  10/200 before the prior pause) and 11 (reactivation probe, 3 seeds, restarted from
  epoch 0 since no checkpoint existed yet), via a new scoped driver
  `run_items_8_11.sh` (sequential, `MAXJOBS=1`) rather than the full
  `run_remaining_items.sh`, because the shared GPU had an unrelated VLLM inference
  process holding ~29.8/32.6GB, leaving too little headroom for the default 6-way
  concurrency. Items 12/13 (class imbalance, label noise) were deliberately deferred,
  not run: they are robustness/generalization extras, not part of the paper's five
  core theoretical contributions, and the user explicitly chose to defer them rather
  than spend the remaining GPU budget there. All 6 runs finished cleanly (exit 0).
- **Item 8 (Selective Backprop, primary cell) results:** $93.31\pm0.11\%$ acc,
  $66.72\pm0.01\%$ backprop saved — the highest savings of *any* method in this
  paper (above TG-IES's $53.63\%$), but at a $1.72$-point accuracy cost relative to
  baseline, the largest accuracy cost of any method/cell anywhere in this paper (next
  largest: TG-IES's own $0.87$ points on CIFAR-100). Read as the opposite end of the
  accuracy/compute trade-off from TG-IES, not a failure of the implementation: SB has
  no calibrated stopping condition tied to a stability argument analogous to
  `cor:kstar`'s certified exclusion horizon. Added to `app:newbaselines` in the ICLR
  draft with this framing; did not change the abstract's four-method-matrix claim
  (baseline/IES shipped/IES literal/TG-IES), which does not involve SB.
- **Item 11 (reactivation probe, primary cell) results:** the probed runs match
  unprobed TG-IES closely ($94.67\pm0.11\%$/$53.66\pm0.10\%$ vs.\ $94.85\pm0.28\%$/
  $53.63\pm0.30\%$), confirming the instrumentation itself doesn't perturb training.
  **Finding: `thm:safe`'s per-instance gradient-floor hypothesis is violated by
  roughly half the excluded set at every check.** Pooling all 9 probed epochs × 3
  seeds (300 true-gradient instances probed each time), mean
  `frac_exceed_eps0` $=47.4\%$, ranging $34$–$59\%$ with no decreasing trend over
  training (if anything it's lowest at epoch 80 and drifts back up by epoch 200).
  Each seed's *median* probed gradient norm sits close to that seed's calibrated
  $\varepsilon_0$ (heavy right tail on the mean, not a uniformly-shifted
  distribution), so $\varepsilon_0$ is a reasonable floor for a *typical* excluded
  instance but a loose one for the population as a whole. This sits in real tension
  with item 7's earlier finding (aggregate trajectory divergence plateaus rather than
  diverging): a near-half per-instance violation rate coexists with bounded aggregate
  drift. We do not have a verified explanation for the reconciliation (a natural but
  unchecked candidate: `thm:safe`'s bound is itself an average, $1/N$-scaled
  quantity, so a bounded-magnitude violation on a large fraction of instances may
  still keep the aggregate small) and flagged it as a named next step rather than
  claim to have closed it.
- **Files produced:** `results/selective_backprop_cifar10_resnet18_..._seed{0,1,2}.csv`
  (complete, 200 epochs), `results/tgies_..._seed{0,1,2}_reactprobe.csv` (complete),
  `results/reactivation_tgies_..._seed{0,1,2}_reactprobe.csv` (9 rows each, the
  per-epoch probe summary). No figures generated (results reported as text/tables,
  not plots, given the small number of probed epochs).
- **Paper:** built a full ICLR 2026-formatted draft from scratch at
  `paper_iclr2026/main.tex` (separate from the existing ICML-format `paper/main.tex`,
  which is left untouched), using the official `iclr2026_conference.sty`/`.bst`
  template. The prior ICML draft's main text ran to ~27 two-column pages, far over
  any reasonable page budget; the ICLR version required substantial triage: main
  text now holds the five theorem statements (full proofs deferred), the TG-IES
  algorithm, the headline results/ablation/K*-sweep, and fits ICLR's strict
  **9-page main-text limit** exactly (verified by compiling and checking where
  `\section*{Impact Statement}` and `\bibliography` land — both now fit on page 9,
  References starts cleanly on page 10). Everything else (full proofs, the
  multi-step bound derivation, extended related work, Adam, the drift-term
  mechanism study, equal-compute analysis, noise-independence validation, percentile
  sensitivity, full efficiency table, full ViT tables, the complete 96-run per-seed
  table, the full paired-statistics appendix, and the two new sections from this
  session's runs) moved to an unrestricted appendix. Anonymized per ICLR double-blind
  rules (no `\iclrfinalcopy`, "Anonymous authors" placeholder). New appendix section
  `app:reactprobe` ("Reactivation Probe: Does the Excluded Set Actually Satisfy the
  Gradient-Floor Hypothesis?") reports item 11's finding in full, with a pointer
  from the Limitations section; `app:newbaselines` extended with the Selective
  Backprop paragraph. Compiled clean with `tectonic` (`paper_iclr2026/main.pdf`, 29
  pages total), no undefined references.
- **Deviations from plan:** none in scope; items 12/13 remain explicitly deferred
  per the user's own choice, not attempted.

### Major-issues review response: K_safe non-binding, reactivation probe in main text, weight-decay tension, drift-term correction, delta grid search — 2026-09-06
- **What happened:** a detailed "Major Issues" review of the ICLR draft (5 points) identified
  real problems, not nitpicks: (1) the certified exclusion horizon $K_{\mathrm{safe}}$ is never
  the binding constraint in any of this paper's 96 runs (the implementation's max_kstar=15
  cap is what's actually used, not the derived value) but this was disclosed only in an
  appendix footnote; (2) the reactivation probe's ~47% per-instance violation finding was
  buried in Appendix R rather than confronted where the K*-trajectory plateau claim is made;
  (3) `thm:bridge`'s shared-interpolating-solution assumption was never flagged as being in
  tension with this paper's weight-decay ($5\times10^{-4}$) training recipe; (4) the paper never
  tested whether a naive grid search over $\delta$ (no theory) would find comparable gains to
  the calibrated value; (5) the drift-term reversal (Appendix J) was disclosed in the abstract
  but Section 6 itself still read as if drift dominates unqualified.
- **Fixes applied (issues 1, 2, 3, 5 — text/framing corrections):**
  - Computed the actual uncapped $K_{\mathrm{safe}}$ for the primary cell using the logged
    $\varepsilon_0$ from the reactivation-probe runs and the paper's own stated $\eta_{31}$:
    **$K_{\mathrm{safe}}\approx1.19$–$1.32$ million** across the 3 seeds, i.e. **5 orders of
    magnitude above the cap of 15**. Added `rem:ksafe-nonbinding` (new remark) + `tab:ksafe`
    (new main-text table) in Section 7, and revised the abstract, intro contribution bullet,
    `sec:selfcalib-scope`, Conclusion, and Limitations (now leads with this as the most
    important limitation) to state plainly that only $\delta$ is genuinely theory-derived in
    practice; $K^*$ is, in every reported run, the hand-set ceiling.
  - Promoted the reactivation-probe finding into main text: `sec:kstartraj`'s plateau claim
    now directly states the ~47.4% per-instance violation rate immediately after it, rather
    than only via a Limitations pointer.
  - Added a third caveat to `rem:bridge-scope` naming the weight-decay/shared-optimum tension
    explicitly (weight decay pulls parameters away from any per-instance loss-minimizing
    $\theta^*$; cross-entropy has no finite minimizer under it) as an unresolved tension with
    `ass:smoothi`'s literal reading.
  - Strengthened Section 6's interpretive paragraph to state the empirical reversal
    (`sec:lrmechanism`'s weak correlation, drift not exceeding curvature+remainder) directly,
    not just via an abstract footnote.
- **Fix applied (issue 4 — new experiment):** ran a real naive $\delta$-grid-search experiment
  to test whether an untheorized sweep matches the calibrated $\delta=0.0077$'s
  accuracy/savings ($94.68\%$/$66.62\%$). 5 points ($\delta\in\{0.003,0.005,0.015,0.03,0.05\}$,
  $K^*{=}1$ fixed matching the "$\delta$ only" ablation protocol, primary cell, seed 0,
  `run_delta_gridsearch.sh`, sequential due to the shared GPU's limited free memory) — all
  completed cleanly. **Result:** $0.003{\to}94.76\%/61.38\%$, $0.005{\to}94.39\%/64.75\%$,
  $0.015{\to}94.52\%/70.47\%$, $0.03{\to}94.27\%/72.98\%$, $0.05{\to}93.92\%/74.48\%$. The
  calibrated point sits on the same accuracy-savings frontier the grid traces (it dominates
  $\delta{=}0.005$ on both axes but is not dominated by, nor clearly better than, $\delta{=}0.015$)
  — **the experiment does not show the derived $\delta$ beats search; it shows calibration
  reaches a comparable point in one run instead of the grid's five.** This is an honest,
  moderating finding, reported as such rather than oversold. Added `app:deltagrid` (new
  appendix section with the full table), a new paragraph in `sec:ablation`, and a rewritten
  Limitations bullet ("calibration is search-competitive, not search-beating").
- **Files produced:** `run_delta_gridsearch.sh`,
  `results/tgies_..._seed0_deltagrid{0.003,0.005,0.015,0.03,0.05}.csv` (all complete, 200
  epochs each).
- **Page budget:** all five fixes added substantial content (~2 new tables, ~600 words) to a
  main text that was already at the exact 9-page ICLR limit with zero slack. Rebalanced via
  a long sequence of targeted trims across nearly every main-text section (abstract, related
  work, setup, results, generalization, conclusion, limitations) plus tighter global spacing
  (`titlesec` spacing reduced further, `arraystretch` to 0.88) — verified after each edit by
  recompiling and checking exactly where `\bibliography` lands. Final state: Limitations
  fully fits on page 9; only the Impact Statement boilerplate (unavoidably short) plus
  References begin on page 10. No content was cut to make room for these fixes — only
  wording tightened; every finding already in the paper is still there.
- **Deviation from plan:** none — this was explicitly a correctness/honesty pass requested by
  the review, not a scope change.

### Second-round review response: K_safe/K* framing, aggregate-cancellation experiment, normalization ablation, plain-language abstract — 2026-09-16
- **What happened:** a second external review raised three items against `paper_iclr2026/main.tex`:
  (1) don't claim the theory "replaces patience" unless a regime is shown where $K_{\mathrm{safe}}$
  actually binds, or reframe the contribution; (2) the reactivation probe's ~47% per-instance
  gradient-floor-violation finding needs the aggregate perturbation
  $\|\frac1N\sum_{i\in P_t}g_i\|$ vs.\ $\frac1N\sum_{i\in P_t}\|g_i\|$ checked directly, not left
  as an unverified "diluted by averaging" guess; (3) add a normalization ablation, $1/N$ vs.\
  $1/|A_t|$, for TG-IES. Investigated each against the actual current draft (not the older
  ICML-format `paper/main.tex`) before doing anything.
- **Item 1 — already resolved, no action needed.** The abstract, intro research question,
  `rem:ksafe-nonbinding`, `sec:selfcalib-scope`, and the Conclusion already state precisely this:
  $K_{\mathrm{safe}}$ never binds, so only $\delta$ is genuinely theory-derived in practice, and
  patience is a fixed, hand-set ceiling like IES's own. This was a prior session's fix (see the
  2026-09-06 entry above); confirmed still correctly stated everywhere, no overclaiming found.
- **Item 2 — infrastructure already built (unlogged), one seed's run was interrupted; resumed,
  fixed a real resume-checkpoint bug, and completed it.** `app:aggregate` (the exact-identity
  derivation, `lem:jensen`, `ass:aggfloor`, `prop:aggmultistep`, `cor:kstaragg`) and
  `reactivation_agg_analysis.py`/`run_reactivation_agg.sh` (new `agg_grad_norm` column on
  `--reactivation_probe_every`, tagged `_reactprobe2`) already existed on disk, dated 2026-09-15,
  with a literal `<<AGGREGATE_RESULTS_PARAGRAPH>>` placeholder left in the paper. 2 of 3 seeds
  (0, 1) had finished; seed 2 was dead at epoch 112/200 with a checkpoint at epoch 80. Resuming it
  hung completely (100% CPU, 0% GPU, 33 min with no progress) with `--num_workers 4`; killed and
  resumed cleanly with `--num_workers 0`, which completed normally (~5 min). Separately, while
  investigating why the first resume's reactivation-probe CSV silently stopped at epoch 100
  instead of reaching 200, found and fixed a real bug: `train.py`'s checkpoint `save_ckpt`/resume
  block never persisted `eps0` (the gradient-norm floor used only by the probe's own trigger
  condition `eps0 is not None`), so any `--reactivation_probe_every` run resumed after its
  one-time calibration epoch silently drops all further probe logging with no error -- core
  training/`delta_tgies`/`kstar` are unaffected (correctly checkpointed), only this diagnostic.
  Added `eps0` to both the save blob and the resume-restore block (`ck.get("eps0")` for
  backward compatibility with old checkpoints). Deleted and cleanly reran seed 2's
  `_reactprobe2` run end-to-end rather than trust a patched resume, to avoid any doubt about
  the number that would go in the paper.
- **Item 2 key results:** pooling all 9 probed epochs $\times$ 3 seeds (27 points; accuracy/savings
  for these runs, $94.73\pm0.02\%$/$53.66\pm0.09\%$, match unprobed TG-IES, confirming the
  instrumentation doesn't perturb training), the mean cancellation ratio
  $\|\mu_t\|/\mathrm{mean}_i\|g_i^{(t)}\| = 0.34$ (range $0.26$--$0.47$) -- real, substantial
  partial cancellation, confirming the "diluted by averaging" half of the reconciliation
  candidate `app:reactprobe` had left unverified. But the honest, sup-based
  $K_{\mathrm{safe}}^{\mathrm{agg}}$ (same conservative floor-over-horizon convention as
  `tab:ksafe`) comes out to $\lfloor5.5\rfloor=5$ -- *smaller* than the $K^*{=}15$ ceiling used
  in every run in this paper. Unlike the per-instance $K_{\mathrm{safe}}$ ($\approx$1.2M, wildly
  non-binding), the aggregate refinement is tight enough to be genuinely violated at this
  paper's own configuration. Read as consistent with, not contradicting,
  `sec:kstartraj`'s empirical plateau: `prop:aggmultistep`'s bound is still exponential in $t$
  (same worst-case triangle-inequality construction as `prop:multistep`), so it can be loose
  enough to be violated on paper while the one actually-realized trajectory stays bounded. This
  **narrows, but does not close**, the per-instance-vs-aggregate gap -- exactly the framing
  `sec:limitations` had already committed to before the number was known.
- **Item 3 — answered from existing code + data, zero new GPU time.** Checked `train.py`'s
  actual per-step loss normalization (`losses.sum()/args.batch_size` for tgies vs.\
  `losses.mean()` for the other 3 methods) against `drop_last_active` (True whenever the active
  set is at least one minibatch). Since every minibatch is then full-sized, the two formulas are
  byte-identical at the per-step level; they can only diverge once the active set falls below
  `batch_size=64`, a case `drop_last` instead discards outright. Checked this empirically
  (no GPU needed) across all 44 distinct TG-IES CSVs in `results/`: minimum active-set size ever
  reached is 2,965 (VGG-16 seed 1), 46$\times$ `batch_size` -- so no run in this paper ever
  reaches a step where $1/N$ and $1/|A_t|$ normalization could numerically differ. A literal
  ablation run under this paper's configuration would measure exactly zero effect by
  construction; testing the real difference would need deliberately pushing the active set below
  one minibatch (far more aggressive $K^*$/$\delta$, or a much smaller `batch_size`), not
  attempted. Added this as a new paragraph in `app:practical` plus a one-sentence pointer from
  `sec:ablation`.
- **Files produced:** `results/{,reactivation_}tgies_cifar10_resnet18_SGD_exponential_learning_rate_seed2_reactprobe2.csv`
  (clean rerun, all 3 seeds now complete), `figures/fig_aggregate_bound.{png,pdf}` (copied to
  `paper_iclr2026/figures/`), `train.py` (`eps0` checkpoint fix).
- **Paper section updated (`paper_iclr2026/main.tex` only; the older `paper/main.tex` ICML
  draft was left untouched):** abstract rewritten in plain language on the same request (dropped
  inline `\cref` clutter and the five-result inventory, kept every substantive honest caveat:
  $K_{\mathrm{safe}}$ non-binding, ~half-violated floor hypothesis, search-competitive not
  search-beating); `app:aggregate`'s results paragraph (previously a placeholder) filled in;
  `app:practical` gained the normalization paragraph; `sec:ablation` gained a one-sentence
  pointer to it. Recompiled clean with `tectonic --keep-intermediates --keep-logs main.tex`
  (zero undefined references -- also caught and fixed one broken `\cref{rem:open}` in the new
  paragraph, a label that only exists in the older ICML draft, not this ICLR one, where that
  remark was merged into `rem:kmismatch`).
- **Deviations from plan:** none against the review's 3 items. The GPU-hang debugging (bad
  `num_workers`, not covered by any existing driver script's retry logic) and the checkpoint
  `eps0` bug were unplanned but necessary detours to trust item 2's number.

<!-- Add new entries above this line, newest at bottom, oldest at top. -->
