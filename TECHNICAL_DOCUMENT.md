# Team Culebs: ball possession from tracking data

More Than A Hack 2027, Phase 1: Barcelona vs Athletic. Model: **shared49_robust_L**. It is our earlier model (x_t1_50_dead37: six temporal convolutional networks over the tactical tracking, trained on open data from two providers, with a 56-parameter dead-ball readout) plus two further ownership networks that decide which team has the ball, and six specialist networks used only where the ball has been unobserved for a second or more. The earlier model stays in this repository as the fallback, and a rule-based model (e110a) is the second fallback. This document describes the submitted file `Culebs.csv`, produced by `shared49_robust_L` as the registered primary.

## 1. Approach

- **Input.** The tactical ATD only: positions of the ball and of tracked players with their team, at 25 frames per second.
- **Features,** computed at 5 Hz:
  - 15 ball and stoppage features;
  - 11 features per team: distances to the ball, counts within 2, 5 and 10 m, nearest-player speed, centroid distance, and a log-likelihood of control;
  - for the ownership networks only, a per-frame encoding of the eight players of each team nearest the ball, from their position and velocity relative to the ball (a permutation-invariant set encoder) plus six fixed kinematic quantities computed from those fields. No parameter is fitted in the six.
- **The earlier model (unchanged).** Six temporal convolutional networks (64 channels, dilations 1, 2, 4, 8, 16), two label heads per network because the providers define stoppages and possession granularity differently. The Home/Away ratio comes from the DFL head, the dead-ball logit is the mean of both heads, and a dead-ball readout (dead logit + 2 tanh(w · x + b), 55 label-free inputs per 0.2 s) refines it.
- **Ownership networks (new).** Two networks (seeds 0 and 1) of the same design plus the player-set encoder. Their Home/Away output is **one parameter shared by both providers' label heads**; each provider keeps its own dead-ball output. Training: 4,615 steps, AdamW with a one-cycle schedule, dropout 0.3, a fixed logit scale of 1.4, and the same synthetic ball and player loss as the earlier model. At inference the Home/Away logit of the earlier model is averaged with the logit of the mean probability of these two networks; the dead-ball evidence, the readout and the decoder are not changed.
- **Gap specialists (new).** Six networks built like the earlier model's, trained under heavier synthetic ball loss. Where the ball has been unobserved for at least one second (five 5 Hz samples), the dead-ball logit and the Home/Away logit are each averaged with the specialists' (read out with the same readout). Elsewhere nothing changes.
- **Decoding** (`rescue_0.8_25`, per half, unchanged):
  - a majority vote over ± 50 frames, with runs under 25 frames merged;
  - a Viterbi decoding proposes finer runs;
  - a finer run is restored if it lasts at least 25 frames and its mean probability is at least 0.8.
- **Everything is fixed at inference.** Twenty-seven weight files, each checked against its sha256 before use; NumPy only; no GPU.

## 2. Evidence and its limits

All figures are the official Spell F1 (`evaluation/possession_metrics.py`), absolute, against the unchanged earlier model (S37), by match, equal weight per match. **No figure is an estimate of the score on the provided match.** An independent audit recomputed every figure below with its own scorer and label path, and a separate reviewer recomputed the essentials from the stored files before reading any summary; both reproduced them exactly.

**Which model each row measures.** External rows use the final networks (fitted on all development data) against the released earlier model. Development rows use networks that never saw the scored match (leave-one-match-out for DFL, leave-one-pair-out for SkillCorner) against the earlier model's matching out-of-fold proxies.

| set | matches | earlier model | this model | difference | matches better |
|---|---:|---:|---:|---|---|
| external, SkillCorner test (not fitted) | 14 | 0.6713 | 0.7346 | +0.063 | 14 / 14 |
| external, Metrica (not fitted) | 3 | 0.7815 | 0.8161 | +0.035 | 3 / 3 |
| **external, all** | 17 | 0.6908 | 0.7490 | **+0.058** [+0.051, +0.065] | 17 / 17 |
| development, DFL (out of fold) | 7 | 0.7064 | 0.7293 | +0.023 | 7 / 7 |
| development, SkillCorner (out of fold) | 6 | 0.6652 | 0.7250 | +0.060 | 6 / 6 |
| development, all | 13 | 0.6873 | 0.7273 | +0.040 | 13 / 13 |

The interval is a 90 % bootstrap over matches. The smallest single-match gain is +0.028. For the ownership networks alone, against a matched comparator (the same fusion with two ordinary player-set networks instead of the shared-ownership ones), the gain is +0.055 externally.

- **Where the gain comes from.** Home/Away accuracy on frames where the ball is truly in play rises by 2.1 points externally, on every match. Dead-ball precision and recall barely change (+0.0015 and −0.0018), because the dead-ball evidence is the earlier model's. The gain is larger at the stricter overlap thresholds (+0.039 at IoU 0.3, +0.082 at 0.7), so possessions are also placed better, not only found.
- **Both label conventions.** On SkillCorner's coarse labels the model predicts fewer possessions, closer to the reference count. On Metrica and DFL's finer labels it predicts more, again closer. Earlier changes of ours gained on one convention and lost on the other; this one gains on all three sources. Fourteen of the seventeen external matches are SkillCorner, the provider closest to the new training signal, so the pooled figure is dominated by it. The fine-label figures, Metrica +0.035 and DFL +0.023 out of fold, are the better guide.
- **Exploratory, not a confirmation:**
  - All 30 external matches were read repeatedly in earlier development, and the 17 external matches were used for three Sprint 49 comparisons before this model was designed. The interval above ignores that selection.
  - The model belongs to a family of twelve studies run in the last two days. Its protocol was written before its own scores, and its effect is about ten times that of the other studies, but the selection cannot be measured.
  - The references are SkillCorner's automatic phases of play and possession labels built from Metrica's events. Neither is built the way the competition's reference is.
- **Degraded tracking.** All degradation here is synthetic: the two fixed doses (0.18 and 0.4047) of our Sprint 37b ball-loss validation, chosen before any of these scores.

  | condition | external, all | Metrica | SkillCorner |
  |---|---|---|---|
  | natural | 0.691 → 0.749 | 0.781 → 0.816 | 0.671 → 0.735 |
  | extra ball loss 18 % | 0.638 → 0.692 | 0.715 → 0.751 | 0.621 → 0.680 |
  | extra ball loss 40 % | 0.555 → 0.613 | 0.625 → 0.669 | 0.539 → 0.601 |

  The earlier model's main known weakness, heavy live ball loss, is reduced here by the specialists but not removed. It is tested only with synthetic loss; the loss on the provided match cannot be sized without using the match.
- **Changes considered and rejected.** Re-selecting the decoder gained on DFL and Metrica and lost on 13 of the 14 SkillCorner test matches, so `rescue_0.8_25` was kept. Combining with the earlier S33 model's probabilities added nothing on natural tracking and was not used.

## 3. Assets used and dropped

| asset | used | note |
|---|---|---|
| tactical ATD | **yes** | the only input at inference |
| Smart Tagging, broadcast ATD, video, homography | no | tried in research; none improved possessions reliably on external data |
| DFL / Sportec open data (CC BY 4.0) | **training** | 7 Bundesliga matches; labels from the BallStatus / BallPossession flags |
| SkillCorner Open Data | **training** (6 matches) and evaluation (14) | labels from phases of play; automatic, not human |
| Metrica sample data | evaluation only | never fitted |

**Provenance of this model.** Our records show no competition label, video, pseudolabel or measured statistic in the training, calibration or selection of any weight. The ball-loss settings come from SkillCorner development matches. The limits of that statement:

- **Feature thresholds.** The shared thresholds are physical priors written in early sprints, when label-free diagnostics of the match were being read. Their independence cannot be fully proven.
- **Readout loss weights.** The readout's designer reported in writing (5 November 2026) how they were chosen: the cost of a missed live stop was doubled after an error audit on external data, with no sweep; the L2 (0.05) and KL (0.1) weights were inherited unchanged from an earlier external-data experiment; and the design revision followed an external evaluation, not an evaluation on the provided match. This is the designer's own statement; we have not independently verified it. The readout file is unchanged.
- **SkillCorner as a source for the Home/Away output.** The earlier model trained its Home/Away output on DFL labels only. The ownership networks train it on both providers' labels. No data source was added; the same six SkillCorner matches are used.
- **The comparison model S33** was trained with a ball-loss setting partly measured on the match. S33 is not part of this model.
- **Earlier scores.** The people who designed these models had seen earlier scores on the published five-minute reference. No such score was used to choose this model.
- **Aggregate exposure.** A historical supplied-match aggregate sat in a source comment of earlier branches and is removed here. A public challenge README baseline-score snippet was returned by a web search after the final recipe was frozen. Neither was used for any choice, threshold or comparison, and no fresh statistic was computed. We do not claim zero exposure.

## 4. Possession rules implemented

- **What is labelled.** A possession is a team's run of H or A; dead ball is D. X is written only outside the graded halves.
- **Loose balls and challenges.** Team evidence follows DFL's convention, partly blended with SkillCorner's: a team keeps possession until the other team gains the ball. The decoder needs sustained evidence before changing team, so very short touches are usually absorbed.
- **Restarts and stoppages** come from the combined dead-ball evidence and are declared only when sustained.
- **When the ball is missing,** the model relies on player positions and motion, and after a second without the ball it also uses the specialists.

## 5. Limitations

- **Evidence is exposed and not competition-like** (section 2). The external improvement over the earlier model is +0.05822 Spell F1 across 17 matches that were read repeatedly in earlier development; those matches do not independently establish performance on the provided Barcelona match. We cannot estimate that effect, and do not.
- **Home/Away asymmetry.** The earlier model is exactly symmetric when the home and away teams are swapped. This model is not: on real tracking 0.4 % (Metrica) to 0.6 % (SkillCorner) of frames change beyond the label swap, because the two ownership networks are not averaged over both team orientations at inference. The measured gains used this same inference. The effect on the provided match is not measured. Making the model symmetric would be a different candidate, and we did not change it.
- **Convention risk.** Providers disagree on what a stoppage is and how short a possession can be. The gain was positive on every labelled match, on three label sources.
- **Heavy missing ball** remains the main known weakness of the family.
- **Short spells.** Recall of spells under 2 s stays low on every provider.
- **Labels.** SkillCorner labels are the automatic output of another vendor's algorithm.

## 6. Compliance with the organisers' rules

- **For this model:** every weight is trained, tuned and calibrated on external data only, and the only input at inference is the provided tracking. It was not trained, tuned, calibrated or selected on the provided video or on any ground truth of the provided match. Its checks on the provided match were limited to the file format and a hash; its predictions were not compared with the video or the published five-minute reference.
- **The official Barcelona reference** (the published five-minute labels and any ground truth of the provided match) was not used to design, tune, calibrate or select this model.
- **Earlier research.** An internal audit of our earlier, rejected research work found activities that need the organisers' guidance under their clarification of 3 November 2026. None of them is used by this model. These incidents have been disclosed and documented by the team; they have **not** been individually approved by Talent Arena or the organisers, we have no ruling on them, and **this document does not claim that they were permitted.**
- **Readout provenance.** The provenance of the dead-ball readout's loss weights, as stated by its designer, is recorded in section 3 and is not independently verified.

## 7. Reproduction

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-lock.txt          (Linux / macOS: .venv/bin/pip ...)
set MTAH_DATA_DIR=<folder with the competition data bundle>  (Linux / macOS: export MTAH_DATA_DIR=...)
.venv\Scripts\python scripts/make_submission.py --candidate shared49_robust_L
.venv\Scripts\python evaluation/validate_submission.py outputs/submission/Culebs.csv
```

- **Output:** `outputs/submission/Culebs.csv`, 199,319 rows, sha256 `7839169700c0cc0060dc6168ea5e414f486fbdfd615c025f379c519649529838`. The script refuses to write a file whose bytes differ, and records every input and code hash in a manifest.
- **Checking it:** the script prints `frozen match True` when the file is byte-identical. In the public repository it also prints `NOT A FINAL RUN`, because our internal approval record is not published there; the bytes are the same.
- **Fallback:** `--candidate x_t1_50_dead37` regenerates the earlier model's file, sha256 `d707e942b4a1a45c038e792b19e69e363373ba1132a3159539cd1c093edb74a1`, from the same bundle.
- **Requirements:** numpy and pandas only, CPU (numpy 2.4.6, pandas 3.1.0, Python 3.11.9, pinned in `requirements-lock.txt`).
- **Platforms checked:** byte-identical output (the sha256 above) was reproduced on Windows from a clean clone, twice, and on Linux (Ubuntu under WSL) from a fresh virtual environment using the public bundle. Other platforms and Python or library versions are not claimed.
- **Weights:** `submission/models/x_t1_50_dead37/` (thirteen unchanged files) and `submission/models/shared49_L/` (fourteen files). Every sha256 is checked before inference.

## 8. Licences and attribution

Three separate things carry separate terms.

1. **Our code:** MIT (`LICENSE`). Files taken unchanged from the organisers (`games.py`, `starter/load_smart_tagging.py`, `evaluation/`) keep their MIT licence (`LICENSE-ORGANISERS`, Metrica Sports).
2. **The training data** is not redistributed. It keeps its own terms:
   - **DFL / Sportec open data:** CC BY 4.0. Bassek, Rein, Weber & Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, *Scientific Data* 12, doi:10.1038/s41597-025-04505-y. (c) DFL / Sportec Solutions. The changes made are listed in the `ATTRIBUTION.md` files.
   - **SkillCorner Open Data** (github.com/SkillCorner/opendata, commit `4340d274`): credit SkillCorner. The repository's MIT notice is reproduced in full in `submission/models/x_t1_50_dead37/ATTRIBUTION.md`. That notice speaks of software, so the team interprets the permission to train on the data and publish derived weights as reasonably clear. **It is not explicitly confirmed**: we did not obtain SkillCorner's separate permission for derived weights (`docs/SPRINT37_SKILLCORNER_LICENCE.md`).
3. **The twenty-seven weight files** carry the licence the team states for them in `LICENSE`, subject to the upstream notices above. The upstream licences do not by themselves set a licence for the weights.

The competition match data is not included and is not covered by any licence here.
