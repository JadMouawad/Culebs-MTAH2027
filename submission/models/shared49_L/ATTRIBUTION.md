# Attribution and licence notes (shared49_robust_L, new weight files)

**Status: a release candidate prepared for the team's review (Sprint 49, Study L), not approved for publication, not selected.**
Publishing these files needs the dated approval version in `submission/APPROVALS.json` naming this candidate, the team's licence wording for
these exact files, and each member's own recorded decision. Do not describe this model as officially compliant: the organisers have not ruled on
the open questions listed in `docs/SPRINT49_L_RIGHTS_AND_PROVENANCE.md`.

## Files in this directory (fourteen)

| file | what it is |
|---|---|
| `SHARED_s0_all.npz`, `SHARED_s1_all.npz` | two ownership networks (seeds 0 and 1), exported to NumPy from two Torch checkpoints trained on all development data |
| `member_0..5__d.npz`, `member_0..5__x.npz` | six clean "T2" gap-specialist networks, each exported twice (DFL head; dead logit averaged over both heads), the same construction as the S37 members |

Every sha256 is in `experiments/configs/shared49_robust_L.json`. The model also loads the thirteen unchanged S37 files in
`submission/models/x_t1_50_dead37/`; their attribution is in `submission/models/x_t1_50_dead37/ATTRIBUTION.md` and applies to this model as well.

## Training data (the same data as S37, nothing added)

**DFL / Sportec open position data**, the same seven Bundesliga matches as S37, each as natural tracking plus synthetic ball and player loss copies
(four versions for the ownership networks, six for the T2 specialists).
- Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.*
  Scientific Data 12, doi:10.1038/s41597-025-04505-y. Data: figshare doi:10.6084/m9.figshare.28196177.v1.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/legalcode.
  (c) Deutsche Fussball Liga (DFL) / Sportec Solutions. Provided by its licensor as-is. Neither DFL nor Sportec Solutions endorses this work.

**SkillCorner Open Data** (https://github.com/SkillCorner/opendata, commit 4340d274572876239c154c90bc507a9b3250a656), the same six development
matches as S37: `tracking_extrapolated.jsonl`, `phases_of_play.csv`, `match.json`. Credit: SkillCorner. The full MIT notice of the repository is
reproduced in `submission/models/x_t1_50_dead37/ATTRIBUTION.md` and applies here unchanged. On 13 November 2026 the repository's LICENSE and README
were fetched again: both are byte-identical at the pinned commit and at the repository's current default branch, and that commit is still its latest.
The notice speaks of software, not data; see `docs/SPRINT37_SKILLCORNER_LICENCE.md` for how the team reads it.

**What is different for these files.** In the two ownership networks, one output row, the one that decides Home or Away at inference, is trained
on **both** providers' labels, including SkillCorner phase-derived team labels. In S37 that row was trained on the DFL head only and the SkillCorner
head was used for dead-ball evidence. The data, files and matches are the same as S37's; no new SkillCorner match and no other dataset was added, so
the upstream rights these files depend on are the same ones S37 already depends on. They bear on these files more directly.

## Changes made before training (CC BY 4.0, "indicate if changes were made")

- Frozen-ball samples removed.
- DFL: natural tracking plus copies with synthetic, externally specified ball and player loss; the amounts follow loss measured on the SkillCorner
  development matches.
- Conversion to 5 Hz features. The ownership networks add a permutation-invariant encoder over the eight nearest players of each team,
  using the seven original node channels plus six fixed kinematic quantities computed from them (distance to the reference point, radial and tangential
  velocity relative to it, speed, the original speed channel and the dot product of position and velocity); no parameter is fitted in those six.
- H/A/D frame labels from BallStatus / BallPossession (DFL) and phases of play (SkillCorner).
- The NumPy files are exports of the trained networks. The weights are not the data, and the data is not included.

## Not used

No PFF data or weights; no Metrica or SkillCorner test match in training; no Study E (expanded SkillCorner), I, J or K checkpoint; no S33 weight;
no provided-match data, labels, video, statistics or pseudolabels in training, tuning or selection of the networks.
The shared feature code's thresholds and the dead-ball readout carry the same disclosed provenance limits as S37 (technical document, section 3).

**Licence of these weight files.** The DFL and SkillCorner terms apply to the data the weights were derived from. They do not by themselves set a
licence for the weights. The wording proposed for review is in `docs/SPRINT49_L_APPROVALS_DRAFT.json`; nothing is in force until the team records it.
