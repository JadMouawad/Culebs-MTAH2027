# Attribution and licence notes (ens6_dead33 release candidate: ens6_rescue80 + Sprint 33 dead readout)

## Files

| file | sha256 | origin |
|---|---|---|
| `readout_all.npz` | `a47ff7999e7ae2f6452cb1ba519684de6f9aaa2d21cc233f5d7b56d5bf3ef140` | 56 parameters (55 weights, 1 bias) and the 55-entry normalisation mean / sd of the Sprint 33 dead readout, fold `all` |

The six ens6 members are used unchanged from `submission/models/ens6_rescue80/`. Their attribution is in that folder's
`ATTRIBUTION.md`.

## Training data of the readout

**DFL / Sportec open position data**, seven matches.
- Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in
  elite soccer.* Scientific Data 12, doi:10.1038/s41597-025-04505-y.
- Data: figshare doi:10.6084/m9.figshare.28196177.v1.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
- © Deutsche Fussball Liga (DFL) / Sportec Solutions.

**SkillCorner Open Data** (https://github.com/SkillCorner/opendata), six development matches.
- MIT licence.
- Credit: SkillCorner.

**Changes made before training** (CC BY 4.0, "indicate if changes were made"):
- frozen-ball samples removed (defreeze);
- synthetic competition-like ball-gap and player-loss corruption (one extra copy per match);
- conversion to 5 Hz tracking features;
- live/dead frame labels:
  - DFL: derived from the BallStatus attribute;
  - SkillCorner: from its published dynamic-event possession data.

**Teacher.** The readout's input includes the dead and team probabilities of the frozen ens6 members, which are
DFL-trained (CC BY 4.0).

**Not used.** No PFF data, PFF-trained weights or PFF-derived predictions; no Metrica data (evaluation only); no
competition data, and no public five-minute reference; no teammate weights.

**Publication.** Publishing these weights requires this attribution. It still needs the individual recorded decisions in
`submission/APPROVALS.json`.
