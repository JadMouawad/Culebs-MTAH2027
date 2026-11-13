# Attribution and licence notes (x_t1_50_dead37)

**Status: the team's intended submission (decided 8 November 2026), not yet approved for publication.** Publishing these files needs the
dated approval version in `submission/APPROVALS.json`, the team's licence wording for the weights, and each member's own recorded decision.
Do not describe this model as officially compliant before the organisers' guidance on the open questions is in.

## Files

Twelve member files `member_0..5__d.npz` and `member_0..5__x.npz`, and `readout_all.npz`. Every sha256 is in
`experiments/configs/x_t1_50_dead37.json`. The twelve files are six trained checkpoints, each exported twice from the same encoder: `__d` with
its DFL-head output layer, `__x` with a dead logit averaged across both label heads.

## Training data

**DFL / Sportec open position data**, seven matches (all members and the readout).
- Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.*
  Scientific Data 12, doi:10.1038/s41597-025-04505-y. Data: figshare doi:10.6084/m9.figshare.28196177.v1.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/legalcode.
  (c) Deutsche Fussball Liga (DFL) / Sportec Solutions. The data is provided by its licensor as-is (CC BY 4.0 section 5). Neither DFL nor
  Sportec Solutions endorses this work.

**SkillCorner Open Data** (https://github.com/SkillCorner/opendata, commit 4340d274572876239c154c90bc507a9b3250a656), six development
matches: `tracking_extrapolated.jsonl`, `phases_of_play.csv` and `match.json` for all members and the readout, and the ball-loss statistics.
Credit: SkillCorner. The repository's only licence file reads as follows.

```
The MIT License

Copyright (c) 2020 SkillCorner. https://skillcorner.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

The licence text speaks of software, not data. See `docs/SPRINT37_SKILLCORNER_LICENCE.md` for how the team reads it.

## Changes made before training (CC BY 4.0, "indicate if changes were made")

- Frozen-ball samples removed (defreeze).
- DFL: the natural tracking plus three copies of each match with synthetic ball loss and correlated player / ball tracking failures (random,
  burst, crowding). The amount follows ball loss measured on the SkillCorner development matches, 8.05 % of live frames (burst measured about 6.5-7 % on
  three matches). The random copy also takes its gaps from those matches (median 4.8 s, log sigma 1.069); burst and crowding keep earlier design shapes.
- Conversion to 5 Hz features; H/A/D frame labels from BallStatus / BallPossession (DFL) and phases of play (SkillCorner).
- The numpy files are exports of the trained networks; the weights are not the data, and the data is not included.

## Not used

No PFF data or weights; no Metrica fitting; no teammate weights; no provided-match data, labels, video, statistics or pseudolabels in
training or tuning. The shared feature code's thresholds are physical priors, written in early sprints; their independence from the match cannot be fully proven.
The readout's loss design and weights come from a teammate's design. The designer reported in writing (5 November 2026) that the missed-live-stop
cost of 2 was set after an error audit on external data, with no sweep, and that the L2 (0.05) and KL (0.1) weights were inherited unchanged
from an earlier external-data experiment. This is the designer's statement; it has not been independently verified.

**Licence of these weight files.** The DFL and SkillCorner terms above apply to the data the weights were derived from. They do not by
themselves set a licence for the weights. The team's chosen wording is recorded in `submission/APPROVALS.json` and stated in the bundle's `LICENSE`.
