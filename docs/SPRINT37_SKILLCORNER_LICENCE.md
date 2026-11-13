# SkillCorner Open Data: licensing determination for x_t1_50_dead37

**Written for:** the Culebs team, deciding whether six networks trained on this data can be released.
**Checked:** 4 November 2026, against commit `4340d274572876239c154c90bc507a9b3250a656`. This is an engineering reading, not legal advice.

## Determination

**Using the data to train and publishing the derived weights is supported by everything SkillCorner publishes, but no text states it for data or for model training.**
Treat the permission as *reasonably clear, not explicit*. We found no additional terms. A team member should accept this reading in writing, or ask SkillCorner. Nothing here
makes the candidate officially compliant.

## What the sources say

| source | what it says | how far it goes |
|---|---|---|
| `LICENSE` at the commit (sha256 `3384cc66…c6eb`; unchanged since 2020) | MIT, "Copyright (c) 2020 SkillCorner". Permission covers "this software and associated documentation files (the "Software")" | The only licence file in the tree. GitHub's API reports the repository as MIT. It does **not** mention data |
| `README.md` at the commit (sha256 `c09886c6…4428`) | "This data has been open sourced in a joint initiative between SkillCorner and PySport"; a goal is to "Provide access to tracking data to researchers and the sports analytics community" | Shows the intent that the data is covered, and names research use. It does not name model training or evaluation |
| same | "if you use the data, we kindly ask that you credit SkillCorner and hope you'll notify us on Twitter" | A request, not a condition. We credit them anyway; the notification is a team choice |
| same | "If you have some feedback, some project research that you want to conduct with our data, reach us on our website" | An invitation, not a restriction |
| Tree search at the commit | No other `LICENSE`, `NOTICE`, `TERMS`, or data-specific licence file. A search of Markdown, YAML, text and citation files finds no licence, redistribution or non-commercial terms, only a "commercial information" contact line | No extra conditions in the repository |
| Hugging Face card of SkillCorner's own `opendata-bodypose` | Tagged `license: mit` | SkillCorner applies MIT to a data release from this same repository. **This is the strongest available signal that MIT is meant to cover data.** We did not use body pose |
| Project site `skillcorner.github.io/opendata` | The README text, including the credit request | No extra conditions found |
| Analytics Cup 2027 page | "the Analytics Cup was created to open source SkillCorner data"; no licence terms on the page | Consistent with the open-source intent; it does not mention training or weights |

**Read on 4 November, after the first version of this document**
- SkillCorner's standard terms (May 2026; PDF sha256 `c508f236…9d6a`). The website footer's "Terms of Service" link opens this same PDF, so we no longer treat the two as separate unread documents. It is a contract with a paying "Partner": the "Agreement" is the Work Order plus these terms (clause 1.1), and the data licence comes "subject to payment of the applicable Fees" (2.1). Clause 3.3 forbids a Partner to redistribute the Data or build a competing product. **It never mentions open data, the GitHub repository, research use or model training**, so it neither grants nor removes anything for this repository. It also does not say the open-data notice survives. Its restrictions would matter if a member obtained the same files through a paid API, an account or a signed Work Order.
- A team review of these terms (`docs/SPRINT37B_SKILLCORNER_TERMS_REVIEW_20261104.md` on `research/sprint-37b-skillcorner-terms-review-20261104`) reached the same reading. Its clause and page references check out against the PDF.

**Not verified**
- Rights of third parties: whether SkillCorner can grant everything in A-League match data and broadcast-derived tracking. No public source addresses it, and MIT gives no warranty.
- The Analytics Cup rules: they bind that competition's entrants, and we did not enter it.

**A stale line.** The README says 10 matches; the tree holds 20, added by commit `abbc3ff` "Add 10 A-League matches". The licence file covers the whole repository, so nothing turns on it.

## Exactly which files were used

Six development matches for training and for the ball-loss statistics; 14 further matches for evaluation only.

| file | used for |
|---|---|
| `{id}_tracking_extrapolated.jsonl` | model input |
| `{id}_phases_of_play.csv` | frame labels (`culebs/external/skillcorner.py`) |
| `{id}_match.json` | team, pitch size |

`dynamic_events.csv` was downloaded and not used for these labels. The tracking `possession` field, aggregates and body pose were not used.

**Integrity.** All 80 local files of the 20 matches match the commit: 60 by git blob hash and 20 tracking files by their Git LFS sha256 (0 mismatches). The data itself is not in our repository.

## Obligations if the weights are published

| obligation | source | status |
|---|---|---|
| Keep the copyright and permission notice | MIT, if weights count as a "substantial portion" (uncertain; we comply anyway) | the full notice is in `submission/models/x_t1_50_dead37/ATTRIBUTION.md` |
| Credit SkillCorner | README, a request | in `ATTRIBUTION.md`, the bundle `LICENSE`, and the technical document |
| Do not redistribute the data | our own rule; not required by MIT | the bundle ships weights only |
| Do not imply warranty or endorsement | MIT disclaimer | wording checked in the technical document |
| DFL: attribution and list of changes | CC BY 4.0 | in `ATTRIBUTION.md` |

**Licence of the weights themselves.** MIT on the training data does not make the weights MIT. Our attribution and technical document give the upstream notices and do not put a licence on these weights; the team chooses that wording in the approval record, subject to those notices. The committed `ens6_dead33` record describes its readout as "CC BY 4.0 + MIT", which the team should read the same way.

**A gap in existing candidates, not changed here.** `ens6_dead33` and `ens6c_dead36` also use this data (readout and statistics). Their `ATTRIBUTION.md` files credit SkillCorner and say MIT but do not reproduce the notice.
If either is published, add it. That text is outside the weights, so no checksum changes, but `ens6_dead33`'s attribution falls under the existing approval: the team should decide.

## Why this matters more for this candidate

| candidate | SkillCorner's role | weights affected |
|---|---|---|
| `ens6_dead33` | readout only (56 parameters) | 1 file |
| `ens6c_dead36` | readout and statistics | 1 file, plus the statistics |
| `x_t1_50_dead37` | **labels for one of two heads, input to the shared encoder**, readout, statistics | **all 13 files** |

If the team or SkillCorner decides that training is not permitted, this candidate cannot be repaired; the DFL-only `ens6c_dead36` still depends on SkillCorner through its readout and statistics.

## Recommended action

1. A team member confirms three things in writing: (a) all 20 matches came only from the public GitHub repository at the pinned commit and its Git LFS objects; (b) no member accepted a commercial licence or Work Order covering these files; (c) the team accepts this reading, or decides to ask. We can show the local files match the commit (above), but only a person can confirm (b) and how each copy was obtained.
2. Optional, and only a human can send it. Suggested text:
   > We are a team using your open data repository (commit 4340d274), matches and phases of play, to train a possession model for the FC Barcelona More Than A Hack 2027 challenge. We would publish the model weights with credit to SkillCorner and your MIT notice. We do not redistribute the data. Is that within your intended use of the repository's MIT licence?

Sending this asks SkillCorner about the project, so whether to send it is the team's decision. Our reading does not rely on a reply.

## Sources

- [LICENSE at the commit](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE)
- [README at the commit](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/README.md)
- [Hugging Face: SkillCorner/opendata-bodypose](https://huggingface.co/datasets/SkillCorner/opendata-bodypose)
- [Project site](https://skillcorner.github.io/opendata/)
- [Analytics Cup 2027](https://skillcorner.com/analytics-cup-2027)
- [SkillCorner standard terms and conditions, May 2026](https://skillcorner.com/legal/standard-terms-and-conditions) and [its page](https://skillcorner.com/legal/standard-terms-and-conditions-2026)
