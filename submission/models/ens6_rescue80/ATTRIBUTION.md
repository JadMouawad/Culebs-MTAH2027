# Attribution and licence notes (ens6_rescue80 release candidate)

The six model members were trained only on the DFL / Sportec open position data:

Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite
soccer. *Scientific Data* 12, doi:10.1038/s41597-025-04505-y. Data: figshare doi:10.6084/m9.figshare.28196177.v1.
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0). (c) Deutsche Fussball Liga (DFL) / Sportec Solutions.

Changes made to the data before training (as required by CC BY 4.0, "indicate if changes were made"): frozen-ball
samples removed (defreeze), synthetic ball dropout and correlated player / ball tracking failures (random, burst and crowding copies), conversion to 5 Hz features
and H/A/D frame labels derived from the BallStatus / BallPossession attributes.

The weights are derived from CC BY 4.0 data; publishing them requires this attribution. No SkillCorner, Metrica, PFF or
competition data was used to train them (SkillCorner and Metrica were used for evaluation only). Publication of the weights
still needs an explicit human decision by the team.
