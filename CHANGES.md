# Research v2

Created a separate follow-up without modifying the original handoff. Implemented and ran 61 configurations: matched constrained/free regularization, log histories, splines, interactions, multiplicative offsets and three tree class-handling alternatives. Selected by 2022 AP, with separate calibration assessment. Retained full candidate ledger and artifacts. Replaced negative subsampling with full-population circular 7/30-day paired bootstrap for AUC and AP. Produced a new paper with explicit retrospective scope.

Matched optimizer verification against scikit-learn: maximum probability difference 4.91e-7.

Selected v2 versus original additive: AUC 0.71444 to 0.85915; AP 0.06824 to 0.28596; Brier 0.021426 to 0.018487; top-decile recall 27.09% to 59.96%.

Selected v2 versus original RF: AUC 0.86011 to 0.85915; AP 0.27709 to 0.28596; Brier 0.018759 to 0.018487; top-decile recall 60.58% to 59.96%. The paper reports this mixed result.

Thirty-day paired AP-difference interval versus original RF: [0.00240, 0.01469]. AUC-difference interval: [-0.00403, 0.00186]. These intervals condition on one previously examined year and are not independent confirmation.
