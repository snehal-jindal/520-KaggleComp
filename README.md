# Predicting Philharmonia Baroque Orchestra subscriptions

For the 2014–15 season, the task was to rank past patrons by how likely they were to buy a subscription. Of the 6,941 patrons in the training data, 332 bought one. The hard part was figuring out who belonged near the top of that list, especially among patrons who had never subscribed. I used AUROC, the competition metric, to judge how well each model ranked buyers ahead of nonbuyers. My final submission scored **0.97676 on Kaggle’s public leaderboard**. The biggest improvement came when I added the lifetime donation fields to the model.

## Building the account data

The submission needed one prediction per account. The raw purchase files were organized by transaction, so a patron who bought several tickets appeared several times. I summarized tickets and subscriptions separately by account before joining them to the account records. That kept the modeling table at one row per patron and preserved accounts with no purchase history. I checked that the training and test IDs were unique, that the sets had no accounts in common, and that the test IDs matched the sample submission. I used the IDs to join and order records.

I wanted purchase history to capture both how often someone bought and how recently they did it. Two patrons might have subscribed three times each, but I’d expect different renewal behavior if one subscribed last season and the other stopped years ago. I kept the time since the last subscription, the number of seasons subscribed, and the current streak. I also built weighted counts for subscriptions and individual tickets. A purchase last season contributed 1. A purchase two seasons back contributed 0.55, and one three seasons back contributed about 0.30.

The code uses the start year in each season label to decide which transactions are available. For the 2014–15 prediction, ticket and subscription records end with the preceding season. Missing transaction summaries start at `-1`. That makes it possible to handle an account with no recorded subscription separately from an account whose last subscription was a long time ago.

| Data source | What it contributed |
| --- | --- |
| Subscription records | Time since the last subscription, number of seasons, current streak, and weighted purchase activity |
| Individual ticket records | Whether a patron had bought a ticket, whether they bought one last season, and weighted ticket activity |
| Account records | Donation history and indicators for incomplete account details |
| ZIP reference file | Approximate distance from the billing ZIP to San Francisco, the East Bay, and the Peninsula |

I converted donation amounts to numbers and extracted five-digit ZIP codes where possible. I matched billing ZIPs to coordinates and capped the distance estimates at 300 km. Accounts without usable coordinates received the capped value. A separate indicator recorded whether the billing ZIP itself was missing.

## What changed between models

I started with a baseline built from subscription history. It averaged about **0.950 AUROC** over the first three validation runs. The score was a starting point, but the ranking errors told me where to spend my time. The biggest group of errors came from comparisons between two patrons who had never subscribed. One eventually bought a subscription, but the baseline had almost the same information about both.

That was my reason for adding individual ticket history, geography, and donations. A patron who bought concert tickets last season has shown some interest even if they have no subscription record. I let the effect of features like ticket activity differ for recent subscribers and patrons who had never subscribed. A ticket can tell me something different about those two groups.

I also used earlier seasons to build a historical purchase score. For each season from 2011–12 through 2013–14, I started with accounts that already had a recorded ticket or subscription purchase. I built their features from transactions before that season and used subscription purchases during the season as the outcome. I fitted a CatBoost model with depth 4, 350 iterations, and a learning rate of 0.035. More recent seasons got more weight because they were closer to 2014–15. The resulting score went into the final logistic regression as a feature. The historical model learned from earlier-season outcomes only.

V1 combined that historical score with a compact set of patron features. In V2, I blended in predictions from separate models for recent and lapsed subscribers. That helped a little locally, but the public score went from **0.97452 to 0.97445**. The two submissions were almost tied. I went back to the group that had caused the most trouble for the baseline and looked for more information about people who had never subscribed.

V3 added lifetime donation amount and count, plus a few account indicators, including whether the billing city was missing. I used `log(1 + x)` on both lifetime donation fields to compress their range. I also let the new features have different effects for patrons with no subscription history. The final estimator was L2-regularized logistic regression with `C=1` and a maximum of 2,000 iterations. With only 332 buyers, I wanted the model to pick up those specific differences without giving too much weight to a handful of unusual accounts. I tried more detailed donation features and boosted trees, but the validation results did not consistently improve on V3.

I treated the lifetime donation totals in the supplied account file as history available for the competition. The exact cutoff is undocumented, so I checked how much the result depended on it. When I masked the lifetime fields for patrons whose first donation was recorded in 2014, the validation gain disappeared. For a forecast made as of a specific date, I’d verify when those donations were recorded before using the totals.

## Validation and final submission

I split the labeled accounts into five folds with about the same proportion of buyers in each. For each fold, I fitted the scaler and logistic regression on four-fifths of the accounts and predicted the remaining fifth. Once all five folds had been scored, I pooled those held-out predictions and calculated one AUROC. I used the same folds for the models I was comparing so the difference wouldn’t come from one model getting an easier split.

V3 averaged **0.98004 AUROC** across 12 runs. In the six additional runs, it averaged **0.98082** against **0.97653** for V1 on the same folds. Those repetitions reuse the 6,941 labeled accounts. I used them to see whether the comparison held when I changed the fold assignments.

| Submission | Main change | Kaggle public AUROC |
| --- | --- | ---: |
| V1 | Compact account features and a historical purchase score | 0.97452 |
| V2 | Separate models for recent and lapsed subscribers | 0.97445 |
| **V3** | **Lifetime donation history and account indicators** | **0.97676** |

I picked V3 because it improved the local comparison and had the highest public score. Each of the 60 V3 fits also predicted the test accounts. I averaged those probabilities for each account and wrote them in sample-submission order. The final file contained 2,975 unique account IDs with one probability between 0 and 1 for each.

## Reproducing V3

Extract the competition data locally and run these commands from the repository with Python 3.10.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python donation_model.py /path/to/for_students submission-v3.csv
```

The per-run scores and the checksum for the submission are in [V3 validation results](results/v3-validation.md).
