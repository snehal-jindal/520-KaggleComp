# Philharmonia Baroque Orchestra subscription prediction

The model estimates whether a past patron will purchase a 2014–15 season subscription. The competition evaluates the ordering of predicted probabilities with AUROC.

## Reproduce

Extract the competition ZIP outside the repository. Pass the directory containing the nine CSV files to the script.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model.py /path/to/for_students submission.csv
```

The script validates the account keys, runs repeated cross-validation, and writes the 2,975 predictions in sample-submission order. The raw data and output CSV stay local.

## Data and modeling

The labeled set has 6,941 patrons and 332 subscription buyers. The test set has 2,975 patrons. All labeled and test accounts join to the account table, and the two sets have disjoint IDs. Subscription and ticket features use seasons before 2014–15. The account features use billing and shipping geography, 2013 donation amount, and first donations dated before 2014. Lifetime donation totals and relationship codes were excluded because their snapshot timing was unclear.

The renewal baseline ranks patrons from subscription recency, streak, frequency, and decayed purchase activity. Its largest source of lost AUROC was ranking first-time buyers among patrons with no subscription history. Ticket activity and proximity to the three main concert markets give the model information for that group. The compact logistic model also allows the effects of geography, ticket activity, and donations to differ for recent subscribers and patrons without subscription history.

The historical prior comes from a shallow CatBoost model trained to predict purchases in the 2011–12 through 2013–14 seasons. For each season, the training population consists of patrons with an earlier ticket or subscription record. Features use records before the season being predicted. More recent historical seasons carry more weight. This prior summarizes purchase patterns from earlier seasons for the logistic model. Earlier seasons do not provide a comparable independent evaluation population for 2014–15, so the reported validation uses the actual 2014–15 training labels.

## Validation and selection

The baseline and finalists use six repetitions of stratified five-fold cross-validation. Each account receives one held-out prediction per repetition. Standardization and logistic regression are fitted inside each fold. The historical prior uses earlier transaction records and never uses 2014–15 labels. Test probabilities average the 30 fitted finalist models.

| Candidate | Mean local AUROC |
| --- | ---: |
| Renewal-history baseline, first three repetitions | 0.9500 |
| Boosted trees on subscription and ticket data, first three repetitions | 0.9624 |
| Compact logistic model | 0.9771 |
| Compact logistic model with historical prior | 0.9773 |

The historical prior improved the compact model slightly across the six repetitions. The gain is small relative to the variation across splits. Adding neighborhood purchase density or more detailed geographic interactions reduced validation AUROC. The wider boosted-tree models also trailed the compact logistic model. The finalists were compared on reused validation folds, so these scores should be read as development estimates.

The selected pipeline retains the compact model with the historical prior. Account IDs are join keys and are excluded from model features.

## Second candidate

The initial submission scored 0.97452 on Kaggle's public split. The second candidate keeps the original model and blends in two separate logistic models for recent and lapsed subscribers. The recent-subscriber model uses the previous season's package, venue, seating, and tier. The lapsed-subscriber model uses recency, earlier purchasing, ticket activity, and geography. A small score adjustment downweights patrons with neither a recent subscription nor a billing location within 100 km of the main concert markets. This includes patrons whose billing location could not be resolved.

The second candidate improved on the original across six development and six fresh cross-validation repetitions. On the fresh splits, mean AUROC was 0.97582 versus 0.97518. The estimated paired improvement was small. The extra ZIP demographics, neighborhood demand estimates, and direct pairwise optimization did not improve validation and were excluded.

```sh
python challenger.py /path/to/for_students submission-v2.csv
```

## Third candidate and timing assumption

V2 scored 0.97445 publicly. V3 adds the supplied lifetime donation amount and donation count, basic account completeness indicators, and interactions for patrons without a subscription history. Its regularized logistic model uses C=1 and averages 60 models across twelve stratified five-fold repetitions.

On six additional validation splits, the initial model averaged 0.97653 and V3 averaged 0.98082. Across all twelve comparison splits, V3 averaged 0.98004. These estimates come from repeated splits of the same 6,941 labeled accounts and are development results.

The recording and competition description present account.csv as donation history available for modeling, so V3 treats the supplied fields as eligible competition inputs. The exact prediction cutoff is undocumented. A sensitivity experiment masking lifetime donation fields for patrons whose first donation was recorded in 2014 removed the improvement. This makes V3 dependent on the availability of 2014 donor information. Its results do not establish performance for a forecast made before those donations occurred.

The original and second submission files remain available separately. V3 can be reproduced with the following command.

```sh
python donation_model.py /path/to/for_students submission-v3.csv
```

V3 scored 0.97676 on the public leaderboard, exceeding V1's 0.97452 and V2's 0.97445. It was selected for final scoring. Further experiments with average gift size, annual donation share, and explicit recent-donor interactions did not show a consistent validation advantage. Public leaderboard performance does not establish the private score.
