"""Refine predictions for recent and lapsed subscribers."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit as log_odds
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model import ID, TARGET, build_features, compact_features, geography, historical_prior, logit, read_inputs


def renewal_details(ids, x, subscriptions):
    last_season = subscriptions.loc[subscriptions.year.eq(TARGET - 1)]
    features = pd.DataFrame(index=ids)
    for column in ["package", "location", "section"]:
        counts = pd.crosstab(last_season[ID], last_season[column]).reindex(ids).fillna(0)
        for value in counts.columns[counts.sum() > 12]:
            features[f"sub_{column}_{value}"] = counts[value]
    features["recent_seats"] = x.sub_seats_1.clip(lower=0)
    features["recent_tier"] = x.sub_subscription_tier_last.where(x.sub_lag.eq(1), 0)
    return features


def predict_fold(base, features, x, labels, training, target):
    model = logit(10).fit(base.iloc[training], labels[training])
    initial = model.predict_proba(base.iloc[target])[:, 1]
    adjusted = log_odds(initial)
    recent = x.sub_lag.eq(1).to_numpy()
    lapsed = x.sub_lag.between(2, 4).to_numpy()
    recent_columns = [
        "streak", "first_season", "season_count", "decay", "donor", "donor_age",
        "donation_amount", "ticket", "closest", "historical_prior",
    ] + [c for c in features if c.startswith(("sub_", "recent_")) and c not in base]
    lapsed_columns = [
        "second", "season_count", "ticket", "ticket_recent", "donor_history",
        "closest", "billing_local", "historical_prior",
    ]
    for cohort, columns in [(recent, recent_columns), (lapsed, lapsed_columns)]:
        fit_rows = training[cohort[training]]
        subset = cohort[target]
        specialist = logit(0.08).fit(features.iloc[fit_rows][columns], labels[fit_rows])
        scores = specialist.predict_proba(features.iloc[target[subset]][columns])[:, 1]
        adjusted[subset] = 0.55 * adjusted[subset] + 0.45 * log_odds(scores)
    far = x.geo_closest.to_numpy() > np.log1p(100)
    adjusted[far[target] & ~recent[target]] -= 1
    return initial, expit(adjusted)


def main(data, output):
    train, test, sample, accounts, history = read_inputs(data)
    ids = train.index.append(test.index)
    x = build_features(ids, accounts, history, TARGET).join(geography(ids, accounts, data))
    base = compact_features(x).assign(historical_prior=historical_prior(history, x))
    features = base.join(renewal_details(ids, x, history["sub"]))
    labels = train.label.to_numpy()
    n = len(labels)
    test_rows = np.arange(n, len(ids))
    test_predictions, validation = [], []
    development_seeds = [27, 73, 126, 58, 158, 244]
    fresh_seeds = [319, 441, 608, 731, 896, 992]
    for seed in development_seeds + fresh_seeds:
        oof = np.empty((2, n))
        for training, held_out in StratifiedKFold(5, shuffle=True, random_state=seed).split(labels, labels):
            target = np.concatenate([held_out, test_rows])
            scores = np.array(predict_fold(base, features, x, labels, training, target))
            oof[:, held_out] = scores[:, :len(held_out)]
            test_predictions.append(scores[1, len(held_out):])
        validation.append(oof)
        print(seed, *(f"{roc_auc_score(labels, p):.6f}" for p in oof), flush=True)

    fresh = np.asarray(validation[len(development_seeds):])
    scores = np.array([[roc_auc_score(labels, p) for p in oof] for oof in fresh])
    print("Fresh split means (baseline, challenger):", scores.mean(axis=0))
    # Resample accounts, keeping the two models paired and preserving label counts.
    rng = np.random.default_rng(1404)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    deltas = []
    mean_oof = fresh.mean(axis=0)
    for _ in range(1000):
        rows = np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
        base_auc, new_auc = (roc_auc_score(labels[rows], p[rows]) for p in mean_oof)
        deltas.append(new_auc - base_auc)
    print("Paired bootstrap 95% interval for change:", np.quantile(deltas, [0.025, 0.975]))

    sample["Predicted"] = pd.Series(np.mean(test_predictions, axis=0), index=test.index).reindex(sample.ID).to_numpy()
    assert len(sample) == 2975 and sample.Predicted.between(0, 1, inclusive="neither").all()
    sample.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.data, args.output)
