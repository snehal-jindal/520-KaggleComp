"""Evaluate the competition's supplied account donation-history fields."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model import TARGET, build_features, compact_features, geography, historical_prior, logit, read_inputs


def account_features(accounts, ids, transactions):
    accounts = accounts.reindex(ids)
    extra = pd.DataFrame(index=ids)
    for column in ["amount.donated.lifetime", "no.donations.lifetime"]:
        extra[column] = np.log1p(pd.to_numeric(accounts[column], errors="coerce").clip(lower=0)).fillna(0)
    extra["relationship_present"] = accounts.relationship.notna().astype(int)
    extra["city_missing"] = accounts["billing.city"].isna().astype(int)
    for column in ["shipping.zip.code", "billing.zip.code"]:
        extra[column + "_extended"] = accounts[column].fillna("").str.len().gt(5).astype(int)
    for column in list(extra):
        extra[column + "_never"] = extra[column] * transactions.sub_lag.eq(-1)
    return extra


def main(data, output):
    train, test, sample, accounts, history = read_inputs(data)
    ids = train.index.append(test.index)
    x = build_features(ids, accounts, history, TARGET).join(geography(ids, accounts, data))
    baseline = compact_features(x).assign(historical_prior=historical_prior(history, x))
    features = baseline.join(account_features(accounts, ids, x))
    labels = train.label.to_numpy()
    n = len(labels)
    predictions = []
    fresh_seeds = [1103, 1229, 1361, 1483, 1601, 1741]
    for seed in [319, 441, 608, 731, 896, 992] + fresh_seeds:
        oof = np.empty((2, n))
        for training, validation in StratifiedKFold(5, shuffle=True, random_state=seed).split(labels, labels):
            base_model = logit(10).fit(baseline.iloc[training], labels[training])
            model = logit(1).fit(features.iloc[training], labels[training])
            oof[0, validation] = base_model.predict_proba(baseline.iloc[validation])[:, 1]
            oof[1, validation] = model.predict_proba(features.iloc[validation])[:, 1]
            predictions.append(model.predict_proba(features.iloc[n:])[:, 1])
        print(seed, *(f"{roc_auc_score(labels, p):.6f}" for p in oof), flush=True)
    sample["Predicted"] = pd.Series(np.mean(predictions, axis=0), index=test.index).reindex(sample.ID).to_numpy()
    assert len(sample) == 2975 and sample.ID.is_unique
    assert sample.Predicted.between(0, 1, inclusive="neither").all()
    sample.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.data, args.output)
