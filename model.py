"""Fit and validate patron renewal models using pre-season records."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ID = "account.id"
TARGET = 2014


def read_inputs(data):
    train = pd.read_csv(data / "train.csv").set_index(ID)
    test = pd.read_csv(data / "test.csv").set_index("ID")
    sample = pd.read_csv(data / "sample_submission.csv")
    accounts = pd.read_csv(data / "account.csv", dtype=str, encoding="latin-1").set_index(ID)
    assert train.index.is_unique and test.index.is_unique and accounts.index.is_unique
    assert train.index.intersection(test.index).empty
    assert set(test.index) == set(sample.ID) and sample.ID.is_unique
    history = {}
    for prefix, name in [("sub", "subscriptions"), ("ticket", "tickets_all")]:
        frame = pd.read_csv(data / f"{name}.csv")
        frame["year"] = frame.season.str[:4].astype(int)
        assert frame.year.max() < TARGET
        history[prefix] = frame
    return train, test, sample, accounts, history


def transaction_features(ids, records, year, prefix):
    past = records.loc[records.year < year].copy()
    past["lag"] = year - past.year
    grouped = past.groupby(ID)
    recent = past.loc[past.lag == past[ID].map(grouped.lag.min())]
    recent_group = recent.groupby(ID)
    seasons = past[[ID, "year"]].drop_duplicates().sort_values("year", ascending=False)
    seasons["offset"] = seasons.groupby(ID).cumcount()
    seasons["consecutive"] = (year - seasons.year - seasons.offset == 1)
    data = {
        "lag": grouped.lag.min(),
        "first": grouped.lag.max(),
        "seasons": grouped.year.nunique(),
        "streak": seasons.groupby(ID).consecutive.sum(),
        "rows": grouped.size(),
        "decay": past.assign(decay=0.55 ** (past.lag - 1)).groupby(ID).decay.sum(),
    }
    for lag in range(1, 6):
        window = past.loc[past.lag <= lag].groupby(ID)
        data[f"seasons_{lag}"] = window.year.nunique()
        data[f"rows_{lag}"] = window.size()
        data[f"seats_{lag}"] = window["no.seats"].sum()
    for column in ["no.seats", "price.level", "subscription_tier"]:
        if column in past:
            numeric = pd.to_numeric(past[column], errors="coerce")
            data[f"{column}_average"] = numeric.groupby(past[ID]).mean()
            data[f"{column}_last"] = numeric.loc[recent.index].groupby(recent[ID]).mean()
    for column in ["location", "section", "package", "marketing.source", "set"]:
        if column in recent:
            data[f"{column}_diversity"] = grouped[column].nunique()
            data[f"{column}_latest_diversity"] = recent_group[column].nunique()
            counts = pd.crosstab(recent[ID], recent[column])
            for value in counts.columns[counts.sum() >= 20]:
                data[f"{column}_{value}"] = counts[value]
    return pd.DataFrame(data).reindex(ids).fillna(-1).add_prefix(f"{prefix}_")


def build_features(ids, accounts, history, year):
    x = pd.concat([
        transaction_features(ids, records, year, prefix)
        for prefix, records in history.items()
    ], axis=1)
    if year == TARGET:
        a = accounts.reindex(ids)
        first_donation = pd.to_datetime(a["first.donated"], errors="coerce").dt.year
        x["donor_before_2014"] = first_donation.lt(TARGET).astype(int)
        x["first_donation_age"] = (TARGET - first_donation).where(first_donation.lt(TARGET), -1)
        x["donation_2013"] = pd.to_numeric(a["amount.donated.2013"], errors="coerce").fillna(-1)
        for kind in ["shipping", "billing"]:
            zipcode = a[f"{kind}.zip.code"].str.extract(r"^\s*(\d{5})", expand=False)
            x[f"{kind}_zip_missing"] = zipcode.isna().astype(int)
            for code in ["94", "945", "946", "947", "949", "941", "943", "940", "950", "951"]:
                x[f"{kind}_{code}"] = zipcode.str.startswith(code, na=False).astype(int)
    return x


def geography(ids, accounts, data):
    zips = pd.read_csv(data / "zipcodes.csv", dtype={"Zipcode": str})
    zips["Zipcode"] = zips.Zipcode.str.zfill(5)
    zips = zips.sort_values("Decommisioned").drop_duplicates("Zipcode").set_index("Zipcode")
    account = accounts.reindex(ids)
    zip_code = account["billing.zip.code"].str.extract(r"^\s*(\d{5})", expand=False)
    geo = zips.reindex(zip_code)
    geo.index = ids
    features = pd.DataFrame(index=ids)
    features["geo_california"] = geo.State.eq("CA").astype(int)
    features["geo_coordinates_missing"] = geo.Lat.isna().astype(int)
    # Approximate centers for the three main Bay Area concert markets.
    for city, latitude, longitude in [
        ("sf", 37.77, -122.42), ("east_bay", 37.87, -122.27), ("peninsula", 37.44, -122.16)
    ]:
        distance = np.hypot(111 * (geo.Lat - latitude), 88 * (geo.Long - longitude))
        features[f"geo_{city}"] = np.log1p(distance.clip(upper=300).fillna(300))
    features["geo_closest"] = features[["geo_sf", "geo_east_bay", "geo_peninsula"]].min(axis=1)
    return features


def compact_features(x):
    lag = x.sub_lag
    ticket_lag = x.ticket_lag
    features = pd.DataFrame({
        "recent": lag.eq(1),
        "second": lag.eq(2),
        "third_fourth": lag.between(3, 4),
        "older": lag.gt(4),
        "ticket": ticket_lag.gt(0),
        "ticket_recent": ticket_lag.eq(1),
        "ticket_decay": np.log1p(x.ticket_decay.clip(lower=0)),
        "streak": np.log1p(x.sub_streak.clip(lower=0)),
        "first_season": lag.eq(1) & x.sub_seasons.eq(1),
        "season_count": np.log1p(x.sub_seasons.clip(lower=0)),
        "decay": np.log1p(x.sub_decay.clip(lower=0)),
        "donor": x.donation_2013.gt(0),
        "donor_history": x.donor_before_2014,
        "donor_age": np.log1p(x.first_donation_age.clip(lower=0)),
        "donation_amount": np.log1p(x.donation_2013.clip(lower=0)),
        "billing_missing": x.billing_zip_missing,
        "billing_local": x.billing_94,
        "shipping_present": 1 - x.shipping_zip_missing,
        "closest": x.geo_closest,
        "sf": x.geo_sf,
        "east_bay": x.geo_east_bay,
        "peninsula": x.geo_peninsula,
    }, index=x.index).astype(float)
    for group, member in {"recent": lag.eq(1), "never": lag.eq(-1)}.items():
        for column in ["ticket", "donor", "billing_local", "closest", "shipping_present"]:
            features[f"{group}_{column}"] = member * features[column]
    return features


def baseline_columns():
    return ["sub_lag", "sub_streak", "sub_seasons", "sub_decay", "sub_seasons_3"]


def report_pairs(y, scores, lags):
    cohort = np.select([lags == 1, lags > 1], ["recent", "lapsed"], default="never")
    pairs = {}
    for buyer in ["recent", "lapsed", "never"]:
        for nonbuyer in ["recent", "lapsed", "never"]:
            diff = scores[(cohort == buyer) & (y == 1), None] - scores[(cohort == nonbuyer) & (y == 0)]
            pairs[buyer, nonbuyer] = (diff < 0).sum() + 0.5 * (diff == 0).sum()
    print("Largest lost pairs (buyer cohort, nonbuyer cohort):", sorted(pairs.items(), key=lambda item: -item[1])[:5])


def cat(depth=4, iterations=400, seed=42):
    return CatBoostClassifier(depth=depth, iterations=iterations, learning_rate=0.035,
                              l2_leaf_reg=6, verbose=False, thread_count=4,
                              random_seed=seed, allow_writing_files=False)


def logit(regularization=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=2000))


def historical_prior(history, current):
    """Estimate historical conversion propensity among previously active patrons."""
    examples, outcomes, weights = [], [], []
    subscriptions = history["sub"]
    for year in range(2011, TARGET):
        eligible = pd.Index(sorted(set().union(*[
            set(records.loc[records.year < year, ID])
            for records in history.values()
        ])), name=ID)
        x = build_features(eligible, None, history, year)
        positives = subscriptions.loc[subscriptions.year == year, ID]
        examples.append(x)
        outcomes.append(eligible.isin(positives).astype(int))
        weights.append(np.full(len(eligible), 0.8 ** (TARGET - year - 1)))
    historical = pd.concat(examples).fillna(-1)
    current = current.reindex(columns=historical.columns, fill_value=-1)
    learner = cat(4, 350)
    learner.fit(historical, np.concatenate(outcomes), sample_weight=np.concatenate(weights))
    return learner.predict_proba(current)[:, 1]


def cross_validate(features, labels, make_model):
    oof_predictions, test_predictions = [], []
    n = len(labels)
    for seed in [27, 73, 126, 58, 158, 244]:
        oof = np.empty(n)
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        for training, validation in folds.split(features.iloc[:n], labels):
            model = make_model()
            model.fit(features.iloc[training], labels[training])
            oof[validation] = model.predict_proba(features.iloc[validation])[:, 1]
            test_predictions.append(model.predict_proba(features.iloc[n:])[:, 1])
        oof_predictions.append(oof)
    scores = [roc_auc_score(labels, oof) for oof in oof_predictions]
    return np.asarray(oof_predictions), np.mean(test_predictions, axis=0), scores


def main(data, output):
    train, test, sample, accounts, history = read_inputs(data)
    ids = train.index.append(test.index)
    x = build_features(ids, accounts, history, TARGET).join(geography(ids, accounts, data))
    y = train.label.to_numpy()
    n = len(y)
    print("Account join coverage:", int(ids.isin(accounts.index).sum()), "/", len(ids))
    for name, rows in [("train", slice(n)), ("test", slice(n, None))]:
        lag = x.sub_lag.iloc[rows]
        print(name, {"recent": int((lag == 1).sum()), "lapsed": int((lag > 1).sum()),
                     "never": int((lag == -1).sum())})

    compact = compact_features(x)
    prior = historical_prior(history, x)
    candidates = [
        ("renewal", x[baseline_columns()], 0.1),
        ("compact", compact, 10),
        ("historical", compact.assign(historical_prior=prior), 10),
    ]
    for name, features, regularization in candidates:
        oof, predictions, scores = cross_validate(features, y, lambda: logit(regularization))
        print(name, np.round(scores, 6), "mean", np.mean(scores), flush=True)
        if name != "compact":
            report_pairs(y, oof.mean(axis=0), x.sub_lag.to_numpy()[:n])

    # The historical candidate is last and uses only past-season data for its prior.
    sample["Predicted"] = pd.Series(predictions, index=test.index).reindex(sample.ID).to_numpy()
    assert sample.Predicted.between(0, 1).all()
    assert list(sample.columns) == ["ID", "Predicted"] and len(sample) == 2975
    sample.to_csv(output, index=False)
    print("Wrote", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.data, args.output)
