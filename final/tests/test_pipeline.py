import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arviz as az
from scipy import stats

from common import load_paths, verify_frozen_input
from data import DATE, CITY, load_daily, roster_prediction, segment_restart
from metrics import WIS_LEVELS, add_scores, city_equal_summary, seasonal_naive_scale
from model import load_model_config, resolve_spec
from predict import block_forecast, one_step_forecast, test_restart, _ar_paths
from splits import load_splits, prepare_fold, assert_no_leakage

PATHS = load_paths()
FROZEN = verify_frozen_input(PATHS)
FRAME = load_daily(PATHS["data_csv"])
SPLITS = load_splits()
CONFIG = load_model_config()


def _spec(name):
    spec = resolve_spec(CONFIG, name)
    spec["priors"] = CONFIG["prior_scales"]
    return spec


def test_frozen_input_hash():
    assert FROZEN["sha256"] == PATHS["data_sha256"], "frozen input hash drifted"


def test_split_boundaries():
    for name in SPLITS["folds"]:
        bundle = prepare_fold(FRAME, SPLITS, name)
        assert_no_leakage(bundle)
        assert bundle["train_frame"][DATE].max() == bundle["window"]["cutoff"]
        assert bundle["test_frame"][DATE].min() == bundle["window"]["test_start"]
        assert bundle["test_frame"][DATE].max() == bundle["window"]["test_end"]
        assert len(bundle["test_frame"]) == 23 * 56, f"{name} test block is not 23 x 56"
        assert set(bundle["train_frame"][CITY]) == set(roster_prediction(FRAME))


def test_preprocessing_ignores_test_rows():
    bundle = prepare_fold(FRAME, SPLITS, "dev1")
    prep_before = dict(bundle["prep"])
    train_before = bundle["train_arrays"]["response"].copy()
    perturbed = FRAME.copy()
    mask = perturbed[DATE] > bundle["window"]["cutoff"]
    perturbed.loc[mask, "usage"] = perturbed.loc[mask, "usage"] * 3.0 + 100.0
    perturbed.loc[mask, "usage_log1p"] = np.log1p(perturbed.loc[mask, "usage"])
    perturbed.loc[mask, "utci_mean_c"] = perturbed.loc[mask, "utci_mean_c"] + 25.0
    rebuilt = prepare_fold(perturbed, SPLITS, "dev1")
    assert prep_before == rebuilt["prep"], "preprocessing changed when only test rows changed"
    assert np.allclose(train_before, rebuilt["train_arrays"]["response"]), "training response changed"
    assert not np.allclose(
        bundle["test_arrays"]["response"], rebuilt["test_arrays"]["response"]
    ), "test perturbation had no effect"


def test_segment_restart_flags():
    city_index = np.array([0, 0, 0, 1, 1, 1])
    dates = pd.to_datetime([
        "2017-01-01", "2017-01-02", "2017-01-04",
        "2017-01-01", "2017-01-02", "2017-01-03",
    ])
    restart = segment_restart(city_index, dates)
    assert restart.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], restart.tolist()


def test_ar_innovation_variance():
    rng = np.random.default_rng(3)
    sample_count, horizon = 400, 60
    nu = np.full(sample_count, 8.0)
    sigma = np.full(sample_count, 0.4)
    error_last = np.zeros(sample_count)
    observed = np.zeros(horizon)
    mu_city = np.zeros((sample_count, horizon))
    restart = np.zeros(horizon)

    samples_white, _ = _ar_paths(
        rng, mu_city, np.zeros(sample_count), sigma, sigma, nu, error_last, restart,
        40, observed,
    )
    expected = sigma[0] ** 2 * 8.0 / (8.0 - 2.0)
    assert abs(samples_white.var() - expected) / expected < 0.10, samples_white.var()

    phi = np.full(sample_count, 0.6)
    sigma_init = sigma * np.sqrt((8.0 / 6.0) / (1.0 - 0.36))
    samples_ar, _ = _ar_paths(
        rng, mu_city, phi, sigma, sigma_init, nu, error_last, restart, 40, observed,
    )
    expected_ar = sigma[0] ** 2 * 8.0 / 6.0 / (1.0 - 0.36)
    tail = samples_ar.reshape(sample_count, 40, horizon)[:, :, 40:].ravel()
    assert abs(tail.var() - expected_ar) / expected_ar < 0.12, tail.var()


def _synthetic_posterior(names, spec, seed=5):
    rng = np.random.default_rng(seed)
    chains, draws = 2, 40
    shape = (chains, draws)
    city_shape = (chains, draws, len(names))
    posterior = {
        "alpha_city": rng.normal(7.0, 0.3, city_shape),
        "beta_temp_city": rng.normal(0.1, 0.05, city_shape),
        "beta_temp_sq_city": rng.normal(-0.05, 0.02, city_shape),
        "beta_rain_city": rng.normal(-0.2, 0.05, city_shape),
        "sigma_city": np.abs(rng.normal(0.4, 0.02, city_shape)),
        "nu": rng.normal(8.0, 0.5, shape),
        "weekday_effect": rng.normal(0.0, 0.1, (chains, draws, 7)),
        "beta_sin_city": rng.normal(0.0, 0.1, city_shape),
        "beta_cos_city": rng.normal(0.0, 0.1, city_shape),
        "beta_trend_city": rng.normal(0.0, 0.02, city_shape),
        "beta_weekend_city": rng.normal(0.0, 0.1, city_shape),
        "phi_city": rng.uniform(0.2, 0.4, city_shape),
    }
    posterior["sigma_init"] = posterior["sigma_city"] * np.sqrt(
        posterior["nu"][:, :, None] / (posterior["nu"][:, :, None] - 2.0)
        / (1.0 - posterior["phi_city"] ** 2)
    )
    dims = {key: ["chain", "draw", "city"] for key in posterior if key != "nu"}
    dims["nu"] = ["chain", "draw"]
    dims["weekday_effect"] = ["chain", "draw", "weekday"]
    return az.from_dict(
        {"posterior": posterior},
        coords={"city": list(names), "weekday": np.arange(7)},
        dims=dims,
    )


def test_model_exposes_prediction_variables():
    from model import build_model, coefficient_names

    bundle = prepare_fold(FRAME, SPLITS, "dev1")
    names = bundle["prep"]["city_names"]
    for model_name in ("M0", "M1", "M2", "A0", "A1"):
        spec = _spec(model_name)
        model = build_model(spec, bundle["train_arrays"], names)
        available = set(model.named_vars)
        expected = set(coefficient_names(spec)) | {"sigma_city", "nu"}
        if spec["ar1"]:
            expected |= {"phi_city", "sigma_init", "z_population", "z_city_sd"}
        missing = expected - available
        assert not missing, f"{model_name} is missing {sorted(missing)}"
        if not spec["ar1"]:
            assert "phi_city" not in available

    synthetic = _synthetic_posterior(names, _spec("M2"))
    present = set(synthetic.posterior.dataset.data_vars)
    needed = set(coefficient_names(_spec("M2"))) | {
        "sigma_city", "nu", "weekday_effect", "phi_city", "sigma_init",
    }
    assert not (needed - present), sorted(needed - present)


def test_block_forecast_ignores_test_usage():
    bundle = prepare_fold(FRAME, SPLITS, "dev1")
    spec = _spec("M2")
    idata = _synthetic_posterior(bundle["prep"]["city_names"], spec)
    baseline, _ = block_forecast(idata, spec, bundle, sample_cap=40, paths=30)
    perturbed = prepare_fold(FRAME, SPLITS, "dev1")
    perturbed["test_frame"]["usage"] = perturbed["test_frame"]["usage"] * 5.0 + 10.0
    perturbed["test_arrays"]["response"] = np.log1p(perturbed["test_frame"]["usage"].to_numpy())
    changed, _ = block_forecast(idata, spec, perturbed, sample_cap=40, paths=30)
    for column in ("median_log", "lower_50", "upper_50", "lower_90", "upper_90"):
        assert np.allclose(
            baseline[column].to_numpy(), changed[column].to_numpy()
        ), f"block forecast changed with test usage in {column}"
    assert not np.allclose(
        baseline["log_predictive_density"].to_numpy(),
        changed["log_predictive_density"].to_numpy(),
    ), "log predictive density must be evaluated at the observed value"

    one_step_baseline, _ = one_step_forecast(idata, spec, bundle, sample_cap=40, paths=30)
    one_step_changed, _ = one_step_forecast(idata, spec, perturbed, sample_cap=40, paths=30)
    assert not np.allclose(
        one_step_baseline["median_log"].to_numpy(), one_step_changed["median_log"].to_numpy()
    ), "one-step forecast should depend on observed test usage"


def test_one_step_uses_previous_observation():
    bundle = prepare_fold(FRAME, SPLITS, "dev2")
    spec = _spec("M2")
    idata = _synthetic_posterior(bundle["prep"]["city_names"], spec)
    frame, _ = one_step_forecast(idata, spec, bundle, sample_cap=40, paths=30)
    assert np.allclose(frame["observed_log"].to_numpy(), bundle["test_arrays"]["response"])
    assert frame["horizon"].min() == 1 and frame["horizon"].max() == 56


def test_restart_marks_first_test_day():
    bundle = prepare_fold(FRAME, SPLITS, "dev1")
    names = bundle["prep"]["city_names"]
    restart = test_restart(bundle["test_frame"], bundle["train_frame"], names)
    assert np.all(restart == 0.0), "a contiguous test block must carry the residual state forward"

    synthetic_test = pd.DataFrame({
        "city": ["A", "A", "A", "B", "B"],
        "date": pd.to_datetime([
            "2017-07-20", "2017-07-21", "2017-07-24", "2017-07-17", "2017-07-18",
        ]),
    })
    synthetic_train = pd.DataFrame({
        "city": ["A", "B"],
        "date": pd.to_datetime(["2017-07-16", "2017-07-10"]),
    })
    flags = test_restart(synthetic_test, synthetic_train, ["A", "B"])
    assert flags.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0], flags.tolist()


def test_metrics_recompute_from_csv():
    bundle = prepare_fold(FRAME, SPLITS, "dev1")
    spec = _spec("M0")
    idata = _synthetic_posterior(bundle["prep"]["city_names"], spec)
    frame, _ = block_forecast(idata, spec, bundle, sample_cap=40, paths=30)
    scored = add_scores(frame)
    scale = seasonal_naive_scale(bundle["train_frame"])
    summary, per_city = city_equal_summary(scored, scale=scale)
    rescored = add_scores(pd.read_csv(io.StringIO(scored.to_csv(index=False))))
    assert np.isclose(rescored["wis_log"].mean(), scored["wis_log"].mean())
    assert np.isclose(summary["wis_log"], scored["wis_log"].mean())
    assert len(per_city) == 23
    for level in WIS_LEVELS:
        tag = int(round(level * 100))
        assert scored[f"covered_{tag:02d}"].dtype == bool


def test_unit_transform():
    values = np.array([0.0, 1.0, 2.0])
    assert np.allclose(np.expm1(np.log1p(values)), values)


def test_wis_penalises_miss_and_width():
    from metrics import weighted_interval_score

    alphas = [0.5]
    tight = weighted_interval_score(
        np.array([0.0]), np.array([0.0]), [np.array([-1.0])], [np.array([1.0])], alphas
    )
    wide = weighted_interval_score(
        np.array([0.0]), np.array([0.0]), [np.array([-5.0])], [np.array([5.0])], alphas
    )
    missed = weighted_interval_score(
        np.array([4.0]), np.array([0.0]), [np.array([-1.0])], [np.array([1.0])], alphas
    )
    assert tight < wide, "wider intervals must not score better"
    assert tight < missed, "a miss must not score better than a hit"


TESTS = [
    test_frozen_input_hash,
    test_split_boundaries,
    test_preprocessing_ignores_test_rows,
    test_segment_restart_flags,
    test_ar_innovation_variance,
    test_model_exposes_prediction_variables,
    test_block_forecast_ignores_test_usage,
    test_one_step_uses_previous_observation,
    test_restart_marks_first_test_day,
    test_metrics_recompute_from_csv,
    test_unit_transform,
    test_wis_penalises_miss_and_width,
]


def main():
    failures = []
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as error:
            failures.append((test.__name__, str(error)))
            print(f"FAIL {test.__name__}: {error}")
    if failures:
        print(f"\n{len(failures)} test(s) failed")
        return 1
    print(f"\n{len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
