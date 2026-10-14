import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baselines import BASELINES, _design
from common import load_paths
from data import load_daily
from metrics import add_scores, city_equal_summary
from model import build_model, load_model_config, linear_predictor, flat_posterior
from predict import block_forecast
from run import resolve_spec_for_run
from splits import load_splits, prepare_fold

OUT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "formal_20260930"


def main():
    data = load_daily(load_paths()["data_csv"])
    splits = load_splits()
    records = []
    bundle = prepare_fold(data, splits, "final")
    predictions = add_scores(pd.concat([function(bundle) for function in BASELINES.values()], ignore_index=True))
    predictions.to_csv(OUT / "supplementary_final_baselines.csv", index=False)
    for (model, task), group in predictions.groupby(["model", "task"]):
        overall, _ = city_equal_summary(group)
        records.append({"model": model, "task": task, "wis_log": overall["city_equal_wis_log"], "coverage_90": overall["city_equal_coverage_90"], "mae_log": overall["city_equal_mae_log"]})
    print("末段基线补充核验，使用现有实现", records, flush=True)
    ratios = []
    for fold in ("dev1", "dev2", "final"):
        bundle = prepare_fold(data, splits, fold)
        for index, city in enumerate(bundle["prep"]["city_names"]):
            selected = bundle["train_arrays"]["city_index"] == index
            design = _design(bundle["train_arrays"])[selected]
            response = bundle["train_arrays"]["response"][selected]
            coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
            residual = response - design @ coefficients
            rho = np.sum(residual[1:] * residual[:-1]) / np.sum(residual[:-1] ** 2)
            factor = np.sqrt(np.sum(rho ** (2 * np.arange(56))))
            ratios.append({"fold": fold, "city": city, "rho": float(rho), "innovation_uncertainty_factor_at_56": float(factor)})
    print("AR-OLS 56 天创新标准差累积系数", pd.DataFrame(ratios).groupby("fold")["innovation_uncertainty_factor_at_56"].agg(["min", "median", "max"]).to_dict(), flush=True)

    stability = []
    spec = resolve_spec_for_run(load_model_config(), "M2")
    for fold in ("dev1", "final"):
        bundle = prepare_fold(data, splits, fold)
        idata = az.from_netcdf(RUNS / "M2" / fold / "posterior.nc")
        predictions, extra = block_forecast(idata, spec, bundle, sample_cap=800, paths=200, seed=31)
        overall, _ = city_equal_summary(add_scores(predictions))
        original = json.loads((RUNS / "M2" / fold / "metrics_overall.json").read_text())["tasks"]["block"]
        stability.append({"fold": fold, "sample_cap": 800, "paths": 200, "seed": 31, "wis_new": overall["city_equal_wis_log"], "wis_original": original["city_equal_wis_log"], "mean_log_density_new": overall["city_equal_mean_log_predictive_density"], "mean_log_density_original": original["city_equal_mean_log_predictive_density"], **extra})
        idata.close()
        print("预测积分稳定性核验", stability[-1], flush=True)

    bundle = prepare_fold(data, splits, "dev1")
    model = build_model(spec, bundle["train_arrays"], bundle["prep"]["city_names"])
    variables = [variable.name for variable in model.free_RVs + model.deterministics]
    with model:
        prior = pm.sample_prior_predictive(draws=1000, random_seed=27, var_names=variables)
    arrays = bundle["train_arrays"]
    count = prior.prior.sizes["chain"] * prior.prior.sizes["draw"]
    prior_idata = xr.DataTree.from_dict({"posterior": prior.prior.dataset})
    mu = linear_predictor(prior_idata, spec, arrays, np.arange(count))
    nu = flat_posterior(prior_idata, "nu")
    phi = flat_posterior(prior_idata, "phi_city")[:, arrays["city_index"]]
    sigma = flat_posterior(prior_idata, "sigma_city")[:, arrays["city_index"]]
    sigma_init = flat_posterior(prior_idata, "sigma_init")[:, arrays["city_index"]]
    rng = np.random.default_rng(27)
    prior_y = np.empty_like(mu)
    previous = np.zeros(count)
    for index in range(len(arrays["response"])):
        scale = sigma_init[:, index] if arrays["restart"][index] else sigma[:, index]
        offset = np.zeros(count) if arrays["restart"][index] else phi[:, index] * previous
        previous = offset + rng.standard_t(nu) * scale
        prior_y[:, index] = mu[:, index] + previous
    prior_result = {"draws": count, "seed": 27, "negative_joint_response_fraction": float((prior_y < 0).mean()), "negative_intercept_fraction": float((flat_posterior(prior_idata, "alpha_city") < 0).mean()), "log_response_quantiles": np.quantile(prior_y, [0.005, 0.05, 0.5, 0.95, 0.995]).tolist(), "observed_log_response_range": [float(arrays["response"].min()), float(arrays["response"].max())]}
    print("完整先验预测核验", prior_result, flush=True)
    (OUT / "followup_checks.json").write_text(json.dumps({"final_baselines_existing_implementation": records, "ar_ols_variance_accumulation": ratios, "predictive_integration_stability": stability, "joint_prior_predictive": prior_result}, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
