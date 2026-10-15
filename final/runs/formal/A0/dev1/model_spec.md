# A0 model specification

Ablation, no pooling: city weather coefficients keep a fixed pre-specified prior scale, climate moderators removed.

- calendar terms: `weekday`
- annual harmonic pairs: 1
- city linear trend: True
- AR(1) residual process: False
- climate moderators: False
- weather coefficient pooling: `independent`
- city intercept parameterisation: `centred`

## Response

`y_it = log(1 + usage_it)` on daily city observations, Student-t likelihood with a
shared degrees-of-freedom parameter and city-specific scale.

## Linear predictor

`mu_it = alpha_i + b_temp_i * T_it + b_temp_sq_i * Q_it + b_rain_i * R_it + calendar + trend`

- `T_it = (utci_mean_c_it - city training mean) / 10`
- `Q_it = T_it^2 - city training mean of T^2`
- `R_it = (log1p(precipitation_mm_it) - city training mean) / pooled training standard deviation`
- calendar: shared sum-to-zero weekday effect (`weekday`), or a city weekend coefficient (`weekend`)
- trend: city slope per 100 days, centred on the training-period mean day number

All centring constants, scales and standard deviations are computed from training rows only.

## Prior scales

| quantity | scale |
|---|---:|
| `alpha_population_sd` | 2.0 |
| `alpha_city_sd` | 2.0 |
| `beta_temp_sd` | 0.5 |
| `beta_temp_sq_sd` | 0.3 |
| `beta_rain_sd` | 0.3 |
| `beta_weekend_sd` | 0.4 |
| `beta_weekday_sd` | 0.4 |
| `beta_season_sd` | 0.8 |
| `beta_trend_population_sd` | 0.3 |
| `beta_trend_city_sd` | 0.2 |
| `moderator_sd` | 0.3 |
| `temp_moderator_sd` | 0.2 |
| `log_sigma_population_mean` | -0.5 |
| `log_sigma_population_sd` | 0.8 |
| `log_sigma_city_sd` | 0.5 |
| `nu_minus_two_rate` | 0.1 |
| `z_population_sd` | 0.5 |
| `z_city_sd` | 0.5 |

## Interpretation

Weather coefficients are observational conditional associations given the calendar and
trend controls. They are not causal effects.
