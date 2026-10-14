# M2 model specification

M1 with a city-specific AR(1) residual process and Student-t innovations.

- calendar terms: `weekday`
- annual harmonic pairs: 1
- city linear trend: True
- AR(1) residual process: True
- climate moderators: True
- weather coefficient pooling: `hierarchical`
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

## AR(1) residual process

`e_it = phi_i * e_i,t-1 + eta_it`, `eta_it ~ StudentT(nu, 0, sigma_i)`, `phi_i = tanh(z_i)`
with a hierarchical prior on `z_i`. The first observation of every contiguous date
segment uses a prescribed wide Student-t initial scale
`sigma_init_i = sigma_i * sqrt(nu / ((nu - 2) * (1 - phi_i^2)))`, the variance-matched
scale implied by a Gaussian stationary law. This is a moment-matched initial scale, not
the exact stationary distribution of the Student-t innovation process. Segments are cut
at any calendar gap larger than one day.

`sigma_i` is the innovation scale, so it is not directly comparable to the residual
scale of models without the AR term.

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
