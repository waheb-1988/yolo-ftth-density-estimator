# Known differences vs the original codes 1–5

Every place where FSDR NBX V3.0 does not do exactly what the original Dataiku codes did, and why. The flag `reproduce_original_bugs` exists only for the code-5 parity test (Phase 7); everything below is fixed by default.

## Code 1 (NBA training and scoring)

| # | Original | V3.0 | Rule |
|---|---|---|---|
| 1 | Contextual bandit (VW `cb_explore_adf`) decides the action | The LLM NBA agent scores the allowed actions; the ⑩ bandit is evidence only | master §9, §10.2 |
| 2 | Missing model score → one `uniform(0, 0.02)` draw for the whole column | Flag `*_MISSING` + median of the nationality group; a missing score never triggers "at risk" | B-05 |
| 3 | Churn = max of the 3 stacked tables (V1.0) | First row of the stack (global, 0-30, 30-60), as the spec says | B-04 |
| 4 | `calcul_evol` called with current / previous inverted | Trends go from the older period to the recent one | B-07 |
| 5 | `dict_segm_actions` literal with repeated keys | Kept literally; last value wins, case-sensitive (RGS_ vs Rgs_) | B-03 |
| 6 | MSISDN merged as float (`astype(float)`) | One canonical text key everywhere; IDs always text (`.csv.gz` contract) | master §2 |
| 7 | Propensity = prior + `uniform(0, 0.02)` | Logged propensity of the sampled action (ε = 0.1), floor 0.01 | §9, open item 12 |

## Codes 3 and 4 (offers)

| # | Original | V3.0 | Rule |
|---|---|---|---|
| 8 | LightFM score in the sort | LLM NBO score (or its evidence fallback) in the same `score` column | B-24, update §4.1 / §4.2 |
| 9 | CROSS SELL VOICE branch repeats `CROSS SELL DATA`: no family filter for CROSS SELL VOICE | Its own filter: DATA + COMBO, L2 | B-22, update §4.1 |
| 10 | LightFM item / user ids from the loop index (the cold-start loop scores user index 0, 1, … of the takers' embeddings) | Real customer ids; ⑪ helper only | §10.2 |
| 11 | `sort_values` with the default quicksort: equal score and price → order not defined | Stable sort (mergesort): a tie keeps the candidate order. The parity test uses continuous scores, where both agree | update test 8 |
| 12 | Code 3 and code 4 run per action / per validity, filtering the whole table per customer | Vectorised over all customers at once (same filters, same passes) | §19.5 |
| 13 | Target 1 = 10k constraint (code 3) + 10k free (code 4); "free" was an arm | V3.0: four groups of 25k: T1C (code 3), T1F (code 4), T2C (CC-v1), T2F (CF-v1) + 20k control | update §2 |

## Code 5 (`input_flytxt`): fixed by default, reproduced with the bug flag for parity (Phase 7)

| # | Original | V3.0 |
|---|---|---|
| a | A slot cleared by F-11 / F-12 keeps its script and opportunity name | Cleared too |
| b | Empty slot → `nbo_LFM_nan` | Empty field |
| c | Favourites can repeat an offer and ignore the purchase count | One row per offer, by price then count |
| d | Two anchors (B-26 vs F-05) | Both kept, never merged |
| e | 2 recommended offers per validity (slots 1–4), favourites / forced in 5–6 | V4.0: 3 per validity (`FSDR_RECO_OFFERS=6`, rule EXT-6): slot 5 = 3rd monthly, slot 6 = 3rd weekly; favourites / forced only fill an empty 5 / 6. `FSDR_RECO_OFFERS=4` = the original layout |

## Data files

| # | Original | V3.0 |
|---|---|---|
| 14 | Dataiku datasets read in memory | One `.csv.gz` per table (FSDR CSV standard), converted once to a partitioned Parquet cache |
| 15 | `float` ids could print as `9.68e+07` | IDs written and read as text; no scientific notation for amounts |
