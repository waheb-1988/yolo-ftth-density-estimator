# FSDR verification audit and business-rule fixes: done

Status on 2026-10-24.

- **Audit:** `docs/fsdr_verification_report.md` holds the independent audit (102 checks: 1 BLOCKER, 21 MAJOR) and
  its re-check after the fixes, appended at the end.
- **Fixes:** all on `feature/agentic-ooredoo`, one commit per item, with a test each. The list is in
  `docs/fsdr_plan.md`, section "Verification audit and fixes (2026-10-24)".
- **Still open (MINOR, not fixed on purpose):** see the re-check section of the report.
  - E5: no warning for unmapped segments (they are listed in the report).
  - O3: tab grouping and names.
  - A5: upper bounds on jsonschema / pyarrow.
  - A9: four settings missing from `.env.example`.
  - B3: `bigint` raw types and extra sample fields.
  - C4: stratification on `Final`.
  - D3: some literals in the logic.
  - F7: float32 in the reference.
  - I1: the training steps are not separate tools.
  - I4: the IPS is rounded to 4 dp.
  - I5: forced promotion is possible.
  - L8: the cache key uses the score features only.
  - P1: no test for BR §3.
  - P2: no example records in the data dictionary.
  - Q11: generated `nb_sub` is always 1.
- **Unverifiable:** C1, because the reference generator `docs/fsdr_reference/fsdr_generate_dummy.py` is not on this
  machine.
- **Open business decisions (unchanged):** Retention_High_Probability_To_Reconnect → RETAIN, and
  RGS_Not_At_Risk_Data_Only_Voice_Appetency → UPSELL VOICE.
- **After this update:** models trained before it are refused by the compatibility check (new last-campaign
  features). Retrain once: `python run.py fsdr-train`, then `fsdr-score`.
