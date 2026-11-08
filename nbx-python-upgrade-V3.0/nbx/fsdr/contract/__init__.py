"""V2.0 input contract (plan §4, §13-§15).

    schemas.py (nbx/fsdr/schemas.py)   the 47 tables: fields, raw types, keys, input folder
    datasource.py                      reads a run folder: 5 input folders or flat, JSON / JSONL / CSV / Parquet
    validation.py                      per-table, cross-table, layout and customer_profile checks
    contract/completeness.py           which tables each pipeline needs this week (master prompt §17.9)
    contract/cache.py                  Parquet cache, hash-partitioned on the line id (plan §15)
    contract/data_agent.py             ① data agent: step T0_load = validate + completeness + cache, or block
"""
