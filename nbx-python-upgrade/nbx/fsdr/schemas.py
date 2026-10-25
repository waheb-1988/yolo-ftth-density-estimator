"""Data contract: the exact raw schema of every input JSON file.

One file per table, file name = table name + ".json" (array) or ".jsonl".
Field names, case and production quirks are kept on purpose
(`UBSCRIPTION_ID`, `activities_auto_m2`, `Access_Method_id`, "Price data",
numbers stored as strings in `btl_catalog` and in the test sample), so the
same code runs on the dummy data and on real exports.

Raw types follow the production tables: string, int, bigint, tinyint,
double, date ("YYYY-MM-DD") and timestamp (ISO 8601). In JSON, every value
may also be null unless the field is a key.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.calendar import MONTHS, WEEKS

INT_TYPES = {"int", "bigint", "tinyint"}
RAW_TYPES = INT_TYPES | {"string", "double", "date", "timestamp"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Field:
    name: str
    raw_type: str
    key: bool = False           # join key: must be present and non-null


@dataclass
class TableSchema:
    name: str
    fields: list[Field]
    description: str
    group: str
    anchor: str | None = None               # "T", "D" or None
    primary_key: tuple[str, ...] = ()       # duplicates on it are reported
    duplicates_blocking: bool = True        # False: duplicates are expected / resolved downstream
    used_by_model: bool = True              # BR §12: some tables are loaded but not used
    family: str | None = None               # e.g. "model_score", "activities"

    def __post_init__(self):
        for f in self.fields:
            if f.raw_type not in RAW_TYPES:
                raise ValueError(f"{self.name}.{f.name}: unknown raw type {f.raw_type}")

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def key_fields(self) -> list[str]:
        return [f.name for f in self.fields if f.key]

    def types(self) -> dict[str, str]:
        return {f.name: f.raw_type for f in self.fields}


def _f(name: str, raw_type: str, key: bool = False) -> Field:
    return Field(name, raw_type, key)


# ---------------------------------------------------------------------------
# Field builders for the repetitive table families
# ---------------------------------------------------------------------------
ACTIVITY_METRICS = [  # (name stem, raw type) -- SQL order of activities_*
    ("Call_Count", "int"), ("Data_Session_Count", "int"), ("Voice_Traffic_min", "double"),
    ("Data_Volume_MB", "double"), ("PG_Total_Amount", "double"), ("PG_Voice_Amount", "double"),
    ("PG_Data_Amount", "double"), ("Onnet_Call_Count", "int"), ("Onnet_Voice_min", "double"),
    ("Offnet_Call_Count", "int"), ("Offnet_Voice_min", "double"), ("Local_Call_Count", "int"),
    ("Local_Voice_min", "double"), ("Active_Days_Count", "int"), ("Active_Days_Voice_Count", "int"),
    ("Active_Days_Data_Count", "int"),
]


def activity_table_name(period: str, anchor: str) -> str:
    """activities_w1_auto ... activities_m3_auto, except M2 auto = activities_auto_m2 (production name)."""
    p = period.lower()
    if anchor == "T":
        return "activities_auto_m2" if p == "m2" else f"activities_{p}_auto"
    return f"activities_{p}_score"


def _activity_fields(period: str) -> list[Field]:
    return [_f("ACCESS_METHOD_VAL", "string", True), _f("ACCESS_METHOD_ID", "bigint", True),
            _f("ACCOUNT_ID", "int", True)] + [_f(f"{stem}_{period}", t) for stem, t in ACTIVITY_METRICS]


def _subs_amt_fields() -> list[Field]:
    return ([_f("ACCESS_METHOD_VAL", "string", True), _f("ACCESS_METHOD_ID", "int", True),
             _f("ACCOUNT_ID", "double", True)]
            + [_f(f"Sub_Amt_{p.lower()}", "double") for p in WEEKS + MONTHS])


def _subscriptions_tg_fields() -> list[Field]:
    out = [_f("ACCESS_METHOD_VAL", "string", True), _f("ACCESS_METHOD_ID", "int", True), _f("ACCOUNT_ID", "double", True)]
    for w in WEEKS:
        out += [_f(f"{k}_Sub_Amt_{w.lower()}", "double") for k in ("Total", "Data", "Voice")]
    for m in MONTHS:
        out += [_f(f"Sub_Count_{m}", "int")] + [_f(f"{k}_Sub_Amt_{m}", "double") for k in ("Total", "Data", "Voice")]
    return out


def _payg_fields(prefix_cols: list[Field]) -> list[Field]:
    return prefix_cols + [_f(f"PG_Total_Amount_{p.lower()}", "double") for p in MONTHS + WEEKS]


def _targeting_auto_fields() -> list[Field]:
    out = [_f("DATE1", "date"), _f("START_DATE", "date"), _f("END_DATE", "date"), _f("MSISDN", "string", True)]
    first, last = range(1, 5), range(5, 7)
    out += [_f(f"CHANNEL_{i}", "string") for i in first]
    out += [_f(f"OPPORTUNITY_NAME_{i}", "string") for i in first]
    out += [_f(f"DESTINATION_OFFER_NAME_{i}", "string") for i in first]
    out += [_f("FLAG_SPLIO", "string")]
    out += [_f(f"OFFER_SCRIPT_EN_{i}", "string") for i in first]
    out += [_f(f"OFFER_SCRIPT_AR_{i}", "string") for i in first]
    out += [_f("SEGMENT", "string"), _f("FILENAME", "string"), _f("ETL_CREATE_DT", "date"), _f("JOB_ID", "int")]
    for i in last:
        out += [_f(f"CHANNEL_{i}", "string"), _f(f"OPPORTUNITY_NAME_{i}", "string"),
                _f(f"DESTINATION_OFFER_NAME_{i}", "string"), _f(f"OFFER_SCRIPT_EN_{i}", "string"),
                _f(f"OFFER_SCRIPT_AR_{i}", "string")]
    return out + [_f("SEGMENT_BTL_TAKER", "string"), _f("SEGMENT_BTL_TAKER_UPSELL", "string")]


_CONFLICT_SIDE = ["Offer", "Gen_Level", "Price", "Data", "Min", "Validity", "Offer_Type"]

# The 11 model-score files share ONE identical structure.
MODEL_SCORE_FIELDS = [
    _f("subscription_id", "string"), _f("access_method_val", "bigint", True), _f("access_method_id", "bigint", True),
    _f("tenure", "bigint"), _f("life_in_nw_range", "string"), _f("proba_0", "double"), _f("proba_1", "double"),
    _f("prediction", "string"),
]

SAMPLE_FIELDS = [  # every column is a string in production
    "access_method_val", "access_method_id", "account_id", "tenure", "activation_date", "plan_name", "payg_revenue",
    "subscription_revenue", "total_revenue", "total_revenue_3m", "voice_traffic", "data_volume", "nationality",
    "region", "ARPU", "ARPU_3m", "voice_trf_group", "data_trf_group", "Final", "channel", "group",
    "strata", "counts",
]
_SAMPLE_KEYS = {"access_method_val", "access_method_id", "account_id", "group"}


def model_score_schema(table: str) -> TableSchema:
    return TableSchema(table, list(MODEL_SCORE_FIELDS), "Model score (one shared structure; proba_1 is the NBA score).",
                       "model_scores", primary_key=("access_method_val", "access_method_id"),
                       duplicates_blocking=False, family="model_score")


def _build_tables() -> dict[str, TableSchema]:
    t: list[TableSchema] = [
        # --- Base & reference --------------------------------------------------
        TableSchema("base_prepaid", [
            _f("UBSCRIPTION_ID", "string"), _f("ACCESS_METHOD_VAL", "string", True), _f("ACCESS_METHOD_ID", "int", True),
            _f("ACCOUNT_ID", "int", True), _f("PARTY_ID", "int", True), _f("access_method_subs_start_dt", "string"),
            _f("CURRENT_PLAN", "string")],
            "Active prepaid consumer lines, snapshot D-2 (BR-01..BR-10).", "base", primary_key=("ACCESS_METHOD_VAL",)),
        TableSchema("consumer_aact_account", [_f("ACCOUNT_ID", "bigint", True), _f("PARTY_ID", "bigint", True)],
                    "Active consumer account -> party (BR-11).", "base", primary_key=("ACCOUNT_ID",)),
        TableSchema("access_method_id_auto", [
            _f("ACCESS_METHOD_ID", "int", True), _f("MSISDN", "string", True), _f("ACTIVATION_DATE", "string")],
            "Line id -> MSISDN with 3-character country prefix (BR-12).", "base",
            primary_key=("ACCESS_METHOD_ID", "MSISDN")),
        *[TableSchema(name, [_f("party_id", "int", True), _f("AGE", "int"), _f("GENDER", "string"),
                             _f("NATIONALITY", "string"), _f("nationality_identifier", "string")],
                      "Customer demographics (BR-15).", "base", primary_key=("party_id",))
          for name in ("party_auto", "party_scoring")],
        TableSchema("location_auto", [
            _f("Access_Method_id", "int", True), _f("Calling_Month", "int"), _f("Calling_Year", "int"),
            _f("REGION", "string"), _f("WILAYAT", "string"), _f("AREA", "string")],
            "Home location, previous calendar month of D-2 (BR-13/14). Not used by the model.", "base",
            primary_key=("Access_Method_id", "REGION", "WILAYAT", "AREA"), used_by_model=False),
        TableSchema("btl_catalog", [_f("btl_name", "string", True)] + [_f(n, "string") for n in (
            "type", "generosity", "gb", "min", "price", "Price data", "Price voice", "service", "validity")],
            "BTL offer catalogue; every column is a string (BR-63..65).", "base", primary_key=("btl_name",)),
        TableSchema("BTL_script_english_arabic_ok", [
            _f("Sl No.", "string"), _f("destination_offer_name", "string", True), _f("English Script", "string"),
            _f("Arabic Script", "string")],
            "English / Arabic script per offer.", "base", primary_key=("destination_offer_name",), used_by_model=False),
        TableSchema("conflicting_offers_h",
                    [_f(f"{c}_A", "string", c == "Offer") for c in _CONFLICT_SIDE]
                    + [_f(f"{c}_B", "string", c == "Offer") for c in _CONFLICT_SIDE]
                    + [_f(n, "string") for n in ("Price_Difference", "Data_Difference", "SAME_VALIDITY", "SAME_OFFER_TYPE")],
                    "Pairs of offers that must never be sent together.", "base", primary_key=("Offer_A", "Offer_B")),
        TableSchema("DEVICEBEHAVIORSEGMENTATION.dv_segment", [_f("ACCESS_METHOD_ID", "int", True), _f("DV_SEGMENT", "string")],
                    "Device behaviour segment. Not used by the model.", "base", primary_key=("ACCESS_METHOD_ID",),
                    used_by_model=False),
        # --- Revenue & usage ---------------------------------------------------
        TableSchema("subs_amt_auto", _subs_amt_fields(), "Subscription amounts, windows on T (BR 4.1).", "usage", "T",
                    ("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID"), family="subs_amt"),
        TableSchema("subs_amt_score", _subs_amt_fields(), "Subscription amounts, windows on D (BR 4.1).", "usage", "D",
                    ("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID"), family="subs_amt"),
        *[TableSchema(activity_table_name(p, a), _activity_fields(p),
                      f"Usage activity {p}, windows on {a} (BR 4.2).", "usage", a,
                      ("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID"), family="activities")
          for a in ("T", "D") for p in WEEKS + MONTHS],
        TableSchema("paygo_activity_auto", _payg_fields([
            _f("ACCESS_METHOD_ID", "bigint", True), _f("ACCOUNT_ID", "int", True), _f("ACCESS_METHOD_VAL", "string", True)]),
            "PAYG amounts on T (BR 4.3). Loaded but not used.", "usage", "T",
            ("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID"), used_by_model=False),
        TableSchema("subscriptions_tg", _subscriptions_tg_fields(),
                    "Offer-pricing subscriptions on D, 2-day lag (BR 4.3, BR-19 SUBSCRIPTION_AMT).", "pricing", "D",
                    ("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID")),
        TableSchema("usage_data_voice_payg_tg", _payg_fields([
            _f("ACCESS_METHOD_ID", "bigint", True), _f("ACCOUNT_ID", "int", True), _f("ACCESS_METHOD_VAL", "string", True)]
            + [_f(f"local_Voice_min_p{i}", "double") for i in (1, 2, 3)]
            + [_f(f"data_volume_mb_p{i}", "double") for i in (1, 2, 3)]),
            "Offer-pricing usage + positive PAYG on D (BR 4.3, BR-33/34).", "pricing", "D",
            ("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID")),
        # --- Campaign ----------------------------------------------------------
        TableSchema("targeting_dataiku_auto", [
            _f("date1", "string"), _f("access_method_id", "int", True), _f("account_id", "int"), _f("segment", "string")]
            + [_f(f"destination_offer_name_{i}", "string") for i in range(1, 7)],
            "Last campaign targeting, date1 in D-8..D-2 (BR-21, BR-36..39).", "campaign", "D",
            ("access_method_id", "date1"), duplicates_blocking=False),
        TableSchema("targeting_auto", _targeting_auto_fields(),
                    "Arbitration targeting, start_date in D-7..D (BR-22). Loaded but not used.", "campaign", "D",
                    duplicates_blocking=False, used_by_model=False),
        TableSchema("subs_btl_auto", [
            _f("access_method_val", "string", True), _f("PRODUCTNAME", "string", True),
            _f("FULFILMENTSUCCESSDATE_TIME", "timestamp"), _f("sub_flag", "tinyint"), _f("nb_sub", "int")],
            "Fulfilments D-7..D-1, MSISDN without prefix (BR-23/25).", "campaign", "D",
            ("access_method_val", "PRODUCTNAME", "FULFILMENTSUCCESSDATE_TIME")),
        TableSchema("sub_btl_3months", [
            _f("access_method_val", "string", True), _f("PRODUCTNAME", "string", True), _f("sub_flag", "tinyint"),
            _f("nb_sub", "int")],
            "Fulfilments D-270..D-1 despite the name (BR-24).", "campaign", "D", ("access_method_val", "PRODUCTNAME")),
        # --- Test sample -------------------------------------------------------
        TableSchema(config.TARGET_SAMPLE_TABLE, [_f(n, "string", n in _SAMPLE_KEYS) for n in SAMPLE_FIELDS],
                    "Campaign test sample: control cg + arms free / cstr / llm, stratified (BR-59..62). All strings.",
                    "sample", "D", ("access_method_val",)),
    ]
    tables = {s.name: s for s in t}
    for m in config.MODEL_SCORE_REGISTRY:
        tables.setdefault(m.table, model_score_schema(m.table))
    return tables


TABLES: dict[str, TableSchema] = _build_tables()

# Accepted alternative file names (production typos) -> canonical table.
TABLE_ALIASES = {"activities_w1_score_hfds": "activities_w1_score"}


def refresh() -> None:
    """Rebuilds TABLES after config.MODEL_SCORE_REGISTRY was changed (tests, plug-in models)."""
    global TABLES
    TABLES = _build_tables()


def model_score_tables() -> list[str]:
    return [m.table for m in config.MODEL_SCORE_REGISTRY]


# ---------------------------------------------------------------------------
# Cross-table key relations (validated: child keys must exist in the parent)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Relation:
    child: str
    child_col: str
    parent: str
    parent_col: str
    blocking: bool = True
    note: str = ""


def relations() -> list[Relation]:
    rels = [
        Relation("base_prepaid", "ACCESS_METHOD_ID", "access_method_id_auto", "ACCESS_METHOD_ID"),
        Relation("base_prepaid", "PARTY_ID", "party_auto", "party_id"),
        Relation("base_prepaid", "ACCOUNT_ID", "consumer_aact_account", "ACCOUNT_ID", blocking=False),
        Relation("targeting_dataiku_auto", "access_method_id", "access_method_id_auto", "ACCESS_METHOD_ID"),
        Relation(config.TARGET_SAMPLE_TABLE, "access_method_id", "base_prepaid", "ACCESS_METHOD_ID"),
        Relation(config.TARGET_SAMPLE_TABLE, "account_id", "consumer_aact_account", "ACCOUNT_ID"),
        Relation("subs_amt_auto", "ACCESS_METHOD_VAL", "base_prepaid", "ACCESS_METHOD_VAL", blocking=False),
        Relation("activities_w1_auto", "ACCESS_METHOD_VAL", "access_method_id_auto", "MSISDN", blocking=False),
        # B6: every fulfilment line must belong to a known line (reported, not blocking: churned lines buy too)
        Relation("subs_btl_auto", "access_method_val", "base_prepaid", "ACCESS_METHOD_VAL", blocking=False),
        Relation("sub_btl_3months", "access_method_val", "base_prepaid", "ACCESS_METHOD_VAL", blocking=False),
        *[Relation("targeting_dataiku_auto", f"destination_offer_name_{i}", "btl_catalog", "btl_name", blocking=False)
          for i in range(1, 7)],
        Relation("conflicting_offers_h", "Offer_A", "btl_catalog", "btl_name", blocking=False),
        Relation("conflicting_offers_h", "Offer_B", "btl_catalog", "btl_name", blocking=False),
    ]
    rels += [Relation(t, "access_method_val", "base_prepaid", "ACCESS_METHOD_VAL", blocking=False,
                      note="model score for a line outside the base") for t in model_score_tables()]
    return rels


# ---------------------------------------------------------------------------
# Raw-type checks and coercion
# ---------------------------------------------------------------------------
def value_matches(raw_type: str, v) -> bool:
    """Does one non-null JSON value match the raw type?"""
    if raw_type == "string":
        return isinstance(v, str)
    if raw_type in INT_TYPES:
        return isinstance(v, (int, np.integer)) and not isinstance(v, bool)
    if raw_type == "double":
        return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    if raw_type == "date":
        return isinstance(v, str) and bool(_DATE_RE.match(v))
    if raw_type == "timestamp":
        if not isinstance(v, str):
            return False
        try:
            datetime.fromisoformat(v)
            return True
        except ValueError:
            return False
    return False


def type_errors(series: pd.Series, raw_type: str, sample: int = 3) -> tuple[int, list]:
    """(number of non-null values with the wrong raw type, a few examples)."""
    s = series[series.notna()]
    if s.empty:
        return 0, []
    ok = s.map(lambda v: value_matches(raw_type, v))
    bad = s[~ok]
    return int(len(bad)), [repr(v) for v in bad.head(sample).tolist()]


def coerce(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Raw (object) frame -> typed frame for the pipelines: ints -> Int64,
    doubles -> float64, strings -> pandas string, dates / timestamps kept as
    strings. Values that cannot be converted become missing (the validation
    report already lists them)."""
    schema = TABLES[table]
    out = df.copy()
    for f in schema.fields:
        if f.name not in out.columns:
            continue
        col = out[f.name]
        if f.raw_type in INT_TYPES:
            out[f.name] = pd.to_numeric(col, errors="coerce").round().astype("Int64")
        elif f.raw_type == "double":
            out[f.name] = pd.to_numeric(col, errors="coerce").astype("float64")
        else:
            out[f.name] = col.astype("string")
    return out


# ---------------------------------------------------------------------------
# JSON Schema export (docs/fsdr_json_schemas/<table>.json)
# ---------------------------------------------------------------------------
_JSON_TYPE = {"string": "string", "int": "integer", "bigint": "integer", "tinyint": "integer",
              "double": "number", "date": "string", "timestamp": "string"}


def json_schema(table: str) -> dict:
    s = TABLES[table]
    props = {}
    for f in s.fields:
        p: dict = {"type": _JSON_TYPE[f.raw_type] if f.key else [_JSON_TYPE[f.raw_type], "null"],
                   "x-raw-type": f.raw_type}
        if f.raw_type == "date":
            p["pattern"] = _DATE_RE.pattern
        if f.raw_type == "timestamp":
            p["format"] = "date-time"
        props[f.name] = p
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"fsdr/{table}.json",
        "title": table,
        "description": s.description,
        "type": "array",
        "items": {"type": "object", "properties": props, "required": s.field_names, "additionalProperties": True},
        "x-schema-version": config.SCHEMA_VERSION,
        "x-group": s.group,
        "x-anchor": s.anchor,
        "x-primary-key": list(s.primary_key),
        "x-used-by-model": s.used_by_model,
    }


def export_json_schemas(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or config.JSON_SCHEMA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in TABLES:
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(json_schema(name), indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(p)
    index = {"schema_version": config.SCHEMA_VERSION, "generated": date.today().isoformat(),
             "tables": {n: {"group": s.group, "anchor": s.anchor, "fields": len(s.fields), "used_by_model": s.used_by_model}
                        for n, s in TABLES.items()}}
    (out_dir / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return paths
