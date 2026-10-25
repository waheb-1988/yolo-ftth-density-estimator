"""Nationality segment -> message languages, and the messenger's handling of
the native-language SMS (agents/agentic/languages.py, council.run_messenger)."""
from __future__ import annotations

import json

import pytest

from nbx.agents.agentic import council, languages
from nbx.agents.agentic.tools import RunContext
from nbx.core import config, insights
from nbx.llms import llm_client


@pytest.mark.parametrize("nat,segment,langs,lead", [
    ("OMN", "Omani", ["Arabic", "English"], "Arabic"),
    ("EGY", "Egyptian", ["Arabic", "English"], "Arabic"),
    ("IND", "Indian", ["Arabic", "English", "Hindi"], "Hindi"),
    ("PAK", "Pakistani", ["Arabic", "English", "Urdu"], "Urdu"),
    ("BGD", "Bangladeshi", ["Arabic", "English", "Bengali"], "Bengali"),
    ("PHL", "Other nationality", ["Arabic", "English"], "English"),
    (None, "Other nationality", ["Arabic", "English"], "English"),
])
def test_language_plan(nat, segment, langs, lead):
    p = languages.plan(nat)
    assert (p["segment"], p["languages"], p["lead_language"]) == (segment, langs, lead)


def test_script_check_rejects_wrong_script_and_arabic_as_urdu():
    assert languages.script_ok(languages.template("hi", None), "hi")
    assert languages.script_ok(languages.template("bn", None), "bn")
    assert languages.script_ok(languages.template("ur", None), "ur")
    assert not languages.script_ok("FSDR: thank you for staying with us", "hi")
    assert not languages.script_ok("شكراً لبقائك معنا في شبكتنا", "ur")  # Arabic, not Urdu


def test_native_currency_is_forced_to_omr():
    text, notes = languages.fix_currency("केवल 5 रुपये में 10GB")
    assert "OMR" in text and "रुपये" not in text and notes
    assert languages.fix_currency("صرف 5 روپے")[0] == "صرف 5 OMR"
    assert languages.fix_currency("Rs 5 only")[0] == "OMR 5 only"


def _customer(nat):
    return next(r["msisdn"] for r in insights.customers_list() if r["nationality"] == nat)


def test_template_message_adds_native_language_only_for_south_asians():
    for nat, expected in (("IND", "Hindi"), ("PAK", "Urdu"), ("BGD", "Bengali"), ("OMN", None), ("EGY", None)):
        ctx = RunContext.load(_customer(nat))
        msg = council.template_message(ctx, "RETAIN", [])
        assert msg["native_language"] == expected
        assert bool(msg["sms_native"]) == bool(expected)
        assert msg["languages"][:2] == ["Arabic", "English"]


@pytest.fixture
def llm_says(monkeypatch):
    def install(reply: dict):
        async def available():
            return True

        async def resolve(model):
            return "fake"

        async def generate(model, prompt, system=None, force_json=False, timeout=120.0):
            if "customer messages" in (system or ""):
                return {"content": json.dumps(reply, ensure_ascii=False), "error": None, "model": model, "duration_ms": 1}
            return {"content": "", "error": "unused", "model": model, "duration_ms": 1}  # other agents fall back

        # A fake *server* LLM: the offline backend would skip the LLM entirely.
        monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
        monkeypatch.setattr(llm_client, "is_available", available)
        monkeypatch.setattr(llm_client, "resolve_model", resolve)
        monkeypatch.setattr(llm_client, "generate", generate)

    return install


async def test_native_sms_comes_from_template_by_default(llm_says):
    llm_says({"sms_en": "Enjoy 10GB data for 5 OMR this month", "sms_ar": "استمتع بـ 10 جيجابايت مقابل 5 ر.ع هذا الشهر",
              "sms_native": "garbled model text", "agent_script": "Greet and present the pack."})
    ctx = RunContext.load(_customer("BGD"))
    msg = await council.run_messenger(ctx, "fake", {"action": "RETAIN", "channel": "SMS"}, [], lambda e: None)
    assert msg["sms_native"] == languages.template("bn", None)
    assert msg["sms_en"].startswith("Enjoy")  # English / Arabic still come from the model


def test_persian_is_not_accepted_as_urdu():
    assert not languages.script_ok("پیشنهاد ویژه با خصم پنج درصد برای شما است", "ur")


async def test_messenger_keeps_valid_native_sms_and_fixes_currency(llm_says, monkeypatch):
    from nbx.core import config

    monkeypatch.setattr(config, "MESSENGER_NATIVE_SOURCE", "llm")
    llm_says({"sms_en": "Enjoy 10GB data for 5 OMR this month", "sms_ar": "استمتع بـ 10 جيجابايت مقابل 5 ر.ع هذا الشهر",
              "sms_native": "इस महीने केवल 5 रुपये में 10GB डेटा का आनंद लें", "agent_script": "Greet and present the pack."})
    ctx = RunContext.load(_customer("IND"))
    events = []
    msg = await council.run_messenger(ctx, "fake", {"action": "UPSELL", "channel": "SMS"}, [], events.append)
    assert msg["native_language"] == "Hindi" and "OMR" in msg["sms_native"] and "रुपये" not in msg["sms_native"]
    assert any("sms_native" in e.get("text", "") for e in events if e["type"] == "auto_fix")


async def test_messenger_replaces_wrong_script_with_native_template(llm_says, monkeypatch):
    from nbx.core import config

    monkeypatch.setattr(config, "MESSENGER_NATIVE_SOURCE", "llm")
    llm_says({"sms_en": "Thank you for staying with FSDR, reply HELP", "sms_ar": "شكراً لبقائك معنا، أرسل مساعدة",
              "sms_native": "Thank you for staying with us (in English, not Urdu)", "agent_script": "Greet the customer."})
    ctx = RunContext.load(_customer("PAK"))
    events = []
    msg = await council.run_messenger(ctx, "fake", {"action": "RETAIN", "channel": "SMS"}, [], events.append)
    assert msg["sms_native"] == languages.template("ur", None) and msg["native_rtl"] is True
    assert any("Urdu SMS failed" in e["text"] for e in events if e["type"] == "policy")


def test_customer_safe_strips_hashtags_and_rejects_jargon_and_mixed_arabic():
    assert council.customer_safe("Great pack for 5 OMR #FSDR #Deal") == ("Great pack for 5 OMR", None)
    assert council.customer_safe("Upgrade now #NBO #MediumRisk")[1] is None  # hashtags alone are just stripped
    assert "internal term" in council.customer_safe("As a high churn risk customer, enjoy 5 OMR")[1]
    assert council.customer_safe("باقة مختارة لك بسعر 5 OMR من FSDR", arabic=True)[1] is None
    assert council.customer_safe("إلى updating 7 أيام للبيانات", arabic=True)[1]  # stray English word
    assert council.customer_safe("باقة ク مختارة لك", arabic=True)[1]  # stray other script


async def test_messenger_falls_back_per_field_on_unsafe_text(llm_says):
    llm_says({"sms_en": "Upgrade now, our NBO for medium churn risk users! #NBO", "sms_ar": "باقة مختارة لك بسعر 5 ر.ع من FSDR",
              "agent_script": "Greet the customer and present the pack."})
    ctx = RunContext.load(_customer("OMN"))
    events = []
    msg = await council.run_messenger(ctx, "fake", {"action": "RETAIN", "channel": "SMS"}, [], events.append)
    assert msg["sms_en"] == council.template_message(ctx, "RETAIN", [])["sms_en"]
    assert msg["sms_ar"].startswith("باقة")  # the clean Arabic SMS is kept
    assert any("sms_en rejected" in e["text"] for e in events if e["type"] == "policy")


def test_the_operator_name_comes_from_config(monkeypatch):
    """A8: no operator name is hard-coded in templates or prompts; OPERATOR_NAME drives them."""
    from nbx.core import config
    monkeypatch.setattr(config, "OPERATOR_NAME", "ACME")
    offer = {"data_gb": "5GB", "voice_mins": 100, "price_omr": 3.0, "validity_days": 30}
    for code in ("hi", "ur", "bn"):
        assert languages.template(code, None).startswith("ACME:") and languages.template(code, offer).startswith("ACME:")
