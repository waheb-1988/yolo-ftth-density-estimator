"""Nationality segment -> outreach languages.

Policy (business rule, not model output):
  - Arabic and English: always.
  - Plus the customer's own language for the three large South-Asian
    communities: Pakistani -> Urdu, Indian -> Hindi, Bangladeshi -> Bengali.
  - Everyone else: Arabic + English only.
Lead language (what goes first / is used for the call script greeting):
Arabic for Arab nationalities, the native language when there is one,
English otherwise.

The native-language SMS written by the model is checked in code: it must
be in the right script, and the price must stay in OMR (small models like
to "localize" the currency into rupees or taka).
"""
from __future__ import annotations

import re

SEGMENTS = {
    "OMN": "Omani",
    "EGY": "Egyptian",
    "IND": "Indian",
    "PAK": "Pakistani",
    "BGD": "Bangladeshi",
}
ARAB_NATIONALITIES = {"OMN", "EGY", "SAU", "ARE", "QAT", "KWT", "BHR", "YEM", "JOR", "SYR", "LBN", "SDN"}

NATIVE = {
    "IND": {"language": "Hindi", "code": "hi", "script": "Devanagari", "rtl": False},
    "PAK": {"language": "Urdu", "code": "ur", "script": "Perso-Arabic (Nastaliq)", "rtl": True},
    "BGD": {"language": "Bengali", "code": "bn", "script": "Bengali", "rtl": False},
}

# Script checks: the text must mostly use the expected Unicode block.
_SCRIPT_RE = {
    "hi": re.compile(r"[ऀ-ॿ]"),
    "bn": re.compile(r"[ঀ-৿]"),
    "ur": re.compile(r"[؀-ۿ]"),
}
# Letters Urdu has and both Arabic and Persian lack (ے ہ ں ٹ ڈ ڑ) -- so neither
# Arabic nor Persian text is accepted as Urdu.
_URDU_ONLY = re.compile("[ےہںٹڈڑ]")

# Currency words the model may substitute for OMR in these languages.
_WRONG_NATIVE_CURRENCY = ["रुपये", "रुपए", "रूपये", "रुपया", "₹", "روپے", "روپیہ", "روپیے", "টাকা", "৳", "Rs.", "Rs", "INR", "PKR", "BDT", "Tk"]

TEMPLATES = {
    "hi": {
        "offer": "{operator}: आपके लिए चुना गया पैक - {data} डेटा, {mins} मिनट, {price} OMR में ({days} दिन)। सक्रिय करने के लिए YES भेजें।",
        "care": "{operator}: हमारे साथ रहने के लिए धन्यवाद। अपने प्लान में मदद चाहिए? HELP भेजें, हम आपको कॉल करेंगे।",
    },
    "ur": {
        "offer": "{operator}: آپ کے لیے منتخب پیکج - {data} ڈیٹا، {mins} منٹ، {price} OMR میں ({days} دن)۔ فعال کرنے کے لیے YES بھیجیں۔",
        "care": "{operator}: ہمارے ساتھ رہنے کا شکریہ۔ اپنے پلان میں مدد چاہیے؟ HELP بھیجیں، ہم آپ کو کال کریں گے۔",
    },
    "bn": {
        "offer": "{operator}: আপনার জন্য বাছাই করা প্যাক - {data} ডেটা, {mins} মিনিট, {price} OMR-এ ({days} দিন)। চালু করতে YES পাঠান।",
        "care": "{operator}: আমাদের সাথে থাকার জন্য ধন্যবাদ। আপনার প্ল্যানে সাহায্য দরকার? HELP পাঠান, আমরা আপনাকে কল করব।",
    },
}


def plan(nationality: str | None) -> dict:
    code = (nationality or "").upper()
    native = NATIVE.get(code)
    languages = ["Arabic", "English"] + ([native["language"]] if native else [])
    if code in ARAB_NATIONALITIES:
        lead = "Arabic"
    elif native:
        lead = native["language"]
    else:
        lead = "English"
    segment = SEGMENTS.get(code) or ("Arab expat" if code in ARAB_NATIONALITIES else "Other nationality")
    return {
        "nationality": code or "unknown",
        "segment": segment,
        "languages": languages,
        "lead_language": lead,
        "native": native,
        "reason": (f"{segment} customer: Arabic and English always"
                   + (f", plus {native['language']} ({native['script']} script)" if native else ", no third language")
                   + f"; lead with {lead}."),
    }


def script_ok(text: str, code: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    hits = sum(1 for ch in letters if _SCRIPT_RE[code].match(ch))
    if hits / len(letters) < 0.5:  # brand names / OMR are Latin, the rest must be native script
        return False
    return code != "ur" or bool(_URDU_ONLY.search(text))


def fix_currency(text: str) -> tuple[str, list[str]]:
    notes = []
    for bad in _WRONG_NATIVE_CURRENCY:
        pattern = rf"(?<![A-Za-z]){re.escape(bad)}(?![A-Za-z])" if bad.isascii() else re.escape(bad)
        if re.search(pattern, text):
            text = re.sub(pattern, "OMR", text)
            notes.append(f"replaced wrong currency '{bad}' with 'OMR'")
    return text, notes


def template(code: str, offer: dict | None) -> str:
    from nbx.core import config

    t = TEMPLATES[code]
    if offer is None:
        return t["care"].format(operator=config.OPERATOR_NAME)
    return t["offer"].format(operator=config.OPERATOR_NAME, data=offer["data_gb"], mins=offer["voice_mins"],
                             price=offer["price_omr"], days=offer["validity_days"])
