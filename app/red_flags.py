"""
red_flags.py
Domain-specific, rule-based red-flag detector.
Catches modern 2024-2026 cyber threat patterns:
- Upfront fees, verification charges, & refundable deposit promises
- Sensitive personal ID harvesting (Aadhaar, PAN, SSN) upfront
- Task-based review/video-liking schemes
- Cryptocurrency/USDT payments
- Messaging-app (WhatsApp/Telegram) recruitment & application routing
- Artificial urgency & deadline pressure tactics
- Bypass selection & no-interview claims
- Field-level attribution to pinpoint where red flags originate
"""

import re

RED_FLAG_RULES = [
    {
        "indicator": "Upfront payment / deposit request",
        "description": "Posting asks the applicant to pay money, purchase credentials, or complete a registration/verification fee before or during hiring.",
        "severity": "High",
        "pattern": (
            r"(?:refundable\s+(?:(?:₹|rs\.?|inr|\$)?\s?\d+[\w\s]{0,25})?(?:fee|deposit|charges?))|"
            r"(?:registration|application|training|security|processing|verification|portal|account|joining|onboarding|documentation)\s+(?:(?:and|&)\s+\w+\s+)?(?:fee|charges?|deposit)|"
            r"(?:pay|complete|transfer|deposit)\s+(?:a\s+)?(?:refundable\s+)?(?:(?:₹|rs\.?|inr|\$)?\s?\d+[\w\s]{0,25})?(?:fee|deposit|charges?)|"
            r"pay\s+(?:₹|rs\.?|inr|\$)\s?\d+|(?:wallet\s+)?deposit\s+(?:of|required|to)|deposit\s+(?:₹|rs\.?|inr|\$)\s?\d+|prepaid\s+tasks?"
        ),
    },
    {
        "indicator": "Fee refund / salary reimbursement promise",
        "description": "Claims that an upfront fee or deposit is 'refundable' or will be returned with the first month's salary. Legitimate corporate employers never demand upfront payments with reimbursement promises.",
        "severity": "High",
        "pattern": (
            r"(?:amount|fee|deposit|money)\s+will\s+be\s+(?:returned|refunded)|"
            r"(?:returned|refunded)\s+with\s+(?:the\s+)?(?:first|1st)\s+(?:month['’]?s?\s+)?salary|"
            r"refundable\s+(?:after|upon)\s+(?:joining|selection)"
        ),
    },
    {
        "indicator": "Sensitive government ID / personal document request upfront",
        "description": "Posting asks applicants for sensitive government IDs (Aadhaar, PAN card, SSN, Passport) during initial application or registration before official onboarding. Often used for identity theft and fraudulent SIM/loan accounts.",
        "severity": "High",
        "pattern": (
            r"\b(?:aadhaar|aadhar|pan\s+card|pan\s+details?|ssn|social\s+security(?:\s+number)?|passport\s+(?:copy|details?)|voter\s+id|driving\s+licen[sc]e)\b"
        ),
    },
    {
        "indicator": "Task-based rating / video-liking scam pattern",
        "description": "Recruitment involves micro-tasks (rating Google Maps/apps, liking YouTube videos, prepaid orders) promising instant daily returns. This is the primary template of modern cyber task fraud.",
        "severity": "High",
        "pattern": r"(?:rate|rating|review)\s+(?:apps?|hotels?|products?|google\s+maps)|like\s+(?:youtube\s+)?videos?|completing\s+(?:simple\s+)?tasks?\s+(?:on|online)|prepaid\s+tasks?|task\s+(?:bonus|commission|tier)",
    },
    {
        "indicator": "Cryptocurrency / USDT wallet or payment reference",
        "description": "Recruiter references cryptocurrency (USDT, Binance, Crypto wallet) for wages or task deposits. Legitimate corporate employers do not disburse payroll in cryptocurrency.",
        "severity": "High",
        "pattern": r"\b(?:usdt|crypto|cryptocurrency|binance|trust\s*wallet|telegram\s+wallet)\b",
    },
    {
        "indicator": "Bank/financial detail request",
        "description": "Posting asks for sensitive banking or financial information upfront.",
        "severity": "High",
        "pattern": r"bank\s+(account|details)|ifsc|debit\s+card|credit\s+card\s+details|upi\s+pin",
    },
    {
        "indicator": "Application / contact routed via messaging app",
        "description": "Recruitment instructions direct candidates to contact a coordinator or register via WhatsApp/Telegram rather than a formal company channel or verified portal.",
        "severity": "Medium",
        "pattern": (
            r"(?:contact|message|reach|send\s+resume|register|apply)\s+(?:the\s+)?(?:recruitment\s+coordinator|hr|team)?[\w\s]{0,25}(?:through|via|on)\s+(?:whatsapp|telegram)|"
            r"\b(?:whatsapp|telegram)\b"
        ),
    },
    {
        "indicator": "Unusually high salary claim",
        "description": "Salary figure looks unusually high for the stated role or effort.",
        "severity": "Medium",
        "pattern": r"₹\s?[5-9]\d,000|₹\s?\d{2,3},\d{3}\s*-\s*₹?\s?\d{2,3},\d{3}|earn\s+up\s+to|rs\.?\s?[2-9]\d{3}\s*(?:\/|\s*per\s*)day",
    },
    {
        "indicator": "Target-based / commission-only earning",
        "description": "Pay depends on targets/commission or merchant orders rather than a fixed salary.",
        "severity": "Medium",
        "pattern": r"target[- ]based|commission[- ]only|incentive[- ]based\s+earning|merchant\s+commission",
    },
    {
        "indicator": "MLM / network marketing language",
        "description": "Wording resembling multi-level-marketing or referral-based earning schemes.",
        "severity": "High",
        "pattern": r"network\s+marketing|mlm|refer\s+and\s+earn|build\s+your\s+team|downline",
    },
    {
        "indicator": "Guaranteed income / job claims",
        "description": "Posting guarantees income, selection, or interview outcome unrealistically.",
        "severity": "Medium",
        "pattern": r"guaranteed\s+(income|job|selection|interview|placement|daily\s+pay)",
    },
    {
        "indicator": "Brand/company impersonation cue",
        "description": "Posting invokes a well-known brand without a verifiable link to it.",
        "severity": "High",
        "pattern": r"\b(amazon|flipkart|meesho|jiomart|swiggy|zomato)\b",
    },
    {
        "indicator": "Vague company identity",
        "description": "Company name/identity is unclear, generic, or unverifiable.",
        "severity": "Medium",
        "pattern": r"a\s+leading\s+company|private\s+company|confidential\s+company|reputed\s+company|mnc\s+company",
    },
    {
        "indicator": "Excessive urgency & pressure tactics",
        "description": "Posting pressures the applicant to act or register immediately under artificial deadlines (e.g. within 24 hours, limited slots).",
        "severity": "Medium",
        "pattern": (
            r"hiring\s+urgently|urgently\s+hiring|limited\s+(?:seats|vacancies|slots)|"
            r"apply\s+immediately|start\s+immediately|join\s+immediately|only\s+today|hurry|"
            r"within\s+(?:12|24|48)\s*hours?|lose\s+(?:their|your)\s+opportunity"
        ),
    },
    {
        "indicator": "Suspicious 'special access' or no-interview claims",
        "description": "Claims of direct selection, bypassing technical interviews, or shortcut hiring processes.",
        "severity": "Medium",
        "pattern": (
            r"special\s+access|prescreen(?:ed|ing)|direct\s+selection|"
            r"no\s+(?:technical\s+)?interview\s*(?:needed|required|is\s+required)?|"
            r"without\s+(?:any\s+)?interview|direct\s+(?:joining|hiring|appointment)"
        ),
    },
]


def _attribute_source_fields(pattern: str, context: dict) -> list:
    """Identify which input field(s) contain the evidence matching a given pattern."""
    if not context:
        return []

    field_labels = [
        ("application_process", "Application Process"),
        ("description", "Job Description"),
        ("requirements", "Requirements"),
        ("benefits", "Benefits"),
        ("title", "Job Title"),
        ("company", "Company"),
        ("contact_details", "Contact Details"),
        ("salary", "Salary"),
    ]
    matched = []
    for key, label in field_labels:
        val = str(context.get(key, "") or "").strip()
        if val and re.search(pattern, val, flags=re.IGNORECASE):
            matched.append(label)
    return matched


def detect_red_flags(raw_text: str, context: dict = None):
    """
    raw_text: the full, human-readable posting text (title + description +
    application_process + requirements + benefits + contact info etc, concatenated).
    context: optional dictionary with specific form fields (e.g. application_process, company, title, description).

    Returns a list of dicts, one per MATCHED rule only:
        {indicator, description, severity, evidence, field}
    """
    text = raw_text or ""
    text_lower = text.lower()
    findings = []
    context = context or {}

    for rule in RED_FLAG_RULES:
        match = re.search(rule["pattern"], text_lower, flags=re.IGNORECASE)
        if match:
            start = max(match.start() - 25, 0)
            end = min(match.end() + 25, len(text))
            evidence = text[start:end].strip()

            src_fields = _attribute_source_fields(rule["pattern"], context)
            field_name = ", ".join(src_fields) if src_fields else "Posting Text"

            findings.append({
                "indicator": rule["indicator"],
                "description": rule["description"],
                "severity": rule["severity"],
                "evidence": f"...{evidence}..." if evidence else match.group(0),
                "field": field_name,
            })

    # Compound rule: "no experience required" AND a high-pay mention together
    no_exp = re.search(r"no\s+previous\s+experience\s+required|no\s+experience\s+(?:required|needed)", text_lower)
    high_pay = re.search(r"₹\s?[3-9]\d,000|₹\s?\d{2,3},\d{3}\s*-\s*₹?\s?\d{2,3},\d{3}|earn\s+up\s+to|lakh|rs\.?\s?[2-9]\d{3}\s*(?:\/|\s*per\s*)day", text_lower)
    if no_exp and high_pay:
        fields = []
        if context.get("requirements") and re.search(r"no\s+(?:previous\s+)?experience", str(context.get("requirements")), re.I):
            fields.append("Requirements")
        if context.get("benefits") and re.search(r"no\s+(?:previous\s+)?experience", str(context.get("benefits")), re.I):
            fields.append("Benefits")
        if context.get("description") and re.search(r"no\s+(?:previous\s+)?experience", str(context.get("description")), re.I):
            fields.append("Job Description")
        findings.append({
            "indicator": "No experience required + unusually high pay",
            "description": "Combination frequently seen in scam postings targeting inexperienced job seekers with unrealistic remuneration.",
            "severity": "High",
            "evidence": f'"{no_exp.group(0)}" + "{high_pay.group(0)}"',
            "field": ", ".join(fields) if fields else "Benefits / Salary",
        })

    # Missing employer check
    comp_input = str(context.get("company", "") or "").strip().lower()
    is_comp_missing = comp_input in ("", "select", "unknown", "none", "n/a")
    if is_comp_missing:
        findings.append({
            "indicator": "Undisclosed / missing employer identity",
            "description": "No company name is provided, making official employer verification impossible.",
            "severity": "Medium",
            "evidence": "Company field is empty or unspecified",
            "field": "Company",
        })

    # Abnormally brief / low-detail check
    words = [w for w in text.strip().split() if len(w) > 1]
    if 0 < len(words) < 25:
        # If posting has < 15 words and no company name, elevate to High severity
        sev = "High" if (len(words) < 15 and is_comp_missing) else "Medium"
        findings.append({
            "indicator": "Abnormally brief / low-detail posting",
            "description": f"The entire submission contains only {len(words)} words. Legitimate job advertisements provide comprehensive duties, prerequisites, and company context.",
            "severity": sev,
            "evidence": text.strip()[:80] + ("..." if len(text.strip()) > 80 else ""),
            "field": "Overall Posting",
        })

    return findings
