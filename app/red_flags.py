"""
red_flags.py
Domain-specific, rule-based red-flag detector.
Catches modern 2024-2026 cyber threat patterns:
- Upfront fees & wallet deposits
- Task-based review/video-liking schemes
- Cryptocurrency/USDT payments
- Brand impersonation & messaging-app recruitment
- Suspicious brevity & missing employer identity
"""

import re

RED_FLAG_RULES = [
    {
        "indicator": "Upfront payment / deposit request",
        "description": "Posting asks the applicant to pay money, purchase credentials, or make a wallet deposit before or during hiring.",
        "severity": "High",
        "pattern": r"(?:registration|application|training|security|processing|portal|account)\s+fee|pay\s+(?:₹|rs\.?|inr|\$)?\s?\d+|(?:wallet\s+)?deposit\s+(?:of|required|to)|deposit\s+(?:₹|rs\.?|inr|\$)?\s?\d+|prepaid\s+tasks?",
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
        "indicator": "Messaging-app-only recruitment",
        "description": "Recruitment/contact happens only via WhatsApp/Telegram rather than a formal company channel.",
        "severity": "Medium",
        "pattern": r"whatsapp|telegram",
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
        "indicator": "Excessive urgency",
        "description": "Posting pressures the applicant to act immediately.",
        "severity": "Low",
        "pattern": r"hiring\s+urgently|limited\s+seats|apply\s+immediately|only\s+today|hurry",
    },
    {
        "indicator": "Suspicious 'special access' claims",
        "description": "Claims of special/insider recruiter access or shortcut selection processes.",
        "severity": "Medium",
        "pattern": r"special\s+access|prescreen(ed|ing)|direct\s+selection|no\s+interview\s+needed|direct\s+joining",
    },
]


def detect_red_flags(raw_text: str, context: dict = None):
    """
    raw_text: the full, human-readable posting text (title + description +
    requirements + benefits + contact info etc, concatenated).
    context: optional dictionary with specific form fields (e.g. company, title, description).

    Returns a list of dicts, one per MATCHED rule only:
        {indicator, description, severity, evidence}
    """
    text = raw_text or ""
    text_lower = text.lower()
    findings = []
    context = context or {}

    for rule in RED_FLAG_RULES:
        match = re.search(rule["pattern"], text_lower, flags=re.IGNORECASE)
        if match:
            start = max(match.start() - 20, 0)
            end = min(match.end() + 20, len(text))
            evidence = text[start:end].strip()
            findings.append({
                "indicator": rule["indicator"],
                "description": rule["description"],
                "severity": rule["severity"],
                "evidence": f"...{evidence}..." if evidence else match.group(0),
            })

    # Compound rule: "no experience required" AND a high-pay mention together
    no_exp = re.search(r"no\s+experience\s+(required|needed)", text_lower)
    high_pay = re.search(r"₹\s?[5-9]\d,000|earn\s+up\s+to|lakh|rs\.?\s?[2-9]\d{3}\s*(?:\/|\s*per\s*)day", text_lower)
    if no_exp and high_pay:
        findings.append({
            "indicator": "No experience required + unusually high pay",
            "description": "Combination often seen in scam postings targeting inexperienced applicants.",
            "severity": "High",
            "evidence": f'"{no_exp.group(0)}" + "{high_pay.group(0)}"',
        })

    # Missing employer check
    comp_input = context.get("company", "").strip().lower()
    is_comp_missing = comp_input in ("", "select", "unknown", "none", "n/a")
    if is_comp_missing:
        findings.append({
            "indicator": "Undisclosed / missing employer identity",
            "description": "No company name is provided, making official employer verification impossible.",
            "severity": "Medium",
            "evidence": "Company field is empty or unspecified",
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
        })

    return findings
