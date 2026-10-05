"""
domain_check.py
Domain & email verification helper for job recruiter contacts.
Checks for consumer email domains, unreachable DNS records, fresh domain age,
and mismatch with advertised employer.
"""

import re
import socket
from datetime import datetime, timezone
import requests

FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me",
    "protonmail.com", "aol.com", "rediffmail.com", "icloud.com", "zoho.com",
    "mail.com", "yandex.com", "gmx.com", "live.com"
}

ENTERPRISE_BRANDS = [
    "amazon", "flipkart", "meesho", "tcs", "infosys", "wipro", "google",
    "microsoft", "apple", "meta", "reliance", "tata", "ibm", "accenture",
    "swiggy", "zomato", "deloitte", "stripe", "uber", "netflix", "adobe"
]


def extract_email(text: str) -> str:
    """Extract first email address found in a string."""
    if not text:
        return ""
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", str(text))
    return match.group(0).lower().strip() if match else ""


def check_domain(email_or_contact: str, company: str = ""):
    """
    Validates recruiter email/domain.
    Returns a list of dicts:
        {"indicator": ..., "description": ..., "severity": ..., "evidence": ...}
    """
    email = extract_email(email_or_contact)
    if not email or "@" not in email:
        return []

    domain = email.split("@")[-1].lower().strip()
    findings = []
    comp_lower = company.lower() if company else ""

    # 1. Free/consumer email check
    if domain in FREE_EMAIL_PROVIDERS:
        is_major_brand = any(brand in comp_lower for brand in ENTERPRISE_BRANDS)
        severity = "High" if is_major_brand else "Medium"
        if is_major_brand:
            desc = f"Recruiter claiming to represent major enterprise '{company}' is using a consumer email ({domain}). Large corporations never hire through public free email accounts."
            indicator = "Free email domain impersonating major enterprise"
        else:
            desc = f"Contact uses a public free email ({domain}) rather than an organizational domain. Common among early-stage startups and micro-agencies, but independent verification is advised."
            indicator = "Consumer / free email used for recruitment"

        findings.append({
            "indicator": indicator,
            "description": desc,
            "severity": severity,
            "evidence": email,
        })
        return findings

    # 2. DNS resolution check (native socket - zero external dependencies)
    try:
        socket.gethostbyname(domain)
    except Exception:
        findings.append({
            "indicator": "Unresolvable or non-existent email domain",
            "description": f"Domain '{domain}' failed DNS lookup. The domain does not have an active IP address or mail server.",
            "severity": "High",
            "evidence": domain,
        })
        return findings

    # 3. RDAP Domain registration age check
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=3)
        if r.status_code == 200:
            data = r.json()
            events = data.get("events", [])
            created = next((e["eventDate"] for e in events if e.get("eventAction") == "registration"), None)
            if created:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - created_dt).days
                if age_days < 180:
                    findings.append({
                        "indicator": f"Newly registered domain ({age_days} days old)",
                        "description": f"Domain '{domain}' was registered very recently ({age_days} days ago). Disposable domains are commonly used for fraudulent campaigns.",
                        "severity": "High",
                        "evidence": f"Created: {created[:10]} ({age_days} days old)",
                    })
    except Exception:
        pass  # Network/RDAP timeout should never block analysis

    # 4. Domain vs Company name consistency
    if company:
        clean_company = re.sub(r"\W", "", comp_lower)
        clean_domain = domain.split(".")[0].replace("-", "").lower()
        if len(clean_company) >= 4 and clean_company[:5] not in clean_domain and clean_domain not in clean_company:
            findings.append({
                "indicator": "Domain mismatch with company name",
                "description": f"Email domain '{domain}' does not appear related to advertised company '{company}'.",
                "severity": "Medium",
                "evidence": f"Company: '{company}' vs Email: '{email}'",
            })

    return findings