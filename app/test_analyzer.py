"""
test_analyzer.py
Comprehensive test suite for the Explainable Job Scam Risk Analyzer.
Runs test cases through Dual-Engine Evaluation and Composite Risk Fusion.

Run with: python test_analyzer.py
"""

import sys
from predictor import predict_posting, load_artifacts
from explainer import explain_posting
from red_flags import detect_red_flags
from domain_check import check_domain

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TEST_CASES = {
    "case_1_legitimate": {
        "title": "Senior Backend Software Engineer (Python/Distributed Systems)",
        "company": "Stripe Technologies",
        "company_profile": "Stripe is building economic infrastructure for the internet.",
        "description": "Design, build, and maintain high-volume API services. Architect distributed systems, collaborate with product managers, write unit tests, and maintain platform availability.",
        "requirements": "5+ years experience with Python, Go, or Java. Solid understanding of PostgreSQL, Redis, RESTful API design, and Kubernetes.",
        "benefits": "Health insurance, 401(k) matching, flexible PTO, wellness stipend. Apply via careers page.",
        "salary": "$150,000",
        "country": "US",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1,
        "contact_details": "recruiting@stripe.com",
    },
    "case_2_obvious_scam": {
        "title": "Online Part-Time Form Filling & Data Entry Assistant",
        "company": "",
        "company_profile": "",
        "description": "URGENT HIRING!! Work only 2-3 hours daily from mobile. Simple copy-paste, captcha typing. No previous experience needed. Daily payout guaranteed without interview!",
        "requirements": "10th pass, smartphone required.",
        "benefits": "One-time mandatory refundable registration fee of Rs 499 required.",
        "salary": "Rs 3500/day",
        "country": "India",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "contact_details": "+91 9876543210 (WhatsApp only)",
    },
    "case_3_subtle_brand_impersonator": {
        "title": "Customer Support Operations Associate",
        "company": "Amazon India",
        "company_profile": "",
        "description": "Amazon India is expanding customer service operations and currently hiring Remote Customer Support Associates to address customer queries via live chat and email. Candidates will assist with package tracking, return requests, and customer feedback. Standard shifts available.",
        "requirements": "Graduate in any discipline. Good written English skills and basic typing speed (30 WPM). Must have personal laptop and broadband connection.",
        "benefits": "ESI, PF benefits, night shift allowance, and internet reimbursement provided. Contact HR Priya directly on Telegram @amazon_hr_onboarding.",
        "salary": "Rs 45,000/month",
        "country": "India",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "contact_details": "amazon_careers_hr@gmail.com",
    },
    "case_4_startup_intern": {
        "title": "Video Editing & Social Media Intern",
        "company": "Nudge Media Studio",
        "company_profile": "Boutique creative studio working with indie creators and brands.",
        "description": "We are looking for a video editor who knows Premiere Pro or DaVinci Resolve. You will edit 3-4 reels and long-form podcasts each week, create simple motion graphics, and collaborate directly with our creative lead.",
        "requirements": "Familiarity with Adobe Premiere Pro or DaVinci Resolve. Portfolio or Google Drive link showcasing past edits is mandatory. Eager to learn storytelling and pacing.",
        "benefits": "Flexible hours, portfolio mentorship, certificate of completion, and potential conversion to full-time junior role based on performance.",
        "salary": "Rs 12,000/month stipend",
        "country": "IN",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 0,
        "contact_details": "nudge.creativeteam@gmail.com",
    },
    "case_6_brevity_classified": {
        "title": "Job",
        "company": "",
        "company_profile": "",
        "description": "this job is good call me immediately for fast joining",
        "requirements": "",
        "benefits": "",
        "salary": "",
        "country": "India",
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
        "contact_details": "9876543210",
    },
    "case_7_crypto_task_scam": {
        "title": "App Optimization & Google Maps Rating Specialist",
        "company": "Global App Review Partner",
        "company_profile": "",
        "description": "Earn money by rating apps on Google Play Store and liking videos on YouTube. Each task takes 5 minutes and pays Rs 150 immediately. Complete 20 prepaid tasks per day to unlock merchant commission bonuses. VIP tasks require account wallet deposit of Rs 1,000 to initiate merchant order release. Payout sent directly to your USDT or UPI account within 10 minutes.",
        "requirements": "No experience required. Smartphone and Telegram app mandatory. Age 18+.",
        "benefits": "Daily cash bonus and task tier incentives. Apply via Telegram DM.",
        "salary": "Rs 2,500 - Rs 6,000 per day",
        "country": "India",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "contact_details": "Telegram: @digital_task_coordinator",
    },
    "case_8_technova_wfh_fee_scam": {
        "title": "Urgent Work From Home – Software Developer",
        "company": "TechNova Digital Solutions",
        "company_profile": "",
        "description": "We are urgently hiring Software Developers for immediate work-from-home opportunities. Freshers and candidates without prior experience are welcome. No technical interview is required and selected candidates can start immediately.\n\nThe selected candidate will work on basic software development, data entry, application testing, and online projects. Complete training will be provided.",
        "requirements": "Basic computer knowledge\nBasic knowledge of programming\nGood communication skills\nSmartphone and laptop required\nCandidates must be available to join immediately",
        "benefits": "₹35,000–₹75,000 monthly income\nPerformance-based incentives\nFlexible working hours\nWork completely from home\nNo previous experience required",
        "application_process": "Interested candidates should contact the recruitment coordinator through WhatsApp to receive the registration form and interview details.\n\nSelected candidates must complete a refundable ₹1,500 registration and verification fee before receiving the joining documents. The amount will be returned with the first month's salary.\n\nCandidates should also provide their Aadhaar/PAN details during registration for verification.\n\nImportant:\nOnly limited vacancies are available. Candidates who do not complete registration within 24 hours may lose their opportunity.",
        "salary": "₹35,000–₹75,000 per month",
        "country": "Remote",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 0,
        "employment_type": "Full-time",
        "required_experience": "Entry Level",
        "required_education": "Any Degree",
        "industry": "Information Technology",
        "function": "Software Development",
        "contact_details": "",
    },
}


def run_test(name, user_input):
    print(f"\n{'='*65}\nTEST CASE: {name}\n{'='*65}")

    ml_label, ml_proba, X_transformed, _ = predict_posting(user_input)
    _, _, feature_names = load_artifacts()

    # Heuristic & Domain checks
    raw_text = " ".join(filter(None, [
        user_input.get("title", ""), user_input.get("company", ""),
        user_input.get("company_profile", ""), user_input.get("description", ""),
        user_input.get("application_process", ""),
        user_input.get("requirements", ""), user_input.get("benefits", ""),
        user_input.get("salary", ""), user_input.get("contact_details", "")
    ]))

    flags = detect_red_flags(raw_text, context=user_input)

    contact = user_input.get("contact_details", "")
    if contact:
        domain_findings = check_domain(contact, company=user_input.get("company", ""))
        flags.extend(domain_findings)

    # Composite Risk Calculation
    high_threats = [f for f in flags if f.get("severity") == "High"]
    medium_threats = [f for f in flags if f.get("severity") == "Medium"]
    app_high_threats = [f for f in high_threats if "Application Process" in f.get("field", "")]

    composite_proba = ml_proba
    is_override = False

    if len(high_threats) >= 2 or app_high_threats:
        composite_proba = max(ml_proba, 0.90 if app_high_threats else 0.85)
        is_override = composite_proba > ml_proba
    elif len(high_threats) == 1:
        composite_proba = max(ml_proba, 0.75)
        is_override = composite_proba > ml_proba
    elif len(medium_threats) >= 2 and ml_proba < 0.35:
        composite_proba = max(ml_proba, 0.40)
        is_override = composite_proba > ml_proba

    if composite_proba < 0.20:
        composite_label = "Low Scam Risk (Likely Legitimate)"
    elif composite_proba < 0.50:
        composite_label = "Moderate Risk / Inconclusive"
    else:
        composite_label = "High Scam Risk (Potentially Fraudulent)"

    print(f"ML Statistical Score:   {ml_proba:.1%}")
    print(f"Composite Scam Risk:    {composite_proba:.1%}")
    print(f"Final Risk Label:       {composite_label}")
    if is_override:
        print(f"[*] SECURITY OVERRIDE TRIGGERED: Escalated due to High-Severity Threat(s) (App Process: {bool(app_high_threats)})")

    print(f"\nDetected Threat Indicators ({len(flags)}):")
    for f in flags:
        field_info = f" [Field: {f['field']}]" if f.get("field") else ""
        print(f"  [{f['severity']}] {f['indicator']}{field_info}")

    explanations, base_value = explain_posting(X_transformed, feature_names, top_n=5)
    print("\nTop SHAP Contributors:")
    for exp in explanations:
        print(f"  {exp['field']}: {exp['term']} -> {exp['direction']} ({exp['shap_value']:+.3f})")


if __name__ == "__main__":
    for name, case in TEST_CASES.items():
        run_test(name, case)
