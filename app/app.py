r"""
app.py
Streamlit front-end for the Explainable Job Scam Risk Analyzer.

Run with:  .\venv\Scripts\streamlit.exe run app/app.py
"""

import streamlit as st
from predictor import predict_posting, load_artifacts
from explainer import explain_posting
from red_flags import detect_red_flags
from domain_check import check_domain
from image_extractor import process_uploaded_image, process_uploaded_images

st.set_page_config(page_title="Job Scam Risk Analyzer", page_icon="🔎", layout="wide")

# -------------------------------------------------------------
# Custom CSS for High-Visibility Submit Button & Enhancements
# -------------------------------------------------------------
st.markdown("""
<style>
/* High-visibility primary submit button */
div[data-testid="stFormSubmitButton"] > button,
div.stButton > button {
    width: 100% !important;
    min-height: 56px !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    box-shadow: 0 6px 20px -2px rgba(99, 102, 241, 0.5) !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-top: 12px !important;
}

div[data-testid="stFormSubmitButton"] > button:hover,
div.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 50%, #1d4ed8 100%) !important;
    box-shadow: 0 10px 25px -2px rgba(99, 102, 241, 0.75) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(255, 255, 255, 0.45) !important;
}

div[data-testid="stFormSubmitButton"] > button:active,
div.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 3px 10px rgba(99, 102, 241, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🔎 Explainable Job Scam Risk Analyzer")
st.caption("Dual-Engine Evaluation: EMSCAD ML Pipeline + Cross-Platform Threat Heuristics")


def get_select_index(options, target_value, default_index=0):
    """Helper to match auto-extracted string to selectbox options index."""
    if not target_value:
        return default_index
    target_lower = str(target_value).lower().strip()
    for idx, opt in enumerate(options):
        opt_lower = opt.lower()
        if opt_lower == target_lower or target_lower in opt_lower or opt_lower in target_lower:
            return idx
    return default_index


# -------------------------------------------------------------
# IMAGE UPLOAD & AUTO-EXTRACTION SECTION (Supports Up to 7 Images)
# -------------------------------------------------------------
with st.expander("📷 Upload Job Description Images / Screenshots (Up to 7 Images)", expanded=False):
    st.markdown(
        "Upload **1 to 7 screenshots or flyers** of a job posting (e.g. WhatsApp chat, Instagram ad, LinkedIn, Telegram flyer, multi-page document). "
        "Gemini Vision AI will automatically analyze the images, extract key fields (Title, Company, Salary, Contacts), and pre-fill the form below."
    )
    
    uploaded_images = st.file_uploader(
        "Upload Up to 7 Images (PNG, JPG, JPEG, WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="job_image_uploader",
        help="Upload up to 7 images to extract details from."
    )

    if uploaded_images:
        if len(uploaded_images) > 7:
            st.warning("⚠️ You uploaded more than 7 images. Only the first 7 images will be processed.")
            uploaded_images = uploaded_images[:7]

        st.markdown(f"**Preview Uploaded Images ({len(uploaded_images)}/7):**")
        cols_per_row = 4 if len(uploaded_images) > 3 else max(len(uploaded_images), 1)
        for row_start in range(0, len(uploaded_images), cols_per_row):
            row_imgs = uploaded_images[row_start : row_start + cols_per_row]
            img_cols = st.columns(len(row_imgs))
            for idx, img in enumerate(row_imgs):
                with img_cols[idx]:
                    st.image(img, caption=f"Image {row_start + idx + 1}: {img.name}", use_container_width=True)

        st.write("Click below to extract and synthesize job details across all uploaded image(s) using Gemini Vision AI.")
        if st.button("✨ Auto-Extract Details into Form", type="primary", use_container_width=True):
            with st.spinner(f"Extracting job details from {len(uploaded_images)} image(s) via Gemini AI..."):
                extracted = process_uploaded_images(uploaded_images)
                if extracted and extracted.get("_error"):
                    st.error(f"⚠️ {extracted['_error']}")
                elif extracted and any(str(extracted.get(k, "")).strip() for k in ["title", "company", "description", "contact_details"]):
                    for k, v in extracted.items():
                        if not k.startswith("_"):
                            st.session_state[f"auto_{k}"] = v
                    st.session_state["image_extracted_success"] = True
                    st.rerun()
                else:
                    st.error(
                        "⚠️ Could not extract job posting details from the image(s). "
                        "Please verify image clarity or enter the details manually in the form below."
                    )

if st.session_state.get("image_extracted_success"):
    c_alert, c_btn = st.columns([4, 1])
    with c_alert:
        st.success("✅ Job details successfully extracted and populated into the form! Review or modify any field below before running the analyzer.")
    with c_btn:
        if st.button("🔄 Clear Auto-Filled Form"):
            for k in list(st.session_state.keys()):
                if k.startswith("auto_") or k == "image_extracted_success":
                    del st.session_state[k]
            st.rerun()


# -------------------------------------------------------------
# MAIN ANALYSIS FORM
# -------------------------------------------------------------
with st.form("scam_analyzer_form"):
    
    # -------------------------------------------------------------
    # 1. ORIGIN & ENTITY IDENTITY
    # -------------------------------------------------------------
    st.subheader("1. Origin & Entity Identity")
    c1, c2 = st.columns(2)
    
    platform_options = [
        "Select", "Telegram Channel", "WhatsApp Group / DM", "Instagram Ad", 
        "Facebook Jobs", "LinkedIn", "Indeed / Job Board", "Company Website"
    ]
    platform_default_idx = get_select_index(platform_options, st.session_state.get("auto_platform", "Select"))

    with c1:
        platform = st.selectbox(
            "Source Platform *",
            platform_options,
            index=platform_default_idx
        )
        company = st.text_input(
            "Company (as advertised) *",
            value=st.session_state.get("auto_company", ""),
            placeholder="e.g. Amazon, Meesho, ABC Corp"
        )
    with c2:
        account_name = st.text_input(
            "Poster / Account Handle",
            value=st.session_state.get("auto_account_name", ""),
            placeholder="e.g. @careers_hr, HR Pooja"
        )
        contact_details = st.text_input(
            "Contact Details / Recruiter Handle *",
            value=st.session_state.get("auto_contact_details", ""),
            placeholder="e.g. Telegram ID, WhatsApp number, or Email (recruiter@company.com)"
        )
        
    company_profile = st.text_area(
        "Company Profile / Overview (Optional)",
        value=st.session_state.get("auto_company_profile", ""),
        height=70
    )

    # -------------------------------------------------------------
    # 2. ROLE & TEXT CONTENT
    # -------------------------------------------------------------
    st.subheader("2. Role & Core Content")
    t1, t2 = st.columns([3, 1])
    with t1:
        title = st.text_input(
            "Job Title *",
            value=st.session_state.get("auto_title", ""),
            placeholder="e.g. Data Entry Assistant, Customer Support"
        )
    with t2:
        target_audience = st.selectbox(
            "Target Audience",
            ["Select", "General / Unspecified", "Students / Freshers", "Homemakers", "Anyone with smartphone"]
        )

    description = st.text_area(
        "Job Description *",
        value=st.session_state.get("auto_description", ""),
        height=130,
        placeholder="Paste the full job posting description here..."
    )

    r1, r2 = st.columns(2)
    with r1:
        requirements = st.text_area(
            "Requirements Listed",
            value=st.session_state.get("auto_requirements", ""),
            height=80
        )
    with r2:
        benefits = st.text_area(
            "Benefits / Daily Perks",
            value=st.session_state.get("auto_benefits", ""),
            height=80
        )

    # -------------------------------------------------------------
    # 3. COMPENSATION & LOGISTICS
    # -------------------------------------------------------------
    st.subheader("3. Compensation, Work Mode & Shifts")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        salary = st.text_input(
            "Salary Stated",
            value=st.session_state.get("auto_salary", ""),
            placeholder="e.g. Rs 3,500/day, $40/hr"
        )
    with l2:
        work_mode_options = ["Select", "Work From Home / Remote", "Hybrid", "On-site"]
        work_mode_default_idx = get_select_index(work_mode_options, st.session_state.get("auto_work_mode", "Select"))
        work_mode = st.selectbox("Work Mode", work_mode_options, index=work_mode_default_idx)
    with l3:
        shifts = st.selectbox("Shifts Listed", ["Select", "Flexible / Part-time Tasks", "Day Shift", "Night Shift", "Not Specified"])
    with l4:
        location = st.text_input(
            "Location / Country",
            value=st.session_state.get("auto_location", ""),
            placeholder="e.g. US, IN, Remote"
        )

    b1, b2 = st.columns(2)
    with b1:
        has_company_logo = st.checkbox(
            "Posting includes verified company logo",
            value=bool(st.session_state.get("auto_has_company_logo", False))
        )
    with b2:
        has_questions = st.checkbox(
            "Posting includes formal screening / application questions",
            value=bool(st.session_state.get("auto_has_questions", False))
        )

    # -------------------------------------------------------------
    # 4. EXPANDABLE ADVANCED METADATA (EMSCAD Categorical Features)
    # -------------------------------------------------------------
    with st.expander("🎓 Advanced Data (Optional / Detailed Metadata)"):
        m1, m2, m3 = st.columns(3)
        with m1:
            employment_type = st.selectbox(
                "Employment Type",
                ["Select", "Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"]
            )
        with m2:
            required_experience = st.selectbox(
                "Required Experience",
                ["Select", "Not Applicable", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Internship", "Unknown"]
            )
        with m3:
            required_education = st.selectbox(
                "Required Education",
                ["Select", "High School or equivalent", "Bachelor's Degree", "Master's Degree", "Associate Degree", "Certification", "Doctorate", "Professional", "Some College Coursework Completed", "Unknown"]
            )
        
        i1, i2 = st.columns(2)
        with i1:
            industry = st.text_input("Industry", placeholder="e.g. Internet, Telecommunications")
        with i2:
            function = st.text_input("Function", placeholder="e.g. Customer Service, Administrative")

    # -------------------------------------------------------------
    # SUBMIT BUTTON (High-Visibility with CSS)
    # -------------------------------------------------------------
    submitted = st.form_submit_button(
        "⚡ Analyze Job Scam Risk (Dual-Engine Evaluation)",
        type="primary",
        use_container_width=True
    )

if submitted:
    # Check if meaningful content was provided
    has_content = any(
        field.strip() for field in [title, company, description, requirements, benefits, contact_details]
    )

    if not has_content:
        st.divider()
        st.subheader("Baseline Prior (No Input Provided)")
        st.info(
            "ℹ️ **Empty Submission — Displaying Baseline Prior**\n\n"
            "You submitted the form without providing any job content (Title, Company, or Description).\n\n"
            "👉 **To analyze a posting:** Please enter at least a **Job Title** or paste the **Job Description** above, or upload an image."
        )
    else:
        # 1. Pre-flight text quality / brevity check
        core_text = " ".join(filter(None, [title, company, description, requirements, benefits]))
        word_count = len([w for w in core_text.strip().split() if len(w) > 1])

        st.divider()

        if word_count < 25:
            st.warning(
                f"⚠️ **Low Information / Abbreviated Posting Warning ({word_count} words)**\n\n"
                "The text entered is very short (< 25 words). Machine learning NLP models trained on full job postings "
                "rely on rich vocabulary and contextual patterns. Abbreviated inputs or 1-line phrases lack standard "
                "duties, prerequisites, and qualifications, causing predictions to lean toward baseline prior or inconclusive scores. "
                "Please review the heuristic indicators and exercise caution."
            )

        # 2. Build clean user input dictionary with robust category handling
        user_input = {
            "title": title.strip(),
            "company_profile": company_profile.strip() if company_profile.strip() else "",
            "description": description.strip(),
            "requirements": requirements.strip(),
            "benefits": benefits.strip(),
            "salary": salary.strip(),
            "country": location.strip(),
            "telecommuting": 1 if "Remote" in work_mode else 0,
            "has_company_logo": int(has_company_logo),
            "has_questions": int(has_questions),
            "employment_type": None if employment_type == "Select" else employment_type,
            "required_experience": None if required_experience == "Select" else required_experience,
            "required_education": None if required_education == "Select" else required_education,
            "industry": industry.strip() if industry.strip() else "Unknown",
            "function": function.strip() if function.strip() else "Unknown",
        }

        # 3. Dual-Engine Evaluations
        # A. ML Ensemble Prediction
        ml_label, ml_proba, X_transformed, input_row = predict_posting(user_input)
        _, _, feature_names = load_artifacts()

        # B. Heuristic Threat & Domain Checks
        raw_text = " ".join(filter(None, [
            title, company, account_name, contact_details,
            company_profile, description, requirements, benefits, salary
        ]))

        context_data = {
            "company": company,
            "title": title,
            "description": description
        }
        flags = detect_red_flags(raw_text, context=context_data)

        if contact_details:
            domain_findings = check_domain(contact_details, company=company)
            flags.extend(domain_findings)

        # 4. Composite Risk Fusion (Cybersecurity Safety Override)
        high_threats = [f for f in flags if f.get("severity") == "High"]
        medium_threats = [f for f in flags if f.get("severity") == "Medium"]

        composite_proba = ml_proba
        is_override_active = False
        override_reason = ""

        if len(high_threats) >= 2:
            composite_proba = max(ml_proba, 0.85)
            is_override_active = (composite_proba > ml_proba)
            override_reason = f"Triggered by {len(high_threats)} High-Severity Threat Indicators ({high_threats[0]['indicator']}, {high_threats[1]['indicator']})."
        elif len(high_threats) == 1:
            composite_proba = max(ml_proba, 0.75)
            is_override_active = (composite_proba > ml_proba)
            override_reason = f"Triggered by High-Severity Threat Indicator: '{high_threats[0]['indicator']}'."
        elif len(medium_threats) >= 2 and ml_proba < 0.35:
            composite_proba = max(ml_proba, 0.40)
            is_override_active = (composite_proba > ml_proba)
            override_reason = "Elevated due to multiple Medium-Severity Threat Indicators."

        st.subheader("Dual-Engine Evaluation Result")

        # Visual Metrics Dashboard
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Composite Scam Risk", f"{composite_proba:.1%}")
        with m_col2:
            st.metric("ML Statistical Score", f"{ml_proba:.1%}")
        with m_col3:
            st.metric("Threat Indicators", f"{len(flags)} detected", f"{len(high_threats)} High-risk" if high_threats else "Normal")

        st.progress(min(max(composite_proba, 0.0), 1.0))

        # Final Classification Verdict
        if composite_proba < 0.20:
            st.success(
                f"🟢 **Low Scam Risk (Likely Legitimate)** — Evaluated Risk: **{composite_proba:.1%}**\n\n"
                "The posting's vocabulary, metadata, verified domain, and structural features closely align with authentic employment listings."
            )
        elif composite_proba < 0.50:
            st.warning(
                f"🟡 **Moderate Risk / Inconclusive** — Evaluated Risk: **{composite_proba:.1%}**\n\n"
                "**Caution Advised:** The posting does not trigger a full scam classification, but exhibits elevated risk signals "
                "or lacks sufficient verifiable information. Verify recruiter credentials independently before sharing personal details."
            )
        else:
            st.error(
                f"🔴 **High Scam Risk (Potentially Fraudulent)** — Evaluated Risk: **{composite_proba:.1%}**\n\n"
                "**High Scam Likelihood:** This posting exhibits strong statistical markers, linguistic cues, or unauthorized deception channels "
                "characteristic of fraudulent or predatory employment schemes."
            )

        if is_override_active:
            st.info(
                f"🛡️ **Security Override Active:** {override_reason}\n\n"
                f"*ML Text & Structural Score:* `{ml_proba:.1%}` ➔ *Composite Risk Score:* `{composite_proba:.1%}`\n\n"
                "**Why this happened:** Scammers frequently copy authentic, professional job descriptions from legitimate company portals. "
                "Even when the text vocabulary appears legitimate, critical deception vectors (such as unauthorized recruiter email domains or upfront fee requests) "
                "override textual signals to ensure applicant safety."
            )

        # 5. Model Explanation (SHAP)
        st.divider()
        st.subheader("Model Explanation (SHAP)")
        st.caption(
            "These are the specific words and fields that most influenced the machine learning model. "
            "A positive contribution pushes toward 'Fraudulent'; negative pushes toward 'Real'."
        )

        explanations, base_value = explain_posting(X_transformed, feature_names)
        for exp in explanations:
            arrow = "🔺" if exp["shap_value"] > 0 else "🔻"
            st.write(
                f"{arrow} **{exp['field']}**: {exp['term']} "
                f"— {exp['direction']} (impact: {exp['shap_value']:+.3f})"
            )

        # 6. Detected Warning Indicators (Red Flags + Domain Verification)
        st.divider()
        st.subheader("Detected Threat Indicators (Heuristic & Domain Engine)")
        st.caption(
            "Independent rule-based checks examining unauthorized recruiter channels, upfront fees, unrealistic pay, "
            "and domain reputation."
        )

        if flags:
            for f in flags:
                sev_color = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}.get(f["severity"], "⚪")
                st.write(f"{sev_color} **{f['indicator']}** ({f['severity']})")
                st.caption(f"{f['description']}")
                st.code(f["evidence"], language=None)
        else:
            st.write("No common red-flag patterns or suspicious contact channels detected.")
            st.caption("Note: Absence of red flags does not guarantee the posting is legitimate.")