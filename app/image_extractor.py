"""
image_extractor.py
Extracts job description details from uploaded images (screenshots, flyers, ads)
using Google Gemini Vision Multimodal API.
Supports 1 to 7 images (multi-page flyers, WhatsApp chat screenshots, etc.).
"""

import os
import re
import json
import base64
from pathlib import Path
import requests

# ==============================================================================
# 🔑 GEMINI API KEY CONFIGURATION
# Set GEMINI_API_KEY as an environment variable or in a local .env file.
# You may also paste your key below for local testing (keep out of git):
# (Get a free key from: https://aistudio.google.com/)
# ==============================================================================
GEMINI_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"


def get_configured_api_key(api_key: str = "") -> str:
    """
    Returns the configured Gemini API key:
    1. Key passed explicitly as parameter (if any)
    2. Hardcoded GEMINI_API_KEY above (if updated locally)
    3. GEMINI_API_KEY environment variable
    4. Local .env file (if present)
    """
    if api_key and api_key.strip():
        return api_key.strip()
    if GEMINI_API_KEY and GEMINI_API_KEY.strip() and GEMINI_API_KEY.strip() != "PASTE_YOUR_GEMINI_API_KEY_HERE":
        return GEMINI_API_KEY.strip()
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key and env_key != "PASTE_YOUR_GEMINI_API_KEY_HERE":
        return env_key

    # Attempt to load from project-root or local .env file
    for env_path in [Path(__file__).resolve().parent.parent / ".env", Path(__file__).resolve().parent / ".env"]:
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip('"\'')
                        if val and val != "PASTE_YOUR_GEMINI_API_KEY_HERE":
                            return val
            except Exception:
                pass
    return ""


def extract_with_gemini_vision(images_input, mime_type: str = "image/png", api_key: str = "") -> dict:
    """
    Use Gemini Vision REST API to directly extract and synthesize structured fields from 1 to 7 images.
    images_input can be:
      - bytes: single image raw bytes
      - list of bytes: [bytes, ...]
      - list of tuples: [(bytes, mime_type), ...]
      - list of Streamlit UploadedFile objects
      - single Streamlit UploadedFile object
    Requires no extra SDK; uses standard requests.
    """
    key = get_configured_api_key(api_key)
    if not key:
        return {
            "_error": (
                "Gemini API key is not configured! Please open 'app/image_extractor.py' "
                "and paste your API key in the GEMINI_API_KEY placeholder at the top of the file."
            )
        }

    # Normalize images_input into a list of (image_bytes, mime_type)
    image_items = []
    if isinstance(images_input, bytes):
        image_items.append((images_input, mime_type))
    elif isinstance(images_input, list):
        for item in images_input:
            if isinstance(item, tuple) and len(item) == 2:
                image_items.append(item)
            elif isinstance(item, bytes):
                image_items.append((item, mime_type))
            elif hasattr(item, "getvalue"):
                m = getattr(item, "type", "image/png") or "image/png"
                image_items.append((item.getvalue(), m))
    elif hasattr(images_input, "getvalue"):
        m = getattr(images_input, "type", "image/png") or "image/png"
        image_items.append((images_input.getvalue(), m))

    if not image_items:
        return {"_error": "No image data received for extraction."}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={key}"

    prompt = (
        f"You are an expert recruitment text analyzer. Extract and synthesize job details across these "
        f"{len(image_items)} job posting image(s) (which may be multi-page flyers, multi-screenshot ads, or message threads). "
        "Combine all information into a single coherent job profile.\n"
        "Return ONLY a valid JSON object with the following keys:\n"
        "{\n"
        '  "title": "Job title or role",\n'
        '  "company": "Company or hiring brand",\n'
        '  "account_name": "Poster handle or HR name",\n'
        '  "contact_details": "Recruiter email, WhatsApp number, Telegram handle, or phone",\n'
        '  "description": "Full job description text combined from all images",\n'
        '  "application_process": "Application steps, how to apply, registration fee, or contact procedure",\n'
        '  "requirements": "Required qualifications, skills, or eligibility",\n'
        '  "benefits": "Benefits, perks, or daily payout details",\n'
        '  "salary": "Stated compensation or pay rate",\n'
        '  "location": "Location or country or Remote",\n'
        '  "work_mode": "Work From Home / Remote, Hybrid, or On-site",\n'
        '  "platform": "Source platform if visible (e.g. Telegram, WhatsApp, Instagram, LinkedIn, Indeed)",\n'
        '  "has_company_logo": true or false\n'
        "}\n"
        "If a field is not present or cannot be identified, leave it as an empty string (or false for has_company_logo). "
        "Do not include markdown code block formatting; return pure JSON."
    )

    parts = [{"text": prompt}]
    for img_bytes, m_type in image_items:
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": m_type or "image/png",
                "data": b64_image
            }
        })

    payload = {
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json"
        }
    }

    models_to_try = ["gemini-3.8-flash","gemini-3.7-flash","gemini-3.6-flash","gemini-3.5-flash"]
    last_error = "Unable to connect or receive candidates from Gemini Vision API"
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        try:
            resp = requests.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts_resp = candidates[0].get("content", {}).get("parts", [{}])
                    raw_text = parts_resp[0].get("text", "") if parts_resp else ""
                    clean_json = raw_text.strip()
                    if clean_json.startswith("```json"):
                        clean_json = clean_json[7:]
                    if clean_json.startswith("```"):
                        clean_json = clean_json[3:]
                    if clean_json.endswith("```"):
                        clean_json = clean_json[:-3]
                    parsed = json.loads(clean_json.strip())
                    if isinstance(parsed, dict) and any(parsed.get(k) for k in ["title", "company", "description"]):
                        return parsed
            else:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", {}).get("message", f"HTTP {resp.status_code}")
                except Exception:
                    err_msg = f"HTTP {resp.status_code}: {resp.text[:150]}"
                last_error = f"{model_name}: {err_msg}"
                print(f"Gemini {model_name} HTTP {resp.status_code}: {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"Gemini {model_name} call failed: {e}")

    return {
        "_error": f"Gemini Vision API request failed ({last_error}). Please check your API key and internet connection."
    }


def extract_raw_text_ocr(image_bytes: bytes, filename: str = "posting.png") -> str:
    """
    Extract raw text from image bytes using available OCR engine:
    1. Native Windows Media OCR via winocr (fast, native, offline, zero rate limit)
    2. Local pytesseract (if available)
    3. Cloud OCR via OCR.space API (fallback)
    """
    # 1. Native Windows OCR (Offline, high accuracy, zero rate limit)
    try:
        import winocr
        pil_img = Image.open(BytesIO(image_bytes))
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")
        res = winocr.recognize_pil_sync(pil_img, "en")
        text = res.get("text", "") if isinstance(res, dict) else ""
        if text and len(text.strip()) > 15:
            return text.strip()

        # Try upscaled if initial text is brief
        w, h = pil_img.size
        scale = 2 if w < 1200 else 1.5
        scaled_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        res_scaled = winocr.recognize_pil_sync(scaled_img, "en")
        text_scaled = res_scaled.get("text", "") if isinstance(res_scaled, dict) else ""
        if len(text_scaled.strip()) > len(text.strip()):
            return text_scaled.strip()
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Windows Media OCR failed: {e}")

    # 2. Try pytesseract if installed locally
    try:
        import pytesseract
        pil_img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(pil_img)
        if text and len(text.strip()) > 10:
            return text.strip()
    except Exception:
        pass

    # 3. Free OCR.space API (remote fallback)
    try:
        payload = {
            "apikey": FREE_OCR_API_KEY,
            "language": "eng",
            "isOverlayRequired": False,
            "OCREngine": 2,
        }
        files = {
            "file": (filename, image_bytes)
        }
        resp = requests.post(FREE_OCR_API_URL, data=payload, files=files, timeout=8)
        if resp.status_code == 200:
            result = resp.json()
            parsed_results = result.get("ParsedResults", [])
            if parsed_results:
                text = parsed_results[0].get("ParsedText", "")
                if text:
                    return text.strip()
    except Exception as e:
        print(f"Cloud OCR failed: {e}")

    return ""


def parse_job_text(raw_text: str) -> dict:
    """
    Heuristically extracts key job posting details from raw OCR text.
    """
    extracted = {
        "title": "",
        "company": "",
        "account_name": "",
        "contact_details": "",
        "description": raw_text.strip(),
        "application_process": "",
        "requirements": "",
        "benefits": "",
        "salary": "",
        "location": "",
        "work_mode": "Select",
        "platform": "Select",
        "has_company_logo": False,
    }

    if not raw_text or not raw_text.strip():
        return extracted

    text_lower = raw_text.lower()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # 1. Platform Detection
    if "telegram" in text_lower or "t.me/" in text_lower:
        extracted["platform"] = "Telegram Channel"
    elif "whatsapp" in text_lower or "wa.me" in text_lower:
        extracted["platform"] = "WhatsApp Group / DM"
    elif "instagram" in text_lower or "insta" in text_lower:
        extracted["platform"] = "Instagram Ad"
    elif "linkedin" in text_lower:
        extracted["platform"] = "LinkedIn"
    elif "indeed" in text_lower:
        extracted["platform"] = "Indeed / Job Board"
    elif "facebook" in text_lower:
        extracted["platform"] = "Facebook Jobs"

    # 2. Contact Details (Email, WhatsApp, Phone, Telegram)
    contacts = []
    # Email
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", raw_text)
    if emails:
        contacts.extend(emails)
    # Phone number (Indian/International format)
    phones = re.findall(r"(?:\+?91[\-\s]?)?[6-9]\d{9}|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", raw_text)
    for p in phones:
        if p not in contacts:
            contacts.append(p)
    # Telegram / Handles
    handles = re.findall(r"(?:@|t\.me\/)[a-zA-Z0-9_]{4,}", raw_text)
    for h in handles:
        if h not in contacts:
            contacts.append(h)
    
    if contacts:
        extracted["contact_details"] = ", ".join(contacts[:3])

    # 3. Work Mode
    if any(k in text_lower for k in ["work from home", "wfh", "remote", "online work"]):
        extracted["work_mode"] = "Work From Home / Remote"
    elif "hybrid" in text_lower:
        extracted["work_mode"] = "Hybrid"
    elif any(k in text_lower for k in ["on-site", "onsite", "office work", "in-office"]):
        extracted["work_mode"] = "On-site"

    # 4. Compensation / Salary
    salary_patterns = [
        r"(?:₹|rs\.?|inr|\$)\s?\d+(?:,\d+)*(?:\s*(?:-|to)\s*(?:₹|rs\.?|inr|\$)?\s?\d+(?:,\d+)*)?(?:\s*(?:\/|\s*per\s*)(?:day|month|hr|hour|year|annum|lpa|task))?",
        r"\b\d+\s*(?:lpa|lakhs?|k\b)\s*(?:per\s+(?:month|annum|year))?",
        r"earn\s+up\s+to\s+[^,.\n]+",
        r"\b\d{4,6}\s*(?:\/|\s*per\s*)(?:month|day)",
    ]
    for sp in salary_patterns:
        sal_match = re.search(sp, raw_text, flags=re.IGNORECASE)
        if sal_match:
            extracted["salary"] = sal_match.group(0).strip()
            break

    # 5. Job Title
    title_match = re.search(
        r"(?:hiring\s+for|urgently\s+hiring|opening\s+for|position|role|job\s+title)[\s:]+([A-Za-z\s/]{3,35})",
        raw_text, flags=re.IGNORECASE
    )
    if title_match:
        extracted["title"] = title_match.group(1).strip()
    else:
        # Check first 3 lines for typical job title phrases
        common_roles = [
            "data entry", "customer support", "virtual assistant", "content writer",
            "telecaller", "executive", "software engineer", "sales associate", "accountant",
            "back office", "chat support", "operator", "intern", "developer"
        ]
        for line in lines[:4]:
            l_lower = line.lower()
            if any(role in l_lower for role in common_roles):
                extracted["title"] = line[:40].strip()
                break
        if not extracted["title"] and lines:
            # Fall back to first short headline line
            if len(lines[0].split()) <= 6:
                extracted["title"] = lines[0].strip()

    # 6. Company Name
    company_match = re.search(
        r"(?:company|organization|at\s+|hiring\s+by)[\s:]+([A-Z][A-Za-z0-9\s&]{2,30})",
        raw_text
    )
    if company_match:
        extracted["company"] = company_match.group(1).strip()
    else:
        known_brands = ["amazon", "meesho", "flipkart", "swiggy", "zomato", "tcs", "infosys", "wipro", "google"]
        for b in known_brands:
            if b in text_lower:
                extracted["company"] = b.capitalize()
                extracted["has_company_logo"] = True
                break

    # 7. Requirements & Benefits sections
    req_match = re.search(
        r"(?:requirements|eligibility|qualifications|skills\s+needed)[\s:]+([^#\n]+(?:\n[*-•\d].+)*)",
        raw_text, flags=re.IGNORECASE
    )
    if req_match:
        extracted["requirements"] = req_match.group(1).strip()[:250]

    ben_match = re.search(
        r"(?:benefits|perks|what\s+you\s+get|incentives)[\s:]+([^#\n]+(?:\n[*-•\d].+)*)",
        raw_text, flags=re.IGNORECASE
    )
    if ben_match:
        extracted["benefits"] = ben_match.group(1).strip()[:250]

    # Application Process section
    app_match = re.search(
        r"(?:application\s+process|how\s+to\s+apply|selection\s+process|to\s+apply|registration\s+process)[\s:]+([^#\n]+(?:\n[*-•\d].+)*)",
        raw_text, flags=re.IGNORECASE
    )
    if app_match:
        extracted["application_process"] = app_match.group(1).strip()[:350]

    # 8. Location
    loc_match = re.search(
        r"(?:location|job\s+location|place)[\s:]+([A-Za-z\s,]{3,25})",
        raw_text, flags=re.IGNORECASE
    )
    if loc_match:
        extracted["location"] = loc_match.group(1).strip()
    elif "remote" in text_lower or "work from home" in text_lower:
        extracted["location"] = "Remote"
    elif "india" in text_lower:
        extracted["location"] = "India"

    return extracted


def process_uploaded_images(file_objs, gemini_api_key: str = "") -> dict:
    """
    Main entry point for processing 1 to 7 uploaded images from Streamlit.
    Directly extracts and synthesizes structured job details using Gemini Vision API.
    Uses the configured GEMINI_API_KEY from the top of this file.
    """
    if not file_objs:
        return {}

    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    # Limit to maximum 7 images
    target_files = file_objs[:7]

    # Directly extract structured fields via Gemini Vision API
    return extract_with_gemini_vision(target_files, api_key=gemini_api_key)


def process_uploaded_image(file_obj, gemini_api_key: str = "") -> dict:
    """Backward-compatible single image processor."""
    if not file_obj:
        return {}
    return process_uploaded_images([file_obj], gemini_api_key=gemini_api_key)

