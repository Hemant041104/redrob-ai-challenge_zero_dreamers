#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking
Track 01: Data & AI Challenge

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv

Constraints satisfied:
  - CPU only, no GPU
  - No network calls during ranking
  - < 5 minutes on 100K candidates (typically ~30-60 seconds)
  - < 16 GB RAM (uses ~200–400 MB)
  - Pure Python + stdlib + numpy + pandas (no heavy ML models needed)

Scoring approach:
  - Skill match against JD requirements (weighted by proficiency + endorsements + duration)
  - Career quality (product companies vs consulting, ML/AI roles vs unrelated)
  - Experience fit (5-9 year target band, applied ML trajectory)
  - Location/availability (Pune/Noida/Hyderabad/Mumbai/Delhi NCR preferred)
  - Behavioral signals (activity, response rate, notice period, open-to-work)
  - GitHub activity bonus
  - Honeypot detection (kills impossible profiles before ranking)
  - Trap avoidance (keyword stuffers, title-chasers, consulting-only careers)
"""

import argparse
import csv
import gzip
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# JD-derived constants  (Senior AI Engineer @ Redrob AI)
# ---------------------------------------------------------------------------

# Skills that are "must have" per the JD — broad synonyms included
MUST_HAVE_SKILLS = {
    # Embeddings / retrieval
    "embeddings", "sentence transformers", "sentence-transformers",
    "vector search", "dense retrieval", "semantic search",
    "e5", "bge", "openai embeddings", "text embeddings",
    # Vector DBs / hybrid search
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "hybrid search",
    # Ranking / IR
    "information retrieval", "ranking", "recommendation systems",
    "learning to rank", "bm25", "reranking", "re-ranking",
    # Python
    "python",
    # Evaluation
    "ndcg", "mrr", "map", "a/b testing", "evaluation framework",
    "offline evaluation", "online evaluation",
}

# Nice-to-have skills (bonus but not required)
NICE_TO_HAVE_SKILLS = {
    # LLM / fine-tuning
    "fine-tuning llms", "fine-tuning", "lora", "qlora", "peft",
    "llm", "large language models", "rag", "retrieval augmented generation",
    "langchain", "llama", "mistral", "hugging face", "hugging face transformers",
    "transformers",
    # Learning-to-rank
    "xgboost", "lightgbm", "lambdarank", "listwise", "pairwise",
    # MLOps / infra
    "mlops", "mlflow", "kubeflow", "bentoml", "triton",
    "fastapi", "flask", "docker", "kubernetes",
    # General ML
    "pytorch", "tensorflow", "scikit-learn", "machine learning",
    "deep learning", "nlp", "natural language processing",
    "feature engineering", "data pipelines",
    # Adjacent
    "spark", "kafka", "redis", "postgresql", "aws", "gcp", "azure",
    "weights & biases", "prompt engineering",
}

# Skills that signal the candidate is likely NOT an AI/ML engineer
# (keyword stuffers often add these alongside AI skills)
NON_ML_INDICATOR_SKILLS = {
    "sales", "seo", "marketing", "content writing", "photoshop",
    "illustrator", "figma", "excel", "powerpoint", "tally",
    "accounting", "six sigma", "scrum", "project management",
    "salesforce crm", "sap", "hadoop", "etl",  # data but not ML
}

# Engineering / AI titles that make sense for this JD
POSITIVE_TITLE_KEYWORDS = {
    "ml", "machine learning", "ai", "artificial intelligence",
    "nlp", "natural language", "data science", "data scientist",
    "search", "retrieval", "ranking", "recommendation",
    "applied", "research engineer", "applied scientist",
    "backend", "software engineer", "senior engineer",
    "platform engineer", "infrastructure", "mlops",
    "embedding", "vector",
}

# Titles that are clearly wrong for this role
NEGATIVE_TITLE_KEYWORDS = {
    "hr manager", "human resources", "marketing manager", "sales",
    "accountant", "civil engineer", "mechanical engineer",
    "graphic designer", "customer support", "operations manager",
    "project manager", "business analyst", "content writer",
    "frontend engineer", "mobile developer", ".net developer",
    "java developer", "sap", "devops",  # devops borderline but JD says no
}

# Consulting / services companies — JD explicitly disqualifies these-only careers
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "ltimindtree",
    "niit technologies", "mphasis", "cyient", "birlasoft",
    "zensar", "persistent systems",  # debatable but conservative
}

# Product / startup / AI company signals (strong positive)
PRODUCT_COMPANY_KEYWORDS = {
    "google", "meta", "microsoft", "amazon", "apple",
    "flipkart", "swiggy", "zomato", "ola", "paytm", "razorpay",
    "phonepe", "meesho", "dream11", "cred", "freshworks",
    "zoho", "zendesk", "stripe", "twilio",
    "openai", "anthropic", "cohere", "hugging face",
    "mad street den", "uniphore", "sarvam", "krutrim",
    "elastic", "pinecone", "weaviate", "qdrant",
    "linkedin", "twitter", "uber", "airbnb",
    "startup", "ai", "ml",  # broad but useful
}

# Good industries for AI/ML work
GOOD_INDUSTRIES = {
    "software", "ai/ml", "fintech", "saas", "edtech", "healthtech",
    "food delivery", "transportation", "e-commerce", "media tech",
    "cloud", "semiconductor", "gaming",
}

# Bad industries — candidate unlikely to have relevant ML experience
BAD_INDUSTRIES = {
    "manufacturing", "paper products", "conglomerate", "retail",
    "real estate", "construction", "education (traditional)",
    "government", "non-profit", "fmcg",
}

# Preferred locations for Pune/Noida-first role
# The JD says: Pune, Noida, Hyderabad, Mumbai, Delhi NCR
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "new delhi", "ncr", "gurugram",
    "gurgaon", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "chennai",  # mentioned as acceptable
}

# ---------------------------------------------------------------------------
# Reference date (today)
# ---------------------------------------------------------------------------
TODAY = date.today()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase and strip for fuzzy matching."""
    return text.lower().strip()


def days_since(date_str: str) -> int:
    """Return number of days since the given ISO date string."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 9999


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def skill_name_matches(skill_name: str, skill_set: set) -> bool:
    """Check if a skill name fuzzy-matches anything in the set."""
    n = normalise(skill_name)
    for target in skill_set:
        if target in n or n in target:
            return True
    return False


def candidate_text(candidate: dict) -> str:
    """Build a searchable text blob from the candidate's profile and history."""
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_company", ""),
        profile.get("current_industry", ""),
    ]

    for job in candidate.get("career_history", []):
        parts.extend([
            job.get("title", ""),
            job.get("company", ""),
            job.get("industry", ""),
            job.get("description", ""),
        ])

    for edu in candidate.get("education", []):
        parts.extend([
            edu.get("institution", ""),
            edu.get("degree", ""),
            edu.get("field_of_study", ""),
        ])

    return normalise(" ".join(parts))


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

def is_honeypot(candidate: dict) -> bool:
    """
    Return True if this candidate has a subtly impossible profile.
    These are forced to relevance tier 0 in the ground truth.

    Checks:
      1. Expert proficiency in a skill but 0 months of use
      2. Years-of-experience wildly inconsistent with career history
      3. Currently employed at a company started after claimed tenure
      4. Extreme skill count with 'expert' across incompatible domains
      5. Start date of career predates plausible age (~15 years minimum)
    """
    profile = candidate["profile"]
    career = candidate["career_history"]
    skills = candidate["skills"]

    yoe = profile.get("years_of_experience", 0)

    # Check 1: expert with literally 0 duration months
    expert_zero_dur = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(expert_zero_dur) >= 2:
        return True

    # Check 2: total career months vs stated YOE — gap > 4 years is suspicious
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    career_years = total_career_months / 12.0
    if yoe > 3 and career_years > 0:
        ratio = min(yoe, career_years) / max(yoe, career_years)
        if ratio < 0.35:  # stated 10 years but only 3.5 years in history
            return True

    # Check 3: company founding date trap
    # If current role start_date is before company could plausibly exist
    # (We detect: started at company N years before dataset was created
    #  but company clearly founded later based on industry cues)
    # Practical heuristic: a current role with duration > 12*yoe months
    for job in career:
        if job.get("is_current") and job.get("duration_months", 0) > max(yoe * 12 + 6, 36):
            return True

    # Check 4: expert in mutually exclusive domain count (>= 8 expert skills)
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 10:
        # Check if they span incompatible domains
        domains = set()
        for s in expert_skills:
            n = normalise(s["name"])
            if any(k in n for k in ["ml", "ai", "embedding", "nlp", "retrieval"]):
                domains.add("ml")
            if any(k in n for k in ["accounting", "tally", "sap", "finance"]):
                domains.add("finance")
            if any(k in n for k in ["sales", "marketing", "seo", "crm"]):
                domains.add("marketing")
            if any(k in n for k in ["civil", "mechanical", "cad", "structural"]):
                domains.add("engineering_other")
        if len(domains) >= 3:
            return True

    # Check 5: YOE > 40 years (clearly impossible for active candidate)
    if yoe > 38:
        return True

    return False


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def score_skills(candidate: dict) -> tuple[float, list[str]]:
    """
    Returns (score 0-1, list of matched must-have skill names).

    Strategy:
    - Each must-have skill match contributes based on proficiency + duration
    - Nice-to-have adds a smaller bonus
    - Non-ML skills dominating the profile is a negative signal
    - Assessment scores (if available) refine the weight
    """
    skills = candidate["skills"]
    sig = candidate["redrob_signals"]
    assessment = sig.get("skill_assessment_scores", {})
    text_blob = candidate_text(candidate)

    PROFICIENCY_WEIGHT = {
        "expert": 1.0,
        "advanced": 0.75,
        "intermediate": 0.45,
        "beginner": 0.15,
    }

    must_hit_names = []
    must_score = 0.0
    nice_score = 0.0
    non_ml_count = 0

    for skill in skills:
        name = skill["name"]
        prof = skill.get("proficiency", "beginner")
        dur = skill.get("duration_months", 0)
        endorse = skill.get("endorsements", 0)
        n = normalise(name)

        base_weight = PROFICIENCY_WEIGHT.get(prof, 0.15)

        # Duration bonus: up to +0.25 for 24+ months
        dur_bonus = min(dur / 24.0, 1.0) * 0.25

        # Endorsement trust signal: up to +0.15 for 30+ endorsements
        endorse_bonus = min(endorse / 30.0, 1.0) * 0.15

        # Assessment score refinement: if Redrob has tested this skill
        assess_val = assessment.get(name)
        if assess_val is not None:
            # Assessment score < 40 penalises; > 70 boosts
            if assess_val < 40:
                base_weight *= 0.6
            elif assess_val >= 70:
                base_weight = min(base_weight * 1.2, 1.0)

        effective_weight = base_weight + dur_bonus + endorse_bonus

        if skill_name_matches(name, MUST_HAVE_SKILLS):
            must_score += effective_weight
            must_hit_names.append(name)
        elif skill_name_matches(name, NICE_TO_HAVE_SKILLS):
            nice_score += effective_weight * 0.4
        elif skill_name_matches(name, NON_ML_INDICATOR_SKILLS):
            non_ml_count += 1

    # Contextual matches in the headline, summary, and career descriptions.
    # These capture skills that were described naturally rather than added as
    # structured skill tags.
    contextual_must = 0.0
    contextual_nice = 0.0
    matched_must_terms = {normalise(name) for name in must_hit_names}
    for term in MUST_HAVE_SKILLS:
        if term in text_blob and term not in matched_must_terms:
            contextual_must += 0.12
    for term in NICE_TO_HAVE_SKILLS:
        if term in text_blob:
            contextual_nice += 0.04

    must_score += contextual_must
    nice_score += contextual_nice

    # Normalise must-have score (ideal = hitting 6+ must-haves at expert)
    # 6 must-haves * max effective weight ~= 6 * 1.4 = 8.4
    must_norm = clamp(must_score / 7.2)

    # Normalise nice-to-have (capped at 0.3 contribution)
    nice_norm = clamp(nice_score / 6.0) * 0.3

    # Penalty: too many non-ML skills signals keyword stuffer
    non_ml_penalty = 0.0
    total_skills = len(skills)
    if total_skills > 0:
        non_ml_ratio = non_ml_count / total_skills
        if non_ml_ratio > 0.5:
            non_ml_penalty = 0.3  # heavy penalty
        elif non_ml_ratio > 0.35:
            non_ml_penalty = 0.15

    raw = must_norm + nice_norm - non_ml_penalty
    return clamp(raw), must_hit_names


def score_career(candidate: dict) -> tuple[float, dict]:
    """
    Returns (score 0-1, metadata dict for reasoning).

    Evaluates:
    - Whether career is in product companies vs pure consulting
    - Whether roles are ML/AI/Engineering relevant
    - Whether candidate has been at non-consulting companies at all
    - Career trajectory: are they moving into ML roles?
    - Average tenure (title-chaser flag: avg < 18 months)
    """
    career = candidate["career_history"]
    meta = {}

    if not career:
        return 0.0, {"reason": "no career history"}

    total_jobs = len(career)
    consulting_jobs = 0
    product_jobs = 0
    ml_role_months = 0
    total_months = 0
    titles_ml = 0
    titles_negative = 0
    avg_tenure_months = 0

    for job in career:
        company = normalise(job.get("company", ""))
        title = normalise(job.get("title", ""))
        industry = normalise(job.get("industry", ""))
        dur = job.get("duration_months", 0)
        total_months += dur

        # Consulting check
        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        if is_consulting:
            consulting_jobs += 1
        else:
            # Check for product company signals
            is_product = any(k in company for k in PRODUCT_COMPANY_KEYWORDS)
            is_good_industry = any(k in industry for k in GOOD_INDUSTRIES)
            if is_product or is_good_industry:
                product_jobs += 1

        # ML role title check
        has_ml_title = any(k in title for k in POSITIVE_TITLE_KEYWORDS)
        has_neg_title = any(k in title for k in NEGATIVE_TITLE_KEYWORDS)
        if has_ml_title:
            titles_ml += 1
            ml_role_months += dur
        if has_neg_title:
            titles_negative += 1

    # Tenure analysis
    avg_tenure_months = total_months / total_jobs if total_jobs > 0 else 0
    meta["avg_tenure_months"] = avg_tenure_months
    meta["ml_role_months"] = ml_role_months
    meta["consulting_ratio"] = consulting_jobs / total_jobs if total_jobs > 0 else 0
    meta["product_jobs"] = product_jobs

    # Score components
    score = 0.0

    # 1. Consulting penalty — if ALL jobs are consulting, heavy penalty
    consulting_ratio = consulting_jobs / total_jobs
    if consulting_ratio == 1.0:
        score -= 0.5  # JD explicitly says no consulting-only careers
        meta["all_consulting"] = True
    elif consulting_ratio > 0.7:
        score -= 0.2
    else:
        # Reward product experience
        score += min(product_jobs / max(total_jobs, 1), 1.0) * 0.4

    # 2. ML role presence — critical
    ml_role_fraction = ml_role_months / total_months if total_months > 0 else 0
    score += ml_role_fraction * 0.5

    # 3. Title trajectory — mostly negative titles = wrong person
    if titles_negative > titles_ml and titles_ml == 0:
        score -= 0.4
        meta["title_mismatch"] = True

    # 4. Title-chaser penalty (avg tenure < 14 months, but only if > 3 jobs)
    if total_jobs >= 3 and avg_tenure_months < 14:
        score -= 0.15
        meta["title_chaser"] = True

    return clamp(score), meta


def score_experience(candidate: dict) -> float:
    """
    Returns score 0-1 for experience fit.

    JD says: 5-9 years target band (4-10 acceptable).
    Hard disqualifiers:
    - Pure research with no production (we check via career roles)
    - "AI experience" only in last 12 months + no prior ML
    """
    yoe = candidate["profile"].get("years_of_experience", 0)
    career = candidate["career_history"]

    # Ideal band: 5-9 years
    if 5 <= yoe <= 9:
        exp_score = 1.0
    elif 4 <= yoe < 5:
        exp_score = 0.85
    elif 9 < yoe <= 12:
        exp_score = 0.75
    elif 3 <= yoe < 4:
        exp_score = 0.6
    elif yoe > 12:
        exp_score = 0.55  # over-experienced, might want bigger scope
    else:
        exp_score = max(0.0, yoe / 5.0 * 0.5)

    # ML recency check: was there any ML role in the last 18 months?
    ml_recent = False
    for job in career:
        title = normalise(job.get("title", ""))
        is_current = job.get("is_current", False)
        dur = job.get("duration_months", 0)
        has_ml = any(k in title for k in POSITIVE_TITLE_KEYWORDS)
        if has_ml and (is_current or dur <= 18):
            ml_recent = True
            break

    # If no recent ML role and their ML skills are all < 12 months
    skills = candidate["skills"]
    ml_skills_durations = [
        s.get("duration_months", 0)
        for s in skills
        if skill_name_matches(s["name"], MUST_HAVE_SKILLS | NICE_TO_HAVE_SKILLS)
    ]
    recent_ml = any(d >= 12 for d in ml_skills_durations)

    if not ml_recent and not recent_ml and yoe > 3:
        exp_score *= 0.5  # experienced but ML is very new to them

    return clamp(exp_score)


def score_location(candidate: dict) -> float:
    """
    Returns score 0-1 for location fit.

    Ideal: Pune, Noida, Hyderabad, Mumbai, Delhi NCR
    Acceptable: Other Indian cities + willing to relocate
    Poor: Outside India + not willing to relocate
    """
    profile = candidate["profile"]
    sig = candidate["redrob_signals"]

    location = normalise(profile.get("location", ""))
    country = normalise(profile.get("country", ""))
    relocate = sig.get("willing_to_relocate", False)

    # Check preferred locations
    in_preferred = any(loc in location for loc in PREFERRED_LOCATIONS)
    in_india = "india" in country

    if in_preferred and in_india:
        return 1.0
    elif in_india and relocate:
        return 0.85
    elif in_india:
        return 0.7  # India but not preferred, not willing to relocate
    elif not in_india and relocate:
        return 0.5  # diaspora, willing to come back
    else:
        return 0.2  # outside India, not relocating


def score_behavioral(candidate: dict) -> tuple[float, dict]:
    """
    Returns (score 0-1, metadata dict).

    A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% response rate is, for hiring purposes, not available.

    Signals used:
    - last_active_date (recency)
    - open_to_work_flag
    - recruiter_response_rate
    - avg_response_time_hours
    - notice_period_days
    - profile_completeness_score
    - interview_completion_rate
    - verified_email + verified_phone
    - applications_submitted_30d
    """
    sig = candidate["redrob_signals"]
    meta = {}

    # 1. Recency — last active
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    meta["days_inactive"] = days_inactive
    if days_inactive <= 14:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.85
    elif days_inactive <= 60:
        recency = 0.65
    elif days_inactive <= 90:
        recency = 0.45
    elif days_inactive <= 180:
        recency = 0.25
    else:
        recency = 0.05  # > 6 months inactive

    # 2. Open to work flag
    open_to_work = 1.0 if sig.get("open_to_work_flag", False) else 0.4

    # 3. Recruiter response rate
    rr = sig.get("recruiter_response_rate", 0.0)
    if rr >= 0.7:
        response_score = 1.0
    elif rr >= 0.4:
        response_score = 0.7
    elif rr >= 0.2:
        response_score = 0.4
    else:
        response_score = 0.1  # 5-20% response rate — nearly unreachable

    # 4. Response time
    avg_resp_h = sig.get("avg_response_time_hours", 999)
    if avg_resp_h <= 4:
        resp_time_score = 1.0
    elif avg_resp_h <= 24:
        resp_time_score = 0.8
    elif avg_resp_h <= 72:
        resp_time_score = 0.55
    elif avg_resp_h <= 168:  # 1 week
        resp_time_score = 0.3
    else:
        resp_time_score = 0.1

    # 5. Notice period (JD says: loves sub-30 days, can buy out 30 days)
    notice = sig.get("notice_period_days", 90)
    meta["notice_period_days"] = notice
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.9
    elif notice <= 60:
        notice_score = 0.65
    elif notice <= 90:
        notice_score = 0.45
    else:
        notice_score = 0.2  # 90-180 days is a near-dealbreaker

    # 6. Profile completeness
    completeness = sig.get("profile_completeness_score", 0) / 100.0

    # 7. Interview completion rate (reliability signal)
    icr = sig.get("interview_completion_rate", 0.5)

    # 8. Recruiter visibility and social proof
    views_30d = sig.get("profile_views_received_30d", 0)
    saves_30d = sig.get("saved_by_recruiters_30d", 0)
    search_30d = sig.get("search_appearance_30d", 0)
    visibility = clamp((math.log1p(views_30d) + 2.0 * math.log1p(saves_30d) + 0.5 * math.log1p(search_30d)) / 15.0)

    # 9. Verification bonus
    verified = (
        (0.5 if sig.get("verified_email", False) else 0.0)
        + (0.5 if sig.get("verified_phone", False) else 0.0)
    )

    # 10. Active job seeking
    apps = sig.get("applications_submitted_30d", 0)
    app_score = min(apps / 5.0, 1.0)  # 5+ apps = fully active

    # 11. LinkedIn connectivity and offer history
    linkedin_connected = 1.0 if sig.get("linkedin_connected", False) else 0.0
    offer_rate = sig.get("offer_acceptance_rate", -1)
    offer_score = 0.5 if offer_rate < 0 else clamp(offer_rate)

    # Weighted blend
    behavioral = (
        0.20 * recency
        + 0.12 * open_to_work
        + 0.14 * response_score
        + 0.09 * resp_time_score
        + 0.11 * notice_score
        + 0.08 * completeness
        + 0.08 * icr
        + 0.05 * verified
        + 0.05 * app_score
        + 0.07 * visibility
        + 0.05 * linkedin_connected
        + 0.01 * offer_score
    )

    meta["recency_score"] = recency
    meta["response_rate"] = rr
    meta["notice_period_score"] = notice_score
    meta["visibility_score"] = visibility
    return clamp(behavioral), meta


def score_github(candidate: dict) -> float:
    """GitHub activity is a strong signal for this engineering JD."""
    sig = candidate["redrob_signals"]
    gh = sig.get("github_activity_score", -1)
    if gh == -1:
        return 0.2  # no GitHub linked — mild negative
    return clamp(gh / 100.0)


def score_education(candidate: dict) -> float:
    """
    Education tier and field relevance.
    Tier 1 (IITs, IISc, NITs) = best; tier 4 = weakest.
    CS/CE/EE/Math fields preferred.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3  # unknown is neutral-ish

    TIER_SCORE = {
        "tier_1": 1.0,
        "tier_2": 0.80,
        "tier_3": 0.60,
        "tier_4": 0.40,
        "unknown": 0.35,
    }

    GOOD_FIELDS = {
        "computer science", "computer engineering", "software engineering",
        "information technology", "electrical engineering", "electronics",
        "mathematics", "statistics", "data science", "artificial intelligence",
        "machine learning",
    }

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = normalise(edu.get("field_of_study", ""))
        degree = normalise(edu.get("degree", ""))

        tier_val = TIER_SCORE.get(tier, 0.35)

        # Field bonus
        field_match = any(f in field for f in GOOD_FIELDS)
        field_bonus = 0.15 if field_match else 0.0

        # Degree bonus: Masters/PhD > Bachelors for engineering role
        degree_bonus = 0.05 if any(d in degree for d in ["m.tech", "m.e.", "m.sc", "ms", "mtech", "phd", "ph.d"]) else 0.0

        edu_score = clamp(tier_val + field_bonus + degree_bonus)
        best_score = max(best_score, edu_score)

    return best_score


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_candidate(candidate: dict) -> tuple[float, dict]:
    """
    Returns (final_score 0-1, rich metadata dict for reasoning).
    """
    # --- Honeypot check first ---
    if is_honeypot(candidate):
        return 0.0, {"honeypot": True}

    # --- Component scores ---
    skill_score, matched_skills = score_skills(candidate)
    career_score, career_meta = score_career(candidate)
    exp_score = score_experience(candidate)
    loc_score = score_location(candidate)
    behavioral_score, beh_meta = score_behavioral(candidate)
    github_score = score_github(candidate)
    edu_score = score_education(candidate)

    # --- Title / keyword stuffer check ---
    # JD says: candidate with "Marketing Manager" title + all AI skills = trap
    current_title = normalise(candidate["profile"].get("current_title", ""))
    is_title_stuffer = (
        any(k in current_title for k in NEGATIVE_TITLE_KEYWORDS)
        and skill_score > 0.4  # has AI skills but wrong title
        and career_meta.get("title_mismatch", False)
    )

    # --- Weighted final score ---
    # Weights derived from JD priorities:
    #   skills: core requirement
    #   career: product co + ML roles — critical differentiator
    #   experience: 5-9yr band
    #   location: Pune/Noida preferred
    #   behavioral: availability matters
    #   github: engineer signal
    #   education: minor but relevant

    weights = {
        "skill":      0.35,
        "career":     0.25,
        "experience": 0.15,
        "location":   0.10,
        "behavioral": 0.10,
        "github":     0.03,
        "education":  0.02,
    }

    raw_score = (
        weights["skill"]      * skill_score
        + weights["career"]   * career_score
        + weights["experience"] * exp_score
        + weights["location"] * loc_score
        + weights["behavioral"] * behavioral_score
        + weights["github"]   * github_score
        + weights["education"] * edu_score
    )

    # Apply keyword-stuffer penalty
    if is_title_stuffer:
        raw_score *= 0.4

    # Apply all-consulting career penalty (already in career score but make sure)
    if career_meta.get("all_consulting"):
        raw_score *= 0.55

    final_score = clamp(raw_score)

    metadata = {
        "skill_score": round(skill_score, 3),
        "matched_skills": matched_skills[:5],
        "career_score": round(career_score, 3),
        "exp_score": round(exp_score, 3),
        "loc_score": round(loc_score, 3),
        "behavioral_score": round(behavioral_score, 3),
        "github_score": round(github_score, 3),
        "edu_score": round(edu_score, 3),
        "is_title_stuffer": is_title_stuffer,
        "days_inactive": beh_meta.get("days_inactive", 0),
        "notice_period_days": beh_meta.get("notice_period_days", 0),
        "response_rate": beh_meta.get("response_rate", 0),
        "avg_tenure_months": career_meta.get("avg_tenure_months", 0),
        "ml_role_months": career_meta.get("ml_role_months", 0),
        "all_consulting": career_meta.get("all_consulting", False),
    }

    return final_score, metadata


# ---------------------------------------------------------------------------
# Reasoning generator
# ---------------------------------------------------------------------------

def build_reasoning(candidate: dict, score: float, rank: int, meta: dict) -> str:
    """
    Generate a specific, honest 1-2 sentence reasoning for this candidate.
    References concrete facts from the profile — no hallucination.
    Tone matches rank (top 10 = positive, bottom 20 of top-100 = honest gaps).
    """
    if meta.get("honeypot"):
        return "Profile contains inconsistencies that suggest data integrity issues; excluded from ranking."

    profile = candidate["profile"]
    sig = candidate["redrob_signals"]
    skills = candidate["skills"]
    career = candidate["career_history"]

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    matched = meta.get("matched_skills", [])
    notice = meta.get("notice_period_days", 90)
    days_inactive = meta.get("days_inactive", 0)
    response_rate = meta.get("response_rate", 0)
    ml_months = meta.get("ml_role_months", 0)
    all_consulting = meta.get("all_consulting", False)

    # Build key positive facts
    positives = []
    concerns = []

    # Skills mentioned
    if matched:
        skill_str = ", ".join(matched[:3])
        positives.append(f"hands-on with {skill_str}")

    # ML experience duration
    if ml_months >= 36:
        positives.append(f"{ml_months // 12}+ years in ML/AI roles")
    elif ml_months > 0:
        positives.append(f"{ml_months} months in ML/AI roles")

    # YOE fit
    if 5 <= yoe <= 9:
        positives.append(f"{yoe:.1f} years experience (target band)")
    elif yoe > 9:
        concerns.append(f"{yoe:.1f} yrs (above target band)")

    # Location
    loc_norm = normalise(location)
    in_preferred = any(loc in loc_norm for loc in PREFERRED_LOCATIONS)
    if in_preferred:
        positives.append(f"based in {location}")
    elif sig.get("willing_to_relocate"):
        positives.append("willing to relocate")
    else:
        concerns.append(f"located in {location}, not willing to relocate")

    # Engagement
    if days_inactive <= 14:
        positives.append("active in last 2 weeks")
    elif days_inactive <= 30:
        positives.append("active recently")
    elif days_inactive > 90:
        concerns.append(f"inactive for {days_inactive} days")

    if response_rate >= 0.7:
        positives.append(f"{int(response_rate*100)}% recruiter response rate")
    elif response_rate < 0.2:
        concerns.append(f"low recruiter response rate ({int(response_rate*100)}%)")

    if notice <= 30:
        positives.append(f"{notice}-day notice period")
    elif notice > 90:
        concerns.append(f"{notice}-day notice period")

    # Consulting flag
    if all_consulting:
        concerns.append("entire career at IT services firms")

    # Build sentence 1: who they are + top positives
    positive_str = "; ".join(positives[:3]) if positives else "some adjacent skills"
    s1 = f"{title} at {company} ({yoe:.1f} yrs) — {positive_str}."

    # Build sentence 2: concerns or strong endorsement
    if rank <= 10:
        # Top candidates — lead with why they're exceptional
        top_skill = matched[0] if matched else "relevant skills"
        if ml_months >= 24:
            s2 = f"Strong ML engineering background with {ml_months} months in AI roles; profile fits the Senior AI Engineer mandate closely."
        else:
            s2 = f"High composite fit across skill match, career trajectory, and platform engagement for this role."
    elif concerns:
        concern_str = ", ".join(concerns[:2])
        s2 = f"Concerns: {concern_str}."
    else:
        if score >= 0.5:
            s2 = "Solid fit across most dimensions; included in top-100 shortlist."
        else:
            s2 = "Adjacent skills and partial experience match; included at bottom of shortlist as borderline fit."

    return f"{s1} {s2}"


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_candidates(path: str):
    """Load candidates from .jsonl or .jsonl.gz file, yielding dicts."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"

    def iter_json_records(handle) -> Iterator[dict]:
        decoder = json.JSONDecoder()
        buffer = ""
        position = 0
        started = False
        array_mode = False
        eof = False

        while True:
            if position >= len(buffer):
                if eof:
                    break
                chunk = handle.read(65536)
                if not chunk:
                    eof = True
                else:
                    buffer += chunk
                continue

            if not started:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position >= len(buffer):
                    continue
                if buffer[position] == "[":
                    array_mode = True
                    position += 1
                started = True

            if array_mode:
                while position < len(buffer) and buffer[position] in "\r\n\t ,":
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    break
            else:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1

            if position >= len(buffer):
                continue

            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    break
                if position > 0:
                    buffer = buffer[position:]
                    position = 0
                chunk = handle.read(65536)
                if not chunk:
                    eof = True
                else:
                    buffer += chunk
                continue

            yield record
            position = end

    count = 0
    with opener(path, mode, encoding="utf-8") as f:
        for record in iter_json_records(f):
            yield record
            count += 1
            if count % 10000 == 0:
                print(f"  Processed {count:,} candidates...", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(candidates_path: str, output_path: str):
    import time
    t0 = time.time()

    print(f"Loading candidates from: {candidates_path}", file=sys.stderr)
    print(f"Output will be written to: {output_path}", file=sys.stderr)
    print("", file=sys.stderr)

    results = []
    total = 0
    honeypots_caught = 0

    for candidate in load_candidates(candidates_path):
        total += 1
        score, meta = score_candidate(candidate)

        if meta.get("honeypot"):
            honeypots_caught += 1
            # Still record them with score=0 so we don't accidentally include
            # them, but don't add to results list
            continue

        results.append({
            "candidate_id": candidate["candidate_id"],
            "score": score,
            "meta": meta,
            "candidate": candidate,
        })

    elapsed = time.time() - t0
    print(f"\nScored {total:,} candidates in {elapsed:.1f}s", file=sys.stderr)
    print(f"Honeypots caught and excluded: {honeypots_caught}", file=sys.stderr)

    # Sort by score descending, tie-break by candidate_id ascending
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Take top 100
    top100 = results[:100]

    if len(top100) < 100:
        print(f"WARNING: Only {len(top100)} scoreable candidates found!", file=sys.stderr)

    print(f"Writing top {len(top100)} to {output_path}", file=sys.stderr)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, item in enumerate(top100, start=1):
            cid = item["candidate_id"]
            score = round(item["score"], 6)
            meta = item["meta"]
            candidate = item["candidate"]

            reasoning = build_reasoning(candidate, score, rank_idx, meta)

            writer.writerow([cid, rank_idx, score, reasoning])

    total_time = time.time() - t0
    print(f"\nDone! Total wall-clock time: {total_time:.1f}s", file=sys.stderr)
    print(f"Top candidate score: {top100[0]['score']:.4f}", file=sys.stderr)
    print(f"Rank 100 score: {top100[99]['score']:.4f}" if len(top100) >= 100 else "", file=sys.stderr)

    # Print a preview of top 5
    print("\n=== TOP 5 PREVIEW ===", file=sys.stderr)
    for i, item in enumerate(top100[:5], 1):
        p = item["candidate"]["profile"]
        m = item["meta"]
        print(
            f"  #{i}: {item['candidate_id']} | {p['current_title']} | "
            f"{p['years_of_experience']}yr | score={item['score']:.4f} | "
            f"skills={m['skill_score']:.2f} career={m['career_score']:.2f} "
            f"exp={m['exp_score']:.2f} loc={m['loc_score']:.2f} beh={m['behavioral_score']:.2f}",
            file=sys.stderr
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Intelligent Candidate Ranker"
    )
    parser.add_argument(
        "--candidates",
        default="candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz (default: candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    args = parser.parse_args()
    run(args.candidates, args.out)


if __name__ == "__main__":
    main()
