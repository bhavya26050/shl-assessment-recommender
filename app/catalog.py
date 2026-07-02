import json
from dataclasses import dataclass
from typing import List, Optional, Dict


KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S"
}


@dataclass
class Assessment:
    entity_id: str
    name: str
    link: str
    job_levels: List[str]
    languages: List[str]
    duration: str
    remote: str
    adaptive: str
    description: str
    keys: List[str]
    test_type_codes: List[str]
    search_text: str


def load_catalog(path: str) -> List[Assessment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assessments = []
    for item in data:
        keys = item.get("keys", [])
        test_type_codes = [KEY_TO_CODE[k] for k in keys if k in KEY_TO_CODE]
        if not test_type_codes:
            test_type_codes = ["K"]  # fallback default
            
        job_levels = item.get("job_levels", [])
        languages = item.get("languages", [])
        description = item.get("description", "")
        name = item.get("name", "")
        
        # Concatenate metadata for embedding-based search
        search_parts = [
            f"Name: {name}",
            f"Description: {description}",
            f"Categories: {', '.join(keys)}",
            f"Job Levels: {', '.join(job_levels)}",
            f"Languages: {', '.join(languages)}"
        ]
        search_text = " | ".join(search_parts)
        
        assessments.append(Assessment(
            entity_id=item.get("entity_id", ""),
            name=name,
            link=item.get("link", ""),
            job_levels=job_levels,
            languages=languages,
            duration=item.get("duration", ""),
            remote=item.get("remote", ""),
            adaptive=item.get("adaptive", ""),
            description=description,
            keys=keys,
            test_type_codes=test_type_codes,
            search_text=search_text
        ))
    return assessments


def get_primary_test_type(assessment: Assessment) -> str:
    if assessment.test_type_codes:
        return assessment.test_type_codes[0]
    return "K"


def format_assessment_context(assessment: Assessment) -> str:
    """Format assessment details for the LLM prompt context."""
    types_str = ", ".join(assessment.keys)
    levels_str = ", ".join(assessment.job_levels)
    langs_str = ", ".join(assessment.languages)
    return (
        f"Name: {assessment.name}\n"
        f"URL: {assessment.link}\n"
        f"Type: {types_str}\n"
        f"Job Levels: {levels_str}\n"
        f"Languages: {langs_str}\n"
        f"Duration: {assessment.duration or 'Not specified'}\n"
        f"Description: {assessment.description}\n"
    )
