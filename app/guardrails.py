import re
from typing import List
from app.schemas import Message


# Patterns that suggest prompt injection
INJECTION_PATTERNS = [
    r"ignore\s+(your|all|previous|above)\s+(instructions|rules|prompts)",
    r"pretend\s+(you\s+are|to\s+be|you're)",
    r"you\s+are\s+now",
    r"forget\s+(your|all|previous)\s+(instructions|rules)",
    r"act\s+as\s+(if|a|an)",
    r"new\s+instructions?",
    r"system\s*:\s*",
    r"\[system\]",
    r"override\s+(your|the)\s+(instructions|rules)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"dan\s+mode",
]

# Topics that are out of scope
OFF_TOPIC_PATTERNS = [
    r"\b(salary|compensation|pay\s+range|wage)\b",
    r"\b(legal|lawsuit|sue|attorney|lawyer)\b",
    r"\b(interview\s+tips?|interview\s+questions?)\b",
    r"\b(resume|cv|cover\s+letter)\b",
    r"\b(weather|sports|news|politics|religion)\b",
    r"\b(recipe|cook|food|restaurant)\b",
    r"\b(write\s+(me|a)\s+(poem|story|essay|code))\b",
]


def detect_injection(text: str) -> bool:
    """Check if the text contains prompt injection attempts."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def detect_off_topic(text: str) -> bool:
    """Check if the text is about an off-topic subject."""
    text_lower = text.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def validate_turn_count(messages: List[Message]) -> bool:
    """Check if conversation is within the 8-turn limit."""
    return len(messages) < 8


def validate_recommendations_in_catalog(names: List[str], catalog_names: set) -> List[str]:
    """Filter recommendations to only include items in the catalog."""
    validated = []
    for name in names:
        # Exact match
        if name in catalog_names:
            validated.append(name)
            continue
        # Case-insensitive match
        for cat_name in catalog_names:
            if name.lower() == cat_name.lower():
                validated.append(cat_name)
                break
    return validated
