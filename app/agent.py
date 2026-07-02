import json
import logging
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from google.api_core.exceptions import ResourceExhausted
from app.config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_OUTPUT_TOKENS, MAX_RECOMMENDATIONS, TOP_K_RETRIEVAL
from app.schemas import Message, ChatResponse, Recommendation
from app.catalog import Assessment, format_assessment_context, get_primary_test_type
from app.retriever import CatalogRetriever
from app.prompts import (
    SYSTEM_PROMPT, EXTRACTION_PROMPT, RECOMMENDATION_PROMPT,
    CLARIFICATION_PROMPT, COMPARISON_PROMPT, REFUSAL_PROMPT
)
from app.guardrails import (
    detect_injection, detect_off_topic, validate_turn_count,
    validate_recommendations_in_catalog
)

logger = logging.getLogger(__name__)


class SHLAgent:
    def __init__(self, retriever: CatalogRetriever):
        self.retriever = retriever
        self.catalog_names = {a.name for a in retriever.catalog}
        self.system_prompt = SYSTEM_PROMPT
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
    
    def _format_conversation(self, messages: List[Message]) -> str:
        return "\n".join([f"{m.role}: {m.content}" for m in messages])
    
    def _call_llm(self, prompt: str) -> str:
        try:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        text = text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {text[:200]}")
            return {}
    
    def _extract_requirements(self, messages: List[Message]) -> Dict[str, Any]:
        """Pass 1: Extract structured requirements from conversation."""
        conversation = self._format_conversation(messages)
        prompt = EXTRACTION_PROMPT.format(conversation=conversation)
        response_text = self._call_llm(prompt)
        result = self._parse_json_response(response_text)
        return result
    
    def _generate_recommendations(self, messages: List[Message], requirements: Dict[str, Any], assessments: List[Assessment]) -> ChatResponse:
        """Pass 2: Generate recommendations using retrieved assessments."""
        conversation = self._format_conversation(messages)
        assessments_context = "\n\n".join([format_assessment_context(a) for a in assessments[:MAX_RECOMMENDATIONS * 2]])
        
        prompt = RECOMMENDATION_PROMPT.format(
            role=requirements.get("role", "Not specified"),
            seniority=requirements.get("seniority", "Not specified"),
            skills=", ".join(requirements.get("skills", [])) or "Not specified",
            test_types=", ".join(requirements.get("test_type_preferences", [])) or "Any",
            intent_summary=requirements.get("user_intent_summary", ""),
            assessments_context=assessments_context,
            conversation=conversation,
        )
        
        response_text = self._call_llm(prompt)
        result = self._parse_json_response(response_text)
        
        recommended_names = result.get("recommended_names", [])
        validated_names = validate_recommendations_in_catalog(recommended_names, self.catalog_names)
        
        recommendations = []
        for name in validated_names[:MAX_RECOMMENDATIONS]:
            assessment = next((a for a in self.retriever.catalog if a.name == name), None)
            if assessment:
                recommendations.append(Recommendation(
                    name=assessment.name,
                    url=assessment.link,
                    test_type=get_primary_test_type(assessment),
                ))
        
        return ChatResponse(
            reply=result.get("reply", "Here are my recommendations based on your requirements."),
            recommendations=recommendations,
            end_of_conversation=result.get("end_of_conversation", False),
        )
    
    def _generate_clarification(self, messages: List[Message], requirements: Dict[str, Any]) -> ChatResponse:
        """Generate a clarifying question."""
        conversation = self._format_conversation(messages)
        prompt = CLARIFICATION_PROMPT.format(
            conversation=conversation,
            role=requirements.get("role", "Not specified"),
            seniority=requirements.get("seniority", "Not specified"),
            skills=", ".join(requirements.get("skills", [])) or "Not specified",
        )
        response_text = self._call_llm(prompt)
        result = self._parse_json_response(response_text)
        
        return ChatResponse(
            reply=result.get("reply", "Could you tell me more about the role you're hiring for?"),
            recommendations=[],
            end_of_conversation=False,
        )
    
    def _generate_comparison(self, messages: List[Message], requirements: Dict[str, Any]) -> ChatResponse:
        """Generate a comparison between assessments."""
        comparison_items = requirements.get("comparison_items", [])
        assessments = self.retriever.find_by_names(comparison_items)
        
        if not assessments:
            # Try to find by searching
            search_query = " ".join(comparison_items)
            assessments = self.retriever.retrieve(search_query, top_k=5)
        
        conversation = self._format_conversation(messages)
        assessments_context = "\n\n".join([format_assessment_context(a) for a in assessments])
        
        prompt = COMPARISON_PROMPT.format(
            assessments_context=assessments_context,
            conversation=conversation,
        )
        
        response_text = self._call_llm(prompt)
        result = self._parse_json_response(response_text)
        
        recommended_names = result.get("recommended_names", [])
        validated_names = validate_recommendations_in_catalog(recommended_names, self.catalog_names)
        
        recommendations = []
        for name in validated_names[:MAX_RECOMMENDATIONS]:
            assessment = next((a for a in self.retriever.catalog if a.name == name), None)
            if assessment:
                recommendations.append(Recommendation(
                    name=assessment.name,
                    url=assessment.link,
                    test_type=get_primary_test_type(assessment),
                ))
        
        return ChatResponse(
            reply=result.get("reply", "Here is a comparison of the assessments you asked about."),
            recommendations=recommendations,
            end_of_conversation=result.get("end_of_conversation", False),
        )

    def _build_retrieval_fallback(self, messages: List[Message], reason: str) -> ChatResponse:
        latest_user_msg = ""
        for msg in reversed(messages):
            if msg.role == "user":
                latest_user_msg = msg.content
                break

        assessments = self.retriever.retrieve(query=latest_user_msg, top_k=MAX_RECOMMENDATIONS)
        recommendations = []
        for assessment in assessments:
            recommendations.append(Recommendation(
                name=assessment.name,
                url=assessment.link,
                test_type=get_primary_test_type(assessment),
            ))

        if recommendations:
            reply = (
                "I’m temporarily unable to use the LLM right now, so I’m falling back to catalog search. "
                "Here are the closest SHL assessments I found."
            )
        else:
            reply = (
                "I’m temporarily unable to use the LLM right now and I couldn’t find a strong catalog match. "
                "Please try again shortly."
            )

        logger.warning("Using retrieval fallback because Gemini request failed: %s", reason)
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=False,
        )
    
    def process(self, messages: List[Message]) -> ChatResponse:
        """Main entry point: process a chat request and return a response."""
        try:
            # Check turn count
            if not validate_turn_count(messages):
                return ChatResponse(
                    reply="We've reached the conversation limit. Let me provide my best recommendations based on our discussion.",
                    recommendations=[],
                    end_of_conversation=True,
                )
            
            # Check the latest user message for guardrails
            latest_user_msg = ""
            for msg in reversed(messages):
                if msg.role == "user":
                    latest_user_msg = msg.content
                    break
            
            # Check for prompt injection
            if detect_injection(latest_user_msg):
                return ChatResponse(
                    reply=REFUSAL_PROMPT,
                    recommendations=[],
                    end_of_conversation=False,
                )
            
            # Check for off-topic
            if detect_off_topic(latest_user_msg) and len(messages) <= 1:
                return ChatResponse(
                    reply=REFUSAL_PROMPT,
                    recommendations=[],
                    end_of_conversation=False,
                )
            
            # Pass 1: Extract requirements
            requirements = self._extract_requirements(messages)
            
            if not requirements:
                return ChatResponse(
                    reply="I'd be happy to help you find the right SHL assessment. Could you tell me about the role you're hiring for?",
                    recommendations=[],
                    end_of_conversation=False,
                )
            
            # Check if off-topic per LLM analysis
            if requirements.get("is_off_topic", False):
                return ChatResponse(
                    reply=REFUSAL_PROMPT,
                    recommendations=[],
                    end_of_conversation=False,
                )
            
            # Handle comparison requests
            if requirements.get("is_comparison_request", False):
                return self._generate_comparison(messages, requirements)
            
            # If not enough context, ask clarifying question
            if not requirements.get("has_sufficient_context", False):
                return self._generate_clarification(messages, requirements)
            
            # Retrieve relevant assessments
            search_query = requirements.get("search_query", "")
            if not search_query:
                # Build query from extracted fields
                parts = []
                if requirements.get("role"):
                    parts.append(requirements["role"])
                if requirements.get("skills"):
                    parts.extend(requirements["skills"])
                if requirements.get("seniority"):
                    parts.append(requirements["seniority"])
                search_query = " ".join(parts)
            
            # Map test type preferences to filter codes
            filter_test_types = None
            type_prefs = requirements.get("test_type_preferences", [])
            if type_prefs:
                pref_map = {
                    "knowledge": "K", "personality": "P", "ability": "A",
                    "situational_judgment": "B", "competency": "C",
                    "simulation": "S", "exercise": "E", "development": "D",
                }
                filter_test_types = {pref_map.get(p.lower(), p) for p in type_prefs if p}
            
            # Map seniority to job level filter
            filter_job_levels = None
            seniority = requirements.get("seniority", "")
            if seniority:
                seniority_map = {
                    "entry": {"Entry-Level", "Graduate"},
                    "junior": {"Entry-Level", "Graduate"},
                    "graduate": {"Graduate"},
                    "mid": {"Mid-Professional", "Professional Individual Contributor"},
                    "senior": {"Mid-Professional", "Professional Individual Contributor", "Manager"},
                    "manager": {"Manager", "Front Line Manager", "Supervisor"},
                    "director": {"Director", "Executive"},
                    "executive": {"Executive", "Director"},
                }
                for key, levels in seniority_map.items():
                    if key in seniority.lower():
                        filter_job_levels = levels
                        break
            
            assessments = self.retriever.retrieve(
                query=search_query,
                top_k=TOP_K_RETRIEVAL,
                filter_job_levels=filter_job_levels,
                filter_test_types=filter_test_types if filter_test_types else None,
            )
            
            # If filters are too restrictive and we get no results, retry without filters
            if not assessments:
                assessments = self.retriever.retrieve(query=search_query, top_k=TOP_K_RETRIEVAL)
            
            # Pass 2: Generate recommendations
            return self._generate_recommendations(messages, requirements, assessments)

        except ResourceExhausted as exc:
            return self._build_retrieval_fallback(messages, str(exc))
        
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            return ChatResponse(
                reply="I apologize, but I encountered an issue processing your request. Could you please rephrase your question about SHL assessments?",
                recommendations=[],
                end_of_conversation=False,
            )

