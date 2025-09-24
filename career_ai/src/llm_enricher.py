"""LLM enrichment utilities compatible with OpenAI/Gemini style APIs.

Safe-by-default: returns structured placeholders unless API keys are provided
or an actual provider call succeeds.

Currently implemented provider: OpenAI Chat Completions (JSON response).
Gemini path falls back to placeholder unless extended.
"""

from __future__ import annotations

import os
from typing import Dict, List, Any

import json
import requests
try:
    from dotenv import load_dotenv
    # Load from current working directory first
    load_dotenv()
    # Also attempt repo-level and project-level .env files
    _here = os.path.dirname(__file__)
    _repo_env = os.path.normpath(os.path.join(_here, "..", "..", ".env"))
    _proj_env = os.path.normpath(os.path.join(_here, "..", ".env"))
    if os.path.exists(_repo_env):
        load_dotenv(_repo_env)
    if os.path.exists(_proj_env):
        load_dotenv(_proj_env)
except Exception:
    # dotenv is optional; continue without it
    pass


class LLMEnricher:
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.api_key = os.getenv("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY", "")

    def get_career_guidance(self, predicted_career: str, user_profile: Dict[str, Any], ml_predictions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = self._build_career_prompt(predicted_career, user_profile, ml_predictions)

        if self.provider == "openai" and self.api_key:
            try:
                data = self._openai_chat_json(prompt)
                # Expecting improved structure: summary, recommendations, next_steps
                return {
                    "summary": data.get("summary", f"Guidance for {predicted_career}"),
                    "recommendations": data.get("recommendations", []),
                    "next_steps": data.get("next_steps", []),
                    "_provider": "openai",
                }
            except Exception as ex:
                return self._placeholder_guidance_improved(predicted_career, user_profile, ml_predictions, error=str(ex))

        if self.provider == "gemini" and self.api_key:
            try:
                data = self._gemini_generate_json(prompt)
                return {
                    "summary": data.get("summary", f"Guidance for {predicted_career}"),
                    "recommendations": data.get("recommendations", []),
                    "next_steps": data.get("next_steps", []),
                    "_provider": "gemini",
                }
            except Exception as ex:
                return self._placeholder_guidance_improved(predicted_career, user_profile, ml_predictions, error=str(ex))

        return self._placeholder_guidance_improved(predicted_career, user_profile, ml_predictions)

    def get_skill_recommendations(self, current_skills: List[str], target_career: str) -> Dict[str, Any]:
        if self.provider == "openai" and self.api_key:
            try:
                prompt = (
                    "You are a career upskilling assistant.\n"
                    f"Target Career: {target_career}\n"
                    f"Existing Skills: {json.dumps(current_skills)}\n"
                    "Return a JSON object with keys: target_career, existing_skills,"
                    " missing_skills (array), resources (array), timeline (string)."
                )
                data = self._openai_chat_json(prompt)
                return {
                    "target_career": data.get("target_career", target_career),
                    "existing_skills": data.get("existing_skills", current_skills),
                    "missing_skills": data.get("missing_skills", []),
                    "resources": data.get("resources", []),
                    "timeline": data.get("timeline", ""),
                }
            except Exception:
                return self._placeholder_skills(current_skills, target_career)

        if self.provider == "gemini" and self.api_key:
            try:
                prompt = (
                    "You are a career upskilling assistant.\n"
                    f"Target Career: {target_career}\n"
                    f"Existing Skills: {json.dumps(current_skills)}\n"
                    "Return a JSON object with keys: target_career, existing_skills,"
                    " missing_skills (array), resources (array), timeline (string)."
                )
                data = self._gemini_generate_json(prompt)
                return {
                    "target_career": data.get("target_career", target_career),
                    "existing_skills": data.get("existing_skills", current_skills),
                    "missing_skills": data.get("missing_skills", []),
                    "resources": data.get("resources", []),
                    "timeline": data.get("timeline", ""),
                }
            except Exception:
                return self._placeholder_skills(current_skills, target_career)

        return self._placeholder_skills(current_skills, target_career)

    def _build_career_prompt(self, career: str, profile: Dict[str, Any], ml_predictions: List[Dict[str, Any]] = None) -> str:
        prompt = (
            "You are an expert career counselor AI.\n"
            f"Predicted Career: {career}\n"
            f"User Profile: {json.dumps(profile)}\n"
        )
        if ml_predictions:
            prompt += f"ML Career Predictions: {json.dumps(ml_predictions)}\n"
        prompt += (
            "Generate a JSON object with these keys: summary (string), recommendations (array of objects with keys: career, why_fit, courses), next_steps (array of strings)."
        )
        return prompt
    def _placeholder_guidance_improved(self, predicted_career: str, user_profile: Dict[str, Any], ml_predictions: List[Dict[str, Any]] = None, error: str | None = None) -> Dict[str, Any]:
        # Use ML predictions if available, else fallback to predicted_career
        recommendations = []
        if ml_predictions:
            for pred in ml_predictions:
                recommendations.append({
                    "career": pred["career"],
                    "why_fit": f"Recommended based on profile and ML probability {pred['probability']}",
                    "courses": ["Course 1", "Course 2"]
                })
            summary = f"Based on your profile and interests, top careers are: {', '.join([p['career'] for p in ml_predictions])}."
        else:
            recommendations = [{
                "career": predicted_career,
                "why_fit": "Recommended based on your profile.",
                "courses": ["Course 1", "Course 2"]
            }]
            summary = f"Guidance for {predicted_career}."
        out = {
            "summary": summary,
            "recommendations": recommendations,
            "next_steps": ["Explore recommended courses", "Build projects", "Apply for internships"],
            "_provider": self.provider,
        }
        if error:
            out["_error"] = error
        return out

    # -------------------------
    # Provider integrations
    # -------------------------

    def _openai_chat_json(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI Chat Completions API and expect JSON content.

        Uses response_format={"type": "json_object"} to enforce JSON output.
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Respond in compact JSON only."},
                {"role": "user", "content": prompt},
            ],
        }

        resp = requests.post(url, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as ex:
            raise RuntimeError(f"Unexpected OpenAI response: {data}") from ex

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # As a fallback, return an empty dict to let caller default fields
            return {}

    def _gemini_generate_json(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini generateContent endpoint and parse JSON from text.

        API: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
        """
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": (
                            "Respond with ONLY valid compact JSON. "
                            "If you include prose, the client will fail.\n" + prompt
                        )}
                    ],
                }
            ]
        }
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as ex:
            raise RuntimeError(f"Unexpected Gemini response: {data}") from ex
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON block crudely if wrapped
            import re
            m = re.search(r"\{[\s\S]*\}$", text.strip())
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            return {}

    # -------------------------
    # Fallbacks / placeholders
    # -------------------------

    def _placeholder_guidance(self, predicted_career: str, error: str | None = None) -> Dict[str, Any]:
        out = {
            "career": predicted_career,
            "fit_reason": "Based on your strengths and interests, this role aligns well.",
            "courses": ["Foundations", "Intermediate", "Advanced"],
            "day_in_life": "Typical responsibilities and tasks for the role.",
            "colleges": ["Sample College A", "Sample College B"],
            "roadmap": "Step-by-step plan to reach the role in 6-12 months.",
            "_provider": self.provider,
        }
        if error:
            out["_error"] = error
        return out

    def _placeholder_skills(self, current_skills: List[str], target_career: str) -> Dict[str, Any]:
        return {
            "target_career": target_career,
            "existing_skills": current_skills,
            "missing_skills": ["Skill A", "Skill B"],
            "resources": ["Course 1", "Course 2"],
            "timeline": "8-12 weeks",
        }