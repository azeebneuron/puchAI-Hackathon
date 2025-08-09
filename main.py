import asyncio
import json
import os
import re
from typing import Annotated, Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from uuid import uuid4
import base64
import io

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field

import httpx
from PIL import Image, ImageDraw, ImageFont

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

# Load environment variables
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER", "919998881729")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# Auth Provider for PuchAI
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# Pydantic Models for Structured LLM Outputs
class UserProfileSchema(BaseModel):
    name: str = Field(description="Full name of the person")
    location: str = Field(description="City or area where person lives")
    skills: List[str] = Field(description="List of skills in English")
    experience_years: int = Field(description="Years of work experience")
    job_preferences: List[str] = Field(description="Preferred job types or categories")
    availability: str = Field(description="Full-time, part-time, or flexible")
    preferred_language: str = Field(description="Language code like 'hi', 'en', 'ta'")

class JobSearchCriteria(BaseModel):
    job_types: List[str] = Field(description="Types of jobs being searched for")
    location: str = Field(description="Preferred work location")
    salary_min: Optional[int] = Field(description="Minimum salary expectation")
    salary_max: Optional[int] = Field(description="Maximum salary expectation")
    urgency: str = Field(description="immediate, within_week, flexible")
    work_type: str = Field(description="full_time, part_time, gig, contract")
    additional_requirements: List[str] = Field(description="Any special requirements")

class JobPostingSchema(BaseModel):
    title: str = Field(description="Job title or position")
    description: str = Field(description="Detailed job description")
    location: str = Field(description="Job location")
    salary_min: int = Field(description="Minimum salary offered")
    salary_max: int = Field(description="Maximum salary offered")
    job_type: str = Field(description="full_time, part_time, gig, contract")
    category: str = Field(description="Job category like security, cleaning, delivery")
    requirements: List[str] = Field(description="Required skills or qualifications")
    contact_info: str = Field(description="Phone number or contact details")
    urgency: str = Field(description="immediate, within_week, flexible")
    benefits: List[str] = Field(description="Any additional benefits offered")

class SafetyAnalysis(BaseModel):
    safety_level: str = Field(description="SAFE, MEDIUM_RISK, HIGH_RISK")
    risk_score: int = Field(description="Risk score from 0-100")
    scam_indicators: List[str] = Field(description="Specific red flags found")
    legitimate_indicators: List[str] = Field(description="Positive signs found")
    recommendation: str = Field(description="PROCEED, CAUTION, AVOID")
    explanation: str = Field(description="Detailed explanation of the analysis")

class SalaryInsights(BaseModel):
    salary_range_min: int = Field(description="Minimum typical salary for this role")
    salary_range_max: int = Field(description="Maximum typical salary for this role")
    market_factors: List[str] = Field(description="Factors affecting salary in this market")
    negotiation_tips: List[str] = Field(description="Tips for salary negotiation")
    growth_potential: str = Field(description="Career growth potential")
    demand_level: str = Field(description="HIGH, MEDIUM, LOW demand for this role")

# Data Models
@dataclass
class UserProfile:
    id: str
    phone: str
    profile_data: UserProfileSchema
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class JobPosting:
    id: str
    posting_data: JobPostingSchema
    posted_by: str
    verified: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

# In-memory storage
USERS: Dict[str, UserProfile] = {}
JOBS: Dict[str, JobPosting] = {}
APPLICATIONS: Dict[str, Dict] = {}

# AI Agent for intelligent processing
class JobKrantiAI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            # Initialize OpenAI client when available
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("⚠️  OpenAI not available - using fallback processing")
    
    async def analyze_user_conversation(self, conversation_text: str, phone: str) -> UserProfileSchema:
        """Analyze conversation to extract user profile information"""
        if self.client:
            return await self._openai_analyze_profile(conversation_text)
        else:
            return await self._fallback_analyze_profile(conversation_text)
    
    async def analyze_job_search_query(self, query: str, user_profile: Optional[UserProfileSchema] = None) -> JobSearchCriteria:
        """Analyze natural language job search query"""
        if self.client:
            return await self._openai_analyze_search(query, user_profile)
        else:
            return await self._fallback_analyze_search(query)
    
    async def analyze_job_posting(self, posting_text: str) -> JobPostingSchema:
        """Analyze natural language job posting"""
        if self.client:
            return await self._openai_analyze_posting(posting_text)
        else:
            return await self._fallback_analyze_posting(posting_text)
    
    async def analyze_job_safety(self, job_description: str, salary: int, company: str) -> SafetyAnalysis:
        """Analyze job posting for safety and scam detection"""
        if self.client:
            return await self._openai_safety_analysis(job_description, salary, company)
        else:
            return await self._fallback_safety_analysis(job_description, salary, company)
    
    async def get_salary_insights(self, job_category: str, location: str, experience: str) -> SalaryInsights:
        """Get comprehensive salary insights"""
        if self.client:
            return await self._openai_salary_insights(job_category, location, experience)
        else:
            return await self._fallback_salary_insights(job_category, location, experience)
    
    # OpenAI-powered implementations
    async def _openai_analyze_profile(self, conversation: str) -> UserProfileSchema:
        """Use OpenAI to analyze user profile from conversation"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing conversations to extract job seeker profiles. 
                        Extract user information from the conversation, handling multiple languages (Hindi, English, regional languages).
                        Be smart about inferring skills, experience, and preferences from natural conversation.
                        Convert local language skills to English equivalents."""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this conversation and extract user profile: {conversation}"
                    }
                ],
                response_format=UserProfileSchema
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"OpenAI error: {e}")
            return await self._fallback_analyze_profile(conversation)
    
    async def _openai_analyze_search(self, query: str, user_profile: Optional[UserProfileSchema]) -> JobSearchCriteria:
        """Use OpenAI to analyze job search query"""
        try:
            context = f"User profile: {user_profile.model_dump_json()}" if user_profile else "No user profile available"
            
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at understanding job search queries in multiple languages.
                        Extract search criteria from natural language, considering the user's profile if available.
                        Handle Hindi, English, and regional languages. Be smart about inferring intent."""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this job search query: '{query}'\n\nContext: {context}"
                    }
                ],
                response_format=JobSearchCriteria
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"OpenAI error: {e}")
            return await self._fallback_analyze_search(query)
    
    async def _openai_analyze_posting(self, posting_text: str) -> JobPostingSchema:
        """Use OpenAI to analyze job posting"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing job postings in multiple languages.
                        Extract structured information from natural language job postings.
                        Handle Hindi, English, and mixed language content. Infer missing information intelligently."""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this job posting: {posting_text}"
                    }
                ],
                response_format=JobPostingSchema
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"OpenAI error: {e}")
            return await self._fallback_analyze_posting(posting_text)
    
    async def _openai_safety_analysis(self, description: str, salary: int, company: str) -> SafetyAnalysis:
        """Use OpenAI for comprehensive safety analysis"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at detecting job scams and analyzing safety risks.
                        Look for red flags like advance payments, unrealistic salaries, vague descriptions, etc.
                        Also identify legitimate indicators. Provide detailed analysis."""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this job for safety:\nDescription: {description}\nSalary: ₹{salary}\nCompany: {company}"
                    }
                ],
                response_format=SafetyAnalysis
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"OpenAI error: {e}")
            return await self._fallback_safety_analysis(description, salary, company)
    
    async def _openai_salary_insights(self, job_category: str, location: str, experience: str) -> SalaryInsights:
        """Use OpenAI for salary insights"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert on Indian job market salary trends.
                        Provide realistic salary ranges and market insights for blue-collar jobs in India.
                        Consider location, experience, and current market conditions."""
                    },
                    {
                        "role": "user",
                        "content": f"Provide salary insights for: {job_category} in {location} with {experience} experience level"
                    }
                ],
                response_format=SalaryInsights
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"OpenAI error: {e}")
            return await self._fallback_salary_insights(job_category, location, experience)
    
    # Fallback implementations (rule-based)
    async def _fallback_analyze_profile(self, conversation: str) -> UserProfileSchema:
        """Fallback profile analysis using rules"""
        # Extract name
        name_patterns = [r"my name is (\w+)", r"i am (\w+)", r"main (\w+) hun"]
        name = "User"
        for pattern in name_patterns:
            match = re.search(pattern, conversation.lower())
            if match:
                name = match.group(1).title()
                break
        
        # Extract location
        location_patterns = [r"from (\w+)", r"live in (\w+)", r"(\w+) mein rehta"]
        location = "Not specified"
        for pattern in location_patterns:
            match = re.search(pattern, conversation.lower())
            if match:
                location = match.group(1).title()
                break
        
        # Extract skills
        skill_keywords = ["security", "guard", "driver", "cook", "clean", "delivery", "plumber", "electrician"]
        skills = [skill for skill in skill_keywords if skill in conversation.lower()]
        
        # Extract experience
        exp_match = re.search(r"(\d+)\s*(?:year|saal)", conversation.lower())
        experience_years = int(exp_match.group(1)) if exp_match else 0
        
        return UserProfileSchema(
            name=name,
            location=location,
            skills=skills or ["general"],
            experience_years=experience_years,
            job_preferences=skills or ["general"],
            availability="flexible",
            preferred_language="hi"
        )
    
    async def _fallback_analyze_search(self, query: str) -> JobSearchCriteria:
        """Fallback search analysis"""
        job_types = []
        location = "Any location"
        urgency = "flexible"
        
        # Simple keyword matching
        if any(word in query.lower() for word in ["security", "guard"]):
            job_types.append("security")
        if any(word in query.lower() for word in ["delivery", "courier"]):
            job_types.append("delivery")
        if any(word in query.lower() for word in ["clean", "maid"]):
            job_types.append("cleaning")
        
        # Extract location
        location_match = re.search(r"in (\w+)", query.lower())
        if location_match:
            location = location_match.group(1).title()
        
        # Check urgency
        if any(word in query.lower() for word in ["urgent", "immediately", "asap"]):
            urgency = "immediate"
        
        return JobSearchCriteria(
            job_types=job_types or ["general"],
            location=location,
            salary_min=None,
            salary_max=None,
            urgency=urgency,
            work_type="full_time",
            additional_requirements=[]
        )
    
    async def _fallback_analyze_posting(self, posting_text: str) -> JobPostingSchema:
        """Fallback posting analysis"""
        # Extract salary
        salary_match = re.search(r"₹\s*(\d+(?:,\d+)*)", posting_text)
        salary = int(salary_match.group(1).replace(',', '')) if salary_match else 15000
        
        # Extract phone
        phone_match = re.search(r"(\d{10})", posting_text)
        contact = phone_match.group(1) if phone_match else "Contact not provided"
        
        # Determine category
        category = "general"
        if any(word in posting_text.lower() for word in ["security", "guard"]):
            category = "security"
        elif any(word in posting_text.lower() for word in ["delivery", "courier"]):
            category = "delivery"
        elif any(word in posting_text.lower() for word in ["clean", "maid"]):
            category = "cleaning"
        
        return JobPostingSchema(
            title=f"{category.title()} Position",
            description=posting_text,
            location="Location not specified",
            salary_min=salary,
            salary_max=salary,
            job_type="full_time",
            category=category,
            requirements=["Experience preferred"],
            contact_info=contact,
            urgency="flexible",
            benefits=[]
        )
    
    async def _fallback_safety_analysis(self, description: str, salary: int, company: str) -> SafetyAnalysis:
        """Fallback safety analysis"""
        risk_score = 0
        scam_indicators = []
        legitimate_indicators = []
        
        # Check for scam patterns
        scam_patterns = [
            r"(?i)pay.*registration.*fee",
            r"(?i)advance.*payment",
            r"(?i)easy.*money",
            r"(?i)guaranteed.*income"
        ]
        
        for pattern in scam_patterns:
            if re.search(pattern, description):
                risk_score += 30
                scam_indicators.append("Contains payment request")
        
        # Check for legitimate indicators
        if re.search(r"\d{10}", description):  # Has phone number
            legitimate_indicators.append("Valid contact number provided")
        
        if salary > 0 and salary < 100000:  # Reasonable salary
            legitimate_indicators.append("Realistic salary range")
        elif salary > 100000:
            risk_score += 25
            scam_indicators.append("Unrealistic high salary")
        
        safety_level = "HIGH_RISK" if risk_score >= 50 else "MEDIUM_RISK" if risk_score >= 25 else "SAFE"
        recommendation = "AVOID" if risk_score >= 50 else "CAUTION" if risk_score >= 25 else "PROCEED"
        
        return SafetyAnalysis(
            safety_level=safety_level,
            risk_score=risk_score,
            scam_indicators=scam_indicators,
            legitimate_indicators=legitimate_indicators,
            recommendation=recommendation,
            explanation=f"Analysis based on {len(scam_indicators)} risk factors and {len(legitimate_indicators)} positive indicators"
        )
    
    async def _fallback_salary_insights(self, job_category: str, location: str, experience: str) -> SalaryInsights:
        """Fallback salary insights"""
        # Basic salary data
        base_salaries = {
            "security": (12000, 25000),
            "delivery": (15000, 30000),
            "cleaning": (8000, 18000),
            "cooking": (10000, 20000),
            "general": (10000, 20000)
        }
        
        min_sal, max_sal = base_salaries.get(job_category.lower(), (10000, 20000))
        
        # Adjust for location
        if location.lower() in ["mumbai", "delhi", "bangalore"]:
            min_sal = int(min_sal * 1.3)
            max_sal = int(max_sal * 1.3)
        
        # Adjust for experience
        if experience == "experienced":
            min_sal = int(min_sal * 1.2)
            max_sal = int(max_sal * 1.2)
        
        return SalaryInsights(
            salary_range_min=min_sal,
            salary_range_max=max_sal,
            market_factors=["Location", "Experience", "Demand"],
            negotiation_tips=["Highlight experience", "Show reliability", "Consider benefits"],
            growth_potential="Good with experience",
            demand_level="MEDIUM"
        )

# Initialize AI agent
ai_agent = JobKrantiAI(OPENAI_API_KEY)

# Seed demo data
def seed_demo_data():
    """Add demo job postings"""
    demo_postings = [
        JobPostingSchema(
            title="Security Guard - Night Shift",
            description="Experienced security guard needed for office complex in Gurgaon Sector 15. Night shift 10PM-6AM. Good salary and benefits.",
            location="Gurgaon, Sector 15",
            salary_min=18000,
            salary_max=20000,
            job_type="full_time",
            category="security",
            requirements=["2+ years experience", "Night shift availability"],
            contact_info="9876543210",
            urgency="within_week",
            benefits=["PF", "ESI", "Overtime pay"]
        ),
        JobPostingSchema(
            title="House Cleaning Service",
            description="Need reliable domestic help for daily house cleaning in Dwarka. 2 hours daily morning shift.",
            location="Delhi, Dwarka",
            salary_min=8000,
            salary_max=10000,
            job_type="part_time",
            category="cleaning",
            requirements=["Reliability", "Local area"],
            contact_info="9123456789",
            urgency="immediate",
            benefits=["Flexible timing"]
        )
    ]
    
    for i, posting_data in enumerate(demo_postings):
        job = JobPosting(
            id=str(uuid4()),
            posting_data=posting_data,
            posted_by="demo_employer",
            verified=True
        )
        JOBS[job.id] = job

seed_demo_data()

# Create FastMCP server
mcp = FastMCP(
    "JobKranti AI - Intelligent Job Platform for Bharat",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# MCP Tools

@mcp.tool
async def validate() -> str:
    """Validate this MCP server for PuchAI integration"""
    return MY_NUMBER

@mcp.tool
async def create_user_profile(
    conversation_text: Annotated[str, Field(description="Natural conversation about user's background, skills, and job preferences")],
    phone: Annotated[str, Field(description="User's WhatsApp phone number")]
) -> str:
    """Create user profile by analyzing natural conversation using AI"""
    
    try:
        # Use AI to analyze the conversation
        profile_data = await ai_agent.analyze_user_conversation(conversation_text, phone)
        
        user_id = str(uuid4())
        profile = UserProfile(
            id=user_id,
            phone=phone,
            profile_data=profile_data
        )
        
        USERS[user_id] = profile
        
        result = f"✅ **Profile Created Successfully!**\n\n"
        result += f"👤 **Name**: {profile_data.name}\n"
        result += f"📍 **Location**: {profile_data.location}\n"
        result += f"🛠️ **Skills**: {', '.join(profile_data.skills)}\n"
        result += f"📅 **Experience**: {profile_data.experience_years} years\n"
        result += f"💼 **Job Preferences**: {', '.join(profile_data.job_preferences)}\n"
        result += f"⏰ **Availability**: {profile_data.availability}\n"
        result += f"🗣️ **Language**: {profile_data.preferred_language}\n"
        result += f"🆔 **Profile ID**: `{user_id}`\n\n"
        result += f"🎯 **Ready to find jobs!** You can now search for opportunities or get salary insights."
        
        return result
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to create profile: {str(e)}"))

@mcp.tool
async def intelligent_job_search(
    search_query: Annotated[str, Field(description="Natural language job search query in any language")],
    user_id: Annotated[str, Field(description="User profile ID for personalized results")] = None,
    max_results: Annotated[int, Field(description="Maximum number of results to return")] = 5
) -> str:
    """AI-powered job search that understands natural language queries"""
    
    try:
        # Get user profile if available
        user_profile = None
        if user_id and user_id in USERS:
            user_profile = USERS[user_id].profile_data
        
        # Use AI to analyze search query
        search_criteria = await ai_agent.analyze_job_search_query(search_query, user_profile)
        
        # Search through available jobs with intelligent matching
        matching_jobs = []
        for job in JOBS.values():
            score = 0
            posting = job.posting_data
            
            # Location matching
            if search_criteria.location.lower() in posting.location.lower() or search_criteria.location.lower() == "any location":
                score += 30
            
            # Job type matching
            for job_type in search_criteria.job_types:
                if job_type.lower() in posting.category.lower() or job_type.lower() in posting.title.lower():
                    score += 40
                    break
            
            # Salary matching
            if search_criteria.salary_min and posting.salary_max >= search_criteria.salary_min:
                score += 20
            if search_criteria.salary_max and posting.salary_min <= search_criteria.salary_max:
                score += 20
            
            # Urgency matching
            if search_criteria.urgency == posting.urgency:
                score += 15
            
            # User skills matching (if profile available)
            if user_profile:
                for skill in user_profile.skills:
                    if skill.lower() in posting.description.lower():
                        score += 10
            
            if score > 20:  # Minimum threshold
                matching_jobs.append((job, score))
        
        # Sort by relevance
        matching_jobs.sort(key=lambda x: x[1], reverse=True)
        matching_jobs = matching_jobs[:max_results]
        
        if not matching_jobs:
            return f"❌ **No jobs found for**: '{search_query}'\n\n" \
                   f"💡 **Try searching for**:\n" \
                   f"• Security guard jobs in Delhi\n" \
                   f"• Delivery work near me\n" \
                   f"• House cleaning jobs\n" \
                   f"• Driver jobs with good salary"
        
        result = f"🔍 **Found {len(matching_jobs)} jobs for**: '{search_query}'\n\n"
        result += f"📊 **Search Analysis**:\n"
        result += f"• Job types: {', '.join(search_criteria.job_types)}\n"
        result += f"• Location: {search_criteria.location}\n"
        result += f"• Urgency: {search_criteria.urgency}\n\n"
        
        for i, (job, score) in enumerate(matching_jobs, 1):
            posting = job.posting_data
            
            # Analyze safety
            safety = await ai_agent.analyze_job_safety(posting.description, posting.salary_max, "Unknown")
            safety_emoji = "🟢" if safety.safety_level == "SAFE" else "🟡" if safety.safety_level == "MEDIUM_RISK" else "🔴"
            
            result += f"**{i}. {safety_emoji} {posting.title}**\n"
            result += f"   📍 {posting.location}\n"
            result += f"   💰 ₹{posting.salary_min:,}"
            if posting.salary_max != posting.salary_min:
                result += f" - ₹{posting.salary_max:,}"
            result += f"/month\n"
            result += f"   ⏰ {posting.job_type.replace('_', ' ').title()}\n"
            result += f"   📱 Contact: {posting.contact_info}\n"
            result += f"   🎯 Match Score: {score}%\n"
            
            if safety.scam_indicators:
                result += f"   ⚠️ Risks: {', '.join(safety.scam_indicators[:2])}\n"
            
            result += f"   📝 {posting.description[:80]}...\n\n"
        
        return result
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Search failed: {str(e)}"))

@mcp.tool
async def post_job_intelligently(
    job_posting_text: Annotated[str, Field(description="Natural language job posting in any language")],
    employer_contact: Annotated[str, Field(description="Employer contact information")]
) -> str:
    """Post a job by analyzing natural language description using AI"""
    
    try:
        # Use AI to analyze and structure the job posting
        posting_data = await ai_agent.analyze_job_posting(job_posting_text)
        
        # Override contact info with provided one
        posting_data.contact_info = employer_contact
        
        job_id = str(uuid4())
        job = JobPosting(
            id=job_id,
            posting_data=posting_data,
            posted_by="employer",
            verified=False
        )
        
        JOBS[job_id] = job
        
        # Find matching candidates
        matching_candidates = []
        for user in USERS.values():
            score = 0
            profile = user.profile_data
            
            # Location match
            if posting_data.location.lower() in profile.location.lower():
                score += 30
            
            # Skills match
            for skill in profile.skills:
                if skill.lower() in posting_data.category.lower() or skill.lower() in posting_data.description.lower():
                    score += 25
                    break
            
            # Experience match
            if profile.experience_years >= 1:  # Some experience
                score += 15
            
            if score > 25:
                matching_candidates.append((user, score))
        
        # Sort candidates by relevance
        matching_candidates.sort(key=lambda x: x[1], reverse=True)
        
        result = f"✅ **Job Posted Successfully!**\n\n"
        result += f"🆔 **Job ID**: `{job_id}`\n"
        result += f"📋 **Title**: {posting_data.title}\n"
        result += f"📍 **Location**: {posting_data.location}\n"
        result += f"💰 **Salary**: ₹{posting_data.salary_min:,}"
        if posting_data.salary_max != posting_data.salary_min:
            result += f" - ₹{posting_data.salary_max:,}"
        result += f"/month\n"
        result += f"⏰ **Type**: {posting_data.job_type.replace('_', ' ').title()}\n"
        result += f"🏷️ **Category**: {posting_data.category.title()}\n"
        result += f"📱 **Contact**: {posting_data.contact_info}\n"
       
        if posting_data.urgency == "immediate":
            result += f"🚨 **URGENT REQUIREMENT**\n"
       
        if posting_data.requirements:
            result += f"📝 **Requirements**: {', '.join(posting_data.requirements)}\n"
       
        if posting_data.benefits:
            result += f"🎁 **Benefits**: {', '.join(posting_data.benefits)}\n"
       
        if matching_candidates:
            result += f"\n🎯 **Found {len(matching_candidates)} Potential Candidates:**\n"
            for user, score in matching_candidates[:3]:  # Show top 3
                profile = user.profile_data
                result += f"• **{profile.name}** ({profile.location}) - {', '.join(profile.skills)} - {profile.experience_years}yr exp\n"
        else:
            result += f"\n📢 **Your job will be visible to all job seekers in {posting_data.location}**"
       
        result += f"\n💡 **Next Steps**: Candidates will contact you directly at {posting_data.contact_info}"
       
        return result
       
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to post job: {str(e)}"))

@mcp.tool
async def generate_smart_resume(
   user_id: Annotated[str, Field(description="User profile ID")],
   target_job_description: Annotated[str, Field(description="Target job description to tailor resume for")]
) -> list[TextContent | ImageContent]:
   """Generate AI-tailored resume based on user profile and target job"""
   
   if user_id not in USERS:
       raise McpError(ErrorData(code=INVALID_PARAMS, message="User profile not found"))
   
   try:
       profile = USERS[user_id].profile_data
       
       # Use AI to analyze target job and tailor resume
       job_analysis = await ai_agent.analyze_job_posting(target_job_description)
       
       # Create tailored resume
       img = Image.new('RGB', (800, 1100), color='white')
       draw = ImageDraw.Draw(img)
       
       # Try to load fonts
       try:
           title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
           header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
           body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
           small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
       except:
           title_font = header_font = body_font = small_font = ImageFont.load_default()
       
       y = 40
       
       # Header with name
       draw.text((50, y), profile.name.upper(), fill='#2C3E50', font=title_font)
       y += 50
       
       # Contact info
       draw.text((50, y), f"📱 {USERS[user_id].phone}  📍 {profile.location}", fill='#34495E', font=body_font)
       y += 35
       
       # Horizontal line
       draw.line([(50, y), (750, y)], fill='#BDC3C7', width=2)
       y += 30
       
       # Professional objective (tailored to job)
       draw.text((50, y), "PROFESSIONAL OBJECTIVE", fill='#2C3E50', font=header_font)
       y += 35
       objective = f"Seeking {job_analysis.title} position in {job_analysis.location} area. "
       objective += f"Bringing {profile.experience_years} years of relevant experience "
       objective += f"in {', '.join(profile.skills[:3])} to contribute to organizational success."
       
       # Word wrap for objective
       words = objective.split()
       lines = []
       current_line = []
       for word in words:
           test_line = ' '.join(current_line + [word])
           if len(test_line) <= 80:  # Approximate characters per line
               current_line.append(word)
           else:
               lines.append(' '.join(current_line))
               current_line = [word]
       if current_line:
           lines.append(' '.join(current_line))
       
       for line in lines:
           draw.text((50, y), line, fill='#2C3E50', font=small_font)
           y += 25
       y += 20
       
       # Experience section
       draw.text((50, y), "WORK EXPERIENCE", fill='#2C3E50', font=header_font)
       y += 35
       
       if profile.experience_years > 0:
           # Tailor experience to match job requirements
           relevant_skills = [skill for skill in profile.skills if skill.lower() in job_analysis.description.lower()]
           if not relevant_skills:
               relevant_skills = profile.skills[:2]
           
           draw.text((50, y), f"• {profile.experience_years} years of professional experience", fill='#2C3E50', font=body_font)
           y += 30
           draw.text((50, y), f"• Specialized in: {', '.join(relevant_skills)}", fill='#2C3E50', font=body_font)
           y += 30
           draw.text((50, y), f"• Proven track record in {profile.location} area", fill='#2C3E50', font=body_font)
           y += 30
       else:
           draw.text((50, y), "• Fresh candidate with strong motivation to learn", fill='#2C3E50', font=body_font)
           y += 30
           draw.text((50, y), f"• Ready to apply {', '.join(profile.skills)} skills", fill='#2C3E50', font=body_font)
           y += 30
       
       y += 20
       
       # Skills section (prioritized based on job requirements)
       draw.text((50, y), "KEY SKILLS", fill='#2C3E50', font=header_font)
       y += 35
       
       # Prioritize skills that match job requirements
       job_relevant_skills = []
       other_skills = []
       for skill in profile.skills:
           if skill.lower() in job_analysis.description.lower() or skill.lower() in job_analysis.category.lower():
               job_relevant_skills.append(skill + " ⭐")  # Mark as relevant
           else:
               other_skills.append(skill)
       
       all_skills = job_relevant_skills + other_skills
       skills_text = " • ".join(all_skills)
       
       # Word wrap skills
       words = skills_text.split()
       lines = []
       current_line = []
       for word in words:
           test_line = ' '.join(current_line + [word])
           if len(test_line) <= 70:
               current_line.append(word)
           else:
               lines.append(' '.join(current_line))
               current_line = [word]
       if current_line:
           lines.append(' '.join(current_line))
       
       for line in lines:
           draw.text((50, y), line, fill='#2C3E50', font=body_font)
           y += 25
       
       y += 30
       
       # Additional qualifications
       draw.text((50, y), "ADDITIONAL QUALIFICATIONS", fill='#2C3E50', font=header_font)
       y += 35
       
       qualifications = [
           f"• Available for {profile.availability} work",
           f"• Preferred communication in {profile.preferred_language.upper()}",
           f"• Local knowledge of {profile.location} area",
           "• Reliable and punctual work ethic"
       ]
       
       # Add job-specific qualifications
       if job_analysis.urgency == "immediate":
           qualifications.insert(1, "• Available for immediate joining")
       
       for qual in qualifications:
           draw.text((50, y), qual, fill='#2C3E50', font=body_font)
           y += 25
       
       # Footer
       y = 1050
       draw.text((50, y), f"Resume generated by JobKranti AI • Tailored for {job_analysis.category.title()} positions", 
                fill='#7F8C8D', font=small_font)
       
       # Convert to base64
       buf = io.BytesIO()
       img.save(buf, format='PNG')
       resume_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
       
       return [
           TextContent(
               type="text",
               text=f"📄 **AI-Tailored Resume Generated!**\n\n"
                    f"👤 **Candidate**: {profile.name}\n"
                    f"🎯 **Tailored for**: {job_analysis.title}\n"
                    f"📍 **Location**: {profile.location}\n"
                    f"📅 **Experience**: {profile.experience_years} years\n"
                    f"⭐ **Highlighted Skills**: {', '.join([s for s in profile.skills if s.lower() in job_analysis.description.lower()][:3])}\n\n"
                    f"🤖 **AI Optimization**: Resume automatically optimized based on job requirements\n"
                    f"💡 **Pro Tip**: Skills matching job requirements are marked with ⭐"
           ),
           ImageContent(
               type="image",
               mimeType="image/png",
               data=resume_base64
           )
       ]
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Resume generation failed: {str(e)}"))

@mcp.tool
async def ai_job_safety_analysis(
   job_description: Annotated[str, Field(description="Complete job description to analyze")],
   salary_offered: Annotated[int, Field(description="Salary amount offered")],
   company_name: Annotated[str, Field(description="Company or employer name")] = "Unknown"
) -> str:
   """AI-powered comprehensive job safety and scam detection analysis"""
   
   try:
       # Use AI for comprehensive safety analysis
       safety_analysis = await ai_agent.analyze_job_safety(job_description, salary_offered, company_name)
       
       result = f"🔍 **AI-Powered Job Safety Analysis**\n\n"
       result += f"🏢 **Company**: {company_name}\n"
       result += f"💰 **Salary**: ₹{salary_offered:,}/month\n\n"
       
       # Safety level with detailed explanation
       if safety_analysis.safety_level == "SAFE":
           result += f"✅ **Safety Assessment: SAFE TO PROCEED**\n"
           result += f"🎯 **Recommendation**: {safety_analysis.recommendation}\n"
       elif safety_analysis.safety_level == "MEDIUM_RISK":
           result += f"⚠️ **Safety Assessment: PROCEED WITH CAUTION**\n"
           result += f"🎯 **Recommendation**: {safety_analysis.recommendation}\n"
       else:
           result += f"🚨 **Safety Assessment: HIGH RISK - AVOID**\n"
           result += f"🎯 **Recommendation**: {safety_analysis.recommendation}\n"
       
       result += f"📊 **AI Risk Score**: {safety_analysis.risk_score}/100\n\n"
       
       # Detailed AI analysis
       result += f"🤖 **AI Analysis**: {safety_analysis.explanation}\n\n"
       
       if safety_analysis.scam_indicators:
           result += f"🚨 **Red Flags Detected**:\n"
           for indicator in safety_analysis.scam_indicators:
               result += f"• {indicator}\n"
           result += "\n"
       
       if safety_analysis.legitimate_indicators:
           result += f"✅ **Positive Indicators**:\n"
           for indicator in safety_analysis.legitimate_indicators:
               result += f"• {indicator}\n"
           result += "\n"
       
       result += f"💡 **General Safety Tips**:\n"
       result += f"• Never pay any registration or processing fees\n"
       result += f"• Verify company details through official channels\n"
       result += f"• Meet in public places for interviews\n"
       result += f"• Trust your instincts if something feels wrong\n"
       result += f"• Check with local authorities if suspicious\n"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Safety analysis failed: {str(e)}"))

@mcp.tool
async def ai_salary_insights(
   job_category: Annotated[str, Field(description="Job category or type")],
   location: Annotated[str, Field(description="City or location")],
   experience_level: Annotated[str, Field(description="Experience level: fresher, experienced, expert")] = "experienced"
) -> str:
   """Get AI-powered salary insights and market intelligence"""
   
   try:
       # Use AI for comprehensive salary analysis
       insights = await ai_agent.get_salary_insights(job_category, location, experience_level)
       
       result = f"💰 **AI-Powered Salary Insights**\n\n"
       result += f"🏷️ **Job Category**: {job_category.title()}\n"
       result += f"📍 **Location**: {location.title()}\n"
       result += f"👤 **Experience Level**: {experience_level.title()}\n\n"
       
       # Salary range
       avg_salary = (insights.salary_range_min + insights.salary_range_max) // 2
       result += f"💵 **Salary Analysis**:\n"
       result += f"• **Range**: ₹{insights.salary_range_min:,} - ₹{insights.salary_range_max:,}/month\n"
       result += f"• **Average**: ₹{avg_salary:,}/month\n"
       result += f"• **Market Demand**: {insights.demand_level}\n\n"
       
       # Market factors
       result += f"📊 **Market Factors**:\n"
       for factor in insights.market_factors:
           result += f"• {factor}\n"
       result += "\n"
       
       # Career growth
       result += f"📈 **Growth Potential**: {insights.growth_potential}\n\n"
       
       # Negotiation tips
       result += f"🎯 **AI-Powered Negotiation Tips**:\n"
       for tip in insights.negotiation_tips:
           result += f"• {tip}\n"
       result += "\n"
       
       # Additional insights based on demand level
       if insights.demand_level == "HIGH":
           result += f"🔥 **Market Opportunity**: High demand! You have strong negotiating power.\n"
       elif insights.demand_level == "MEDIUM":
           result += f"⚖️ **Market Opportunity**: Balanced market. Focus on highlighting unique skills.\n"
       else:
           result += f"💡 **Market Opportunity**: Consider skill development or location flexibility.\n"
       
       # Experience-based advice
       if experience_level == "fresher":
           result += f"🌱 **Fresher Tip**: Gain 2-3 years experience to increase salary by 40-60%\n"
       elif experience_level == "experienced":
           result += f"⭐ **Experienced Advantage**: Your experience commands premium in this market\n"
       else:
           result += f"🏆 **Expert Level**: Consider leadership roles or specialized positions\n"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Salary insights failed: {str(e)}"))

@mcp.tool
async def ai_conversation_analyzer(
   conversation_text: Annotated[str, Field(description="Any conversation text to analyze for job-related insights")],
   analysis_type: Annotated[str, Field(description="Type of analysis: profile, job_search, job_posting, or general")] = "general"
) -> str:
   """Universal AI analyzer that can understand any conversation and extract job-related insights"""
   
   try:
       result = f"🤖 **AI Conversation Analysis**\n\n"
       result += f"📝 **Input**: {conversation_text[:100]}...\n"
       result += f"🔍 **Analysis Type**: {analysis_type.title()}\n\n"
       
       if analysis_type == "profile":
           # Analyze for profile creation
           profile_data = await ai_agent.analyze_user_conversation(conversation_text, "unknown")
           result += f"👤 **Profile Insights Extracted**:\n"
           result += f"• **Name**: {profile_data.name}\n"
           result += f"• **Location**: {profile_data.location}\n"
           result += f"• **Skills**: {', '.join(profile_data.skills)}\n"
           result += f"• **Experience**: {profile_data.experience_years} years\n"
           result += f"• **Job Preferences**: {', '.join(profile_data.job_preferences)}\n"
           result += f"• **Availability**: {profile_data.availability}\n"
           result += f"• **Language**: {profile_data.preferred_language}\n"
           
       elif analysis_type == "job_search":
           # Analyze for job search intent
           search_criteria = await ai_agent.analyze_job_search_query(conversation_text)
           result += f"🔍 **Search Intent Detected**:\n"
           result += f"• **Job Types**: {', '.join(search_criteria.job_types)}\n"
           result += f"• **Location**: {search_criteria.location}\n"
           result += f"• **Salary Range**: ₹{search_criteria.salary_min or 'Not specified'} - ₹{search_criteria.salary_max or 'Not specified'}\n"
           result += f"• **Work Type**: {search_criteria.work_type}\n"
           result += f"• **Urgency**: {search_criteria.urgency}\n"
           result += f"• **Requirements**: {', '.join(search_criteria.additional_requirements) if search_criteria.additional_requirements else 'None specified'}\n"
           
       elif analysis_type == "job_posting":
           # Analyze for job posting structure
           posting_data = await ai_agent.analyze_job_posting(conversation_text)
           result += f"💼 **Job Posting Structure**:\n"
           result += f"• **Title**: {posting_data.title}\n"
           result += f"• **Category**: {posting_data.category}\n"
           result += f"• **Location**: {posting_data.location}\n"
           result += f"• **Salary**: ₹{posting_data.salary_min:,} - ₹{posting_data.salary_max:,}\n"
           result += f"• **Type**: {posting_data.job_type}\n"
           result += f"• **Urgency**: {posting_data.urgency}\n"
           result += f"• **Requirements**: {', '.join(posting_data.requirements)}\n"
           result += f"• **Benefits**: {', '.join(posting_data.benefits) if posting_data.benefits else 'None mentioned'}\n"
           
       else:
           # General analysis - try to understand intent
           result += f"🎯 **General Analysis**:\n"
           
           # Try to detect what the conversation is about
           if any(word in conversation_text.lower() for word in ["job", "work", "naukri", "kaam"]):
               result += f"• **Topic**: Job-related conversation detected\n"
               
               # Try profile analysis
               try:
                   profile_data = await ai_agent.analyze_user_conversation(conversation_text, "unknown")
                   result += f"• **Profile Elements Found**: {profile_data.name}, {profile_data.location}, {len(profile_data.skills)} skills\n"
               except:
                   pass
               
               # Try search analysis
               try:
                   search_criteria = await ai_agent.analyze_job_search_query(conversation_text)
                   result += f"• **Search Intent**: Looking for {', '.join(search_criteria.job_types)} in {search_criteria.location}\n"
               except:
                   pass
               
           else:
               result += f"• **Topic**: General conversation (no specific job intent detected)\n"
       
       result += f"\n💡 **Suggested Actions**:\n"
       if analysis_type == "profile":
           result += f"• Use `create_user_profile` tool to create structured profile\n"
       elif analysis_type == "job_search":
           result += f"• Use `intelligent_job_search` tool to find matching jobs\n"
       elif analysis_type == "job_posting":
           result += f"• Use `post_job_intelligently` tool to create job listing\n"
       else:
           result += f"• Specify analysis_type for more detailed insights\n"
           result += f"• Available types: profile, job_search, job_posting\n"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Conversation analysis failed: {str(e)}"))

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for load balancers and monitoring"""
    return JSONResponse({
        "status": "healthy",
        "service": "JobKranti MCP Server",
        "version": "1.0.0",
        "port": 8086,
        "tools_available": 7,
        "ai_enabled": bool(OPENAI_API_KEY)
    })

@mcp.custom_route("/", methods=["GET"])
async def root_endpoint(request: Request) -> JSONResponse:
    """Root endpoint with server information"""
    return JSONResponse({
        "service": "JobKranti AI - Intelligent Job Platform for Bharat",
        "status": "running",
        "mcp_endpoint": "/mcp",  # Updated from /sse to /mcp
        "health_endpoint": "/health",
        "auth_required": True,
        "transport": "streamable-http",
        "tools": [
            "validate",
            "create_user_profile", 
            "intelligent_job_search",
            "post_job_intelligently",
            "generate_smart_resume",
            "ai_job_safety_analysis",
            "ai_conversation_analyzer"
        ]
    })

@mcp.custom_route("/info", methods=["GET"])
async def info_endpoint(request: Request) -> PlainTextResponse:
    """Simple text endpoint for basic checks"""
    return PlainTextResponse("JobKranti MCP Server is running! Ready for PuchAI connection.")

async def main():
    print("🚀 Starting JobKranti AI-Powered MCP server on http://0.0.0.0:8086")
    print("🤖 AI Agent initialized with intelligent conversation analysis")
    print("🔗 Connect from PuchAI using your auth token")
    print("📱 Demo data loaded with sample job postings")
    print("🌐 Custom routes available: /health, /, /info")
    print("🔗 MCP endpoint available at: /mcp")  # Updated info
    
    if OPENAI_API_KEY:
        print("✅ OpenAI API connected - Advanced AI features enabled")
    else:
        print("⚠️  OpenAI API not configured - Using fallback rule-based processing")
    
    # Use the correct transport with default path
    await mcp.run_async(transport="http", host="0.0.0.0", port=8086)
    # This automatically creates the /mcp endpoint

if __name__ == "__main__":
    asyncio.run(main())