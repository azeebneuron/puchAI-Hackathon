import asyncio
import json
import os
import re
import base64
import io
import sqlite3
import aiosqlite
from typing import Annotated, Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from uuid import uuid4
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

# Load environment variables
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER", "919998881729")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY in your .env file"

# Enhanced Auth Provider
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="jobkranti-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# Enhanced Enums and Constants
class JobUrgency(str, Enum):
    IMMEDIATE = "immediate"
    TODAY = "today"
    WITHIN_WEEK = "within_week"
    WITHIN_MONTH = "within_month"
    FLEXIBLE = "flexible"

class WorkType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    GIG = "gig"
    CONTRACT = "contract"
    HOURLY = "hourly"
    DAILY = "daily"

class JobCategory(str, Enum):
    SECURITY = "security"
    DELIVERY = "delivery"
    CLEANING = "cleaning"
    COOKING = "cooking"
    DRIVING = "driving"
    CONSTRUCTION = "construction"
    PLUMBING = "plumbing"
    ELECTRICAL = "electrical"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    MANUFACTURING = "manufacturing"
    GENERAL = "general"

# Supported Indian Languages (OpenAI supports all these)
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "en": "English", 
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese"
}

# Enhanced Pydantic Models for Structured LLM Outputs
class UserProfileSchema(BaseModel):
    name: str = Field(description="Full name extracted from conversation")
    location: str = Field(description="City/area where person lives")
    skills: List[str] = Field(description="Skills mentioned (auto-translated to English)")
    experience_years: int = Field(description="Years of work experience mentioned")
    job_preferences: List[str] = Field(description="Types of jobs person is interested in")
    availability: str = Field(description="When person can work - timing/schedule")
    preferred_language: str = Field(description="Primary language detected from conversation")
    secondary_languages: List[str] = Field(default=[], description="Other languages person speaks")
    contact_preference: str = Field(default="voice", description="Prefers voice calls or text")
    salary_expectation_min: Optional[int] = Field(default=None, description="Minimum salary mentioned")
    salary_expectation_max: Optional[int] = Field(default=None, description="Maximum salary mentioned")
    education_level: str = Field(default="basic", description="Education level if mentioned")
    has_vehicle: bool = Field(default=False, description="Owns bike/car for delivery jobs")
    willing_to_relocate: bool = Field(default=False, description="Open to moving cities")
    immediate_need: bool = Field(default=False, description="Urgently needs job for survival")
    family_situation: str = Field(default="", description="Family context affecting job needs")

class JobSearchCriteria(BaseModel):
    job_types: List[str] = Field(description="Types of jobs being searched for")
    location: str = Field(description="Preferred work location")
    radius_km: int = Field(default=10, description="How far willing to travel")
    salary_min: Optional[int] = Field(description="Minimum salary needed")
    salary_max: Optional[int] = Field(description="Maximum salary expected")
    urgency: str = Field(description="How urgently job is needed")
    work_type: str = Field(description="Type of employment preferred")
    preferred_shift: List[str] = Field(default=[], description="Time preferences")
    additional_requirements: List[str] = Field(description="Special needs or requirements")
    survival_mode: bool = Field(default=False, description="Will take any available job")
    language_detected: str = Field(description="Language user communicated in")

class JobPostingSchema(BaseModel):
    title: str = Field(description="Job title in English")
    description: str = Field(description="Detailed job description")
    location: str = Field(description="Specific job location")
    salary_min: int = Field(description="Minimum salary offered per month")
    salary_max: int = Field(description="Maximum salary offered per month")
    job_type: str = Field(description="Employment type")
    category: str = Field(description="Job category for matching")
    requirements: List[str] = Field(description="Skills/qualifications needed")
    contact_info: str = Field(description="How to contact employer")
    urgency: str = Field(description="How quickly position needs to be filled")
    benefits: List[str] = Field(description="Perks and benefits offered")
    working_hours: str = Field(description="Shift timings")
    original_language: str = Field(description="Language job was posted in")

class ConversationalFollowUp(BaseModel):
    follow_up_questions: List[str] = Field(description="Questions to ask for missing info")
    missing_info: List[str] = Field(description="What information is still needed")
    conversation_stage: str = Field(description="Where we are in the conversation flow")
    next_action: str = Field(description="What should happen next")
    response_in_user_language: str = Field(description="Response in user's preferred language")

class JobMatchAnalysis(BaseModel):
    match_score: int = Field(description="Relevance score 0-100")
    match_reasons: List[str] = Field(description="Why this job matches")
    concerns: List[str] = Field(description="Potential issues or mismatches")
    recommendation: str = Field(description="Should apply, maybe, or skip")

# Enhanced Data Models
@dataclass
class UserProfile:
    id: str
    phone: str
    profile_data: UserProfileSchema
    conversation_history: List[Dict] = None
    voice_samples: List[str] = None
    created_at: datetime = None
    last_active: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()
        if self.conversation_history is None:
            self.conversation_history = []
        if self.voice_samples is None:
            self.voice_samples = []

@dataclass
class JobPosting:
    id: str
    posting_data: JobPostingSchema
    posted_by: str
    verified: bool = False
    views: int = 0
    applications: int = 0
    created_at: datetime = None
    expires_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            self.expires_at = datetime.now() + timedelta(days=30)

@dataclass
class ConversationSession:
    user_id: str
    session_id: str
    messages: List[Dict]
    current_intent: str
    language: str
    last_interaction: datetime
    
    def __post_init__(self):
        if self.last_interaction is None:
            self.last_interaction = datetime.now()

# Database file path
DB_PATH = Path("jobkranti.db")

# In-memory storage for fast access (cache layer)
USERS: Dict[str, UserProfile] = {}
JOBS: Dict[str, JobPosting] = {}
CONVERSATIONS: Dict[str, ConversationSession] = {}
APPLICATIONS: Dict[str, Dict] = {}

class DataManager:
    """Handles data persistence with SQLite + in-memory caching"""
    
    @staticmethod
    async def init_database():
        """Initialize SQLite database with required tables"""
        async with aiosqlite.connect(DB_PATH) as db:
            # Users table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    phone TEXT UNIQUE,
                    profile_data TEXT,
                    conversation_history TEXT,
                    voice_samples TEXT,
                    created_at TIMESTAMP,
                    last_active TIMESTAMP
                )
            ''')
            
            # Jobs table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    posting_data TEXT,
                    posted_by TEXT,
                    verified BOOLEAN DEFAULT FALSE,
                    views INTEGER DEFAULT 0,
                    applications INTEGER DEFAULT 0,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')
            
            # Conversations table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    messages TEXT,
                    current_intent TEXT,
                    language TEXT,
                    last_interaction TIMESTAMP
                )
            ''')
            
            # Applications table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS applications (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    job_id TEXT,
                    application_data TEXT,
                    status TEXT,
                    applied_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (job_id) REFERENCES jobs (id)
                )
            ''')
            
            await db.commit()
            print("✅ Database initialized successfully")
    
    @staticmethod
    async def save_user_profile(user: UserProfile):
        """Save user profile to database and update cache"""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO users 
                    (id, phone, profile_data, conversation_history, voice_samples, created_at, last_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user.id,
                    user.phone,
                    json.dumps(asdict(user.profile_data), default=str),
                    json.dumps(user.conversation_history, default=str),
                    json.dumps(user.voice_samples),
                    user.created_at.isoformat(),
                    user.last_active.isoformat()
                ))
                await db.commit()
            
            # Update cache
            USERS[user.id] = user
            print(f"✅ User {user.id} saved to database")
            
        except Exception as e:
            print(f"❌ Error saving user {user.id}: {e}")
    
    @staticmethod
    async def save_job_posting(job: JobPosting):
        """Save job posting to database and update cache"""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO jobs 
                    (id, posting_data, posted_by, verified, views, applications, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job.id,
                    json.dumps(asdict(job.posting_data), default=str),
                    job.posted_by,
                    job.verified,
                    job.views,
                    job.applications,
                    job.created_at.isoformat(),
                    job.expires_at.isoformat()
                ))
                await db.commit()
            
            # Update cache
            JOBS[job.id] = job
            print(f"✅ Job {job.id} saved to database")
            
        except Exception as e:
            print(f"❌ Error saving job {job.id}: {e}")
    
    @staticmethod
    async def save_conversation(conversation: ConversationSession):
        """Save conversation to database and update cache"""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO conversations 
                    (session_id, user_id, messages, current_intent, language, last_interaction)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    conversation.session_id,
                    conversation.user_id,
                    json.dumps(conversation.messages, default=str),
                    conversation.current_intent,
                    conversation.language,
                    conversation.last_interaction.isoformat()
                ))
                await db.commit()
            
            # Update cache
            CONVERSATIONS[conversation.session_id] = conversation
            
        except Exception as e:
            print(f"❌ Error saving conversation {conversation.session_id}: {e}")
    
    @staticmethod
    async def load_all_data():
        """Load all data from database into memory cache on startup"""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                # Load users
                async with db.execute("SELECT * FROM users") as cursor:
                    async for row in cursor:
                        try:
                            user = UserProfile(
                                id=row[0],
                                phone=row[1],
                                profile_data=UserProfileSchema(**json.loads(row[2])),
                                conversation_history=json.loads(row[3]) if row[3] else [],
                                voice_samples=json.loads(row[4]) if row[4] else [],
                                created_at=datetime.fromisoformat(row[5]),
                                last_active=datetime.fromisoformat(row[6])
                            )
                            USERS[user.id] = user
                        except Exception as e:
                            print(f"❌ Error loading user {row[0]}: {e}")
                
                # Load jobs
                async with db.execute("SELECT * FROM jobs") as cursor:
                    async for row in cursor:
                        try:
                            job = JobPosting(
                                id=row[0],
                                posting_data=JobPostingSchema(**json.loads(row[1])),
                                posted_by=row[2],
                                verified=bool(row[3]),
                                views=row[4],
                                applications=row[5],
                                created_at=datetime.fromisoformat(row[6]),
                                expires_at=datetime.fromisoformat(row[7])
                            )
                            JOBS[job.id] = job
                        except Exception as e:
                            print(f"❌ Error loading job {row[0]}: {e}")
                
                # Load conversations
                async with db.execute("SELECT * FROM conversations") as cursor:
                    async for row in cursor:
                        try:
                            conversation = ConversationSession(
                                session_id=row[0],
                                user_id=row[1],
                                messages=json.loads(row[2]) if row[2] else [],
                                current_intent=row[3],
                                language=row[4],
                                last_interaction=datetime.fromisoformat(row[5])
                            )
                            CONVERSATIONS[conversation.session_id] = conversation
                        except Exception as e:
                            print(f"❌ Error loading conversation {row[0]}: {e}")
            
            print(f"✅ Loaded {len(USERS)} users, {len(JOBS)} jobs, {len(CONVERSATIONS)} conversations from database")
            
        except Exception as e:
            print(f"❌ Error loading data from database: {e}")
    
    @staticmethod
    async def get_user_by_phone(phone: str) -> Optional[UserProfile]:
        """Find user by phone number"""
        # First check cache
        for user in USERS.values():
            if user.phone == phone:
                return user
        
        # If not in cache, check database
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute("SELECT * FROM users WHERE phone = ?", (phone,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        user = UserProfile(
                            id=row[0],
                            phone=row[1],
                            profile_data=UserProfileSchema(**json.loads(row[2])),
                            conversation_history=json.loads(row[3]) if row[3] else [],
                            voice_samples=json.loads(row[4]) if row[4] else [],
                            created_at=datetime.fromisoformat(row[5]),
                            last_active=datetime.fromisoformat(row[6])
                        )
                        USERS[user.id] = user  # Add to cache
                        return user
        except Exception as e:
            print(f"❌ Error finding user by phone {phone}: {e}")
        
        return None
    
    @staticmethod
    async def backup_data():
        """Create JSON backup of all data"""
        try:
            backup_data = {
                'users': {},
                'jobs': {},
                'conversations': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Backup users
            for user_id, user in USERS.items():
                backup_data['users'][user_id] = {
                    'id': user.id,
                    'phone': user.phone,
                    'profile_data': asdict(user.profile_data),
                    'conversation_history': user.conversation_history,
                    'voice_samples': user.voice_samples,
                    'created_at': user.created_at.isoformat(),
                    'last_active': user.last_active.isoformat()
                }
            
            # Backup jobs
            for job_id, job in JOBS.items():
                backup_data['jobs'][job_id] = {
                    'id': job.id,
                    'posting_data': asdict(job.posting_data),
                    'posted_by': job.posted_by,
                    'verified': job.verified,
                    'views': job.views,
                    'applications': job.applications,
                    'created_at': job.created_at.isoformat(),
                    'expires_at': job.expires_at.isoformat()
                }
            
            # Backup conversations
            for session_id, conversation in CONVERSATIONS.items():
                backup_data['conversations'][session_id] = {
                    'session_id': conversation.session_id,
                    'user_id': conversation.user_id,
                    'messages': conversation.messages,
                    'current_intent': conversation.current_intent,
                    'language': conversation.language,
                    'last_interaction': conversation.last_interaction.isoformat()
                }
            
            backup_file = f"jobkranti_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            print(f"✅ Data backed up to {backup_file}")
            return backup_file
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return None

# Initialize data manager
data_manager = DataManager()

# Advanced AI Agent with Voice and Multilingual Support
class JobKrantiAdvancedAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        if api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("⚠️ OpenAI not available")
    
    async def transcribe_voice_message(self, audio_data: bytes, language_hint: str = "hi") -> str:
        """Convert voice message to text using OpenAI Whisper"""
        if not self.client:
            return "Voice transcription not available - OpenAI not configured"
        
        try:
            # Create a temporary file-like object
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "voice_message.mp3"
            
            # Use Whisper for transcription
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language_hint if language_hint != "auto" else None,
                response_format="text"
            )
            
            return transcript
            
        except Exception as e:
            print(f"Voice transcription error: {e}")
            return f"Sorry, couldn't understand the voice message. Error: {str(e)}"
    
    async def generate_voice_response(self, text: str, language: str = "hi") -> bytes:
        """Convert text to speech using OpenAI TTS"""
        if not self.client:
            return b""
        
        try:
            # Choose voice based on language
            voice_map = {
                "hi": "alloy",  # Good for Hindi
                "en": "nova",   # Clear English
                "ta": "echo",   # Works well for Tamil
                "te": "fable",  # Good for Telugu
                "bn": "onyx",   # Bengali
                "mr": "shimmer" # Marathi
            }
            
            voice = voice_map.get(language, "alloy")
            
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            return response.content
            
        except Exception as e:
            print(f"Voice generation error: {e}")
            return b""
    
    async def detect_language_and_intent(self, text: str) -> Dict[str, Any]:
        """Detect language and conversation intent"""
        if not self.client:
            return {"language": "hi", "intent": "general", "confidence": 0.5}
        
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in Indian languages and job-related conversations. 
                        Analyze the text and return JSON with:
                        - language: detected language code (hi, en, ta, te, bn, mr, gu, kn, ml, pa, ur)
                        - intent: what user wants (profile_creation, job_search, job_posting, question, general)
                        - confidence: 0-1 confidence score
                        - key_info: important details extracted"""
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze this text: {text}"
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(completion.choices[0].message.content)
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return {"language": "hi", "intent": "general", "confidence": 0.5}
    
    async def generate_conversational_response(self, user_input: str, conversation_history: List[Dict], intent: str, language: str) -> ConversationalFollowUp:
        """Generate intelligent conversational responses with follow-ups"""
        if not self.client:
            return self._fallback_conversation_response(user_input, language)
        
        try:
            # Build context from conversation history
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]])
            
            system_prompt = f"""You are JobKranti AI, helping people in India find jobs and employers find workers. 
            
            Current conversation intent: {intent}
            User's language: {SUPPORTED_LANGUAGES.get(language, language)}
            
            Guidelines:
            1. Be conversational, friendly, and empathetic
            2. Ask relevant follow-up questions to gather missing info
            3. Handle code-mixing (Hindi-English, Tamil-English etc.)
            4. Understand survival needs - some people need ANY job immediately
            5. Be sensitive to economic pressures and family situations
            6. Guide conversation toward actionable outcomes
            7. Respond primarily in the user's detected language
            
            For job seekers: gather name, location, skills, experience, availability, salary needs
            For employers: gather job details, location, salary, requirements, urgency
            
            Always be practical and helpful. Understand that many users are not tech-savvy."""
            
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Previous conversation:\n{context}\n\nUser's new message: {user_input}"}
                ],
                response_format=ConversationalFollowUp
            )
            
            return completion.choices[0].message.parsed
            
        except Exception as e:
            print(f"Conversational AI error: {e}")
            return self._fallback_conversation_response(user_input, language)
    
    async def intelligent_profile_extraction(self, conversation_text: str, language: str) -> UserProfileSchema:
        """Extract comprehensive user profile from natural conversation"""
        if not self.client:
            return await self._fallback_analyze_profile(conversation_text, language)
        
        try:
            system_prompt = f"""Extract user profile information from this conversation in {SUPPORTED_LANGUAGES.get(language, language)}.
            
            Handle these scenarios intelligently:
            - Code-mixed conversations (Hindi-English, Tamil-English, etc.)
            - Colloquial terms and local expressions
            - Implied information (e.g., "need job fast" = immediate_need = true)
            - Family situations affecting job needs
            - Skills mentioned in local languages (convert to English)
            - Location nicknames (e.g., "Gurgaon" = "Gurugram")
            
            Be smart about inferring missing information from context."""
            
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract profile from: {conversation_text}"}
                ],
                response_format=UserProfileSchema
            )
            
            return completion.choices[0].message.parsed
            
        except Exception as e:
            print(f"Profile extraction error: {e}")
            return await self._fallback_analyze_profile(conversation_text, language)
    
    async def intelligent_job_matching(self, user_profile: UserProfileSchema, available_jobs: List[JobPosting]) -> List[tuple]:
        """Advanced job matching with AI scoring"""
        if not self.client:
            return await self._fallback_job_matching(user_profile, available_jobs)
        
        scored_jobs = []
        
        for job in available_jobs:
            try:
                analysis_prompt = f"""Analyze job match between:
                
                User Profile:
                - Skills: {user_profile.skills}
                - Experience: {user_profile.experience_years} years
                - Location: {user_profile.location}
                - Salary expectation: ₹{user_profile.salary_expectation_min}-{user_profile.salary_expectation_max}
                - Availability: {user_profile.availability}
                - Immediate need: {user_profile.immediate_need}
                - Languages: {user_profile.preferred_language}, {user_profile.secondary_languages}
                
                Job Posting:
                - Title: {job.posting_data.title}
                - Location: {job.posting_data.location}
                - Salary: ₹{job.posting_data.salary_min}-{job.posting_data.salary_max}
                - Requirements: {job.posting_data.requirements}
                - Category: {job.posting_data.category}
                - Urgency: {job.posting_data.urgency}
                
                Consider location proximity, skill match, salary fit, and urgency alignment."""
                
                completion = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert job matcher for blue-collar workers in India."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    response_format=JobMatchAnalysis
                )
                
                analysis = completion.choices[0].message.parsed
                scored_jobs.append((job, analysis.match_score, analysis))
                
            except Exception as e:
                print(f"Job matching error for {job.id}: {e}")
                # Fallback to basic scoring
                basic_score = await self._calculate_basic_match_score(user_profile, job)
                scored_jobs.append((job, basic_score, None))
        
        # Sort by score descending
        scored_jobs.sort(key=lambda x: x[1], reverse=True)
        return scored_jobs
    
    async def _calculate_basic_match_score(self, profile: UserProfileSchema, job: JobPosting) -> int:
        """Fallback basic scoring algorithm"""
        score = 0
        
        # Location match (40 points)
        if profile.location.lower() in job.posting_data.location.lower():
            score += 40
        
        # Skills match (30 points)
        for skill in profile.skills:
            if skill.lower() in job.posting_data.description.lower():
                score += 15
                break
        
        # Salary compatibility (20 points)
        if profile.salary_expectation_min and job.posting_data.salary_max >= profile.salary_expectation_min:
            score += 20
        
        # Urgency match (10 points)
        if profile.immediate_need and job.posting_data.urgency in ["immediate", "today"]:
            score += 10
        
        return min(score, 100)
    
    # Fallback methods for when OpenAI is not available
    # Fallback methods for when OpenAI is not available
    async def _fallback_conversation_response(self, user_input: str, language: str) -> ConversationalFollowUp:
       """Basic rule-based conversation handling"""
       return ConversationalFollowUp(
           follow_up_questions=["Could you tell me more about your experience?"],
           missing_info=["experience", "location", "skills"],
           conversation_stage="information_gathering",
           next_action="ask_follow_up",
           response_in_user_language="Please tell me about your work experience."
       )
   
    async def _fallback_analyze_profile(self, conversation: str, language: str) -> UserProfileSchema:
       """Basic profile extraction without AI"""
       # Simple regex-based extraction
       name_match = re.search(r"(?:name|naam|नाम).{0,5}(\w+)", conversation.lower())
       name = name_match.group(1).title() if name_match else "User"
       
       location_match = re.search(r"(?:from|live|रहता|रहती).{0,10}(\w+)", conversation.lower())
       location = location_match.group(1).title() if location_match else "Not specified"
       
       return UserProfileSchema(
           name=name,
           location=location,
           skills=["General"],
           experience_years=0,
           job_preferences=["Any"],
           availability="Flexible",
           preferred_language=language
       )
   
    async def _fallback_job_matching(self, profile: UserProfileSchema, jobs: List[JobPosting]) -> List[tuple]:
       """Basic job matching without AI"""
       scored_jobs = []
       for job in jobs:
           score = await self._calculate_basic_match_score(profile, job)
           scored_jobs.append((job, score, None))
       
       scored_jobs.sort(key=lambda x: x[1], reverse=True)
       return scored_jobs

# Initialize AI agent
ai_agent = JobKrantiAdvancedAI(OPENAI_API_KEY)

# Seed demo data with more diverse jobs
async def seed_enhanced_demo_data():
   """Add comprehensive demo job postings in multiple languages and save to DB"""
   demo_postings = [
       JobPostingSchema(
           title="Security Guard - Delhi NCR",
           description="रात की शिफ्ट के लिए सिक्यूरिटी गार्ड चाहिए। अच्छी तनख्वाह और सुविधाएं।",
           location="Gurgaon, Sector 15",
           salary_min=18000,
           salary_max=22000,
           job_type="full_time",
           category="security",
           requirements=["2+ years experience", "Night shift availability", "Physical fitness"],
           contact_info="9876543210",
           urgency="within_week",
           benefits=["PF", "ESI", "Overtime pay", "Free meals"],
           working_hours="10 PM - 6 AM",
           original_language="hi"
       ),
       JobPostingSchema(
           title="Delivery Partner - Zomato",
           description="Food delivery के लिए partner चाहिए। Own bike required. Daily payment.",
           location="Mumbai, Andheri",
           salary_min=25000,
           salary_max=35000,
           job_type="gig",
           category="delivery",
           requirements=["Own bike", "Smartphone", "Driving license"],
           contact_info="9123456789",
           urgency="immediate",
           benefits=["Fuel allowance", "Daily payment", "Flexible hours"],
           working_hours="Flexible - 6 hours minimum",
           original_language="hi"
       ),
       JobPostingSchema(
           title="Maid for Household Work",
           description="Need reliable maid for daily cleaning and cooking. Good family, respectful environment.",
           location="Bangalore, Koramangala",
           salary_min=12000,
           salary_max=15000,
           job_type="part_time",
           category="cleaning",
           requirements=["Cooking skills", "Cleaning experience", "Trustworthy"],
           contact_info="9887766554",
           urgency="immediate",
           benefits=["Food provided", "Festival bonus", "Flexible timing"],
           working_hours="8 AM - 2 PM",
           original_language="en"
       ),
       JobPostingSchema(
           title="Factory Worker - Manufacturing",
           description="Production line worker needed. No experience required, training provided.",
           location="Chennai, Sriperumbudur",
           salary_min=16000,
           salary_max=20000,
           job_type="full_time",
           category="manufacturing",
           requirements=["Physical fitness", "Willingness to learn", "Day shift"],
           contact_info="9556677889",
           urgency="within_week",
           benefits=["ESI", "PF", "Annual bonus", "Transport"],
           working_hours="8 AM - 5 PM",
           original_language="ta"
       ),
       JobPostingSchema(
           title="Construction Helper",
           description="निर्माण कार्य में सहायक चाहिए। दैनिक मजदूरी। काम तुरंत शुरू।",
           location="Pune, Hinjawadi",
           salary_min=500,
           salary_max=700,
           job_type="daily",
           category="construction",
           requirements=["Physical strength", "No experience needed"],
           contact_info="9334455667",
           urgency="immediate",
           benefits=["Daily payment", "Overtime available"],
           working_hours="7 AM - 5 PM",
           original_language="hi"
       ),
       JobPostingSchema(
           title="Cook for Restaurant",
           description="अनुभवी रसोइया चाहिए। North Indian और Chinese खाना बनाना आना चाहिए।",
           location="Delhi, Karol Bagh",
           salary_min=20000,
           salary_max=25000,
           job_type="full_time",
           category="cooking",
           requirements=["3+ years cooking experience", "North Indian cuisine", "Chinese dishes"],
           contact_info="9445566778",
           urgency="within_week",
           benefits=["Food provided", "Accommodation available", "Tips"],
           working_hours="11 AM - 11 PM with breaks",
           original_language="hi"
       )
   ]
   
   for i, posting_data in enumerate(demo_postings):
       job = JobPosting(
           id=str(uuid4()),
           posting_data=posting_data,
           posted_by="demo_employer",
           verified=True
       )
       # Save to database
       await data_manager.save_job_posting(job)

# Create FastMCP server
mcp = FastMCP(
   "JobKranti AI - Voice-First Multilingual Job Platform for Bharat",
   auth=SimpleBearerAuthProvider(TOKEN),
)

# Enhanced MCP Tools

@mcp.tool
async def validate() -> str:
   """Validate this MCP server for PuchAI integration"""
   return MY_NUMBER

@mcp.tool
async def process_voice_message(
   audio_data_base64: Annotated[str, Field(description="Base64 encoded audio data from WhatsApp voice message")],
   user_phone: Annotated[str, Field(description="User's WhatsApp phone number")],
   language_hint: Annotated[str, Field(description="Language hint if known (hi, en, ta, etc.)")] = "auto"
) -> list[TextContent | ImageContent]:
   """Process voice message from WhatsApp - transcribe, understand, and respond with voice"""
   
   try:
       # Decode audio data
       audio_bytes = base64.b64decode(audio_data_base64)
       
       # Transcribe voice to text
       transcribed_text = await ai_agent.transcribe_voice_message(audio_bytes, language_hint)
       
       if not transcribed_text or "error" in transcribed_text.lower():
           return [TextContent(
               type="text",
               text="❌ Sorry, I couldn't understand the voice message. Please try speaking clearly or send a text message."
           )]
       
       # Detect language and intent
       language_analysis = await ai_agent.detect_language_and_intent(transcribed_text)
       detected_language = language_analysis.get("language", "hi")
       intent = language_analysis.get("intent", "general")
       
       # Get or create conversation session
       session_id = f"{user_phone}_{datetime.now().strftime('%Y%m%d')}"
       if session_id not in CONVERSATIONS:
           CONVERSATIONS[session_id] = ConversationSession(
               user_id=user_phone,
               session_id=session_id,
               messages=[],
               current_intent=intent,
               language=detected_language
           )
       
       session = CONVERSATIONS[session_id]
       session.messages.append({"role": "user", "content": transcribed_text, "timestamp": datetime.now()})
       session.last_interaction = datetime.now()
       
       # Save conversation to database
       await data_manager.save_conversation(session)
       
       # Generate conversational response
       follow_up = await ai_agent.generate_conversational_response(
           transcribed_text, 
           session.messages, 
           intent, 
           detected_language
       )
       
       # Add AI response to conversation
       session.messages.append({
           "role": "assistant", 
           "content": follow_up.response_in_user_language,
           "timestamp": datetime.now()
       })
       
       # Save updated conversation
       await data_manager.save_conversation(session)
       
       # Generate voice response
       voice_response_bytes = await ai_agent.generate_voice_response(
           follow_up.response_in_user_language, 
           detected_language
       )
       
       result_text = f"🎤 **Voice Message Processed**\n\n"
       result_text += f"🗣️ **You said** ({SUPPORTED_LANGUAGES.get(detected_language, detected_language)}): \"{transcribed_text}\"\n\n"
       result_text += f"🤖 **AI Response**: {follow_up.response_in_user_language}\n\n"
       result_text += f"🎯 **Intent Detected**: {intent}\n"
       result_text += f"🌐 **Language**: {SUPPORTED_LANGUAGES.get(detected_language, detected_language)}\n"
       
       if follow_up.follow_up_questions:
           result_text += f"\n❓ **Follow-up Questions**:\n"
           for q in follow_up.follow_up_questions[:3]:
               result_text += f"• {q}\n"
       
       if follow_up.missing_info:
           result_text += f"\n📝 **Still Need**: {', '.join(follow_up.missing_info)}\n"
       
       result_text += f"\n💡 **Next Step**: {follow_up.next_action}\n"
       result_text += f"🆔 **Session**: {session_id}"
       
       content_list = [TextContent(type="text", text=result_text)]
       
       # Add voice response if generated
       if voice_response_bytes:
           voice_base64 = base64.b64encode(voice_response_bytes).decode('utf-8')
           content_list.append(ImageContent(
               type="image",
               mimeType="audio/mpeg",
               data=voice_base64
           ))
       
       return content_list
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Voice processing failed: {str(e)}"))

@mcp.tool
async def create_profile_conversationally(
   conversation_text: Annotated[str, Field(description="Natural conversation in any Indian language")],
   phone: Annotated[str, Field(description="User's WhatsApp phone number")],
   language: Annotated[str, Field(description="Primary language used in conversation")] = "hi"
) -> str:
   """Create user profile through intelligent conversation analysis"""
   
   try:
       # Check if user already exists
       existing_user = await data_manager.get_user_by_phone(phone)
       if existing_user:
           return f"✅ **Profile already exists for {phone}**\n\nProfile ID: {existing_user.id}\nName: {existing_user.profile_data.name}\n\nUse the job search tool to find opportunities!"
       
       # Extract comprehensive profile
       profile_data = await ai_agent.intelligent_profile_extraction(conversation_text, language)
       
       user_id = str(uuid4())
       profile = UserProfile(
           id=user_id,
           phone=phone,
           profile_data=profile_data
       )
       
       # Save to database
       await data_manager.save_user_profile(profile)
       
       # Generate response in user's language
       response_prompt = f"""Generate a friendly profile confirmation message in {SUPPORTED_LANGUAGES.get(language, 'Hindi')} for this user profile:
       Name: {profile_data.name}
       Location: {profile_data.location}
       Skills: {', '.join(profile_data.skills)}
       Experience: {profile_data.experience_years} years
       
       Make it conversational and encouraging. Ask if anything needs to be corrected."""
       
       if ai_agent.client:
           try:
               completion = ai_agent.client.chat.completions.create(
                   model="gpt-4o-mini",
                   messages=[
                       {"role": "system", "content": "You are JobKranti AI. Generate friendly, encouraging messages in Indian languages."},
                       {"role": "user", "content": response_prompt}
                   ]
               )
               ai_response = completion.choices[0].message.content
           except:
               ai_response = "Profile created successfully! कुछ भी बदलना हो तो बताइए।"
       else:
           ai_response = "Profile created successfully! कुछ भी बदलना हो तो बताइए।"
       
       result = f"✅ **Profile Created Successfully!**\n\n"
       result += f"👤 **Name**: {profile_data.name}\n"
       result += f"📍 **Location**: {profile_data.location}\n"
       result += f"🛠️ **Skills**: {', '.join(profile_data.skills)}\n"
       result += f"📅 **Experience**: {profile_data.experience_years} years\n"
       result += f"💼 **Job Preferences**: {', '.join(profile_data.job_preferences)}\n"
       result += f"⏰ **Availability**: {profile_data.availability}\n"
       result += f"🗣️ **Language**: {SUPPORTED_LANGUAGES.get(profile_data.preferred_language, profile_data.preferred_language)}\n"
       
       if profile_data.immediate_need:
           result += f"🚨 **URGENT**: Job needed immediately for survival\n"
       
       if profile_data.salary_expectation_min:
           result += f"💰 **Salary Expectation**: ₹{profile_data.salary_expectation_min:,} - ₹{profile_data.salary_expectation_max or 50000:,}\n"
       
       result += f"\n🤖 **AI Response**: {ai_response}\n"
       result += f"🆔 **Profile ID**: `{user_id}`\n\n"
       result += f"🎯 **Ready to find jobs!** Use the job search tool to find opportunities."
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Profile creation failed: {str(e)}"))

@mcp.tool
async def smart_job_search(
   search_query: Annotated[str, Field(description="Natural language job search in any Indian language")],
   user_id: Annotated[str, Field(description="User profile ID for personalized results")] = None,
   max_results: Annotated[int, Field(description="Maximum number of results")] = 5,
   include_voice_response: Annotated[bool, Field(description="Generate voice response")] = False
) -> list[TextContent | ImageContent]:
   """AI-powered job search with voice support and multilingual understanding"""
   
   try:
       # Detect language and analyze search query
       language_analysis = await ai_agent.detect_language_and_intent(search_query)
       detected_language = language_analysis.get("language", "hi")
       
       # Get user profile if available
       user_profile = None
       if user_id and user_id in USERS:
           user_profile = USERS[user_id].profile_data
       
       # Analyze search criteria using AI
       search_criteria_prompt = f"""Analyze this job search query in {SUPPORTED_LANGUAGES.get(detected_language, 'Hindi')}:
       "{search_query}"
       
       Extract:
       - What types of jobs they want
       - Preferred location
       - Salary expectations
       - Urgency level
       - Any special requirements
       
       Handle colloquial terms and code-mixing. If someone says "koi bhi kaam" or "any job", set survival_mode=true."""
       
       if ai_agent.client:
           try:
               completion = ai_agent.client.beta.chat.completions.parse(
                   model="gpt-4o",
                   messages=[
                       {"role": "system", "content": "You are an expert at understanding job search queries in Indian languages."},
                       {"role": "user", "content": search_criteria_prompt}
                   ],
                   response_format=JobSearchCriteria
               )
               search_criteria = completion.choices[0].message.parsed
           except:
               # Fallback basic parsing
               search_criteria = JobSearchCriteria(
                   job_types=["general"],
                   location="any",
                   urgency="flexible",
                   work_type="full_time",
                   survival_mode="koi bhi" in search_query.lower() or "any job" in search_query.lower(),
                   language_detected=detected_language
               )
       else:
           search_criteria = JobSearchCriteria(
               job_types=["general"],
               location="any", 
               urgency="flexible",
               work_type="full_time",
               survival_mode=False,
               language_detected=detected_language
           )
       
       # Get available jobs
       available_jobs = list(JOBS.values())
       
       # Use AI for intelligent job matching if user profile available
       if user_profile and ai_agent.client:
           scored_jobs = await ai_agent.intelligent_job_matching(user_profile, available_jobs)
       else:
           # Basic matching for users without profiles
           scored_jobs = []
           for job in available_jobs:
               score = 50  # Base score
               
               # Location matching
               if search_criteria.location.lower() != "any":
                   if search_criteria.location.lower() in job.posting_data.location.lower():
                       score += 30
               else:
                   score += 20  # Any location is okay
               
               # Job type matching
               for job_type in search_criteria.job_types:
                   if job_type.lower() in job.posting_data.category.lower() or job_type.lower() in job.posting_data.title.lower():
                       score += 25
                       break
               
               # Urgency matching
               if search_criteria.urgency == job.posting_data.urgency:
                   score += 15
               
               # Survival mode - show any available job
               if search_criteria.survival_mode:
                   score += 20
               
               scored_jobs.append((job, score, None))
           
           scored_jobs.sort(key=lambda x: x[1], reverse=True)
       
       # Filter and limit results
       matching_jobs = scored_jobs[:max_results]
       
       if not matching_jobs:
           no_jobs_response = f"❌ **No jobs found for**: '{search_query}'\n\n"
           if detected_language == "hi":
               no_jobs_response += f"💡 **कोशिश करें**:\n• Security guard jobs\n• Delivery work\n• House cleaning\n• कोई भी काम"
           else:
               no_jobs_response += f"💡 **Try searching for**:\n• Security guard jobs\n• Delivery work\n• House cleaning\n• Any available work"
           
           return [TextContent(type="text", text=no_jobs_response)]
       
       # Build response
       result = f"🔍 **Found {len(matching_jobs)} jobs for**: '{search_query}'\n\n"
       result += f"🌐 **Language**: {SUPPORTED_LANGUAGES.get(detected_language, detected_language)}\n"
       result += f"📊 **Search Analysis**:\n"
       result += f"• Job types: {', '.join(search_criteria.job_types)}\n"
       result += f"• Location: {search_criteria.location}\n"
       result += f"• Urgency: {search_criteria.urgency}\n"
       
       if search_criteria.survival_mode:
           result += f"🚨 **SURVIVAL MODE**: Showing any available work\n"
       
       result += "\n"
       
       for i, (job, score, analysis) in enumerate(matching_jobs, 1):
           posting = job.posting_data
           
           # Determine emoji based on job safety and quality
           if score >= 80:
               safety_emoji = "🟢"
           elif score >= 60:
               safety_emoji = "🟡"
           else:
               safety_emoji = "🔴"
           
           result += f"**{i}. {safety_emoji} {posting.title}**\n"
           result += f"   📍 {posting.location}\n"
           result += f"   💰 ₹{posting.salary_min:,}"
           if posting.salary_max != posting.salary_min:
               result += f" - ₹{posting.salary_max:,}"
           result += f"/month\n"
           result += f"   ⏰ {posting.working_hours}\n"
           result += f"   📱 Contact: {posting.contact_info}\n"
           result += f"   🎯 Match Score: {score}%\n"
           
           if analysis and analysis.match_reasons:
               result += f"   ✅ Why it matches: {', '.join(analysis.match_reasons[:2])}\n"
           
           if posting.urgency in ["immediate", "today"]:
               result += f"   🚨 **URGENT REQUIREMENT**\n"
           
           result += f"   📝 {posting.description[:100]}...\n\n"
       
       content_list = [TextContent(type="text", text=result)]
       
       # Generate voice response if requested
       if include_voice_response and ai_agent.client:
           voice_summary = f"Found {len(matching_jobs)} jobs for you. Top match is {matching_jobs[0][0].posting_data.title} in {matching_jobs[0][0].posting_data.location} for ₹{matching_jobs[0][0].posting_data.salary_min} per month."
           
           try:
               voice_bytes = await ai_agent.generate_voice_response(voice_summary, detected_language)
               if voice_bytes:
                   voice_base64 = base64.b64encode(voice_bytes).decode('utf-8')
                   content_list.append(ImageContent(
                       type="image",
                       mimeType="audio/mpeg", 
                       data=voice_base64
                   ))
           except Exception as e:
               print(f"Voice generation error: {e}")
       
       return content_list
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Job search failed: {str(e)}"))

async def _fallback_parse_job_posting(description: str, contact: str, language: str) -> JobPostingSchema:
   """Fallback job posting parser without AI"""
   
   # Extract salary
   salary_patterns = [
       r"₹\s*(\d+(?:,\d+)*)",
       r"(\d+)k",
       r"(\d+)\s*हजार",
       r"(\d+)\s*thousand"
   ]
   
   salary = 15000  # Default
   for pattern in salary_patterns:
       match = re.search(pattern, description.lower())
       if match:
           amount = int(match.group(1).replace(',', ''))
           if 'k' in pattern or 'हजार' in pattern or 'thousand' in pattern:
               amount *= 1000
           salary = amount
           break
   
   # Determine category
   category = "general"
   category_keywords = {
       "security": ["security", "guard", "सिक्यूरिटी", "गार्ड"],
       "delivery": ["delivery", "courier", "डिलीवरी"],
       "cleaning": ["clean", "maid", "साफ", "सफाई"],
       "cooking": ["cook", "chef", "खाना", "पकाना"],
       "construction": ["construction", "निर्माण", "मजदूर"]
   }
   
   for cat, keywords in category_keywords.items():
       if any(keyword in description.lower() for keyword in keywords):
           category = cat
           break
   
   # Determine urgency
   urgency = "flexible"
   if any(word in description.lower() for word in ["urgent", "तुरंत", "immediately", "abhi"]):
       urgency = "immediate"
   
   return JobPostingSchema(
       title=f"{category.title()} Position",
       description=description,
       location="Location not specified",
       salary_min=salary,
       salary_max=salary,
       job_type="full_time",
       category=category,
       requirements=["Experience preferred"],
       contact_info=contact,
       urgency=urgency,
       benefits=[],
       working_hours="Standard hours",
       original_language=language
   )

@mcp.tool
async def post_job_with_ai(
   job_description: Annotated[str, Field(description="Natural language job posting in any Indian language")],
   employer_contact: Annotated[str, Field(description="Employer contact information")],
   employer_language: Annotated[str, Field(description="Employer's preferred language")] = "hi"
) -> str:
   """Post a job using AI to parse natural language descriptions"""
   
   try:
       # Detect language and parse job posting
       language_analysis = await ai_agent.detect_language_and_intent(job_description)
       detected_language = language_analysis.get("language", employer_language)
       
       # Use AI to structure the job posting
       if ai_agent.client:
           try:
               system_prompt = f"""Parse this job posting in {SUPPORTED_LANGUAGES.get(detected_language, 'Hindi')} and extract structured information.
               
               Handle:
               - Local terms (e.g., "काम", "नौकरी", "मजदूरी")
               - Salary in various formats (₹15000, 15k, पंद्रह हजार)
               - Location nicknames and areas
               - Urgency indicators ("तुरंत", "urgent", "abhi chahiye")
               - Benefits mentioned casually
               """
               
               completion = ai_agent.client.beta.chat.completions.parse(
                   model="gpt-4o",
                   messages=[
                       {"role": "system", "content": system_prompt},
                       {"role": "user", "content": f"Parse this job posting: {job_description}"}
                   ],
                   response_format=JobPostingSchema
               )
               
               posting_data = completion.choices[0].message.parsed
               posting_data.contact_info = employer_contact
               posting_data.original_language = detected_language
               
           except Exception as e:
               print(f"AI parsing error: {e}")
               # Fallback parsing
               posting_data = await _fallback_parse_job_posting(job_description, employer_contact, detected_language)
       else:
           posting_data = await _fallback_parse_job_posting(job_description, employer_contact, detected_language)
       
       job_id = str(uuid4())
       job = JobPosting(
           id=job_id,
           posting_data=posting_data,
           posted_by="employer",
           verified=False
       )
       
       # Save to database
       await data_manager.save_job_posting(job)
       
       # Find potential candidates using AI matching
       potential_candidates = []
       if USERS:
           for user_id, user in USERS.items():
               try:
                   # Calculate match score
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
                   
                   # Salary compatibility
                   if profile.salary_expectation_max and posting_data.salary_min <= profile.salary_expectation_max:
                       score += 20
                   
                   # Immediate need match
                   if profile.immediate_need and posting_data.urgency in ["immediate", "today"]:
                       score += 25
                   
                   if score > 30:
                       potential_candidates.append((user, score))
                       
               except Exception as e:
                   print(f"Candidate matching error: {e}")
       
       # Sort candidates by score
       potential_candidates.sort(key=lambda x: x[1], reverse=True)
       
       # Generate response in employer's language
       if detected_language == "hi":
           success_msg = "नौकरी सफलतापूर्वक पोस्ट की गई!"
           job_details = f"""
📋 **पद**: {posting_data.title}
📍 **स्थान**: {posting_data.location}
💰 **वेतन**: ₹{posting_data.salary_min:,} - ₹{posting_data.salary_max:,}/महीना
⏰ **समय**: {posting_data.working_hours}
📱 **संपर्क**: {posting_data.contact_info}
"""
       else:
           success_msg = "Job posted successfully!"
           job_details = f"""
📋 **Position**: {posting_data.title}
📍 **Location**: {posting_data.location}
💰 **Salary**: ₹{posting_data.salary_min:,} - ₹{posting_data.salary_max:,}/month
⏰ **Hours**: {posting_data.working_hours}
📱 **Contact**: {posting_data.contact_info}
"""
       
       result = f"✅ **{success_msg}**\n\n"
       result += f"🆔 **Job ID**: `{job_id}`\n"
       result += job_details
       
       if posting_data.urgency in ["immediate", "today"]:
           result += f"🚨 **URGENT REQUIREMENT**\n"
       
       if posting_data.benefits:
           result += f"🎁 **Benefits**: {', '.join(posting_data.benefits)}\n"
       
       if potential_candidates:
           result += f"\n🎯 **Found {len(potential_candidates)} Potential Candidates:**\n"
           for (user, score) in potential_candidates[:3]:
               profile = user.profile_data
               result += f"• **{profile.name}** ({profile.location}) - {', '.join(profile.skills[:2])} - {profile.experience_years}yr exp - Score: {score}%\n"
               if profile.immediate_need:
                   result += f"  🚨 *Needs job immediately*\n"
       else:
           if detected_language == "hi":
               result += f"\n📢 **आपकी नौकरी {posting_data.location} के सभी जॉब सीकर्स को दिखेगी**"
           else:
               result += f"\n📢 **Your job will be visible to all job seekers in {posting_data.location}**"
       
       if detected_language == "hi":
           result += f"\n💡 **अगला कदम**: उम्मीदवार आपसे {posting_data.contact_info} पर संपर्क करेंगे"
       else:
           result += f"\n💡 **Next Steps**: Candidates will contact you directly at {posting_data.contact_info}"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Job posting failed: {str(e)}"))

@mcp.tool
async def conversational_job_assistant(
   user_message: Annotated[str, Field(description="User's message in any language")],
   user_phone: Annotated[str, Field(description="User's phone number")],
   conversation_context: Annotated[str, Field(description="Previous conversation context")] = ""
) -> str:
   """Intelligent conversational assistant that handles job-related queries with follow-ups"""
   
   try:
       # Detect language and intent
       language_analysis = await ai_agent.detect_language_and_intent(user_message)
       detected_language = language_analysis.get("language", "hi")
       intent = language_analysis.get("intent", "general")
       
       # Get or create conversation session
       session_id = f"{user_phone}_{datetime.now().strftime('%Y%m%d')}"
       if session_id not in CONVERSATIONS:
           CONVERSATIONS[session_id] = ConversationSession(
               user_id=user_phone,
               session_id=session_id,
               messages=[],
               current_intent=intent,
               language=detected_language
           )
       
       session = CONVERSATIONS[session_id]
       session.messages.append({"role": "user", "content": user_message})
       session.last_interaction = datetime.now()
       
       # Generate contextual response
       if ai_agent.client:
           try:
               # Build conversation context
               recent_messages = session.messages[-10:]  # Last 10 messages
               context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
               
               system_prompt = f"""You are JobKranti AI, a helpful assistant for India's job marketplace.
               
               User's language: {SUPPORTED_LANGUAGES.get(detected_language, detected_language)}
               Current intent: {intent}
               
               Guidelines:
               1. Respond primarily in the user's detected language
               2. Be empathetic to economic pressures and family situations
               3. Ask relevant follow-up questions to help users
               4. Guide toward actionable outcomes (profile creation, job search, job posting)
               5. Handle urgent survival needs sensitively
               6. Support blue-collar, gig, and informal workers
               7. Use simple, respectful language
               8. Offer specific help based on their situation
               
               Current conversation context: {context}
               
               Provide helpful, actionable guidance."""
               
               completion = ai_agent.client.chat.completions.create(
                   model="gpt-4o",
                   messages=[
                       {"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_message}
                   ]
               )
               
               ai_response = completion.choices[0].message.content
               
           except Exception as e:
               print(f"Conversational AI error: {e}")
               ai_response = await _fallback_conversational_response(user_message, detected_language, intent)
       else:
           ai_response = await _fallback_conversational_response(user_message, detected_language, intent)
       
       # Add AI response to session
       session.messages.append({"role": "assistant", "content": ai_response})
       
       # Save conversation to database
       await data_manager.save_conversation(session)
       
       # Format final response
       result = f"🤖 **JobKranti AI Assistant**\n\n"
       result += f"🗣️ **You**: {user_message}\n\n"
       result += f"💬 **Response**: {ai_response}\n\n"
       result += f"🌐 **Language**: {SUPPORTED_LANGUAGES.get(detected_language, detected_language)}\n"
       result += f"🎯 **Intent**: {intent}\n"
       result += f"🆔 **Session**: {session_id}"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Conversation failed: {str(e)}"))

async def _fallback_conversational_response(message: str, language: str, intent: str) -> str:
   """Fallback conversational responses"""
   
   responses = {
       "hi": {
           "job_search": "मैं आपको नौकरी खोजने में मदद कर सकता हूं। आप किस तरह का काम ढूंढ रहे हैं?",
           "profile": "आपकी प्रोफाइल बनाने के लिए, कृपया अपना नाम, कहां रहते हैं, और क्या काम करते हैं बताइए।",
           "general": "मैं JobKranti AI हूं। मैं नौकरी खोजने और देने में आपकी मदद कर सकता हूं। आप क्या चाहते हैं?"
       },
       "en": {
           "job_search": "I can help you find a job. What kind of work are you looking for?",
           "profile": "To create your profile, please tell me your name, where you live, and what work you do.",
           "general": "I'm JobKranti AI. I can help you find jobs or post job openings. What do you need?"
       }
   }
   
   lang_responses = responses.get(language, responses["en"])
   return lang_responses.get(intent, lang_responses["general"])

@mcp.tool
async def emergency_job_finder(
   user_message: Annotated[str, Field(description="Urgent job search request in any language")],
   user_phone: Annotated[str, Field(description="User's phone number")],
   location: Annotated[str, Field(description="User's current location")] = ""
) -> str:
   """Emergency job finder for people who need work immediately for survival"""
   
   try:
       # Detect language
       language_analysis = await ai_agent.detect_language_and_intent(user_message)
       detected_language = language_analysis.get("language", "hi")
       
       # Find all immediate/urgent jobs
       urgent_jobs = []
       for job in JOBS.values():
           posting = job.posting_data
           if posting.urgency in ["immediate", "today"]:
               urgent_jobs.append(job)
       
       # Sort by salary (higher first) for survival situations
       urgent_jobs.sort(key=lambda x: x.posting_data.salary_max, reverse=True)
       
       if not urgent_jobs:
           if detected_language == "hi":
               return """🚨 **तुरंत काम चाहिए - Emergency Support**

फिलहाल कोई तुरंत वाली नौकरी उपलब्ध नहीं है। लेकिन यहां कुछ तुरंत कमाई के तरीके हैं:

📞 **तुरंत संपर्क करें**:
- Zomato/Swiggy delivery: Call local office
- Ola/Uber driver: Apply online
- Construction daily labor: पास के साइट पर जाएं
- House cleaning: आस-पास के घरों में पूछें

💡 **आज ही कमाई शुरू**:
- Daily wage labor: ₹400-600/day
- Food delivery: ₹800-1200/day
- House help: ₹300-500/day

🆘 **Emergency Help**: अगर बहुत ज्यादा परेशानी है तो local NGO या helpline से संपर्क करें।

हम आपकी प्रोफाइल बना देते हैं ताकि नई नौकरियां आते ही आपको पता चल जाए।"""
           else:
               return """🚨 **Need Work Immediately - Emergency Support**

No urgent jobs available right now, but here are immediate earning options:

📞 **Contact Immediately**:
- Zomato/Swiggy delivery: Call local office
- Ola/Uber driver: Apply online  
- Construction daily labor: Visit nearby sites
- House cleaning: Ask nearby homes

💡 **Start Earning Today**:
- Daily wage labor: ₹400-600/day
- Food delivery: ₹800-1200/day
- House help: ₹300-500/day

🆘 **Emergency Help**: If in severe distress, contact local NGO or helpline.

Let me create your profile so you get notified immediately when new jobs come."""
       
       # Show urgent jobs
       result = f"🚨 **EMERGENCY JOB SEARCH** 🚨\n\n"
       
       if detected_language == "hi":
           result += f"तुरंत काम चाहिए - यहां हैं {len(urgent_jobs)} नौकरियां:\n\n"
       else:
           result += f"Need work immediately - Here are {len(urgent_jobs)} urgent jobs:\n\n"
       
       for i, job in enumerate(urgent_jobs[:5], 1):
           posting = job.posting_data
           result += f"**{i}. 🔥 {posting.title}**\n"
           result += f"   📍 {posting.location}\n"
           result += f"   💰 ₹{posting.salary_min:,}/month (₹{posting.salary_min//30:,}/day)\n"
           result += f"   📱 **CALL NOW**: {posting.contact_info}\n"
           result += f"   ⏰ Start: {posting.urgency.upper()}\n"
           result += f"   📝 {posting.description[:80]}...\n\n"
       
       if detected_language == "hi":
           result += f"🔥 **तुरंत action लें**:\n"
           result += f"1. ऊपर दिए गए नंबरों पर अभी कॉल करें\n"
           result += f"2. अपना नाम, location और experience बताएं\n"
           result += f"3. कब join कर सकते हैं बताएं\n\n"
           result += f"💪 **हिम्मत रखें** - काम मिल जाएगा!"
       else:
           result += f"🔥 **Take immediate action**:\n"
           result += f"1. Call the numbers above RIGHT NOW\n"
           result += f"2. Tell them your name, location and experience\n"
           result += f"3. Say when you can join\n\n"
           result += f"💪 **Stay strong** - You will find work!"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Emergency job search failed: {str(e)}"))

@mcp.tool
async def multilingual_job_matchmaker(
   user_profile_id: Annotated[str, Field(description="User profile ID")],
   employer_requirements: Annotated[str, Field(description="What employer is looking for in any language")],
   employer_language: Annotated[str, Field(description="Employer's preferred language")] = "hi"
) -> str:
   """AI-powered matchmaking between job seekers and employers across languages"""
   
   if user_profile_id not in USERS:
       raise McpError(ErrorData(code=INVALID_PARAMS, message="User profile not found"))
   
   try:
       user_profile = USERS[user_profile_id].profile_data
       
       # Use AI to analyze compatibility
       if ai_agent.client:
           try:
               matchmaking_prompt = f"""Analyze compatibility between job seeker and employer requirements:

JOB SEEKER PROFILE:
- Name: {user_profile.name}
- Location: {user_profile.location}
- Skills: {', '.join(user_profile.skills)}
- Experience: {user_profile.experience_years} years
- Availability: {user_profile.availability}
- Salary expectation: ₹{user_profile.salary_expectation_min}-{user_profile.salary_expectation_max}
- Languages: {user_profile.preferred_language}, {', '.join(user_profile.secondary_languages)}
- Immediate need: {user_profile.immediate_need}

EMPLOYER REQUIREMENTS (in {SUPPORTED_LANGUAGES.get(employer_language, employer_language)}):
{employer_requirements}

Provide detailed match analysis, potential concerns, and recommendations for both parties."""

               completion = ai_agent.client.chat.completions.create(
                   model="gpt-4o",
                   messages=[
                       {"role": "system", "content": "You are an expert job matchmaker for Indian workers and employers."},
                       {"role": "user", "content": matchmaking_prompt}
                   ]
               )
               
               ai_analysis = completion.choices[0].message.content
               
           except Exception as e:
               print(f"AI matchmaking error: {e}")
               ai_analysis = "Basic compatibility analysis available."
       else:
           ai_analysis = "AI analysis not available - showing basic profile match."
       
       # Generate response in employer's language
       if employer_language == "hi":
           result = f"🤝 **Job Seeker - Employer Matchmaking**\n\n"
           result += f"👤 **उम्मीदवार की जानकारी**:\n"
           result += f"• नाम: {user_profile.name}\n"
           result += f"• स्थान: {user_profile.location}\n"
           result += f"• कौशल: {', '.join(user_profile.skills)}\n"
           result += f"• अनुभव: {user_profile.experience_years} साल\n"
           result += f"• उपलब्धता: {user_profile.availability}\n"
           result += f"• वेतन अपेक्षा: ₹{user_profile.salary_expectation_min or 'फ्लेक्सिबल'} - ₹{user_profile.salary_expectation_max or 'बातचीत के बाद'}\n"
           
           if user_profile.immediate_need:
               result += f"🚨 **तुरंत काम चाहिए**\n"
               
           result += f"\n💼 **आपकी जरूरतें**: {employer_requirements}\n\n"
           result += f"🤖 **AI विश्लेषण**: {ai_analysis}\n\n"
           result += f"📱 **संपर्क**: इस उम्मीदवार से बात करने के लिए, उनका फोन नंबर: {USERS[user_profile_id].phone}"
       else:
           result = f"🤝 **Job Seeker - Employer Matchmaking**\n\n"
           result += f"👤 **Candidate Information**:\n"
           result += f"• Name: {user_profile.name}\n"
           result += f"• Location: {user_profile.location}\n"
           result += f"• Skills: {', '.join(user_profile.skills)}\n"
           result += f"• Experience: {user_profile.experience_years} years\n"
           result += f"• Availability: {user_profile.availability}\n"
           result += f"• Salary expectation: ₹{user_profile.salary_expectation_min or 'Flexible'} - ₹{user_profile.salary_expectation_max or 'Negotiable'}\n"
           
           if user_profile.immediate_need:
               result += f"🚨 **Needs work immediately**\n"
               
           result += f"\n💼 **Your requirements**: {employer_requirements}\n\n"
           result += f"🤖 **AI Analysis**: {ai_analysis}\n\n"
           result += f"📱 **Contact**: To speak with this candidate, their phone: {USERS[user_profile_id].phone}"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Matchmaking failed: {str(e)}"))

@mcp.tool
async def get_job_market_insights(
   location: Annotated[str, Field(description="City or area name")],
   job_category: Annotated[str, Field(description="Type of job (security, delivery, cleaning, etc.)")] = "general",
   language: Annotated[str, Field(description="Response language preference")] = "hi"
) -> str:
   """Get real-time job market insights and salary trends for specific location and job type"""
   
   try:
       # Analyze current job market from our data
       location_jobs = []
       category_jobs = []
       
       for job in JOBS.values():
           posting = job.posting_data
           
           if location.lower() in posting.location.lower():
               location_jobs.append(posting)
           
           if job_category.lower() in posting.category.lower() or job_category == "general":
               category_jobs.append(posting)
       
       # Calculate insights
       if location_jobs:
           avg_salary = sum(job.salary_max for job in location_jobs) // len(location_jobs)
           min_salary = min(job.salary_min for job in location_jobs)
           max_salary = max(job.salary_max for job in location_jobs)
           urgent_jobs = len([job for job in location_jobs if job.urgency in ["immediate", "today"]])
       else:
           avg_salary, min_salary, max_salary, urgent_jobs = 18000, 12000, 25000, 0
       
       # Generate insights using AI if available
       if ai_agent.client:
           try:
               insights_prompt = f"""Provide job market insights for {job_category} jobs in {location}, India.
               
               Current data:
               - Available jobs: {len(location_jobs)}
               - Average salary: ₹{avg_salary}
               - Salary range: ₹{min_salary} - ₹{max_salary}
               - Urgent openings: {urgent_jobs}
               
               Provide insights in {SUPPORTED_LANGUAGES.get(language, language)} about:
               1. Market demand
               2. Salary trends
               3. Best opportunities
               4. Tips for job seekers
               5. Future outlook"""
               
               completion = ai_agent.client.chat.completions.create(
                   model="gpt-4o-mini",
                   messages=[
                       {"role": "system", "content": "You are a job market analyst for Indian blue-collar workers."},
                       {"role": "user", "content": insights_prompt}
                   ]
               )
               
               ai_insights = completion.choices[0].message.content
               
           except Exception as e:
               print(f"AI insights error: {e}")
               ai_insights = "Market analysis in progress..."
       else:
           ai_insights = "Basic market data available."
       
       # Format response based on language
       if language == "hi":
           result = f"📊 **{location} में {job_category} जॉब मार्केट इनसाइट्स**\n\n"
           result += f"📈 **बाजार की स्थिति**:\n"
           result += f"• उपलब्ध नौकरियां: {len(location_jobs)}\n"
           result += f"• औसत वेतन: ₹{avg_salary:,}/महीना\n"
           result += f"• वेतन रेंज: ₹{min_salary:,} - ₹{max_salary:,}\n"
           result += f"• तुरंत भर्ती: {urgent_jobs} पोजीशन\n\n"
           result += f"🤖 **AI विश्लेषण**:\n{ai_insights}\n\n"
           result += f"💡 **सुझाव**:\n"
           if urgent_jobs > 0:
               result += f"• {urgent_jobs} तुरंत नौकरियां उपलब्ध - जल्दी अप्लाई करें\n"
           result += f"• औसत वेतन से {10 if avg_salary > 20000 else 5}% ज्यादा मांगें\n"
           result += f"• स्किल डेवलपमेंट पर फोकस करें"
       else:
           result = f"📊 **Job Market Insights for {job_category} in {location}**\n\n"
           result += f"📈 **Market Overview**:\n"
           result += f"• Available jobs: {len(location_jobs)}\n"
           result += f"• Average salary: ₹{avg_salary:,}/month\n"
           result += f"• Salary range: ₹{min_salary:,} - ₹{max_salary:,}\n"
           result += f"• Immediate openings: {urgent_jobs} positions\n\n"
           result += f"🤖 **AI Analysis**:\n{ai_insights}\n\n"
           result += f"💡 **Recommendations**:\n"
           if urgent_jobs > 0:
               result += f"• {urgent_jobs} urgent jobs available - apply quickly\n"
           result += f"• Negotiate {10 if avg_salary > 20000 else 5}% above average salary\n"
           result += f"• Focus on skill development"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Market insights failed: {str(e)}"))

@mcp.tool
async def backup_all_data() -> str:
   """Create a backup of all data (for hackathon safety)"""
   try:
       backup_file = await data_manager.backup_data()
       if backup_file:
           return f"✅ **Data Backup Created**\n\nFile: {backup_file}\nUsers: {len(USERS)}\nJobs: {len(JOBS)}\nConversations: {len(CONVERSATIONS)}"
       else:
           return "❌ **Backup Failed** - Check server logs"
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Backup failed: {str(e)}"))

@mcp.tool
async def get_platform_statistics() -> str:
   """Get comprehensive platform statistics for demo/presentation"""
   try:
       # Calculate statistics
       total_users = len(USERS)
       total_jobs = len(JOBS)
       total_conversations = len(CONVERSATIONS)
       
       # Language distribution
       language_stats = {}
       for user in USERS.values():
           lang = user.profile_data.preferred_language
           language_stats[lang] = language_stats.get(lang, 0) + 1
       
       # Job category distribution
       category_stats = {}
       for job in JOBS.values():
           cat = job.posting_data.category
           category_stats[cat] = category_stats.get(cat, 0) + 1
       
       # Urgent jobs count
       urgent_jobs = len([job for job in JOBS.values() if job.posting_data.urgency in ["immediate", "today"]])
       
       # Users needing immediate work
       survival_users = len([user for user in USERS.values() if user.profile_data.immediate_need])
       
       # Active conversations (last 24 hours)
       yesterday = datetime.now() - timedelta(hours=24)
       active_conversations = len([conv for conv in CONVERSATIONS.values() if conv.last_interaction > yesterday])
       
       result = f"📊 **JobKranti Platform Statistics**\n\n"
       result += f"👥 **Users**: {total_users}\n"
       result += f"💼 **Jobs Posted**: {total_jobs}\n"
       result += f"💬 **Conversations**: {total_conversations}\n"
       result += f"🔥 **Active (24h)**: {active_conversations}\n\n"
       
       result += f"🚨 **Urgent Needs**:\n"
       result += f"• Emergency job seekers: {survival_users}\n"
       result += f"• Immediate job openings: {urgent_jobs}\n\n"
       
       if language_stats:
           result += f"🌍 **Language Distribution**:\n"
           for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
               lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
               result += f"• {lang_name}: {count} users\n"
           result += "\n"
       
       if category_stats:
           result += f"💼 **Job Categories**:\n"
           for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
               result += f"• {cat.title()}: {count} jobs\n"
           result += "\n"
       
       result += f"🏆 **Hackathon Demo Ready!**\n"
       result += f"Database: {'✅ SQLite' if DB_PATH.exists() else '❌ Memory only'}\n"
       result += f"Voice Processing: {'✅ OpenAI' if OPENAI_API_KEY else '❌ Not configured'}\n"
       result += f"Multilingual AI: {'✅ 13 languages' if OPENAI_API_KEY else '❌ Limited'}"
       
       return result
       
   except Exception as e:
       raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Statistics failed: {str(e)}"))

# Custom routes for health checks and information
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
   """Enhanced health check with system status"""
   return JSONResponse({
       "status": "healthy",
       "service": "JobKranti AI - Voice-First Multilingual Job Platform",
       "version": "2.0.0",
       "port": 8086,
       "features": {
           "voice_processing": bool(OPENAI_API_KEY),
           "multilingual_support": True,
           "ai_powered": bool(OPENAI_API_KEY),
           "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
           "tools_available": 10
       },
       "stats": {
           "total_users": len(USERS),
           "total_jobs": len(JOBS),
           "active_conversations": len(CONVERSATIONS),
           "supported_languages": len(SUPPORTED_LANGUAGES)
       }
   })

@mcp.custom_route("/", methods=["GET"])
async def root_endpoint(request: Request) -> JSONResponse:
   """Enhanced root endpoint with comprehensive information"""
   return JSONResponse({
       "service": "JobKranti AI - Voice-First Multilingual Job Platform for Bharat",
       "status": "running",
       "description": "Advanced AI-powered job marketplace with voice support for India's blue-collar workforce",
       "mcp_endpoint": "/mcp",
       "health_endpoint": "/health",
       "auth_required": True,
       "transport": "streamable-http",
       "features": {
           "voice_transcription": "OpenAI Whisper",
           "voice_synthesis": "OpenAI TTS", 
           "multilingual_ai": "GPT-4o with Indian languages",
           "conversation_flow": "Intelligent follow-ups",
           "emergency_job_finder": "Survival mode job search",
           "ai_matchmaking": "Smart candidate-employer matching",
           "persistent_storage": "SQLite with backup"
       },
       "supported_languages": SUPPORTED_LANGUAGES,
       "tools": [
           {
               "name": "process_voice_message",
               "description": "Process WhatsApp voice messages with transcription and voice response"
           },
           {
               "name": "create_profile_conversationally", 
               "description": "Create user profiles from natural conversation"
           },
           {
               "name": "smart_job_search",
               "description": "AI-powered job search with voice support"
           },
           {
               "name": "post_job_with_ai",
               "description": "Post jobs using natural language parsing"
           },
           {
               "name": "conversational_job_assistant",
               "description": "Intelligent job-related conversations with follow-ups"
           },
           {
               "name": "emergency_job_finder",
               "description": "Urgent job search for survival situations"
           },
           {
               "name": "multilingual_job_matchmaker",
               "description": "AI-powered candidate-employer matching"
           },
           {
               "name": "get_job_market_insights",
               "description": "Real-time job market analysis and trends"
           },
           {
               "name": "backup_all_data",
               "description": "Create data backup for safety"
           },
           {
               "name": "get_platform_statistics",
               "description": "Comprehensive platform statistics"
           }
       ]
   })

@mcp.custom_route("/demo", methods=["GET"])
async def demo_endpoint(request: Request) -> JSONResponse:
   """Demo data and examples for testing"""
   return JSONResponse({
       "demo_conversations": {
           "job_seeker_hindi": "नमस्ते, मुझे काम चाहिए। मैं राहुल हूं, दिल्ली में रहता हूं। Security guard का काम करता था 3 साल। अच्छी salary चाहिए।",
           "job_seeker_english": "Hi, I need a job urgently. I'm Priya from Mumbai. I can do cleaning work, cooking also. Have 2 years experience.",
           "employer_hindi": "मुझे एक delivery boy चाहिए Mumbai में। रोज 1000 rupees दूंगा। Own bike होना चाहिए। तुरंत join करना है।",
           "emergency_request": "भाई, मुझे आज ही कोई काम चाहिए। घर में खाना नहीं है। कोई भी काम करूंगा।"
       },
       "sample_voice_scenarios": [
           "WhatsApp voice message asking for maid work in Hindi",
           "Tamil voice message from construction worker",
           "Emergency Bengali voice message needing immediate work",
           "Employer voice message posting delivery job in Hinglish"
       ],
       "demo_jobs": len(JOBS),
       "languages_supported": len(SUPPORTED_LANGUAGES),
       "voice_features": {
           "transcription": "Whisper API supports all Indian languages",
           "synthesis": "TTS with Indian accent support",
           "real_time": "Sub-second processing for conversations"
       }
   })

# Enhanced main function with database initialization
async def main():
   print("🚀 Starting JobKranti AI-Powered Voice-First Multilingual MCP Server")
   print("🎤 Voice Processing: OpenAI Whisper (Transcription) + TTS (Voice Response)")
   print("🧠 AI Engine: GPT-4o with advanced Indian language support")
   print("🌍 Languages: 13 Indian languages + English")
   print("💾 Storage: SQLite database with in-memory caching")
   print("💼 Focus: Blue-collar, gig workers, emergency job needs")
   print("📱 Integration: WhatsApp voice messages, PuchAI ready")
   
   # Initialize database
   try:
       await data_manager.init_database()
       await data_manager.load_all_data()
       
       # Add demo data if database is empty
       if len(JOBS) == 0:
           print("📊 Loading demo data...")
           await seed_enhanced_demo_data()
           print(f"✅ {len(JOBS)} demo jobs loaded")
       else:
           print(f"📊 Loaded existing data: {len(USERS)} users, {len(JOBS)} jobs")
           
   except Exception as e:
       print(f"⚠️ Database initialization failed: {e}")
       print("🔄 Running in memory-only mode")
       # Load demo data in memory for demonstration
       await seed_enhanced_demo_data()
   
   print("🔗 Server: http://0.0.0.0:8086")
   print("🔗 MCP Endpoint: /mcp")
   print("🩺 Health Check: /health")
   print("🎮 Demo: /demo")
   
   if OPENAI_API_KEY:
       print("✅ OpenAI API connected - Full AI features enabled")
       print("   • Voice transcription (Whisper)")
       print("   • Voice synthesis (TTS)")
       print("   • Multilingual conversations (GPT-4o)")
       print("   • Intelligent matching and insights")
   else:
       print("⚠️  OpenAI API not configured - Fallback mode")
   
   print("🏆 Ready for PuchAI Hackathon 2025!")
   
   # Use the correct transport with default path
   await mcp.run_async(transport="http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
   asyncio.run(main())