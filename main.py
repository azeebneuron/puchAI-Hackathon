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
            
            await db.commit()
            print("✅ Database initialized successfully")
    
    @staticmethod
    async def save_user_profile(user: UserProfile):
        """Save user profile to database and update cache - FIXED VERSION"""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO users 
                    (id, phone, profile_data, conversation_history, voice_samples, created_at, last_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user.id,
                    user.phone,
                    json.dumps(user.profile_data.model_dump(), default=str),  # FIXED: use model_dump()
                    json.dumps(user.conversation_history, default=str),
                    json.dumps(user.voice_samples),
                    user.created_at.isoformat(),
                    user.last_active.isoformat()
                ))
                await db.commit()  # CRITICAL: Make sure this executes
            
            # Update cache
            USERS[user.id] = user
            print(f"✅ User {user.id} saved to database")
            
        except Exception as e:
            print(f"❌ Error saving user {user.id}: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    async def save_job_posting(job: JobPosting):
        """Save job posting to database and update cache - FIXED VERSION"""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO jobs 
                    (id, posting_data, posted_by, verified, views, applications, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job.id,
                    json.dumps(job.posting_data.model_dump(), default=str),  # FIXED: use model_dump()
                    job.posted_by,
                    job.verified,
                    job.views,
                    job.applications,
                    job.created_at.isoformat(),
                    job.expires_at.isoformat()
                ))
                await db.commit()  # CRITICAL: Make sure this executes
            
            # Update cache
            JOBS[job.id] = job
            print(f"✅ Job {job.id} saved to database")
            
            # Debug: Check if data was actually saved
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute("SELECT COUNT(*) FROM jobs") as cursor:
                    count = await cursor.fetchone()
                    print(f"📊 Total jobs in database: {count[0]}")
            
        except Exception as e:
            print(f"❌ Error saving job {job.id}: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            print(f"✅ Loaded {len(USERS)} users, {len(JOBS)} jobs from database")
            
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
    
    def get_available_jobs_summary(self) -> str:
        """Get summary of available jobs for AI context"""
        try:
            jobs_summary = []
            for job in list(JOBS.values())[:5]:  # Show first 5 jobs
                posting = job.posting_data
                jobs_summary.append(f"- {posting.title} in {posting.location} - ₹{posting.salary_min}/month")
            return "\n".join(jobs_summary) if jobs_summary else "Sample jobs: Maid in Bangalore, Security guard in Delhi, Delivery in Mumbai"
        except:
            return "Various jobs available across India"
    
    async def transcribe_voice_message(self, audio_data: bytes, language_hint: str = "hi") -> str:
        """Convert voice message to text using OpenAI Whisper"""
        if not self.client:
            return "Voice transcription not available - OpenAI not configured"
        
        try:
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "voice_message.mp3"
            
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

# Initialize AI agent
ai_agent = JobKrantiAdvancedAI(OPENAI_API_KEY)

# Enhanced job posting helper function
async def create_job_posting_from_message(message: str, contact: str, language: str = "hi") -> str:
    """Create job posting from natural language message"""
    try:
        if ai_agent.client:
            completion = ai_agent.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": f"Parse job posting in {SUPPORTED_LANGUAGES.get(language, 'Hindi')} and extract details."
                    },
                    {"role": "user", "content": f"Parse: {message}"}
                ],
                response_format=JobPostingSchema
            )
            
            posting_data = completion.choices[0].message.parsed
            posting_data.contact_info = contact
            posting_data.original_language = language
        else:
            # Fallback parsing
            posting_data = JobPostingSchema(
                title="Job Posting",
                description=message,
                location="Location not specified",
                salary_min=15000,
                salary_max=20000,
                job_type="full_time",
                category="general",
                requirements=["Experience preferred"],
                contact_info=contact,
                urgency="flexible",
                benefits=[],
                working_hours="Standard hours",
                original_language=language
            )
        
        job_id = str(uuid4())
        job = JobPosting(
            id=job_id,
            posting_data=posting_data,
            posted_by="employer",
            verified=False
        )
        
        await data_manager.save_job_posting(job)
        return f"✅ Job posted successfully! ID: {job_id}\n📋 {posting_data.title}\n📍 {posting_data.location}\n💰 ₹{posting_data.salary_min:,}/month"
        
    except Exception as e:
        return f"❌ Failed to create job posting: {str(e)}"

# Search jobs helper function
async def search_jobs_for_user(query: str, max_results: int = 5) -> str:
    """Search available jobs - show EMPLOYER postings where job seekers can work"""
    try:
        matching_jobs = []
        
        print(f"🔍 Searching {len(JOBS)} jobs for job seeker: {query}")
        
        for job in JOBS.values():
            posting = job.posting_data
            score = 0
            
            # Location matching
            query_lower = query.lower()
            if "indiranagar" in query_lower or "इंद्रानगर" in query_lower:
                if "indiranagar" in posting.location.lower() or "इंद्रानगर" in posting.location.lower():
                    score += 40
                    print(f"✅ Location match: {posting.title} in {posting.location}")
            
            # For general job search, show ALL jobs where people can work
            if any(term in query_lower for term in ["job", "naukri", "काम", "work", "नौकरी"]):
                score += 30  # All job postings are potential work opportunities
                print(f"✅ General job match: {posting.title}")
            
            # Category specific matching
            if "maid" in query_lower and "cleaning" in posting.category.lower():
                score += 30
            if "security" in query_lower and "security" in posting.category.lower():
                score += 30
            if "delivery" in query_lower and "delivery" in posting.category.lower():
                score += 30
            
            if score > 0:
                matching_jobs.append((job, score))
                print(f"📝 Job scored {score}: {posting.title} - {posting.location}")
        
        matching_jobs.sort(key=lambda x: x[1], reverse=True)
        matching_jobs = matching_jobs[:max_results]
        
        if not matching_jobs:
            print("❌ No matching jobs found")
            return f"❌ No jobs found in Indiranagar\n\n💡 Try searching in: Bangalore, Delhi, Mumbai"
        
        result = f"🔍 **Found {len(matching_jobs)} job opportunities:**\n\n"
        
        for i, (job, score) in enumerate(matching_jobs, 1):
            posting = job.posting_data
            result += f"**{i}. {posting.title} Position**\n"
            result += f"   📍 {posting.location}\n"
            result += f"   💰 ₹{posting.salary_min:,}/month\n"
            result += f"   📱 Apply: {posting.contact_info}\n"
            result += f"   📝 {posting.description[:60]}...\n\n"
        
        return result
        
    except Exception as e:
        print(f"❌ Job search error: {e}")
        return f"Search error: {str(e)}"

# Emergency job options
async def get_emergency_job_options(location: str = "India", language: str = "hi") -> str:
    """Provide emergency job options for survival"""
    
    if language == "hi":
        return f"""🚨 **तुरंत काम की जरूरत - Emergency Options**

📞 **आज ही संपर्क करें**:
- Zomato/Swiggy delivery: Local office call करें
- Ola/Uber driver: Online apply करें  
- Construction daily labor: पास के sites पर जाएं
- House cleaning: आस-पास के घरों में पूछें
- Security guard: Local agencies से contact करें

💡 **आज से कमाई शुरू**:
- Daily wage labor: ₹400-600/day
- Food delivery: ₹800-1200/day  
- House help: ₹300-500/day
- Night security: ₹500-800/day

🆘 **Emergency Help**: 
अगर serious problem है तो local NGO या helpline:
- Helpline: 1091 (Women), 1098 (Child)
- Local employment exchange office

💪 **हिम्मत रखें** - काम जरूर मिलेगा!"""
    else:
        return f"""🚨 **Need Work Immediately - Emergency Options**

📞 **Contact Today**:
- Zomato/Swiggy delivery: Call local office
- Ola/Uber driver: Apply online
- Construction daily labor: Visit nearby sites  
- House cleaning: Ask nearby homes
- Security guard: Contact local agencies

💡 **Start Earning Today**:
- Daily wage labor: ₹400-600/day
- Food delivery: ₹800-1200/day
- House help: ₹300-500/day
- Night security: ₹500-800/day

🆘 **Emergency Help**:
If in serious distress, contact:
- Helpline: 1091 (Women), 1098 (Child)  
- Local employment exchange office

💪 **Stay Strong** - You will find work!"""

# Seed demo data
async def seed_demo_data():
    """Add demo job postings"""
    demo_postings = [
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
        )
    ]
    
    for posting_data in demo_postings:
        job = JobPosting(
            id=str(uuid4()),
            posting_data=posting_data,
            posted_by="demo_employer",
            verified=True
        )
        await data_manager.save_job_posting(job)

# Create FastMCP server
mcp = FastMCP(
    "JobKranti AI - Voice-First Multilingual Job Platform for Bharat",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# Main AI-powered tool that handles everything
@mcp.tool
async def jobkranti_ai_assistant(
    user_message: Annotated[str, Field(description="User's complete message in any Indian language")],
    user_phone: Annotated[str, Field(description="User's WhatsApp phone number")] = "unknown"
) -> str:
    """JobKranti AI - Intelligent assistant for all job-related needs in India. 
    Handles hiring (maids, drivers, security, delivery), job searching, emergency work needs in multiple Indian languages."""
    
    try:
        if not ai_agent.client:
            return "JobKranti AI ready! Tell me: need to hire someone or find work?"
        
        # Let GPT-4o analyze and decide what to do
        system_prompt = f"""You are JobKranti AI, India's smartest job marketplace assistant.

ANALYZE THIS MESSAGE: "{user_message}"
USER PHONE: {user_phone}

AVAILABLE JOBS IN SYSTEM:
{ai_agent.get_available_jobs_summary()}

YOUR CAPABILITIES:
1. Help employers hire workers (maids, drivers, security guards, delivery, construction, cooks, etc.)
2. Help job seekers find work (any type of blue-collar jobs)
3. Handle emergency job requests (survival situations - "koi bhi kaam")
4. Create job postings automatically
5. Search available jobs intelligently
6. Provide market insights and salary info

DECISION FRAMEWORK:
- If user wants to HIRE someone (maid chahiye, driver needed, etc.) → Create job posting
- If user wants WORK (job chahiye, kaam chahiye, work needed) → Search jobs or emergency help
- If user says "koi bhi kaam" or desperate → Emergency survival options
- If unclear → Ask clarifying questions

LANGUAGE HANDLING:
- Respond in user's language (Hindi/English/mixed)
- Understand colloquial terms and code-mixing
- No need for exact coordinates - city/area names are fine

EXAMPLES:
- "Maid chahiye" → Create maid job posting + show available candidates
- "Koi bhi kaam chahiye" → Emergency job options + survival tips
- "Driver job Delhi" → Search driver jobs in Delhi
- "Security guard 20000 salary" → Search security jobs with salary filter

Take appropriate action and provide complete helpful response."""

        # First, let AI analyze the intent
        analysis_response = ai_agent.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze and handle: {user_message}"}
            ]
        )
        
        ai_analysis = analysis_response.choices[0].message.content
        
        # Now let AI decide specific actions with function calling
        functions = [
            {
                "name": "create_job_posting",
                "description": "Create job posting when employer needs to hire someone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_description": {"type": "string", "description": "The job posting details"},
                        "employer_contact": {"type": "string", "description": "Contact info"}
                    },
                    "required": ["job_description", "employer_contact"]
                }
            },
            {
                "name": "search_jobs",
                "description": "Search jobs when someone needs work",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {"type": "string", "description": "What kind of job to search for"},
                        "max_results": {"type": "integer", "default": 5}
                    },
                    "required": ["search_query"]
                }
            },
            {
                "name": "emergency_job_help",
                "description": "Provide emergency job options for urgent needs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "User's location"},
                        "language": {"type": "string", "description": "Response language"}
                    }
                }
            }
        ]
        
        # Let GPT-4o decide which function to call
        function_response = ai_agent.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are JobKranti AI. Based on user message, call appropriate function to help them."
                },
                {"role": "user", "content": user_message}
            ],
            functions=functions,
            function_call="auto"
        )
        
        # Execute the function GPT-4o chose
        message = function_response.choices[0].message
        
        if message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)
            
            if function_name == "create_job_posting":
                result = await create_job_posting_from_message(
                    function_args["job_description"],
                    function_args.get("employer_contact", user_phone)
                )
                
                # Also search for potential candidates
                candidate_search = await search_jobs_for_user("worker available", 3)
                
                return f"{result}\n\n🔍 **Looking for candidates:**\n{candidate_search}"
                
            elif function_name == "search_jobs":
                # When someone searches for jobs, show them EMPLOYER postings where they can work
                result = await search_jobs_for_user(
                    function_args["search_query"], 
                    function_args.get("max_results", 5)
                )
                
                # If no results, show ALL available job postings
                if "❌" in result or "No jobs found" in result:
                    all_jobs_text = ""
                    for job in list(JOBS.values())[:3]:
                        posting = job.posting_data
                        all_jobs_text += f"• {posting.title} in {posting.location} - ₹{posting.salary_min}/month\n"
                    
                    if all_jobs_text:
                        result = f"🔍 **Available work opportunities:**\n\n{all_jobs_text}\n💡 Contact employers directly to apply!"
                
                return result
                
            elif function_name == "emergency_job_help":
                result = await get_emergency_job_options(
                    function_args.get("location", "India"),
                    function_args.get("language", "hi")
                )
                return result
        
        # If no function called, provide conversational response
        return ai_analysis
        
    except Exception as e:
        # Fallback response
        if "maid" in user_message.lower() or "काम" in user_message:
            return f"मैं समझ गया! आप {'maid hire करना चाहते हैं' if 'maid' in user_message.lower() else 'काम ढूंढ रहे हैं'}। मैं आपकी मदद करूंगा।"
        
        return f"JobKranti AI यहाँ है! मैं job से related सभी चीजों में मदद कर सकता हूँ। बताइए आपको क्या चाहिए? Error: {str(e)}"

@mcp.tool
async def validate() -> str:
   """Validate JobKranti AI server for PuchAI integration"""
   return MY_NUMBER 

@mcp.tool
async def process_voice_message(
   audio_data_base64: Annotated[str, Field(description="Base64 encoded audio data from WhatsApp voice message")],
   user_phone: Annotated[str, Field(description="User's WhatsApp phone number")],
   language_hint: Annotated[str, Field(description="Language hint if known")] = "auto"
) -> list[TextContent | ImageContent]:
   """Process WhatsApp voice messages - transcribe and respond with voice"""
   
   try:
       # Decode audio data
       audio_bytes = base64.b64decode(audio_data_base64)
       
       # Transcribe voice to text
       transcribed_text = await ai_agent.transcribe_voice_message(audio_bytes, language_hint)
       
       if not transcribed_text or "error" in transcribed_text.lower():
           return [TextContent(
               type="text",
               text="❌ Sorry, couldn't understand the voice message. Please try speaking clearly."
           )]
       
       # Process the transcribed message
       ai_response = await jobkranti_ai_assistant(transcribed_text, user_phone)
       
       # Generate voice response
       voice_response_bytes = await ai_agent.generate_voice_response(ai_response, "hi")
       
       result_text = f"🎤 **Voice Message Processed**\n\n"
       result_text += f"🗣️ **You said**: \"{transcribed_text}\"\n\n"
       result_text += f"🤖 **AI Response**: {ai_response}"
       
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
       return [TextContent(
           type="text",
           text=f"Voice processing failed: {str(e)}"
       )]

@mcp.tool
async def create_user_profile(
   conversation_text: Annotated[str, Field(description="Natural conversation about user's background")],
   phone: Annotated[str, Field(description="User's phone number")]
) -> str:
   """Create user profile from conversation"""
   
   try:
       # Check if user already exists
       existing_user = await data_manager.get_user_by_phone(phone)
       if existing_user:
           return f"✅ Profile exists for {phone}: {existing_user.profile_data.name}"
       
       # Create basic profile
       profile_data = UserProfileSchema(
           name="User",
           location="Not specified",
           skills=["General work"],
           experience_years=0,
           job_preferences=["Any work"],
           availability="Flexible",
           preferred_language="hi"
       )
       
       # Use AI to extract better details if available
       if ai_agent.client:
           try:
               completion = ai_agent.client.beta.chat.completions.parse(
                   model="gpt-4o",
                   messages=[
                       {"role": "system", "content": "Extract user profile from conversation"},
                       {"role": "user", "content": f"Extract profile: {conversation_text}"}
                   ],
                   response_format=UserProfileSchema
               )
               profile_data = completion.choices[0].message.parsed
           except:
               pass
       
       user_id = str(uuid4())
       profile = UserProfile(
           id=user_id,
           phone=phone,
           profile_data=profile_data
       )
       
       await data_manager.save_user_profile(profile)
       
       return f"✅ Profile created for {profile_data.name}! ID: {user_id}"
       
   except Exception as e:
       return f"❌ Profile creation failed: {str(e)}"

@mcp.tool
async def get_platform_stats() -> str:
   """Get JobKranti platform statistics"""
   
   try:
       total_users = len(USERS)
       total_jobs = len(JOBS)
       
       urgent_jobs = len([job for job in JOBS.values() 
                         if job.posting_data.urgency in ["immediate", "today"]])
       
       result = f"📊 **JobKranti Platform Stats**\n\n"
       result += f"👥 **Users**: {total_users}\n"
       result += f"💼 **Jobs**: {total_jobs}\n"
       result += f"🚨 **Urgent Jobs**: {urgent_jobs}\n"
       result += f"🌍 **Languages**: {len(SUPPORTED_LANGUAGES)}\n"
       result += f"💾 **Database**: {'✅ Active' if DB_PATH.exists() else '❌ Memory only'}\n"
       result += f"🤖 **AI**: {'✅ OpenAI' if OPENAI_API_KEY else '❌ Limited'}\n"
       result += f"🏆 **Status**: Ready for Hackathon Demo!"
       
       return result
       
   except Exception as e:
       return f"Stats unavailable: {str(e)}"

@mcp.tool
async def emergency_survival_jobs(
   location: Annotated[str, Field(description="User's location")] = "India",
   language: Annotated[str, Field(description="Response language")] = "hi"
) -> str:
   """Emergency job options for people who need work immediately for survival"""
   
   return await get_emergency_job_options(location, language)

# Custom routes
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
   """Health check endpoint"""
   return JSONResponse({
       "status": "healthy",
       "service": "JobKranti AI - Voice-First Job Platform",
       "version": "2.0.0",
       "features": {
           "voice_processing": bool(OPENAI_API_KEY),
           "multilingual": True,
           "ai_powered": bool(OPENAI_API_KEY),
           "database": DB_PATH.exists()
       },
       "stats": {
           "users": len(USERS),
           "jobs": len(JOBS),
           "languages": len(SUPPORTED_LANGUAGES)
       }
   })

@mcp.custom_route("/", methods=["GET"])
async def root_endpoint(request: Request) -> JSONResponse:
   """Root endpoint information"""
   return JSONResponse({
       "service": "JobKranti AI - Voice-First Multilingual Job Platform",
       "description": "AI-powered job marketplace for India's blue-collar workforce",
       "mcp_endpoint": "/mcp",
       "features": [
           "Voice message processing",
           "13+ Indian languages", 
           "Emergency job finder",
           "Smart job matching",
           "Employer-worker connection"
       ],
       "demo_ready": True
   })

# Main function with database initialization
async def main():
   print("🚀 JobKranti AI - Voice-First Multilingual Job Platform")
   print("🎯 Target: India's blue-collar workforce")
   print("🎤 Voice: OpenAI Whisper + TTS")
   print("🧠 AI: GPT-4o with Indian languages")
   print("💾 Storage: SQLite + in-memory cache")
   print("📱 Integration: WhatsApp ready")
   
   # Initialize database
   try:
       await data_manager.init_database()
       await data_manager.load_all_data()
       
       # Add demo data if needed
       if len(JOBS) == 0:
           print("📊 Loading demo data...")
           await seed_demo_data()
           print(f"✅ {len(JOBS)} demo jobs loaded")
       else:
           print(f"📊 Existing data: {len(USERS)} users, {len(JOBS)} jobs")
           
   except Exception as e:
       print(f"⚠️ Database error: {e}")
       print("🔄 Running in memory mode")
       await seed_demo_data()
   
   print("\n🔗 Server: http://0.0.0.0:8086")
   print("🔗 MCP Endpoint: /mcp") 
   print("🩺 Health: /health")
   
   if OPENAI_API_KEY:
       print("✅ OpenAI connected - Full AI features")
   else:
       print("⚠️ OpenAI missing - Limited features")
   
   print("\n🏆 Ready for PuchAI Hackathon 2025!")
   print("💬 Main tool: jobkranti_ai_assistant")
   print("🎤 Voice tool: process_voice_message")
   print("📊 Stats tool: get_platform_stats")
   
   # Start server
   await mcp.run_async(transport="http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
   asyncio.run(main())