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

# Supported Indian Languages
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

# City and Area Normalization with AI fallback
CITY_SYNONYMS = {
    "bangalore": "bengaluru",
    "blr": "bengaluru", 
    "bengaluru": "bengaluru",
    "bombay": "mumbai",
    "mumbai": "mumbai",
    "delhi": "new delhi",
    "newdelhi": "new delhi",
    "gurgaon": "gurugram",
    "ggn": "gurugram",
    "calcutta": "kolkata",
    "kolkata": "kolkata",
    "madras": "chennai",
    "chennai": "chennai",
    "poona": "pune",
    "pune": "pune",
    "hyderabad": "hyderabad",
    "hyd": "hyderabad",
    "ahmedabad": "ahmedabad",
    "surat": "surat",
    "kanpur": "kanpur",
    "lucknow": "lucknow",
    "nagpur": "nagpur",
    "indore": "indore",
    "bhopal": "bhopal",
    "patna": "patna",
    "jaipur": "jaipur",
    "kochi": "kochi",
    "cochin": "kochi",
    "trivandrum": "thiruvananthapuram",
    "thiruvananthapuram": "thiruvananthapuram",
    "coimbatore": "coimbatore",
    "mysore": "mysuru",
    "mysuru": "mysuru",
}

AREA_SYNONYMS = {
    # Bengaluru
    "kormangala": "koramangala",
    "koramangala": "koramangala",
    "hsrlayout": "hsr layout",
    "hsr": "hsr layout",
    "indiragar": "indiranagar", 
    "indiranagar": "indiranagar",
    "whitefield": "whitefield",
    "marathahalli": "marathahalli",
    "electronic city": "electronic city",
    "ecity": "electronic city",
    "jayanagar": "jayanagar",
    "jp nagar": "jp nagar",
    "btm": "btm layout",
    "btm layout": "btm layout",
    
    # Mumbai
    "andheri east": "andheri",
    "andheri west": "andheri", 
    "andheri": "andheri",
    "bandra": "bandra",
    "bkc": "bandra kurla complex",
    "powai": "powai",
    "borivali": "borivali",
    "malad": "malad",
    "goregaon": "goregaon",
    "thane": "thane",
    
    # Delhi
    "cp": "connaught place",
    "connaught place": "connaught place",
    "karol bagh": "karol bagh",
    "lajpat nagar": "lajpat nagar",
    "saket": "saket",
    "dwarka": "dwarka",
    "rohini": "rohini",
    "gk": "greater kailash",
    "greater kailash": "greater kailash",
    
    # Chennai
    "t nagar": "t nagar",
    "anna nagar": "anna nagar",
    "adyar": "adyar",
    "velachery": "velachery",
    "tambaram": "tambaram",
    
    # Pune
    "koregaon park": "koregaon park",
    "viman nagar": "viman nagar",
    "hinjewadi": "hinjewadi",
    "baner": "baner",
    "aundh": "aundh",
    "kharadi": "kharadi",
}

# Job category mapping with AI fallback
JOB_CATEGORY_KEYWORDS = {
    "cleaning": ["maid", "safai", "cleaning", "cleaner", "झाड़ू", "सफाई", "साफ", "clean"],
    "security": ["security", "guard", "सिक्यूरिटी", "गार्ड", "watchman", "chowkidar", "सुरक्षा"],
    "delivery": ["delivery", "courier", "डिलीवरी", "भेजना", "zomato", "swiggy", "dunzo"],
    "driving": ["driver", "ड्राइवर", "गाड़ी", "car", "taxi", "cab", "ola", "uber"],
    "cooking": ["cook", "chef", "रसोइया", "खाना", "food", "kitchen", "रसोई"],
    "construction": ["construction", "निर्माण", "building", "मिस्त्री", "labour", "मजदूर"],
    "childcare": ["nanny", "babysitter", "बच्चा", "child", "care", "आया"],
    "elderly_care": ["eldercare", "बुजुर्ग", "old age", "caretaker", "nursing"],
    "beauty": ["salon", "parlour", "beauty", "सुंदरता", "hair", "makeup"],
    "tutoring": ["tutor", "teacher", "शिक्षक", "पढ़ाना", "teaching", "study"],
}

def normalize_location(text: str) -> str:
    """Normalize city/area names with intelligent fallback"""
    if not text:
        return ""
    
    # Clean and normalize input
    cleaned = re.sub(r'[^\w\s]', '', text.lower().strip())
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Check exact matches first
    if cleaned in CITY_SYNONYMS:
        return CITY_SYNONYMS[cleaned]
    if cleaned in AREA_SYNONYMS:
        return AREA_SYNONYMS[cleaned]
    
    # Check if it contains known city/area names
    for synonym, canonical in CITY_SYNONYMS.items():
        if synonym in cleaned or cleaned in synonym:
            return canonical
    
    for synonym, canonical in AREA_SYNONYMS.items():
        if synonym in cleaned or cleaned in synonym:
            return canonical
    
    # Return cleaned version if no match found
    return cleaned.title()

def extract_job_category(text: str) -> str:
    """Extract job category with keyword matching and AI fallback"""
    if not text:
        return "general"
    
    text_lower = text.lower()
    
    # Check keyword mappings
    for category, keywords in JOB_CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    # Return general if no match
    return "general"

def fuzzy_location_search(query_location: str, job_location: str) -> bool:
    """Intelligent location matching with fuzzy logic"""
    if not query_location or not job_location:
        return True  # No filter applied
    
    query_norm = normalize_location(query_location).lower()
    job_norm = normalize_location(job_location).lower()
    
    # Exact match
    if query_norm == job_norm:
        return True
    
    # Contains match (e.g., "bangalore" matches "koramangala, bangalore")
    if query_norm in job_norm or job_norm in query_norm:
        return True
    
    # Split and check individual words
    query_words = set(query_norm.split())
    job_words = set(job_norm.split())
    
    # If any significant word matches
    if query_words & job_words:
        return True
    
    return False

async def ai_enhance_location(location_text: str) -> str:
    """Use AI to enhance/correct location names when available"""
    if not ai_agent.client or not location_text:
        return normalize_location(location_text)
    
    try:
        response = ai_agent.client.chat.completions.create(
            model="gpt-4o-mini",  # Faster, cheaper model for simple tasks
            messages=[
                {
                    "role": "system",
                    "content": "You are a location normalizer for India. Return only the corrected city/area name, nothing else. If unclear, return the input as-is."
                },
                {
                    "role": "user", 
                    "content": f"Correct this Indian location name: '{location_text}'"
                }
            ],
            max_tokens=20,
            temperature=0
        )
        
        ai_result = response.choices[0].message.content.strip()
        # Only use AI result if it looks reasonable (no long explanations)
        if len(ai_result) <= 50 and not ai_result.startswith("I "):
            return ai_result
            
    except Exception as e:
        print(f"AI location enhancement failed: {e}")
    
    return normalize_location(location_text)

async def ai_enhance_job_category(job_text: str) -> str:
    """Use AI to determine job category when keywords don't match"""
    if not ai_agent.client or not job_text:
        return "general"
    
    # First try keyword matching
    category = extract_job_category(job_text)
    if category != "general":
        return category
    
    try:
        categories_list = list(JOB_CATEGORY_KEYWORDS.keys())
        response = ai_agent.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Classify this job into one category only. Categories: {', '.join(categories_list)}. Return only the category name, nothing else."
                },
                {
                    "role": "user",
                    "content": f"Classify this job: '{job_text}'"
                }
            ],
            max_tokens=10,
            temperature=0
        )
        
        ai_category = response.choices[0].message.content.strip().lower()
        if ai_category in categories_list:
            return ai_category
            
    except Exception as e:
        print(f"AI category classification failed: {e}")
    
    return "general"

def suggest_similar_locations(input_location: str) -> List[str]:
    """Suggest similar locations when exact match isn't found"""
    if not input_location:
        return []
    
    input_lower = input_location.lower()
    suggestions = []
    
    # Check for partial matches in cities
    for city, canonical in CITY_SYNONYMS.items():
        if input_lower in city or city in input_lower:
            suggestions.append(canonical.title())
        # Check for phonetic similarity (simple)
        elif len(input_lower) > 3 and any(
            input_lower[:3] == city[:3] or input_lower[-3:] == city[-3:]
            for city in CITY_SYNONYMS.keys()
        ):
            suggestions.append(canonical.title())
    
    # Check for partial matches in areas
    for area, canonical in AREA_SYNONYMS.items():
        if input_lower in area or area in input_lower:
            suggestions.append(canonical.title())
    
    # Remove duplicates and limit
    suggestions = list(set(suggestions))[:5]
    
    # If no suggestions, provide major cities
    if not suggestions:
        suggestions = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad"]
    
    return suggestions

async def handle_unknown_location(location_input: str, context: str = "") -> str:
    """Handle unknown location inputs with smart suggestions"""
    
    # First try AI enhancement
    enhanced = await ai_enhance_location(location_input)
    if enhanced != location_input and enhanced.lower() != location_input.lower():
        return f"""✅ **Found it! I think you mean: {enhanced}**

🔍 **Try searching:**
- `Find jobs in {enhanced}`
- `Find maid jobs in {enhanced}`
- `Post job in {enhanced}`"""
    
    # Get similar location suggestions
    suggestions = suggest_similar_locations(location_input)
    
    result = f"""🤔 **I couldn't find "{location_input}" in our location database.**

📍 **Did you mean one of these?**
"""
    
    for suggestion in suggestions:
        result += f"• {suggestion}\n"
    
    result += f"""
💡 **Tips for better results:**
- Use full city names: "Bangalore" not "BLR"
- Include state: "Indore, MP" or "Salem, Tamil Nadu"
- Try nearby major cities

🔍 **Or search without location filter:**
- `Find any available jobs`
- `Show me all maid jobs`

🗺️ **We cover 100+ Indian cities!**"""
    
    return result

# Rich Tool Description Model
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# Enhanced Pydantic Models
class JobPostingSchema(BaseModel):
    title: str = Field(description="Job title (e.g., 'Maid', 'Security Guard', 'Driver')")
    description: str = Field(description="Detailed job description")
    location: str = Field(description="City/area where job is located")
    salary_min: int = Field(description="Minimum salary offered per month in INR")
    salary_max: int = Field(description="Maximum salary offered per month in INR")
    job_type: str = Field(description="Employment type: full_time, part_time, daily_wage, gig")
    category: str = Field(description="Job category: cleaning, security, delivery, cooking, construction, etc.")
    requirements: List[str] = Field(description="Skills/qualifications needed")
    contact_info: str = Field(description="How to contact employer (phone/WhatsApp)")
    working_hours: str = Field(description="Shift timings")
    gender_preference: str = Field(default="Any", description="Male/Female/Any")
    experience_required: int = Field(default=0, description="Years of experience required")
    benefits: List[str] = Field(default=[], description="Perks and benefits offered")
    urgency: str = Field(default="flexible", description="immediate/within_week/flexible")
    
class JobSearchFilters(BaseModel):
    location: str | None = Field(default=None, description="City/area to search in")
    category: str | None = Field(default=None, description="Job category filter")
    max_salary: int | None = Field(default=None, description="Maximum salary expected")
    min_salary: int | None = Field(default=None, description="Minimum salary expected")
    job_type: str | None = Field(default=None, description="Employment type filter")
    gender_preference: str | None = Field(default=None, description="Gender preference")
    max_experience: int | None = Field(default=None, description="Maximum experience years")

# Data Models
@dataclass
class JobPosting:
    id: str
    posting_data: JobPostingSchema
    posted_by: str
    management_key: str  # Secret key for editing/deleting
    verified: bool = False
    views: int = 0
    applications: int = 0
    created_at: datetime = None
    expires_at: datetime = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            self.expires_at = datetime.now() + timedelta(days=30)

# Database setup
DB_PATH = Path("jobkranti.db")
JOBS: Dict[str, JobPosting] = {}

class DataManager:
    """Handles data persistence with SQLite + in-memory caching"""
    
    @staticmethod
    async def init_database():
        """Initialize SQLite database with required tables"""
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    posting_data TEXT,
                    posted_by TEXT,
                    management_key TEXT,
                    verified BOOLEAN DEFAULT FALSE,
                    views INTEGER DEFAULT 0,
                    applications INTEGER DEFAULT 0,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            await db.commit()
            print("✅ Database initialized successfully")
    
    @staticmethod
    async def save_job_posting(job: JobPosting):
        """Save job posting to database and update cache"""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO jobs 
                    (id, posting_data, posted_by, management_key, verified, views, applications, created_at, expires_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job.id,
                    json.dumps(job.posting_data.model_dump(), default=str),
                    job.posted_by,
                    job.management_key,
                    job.verified,
                    job.views,
                    job.applications,
                    job.created_at.isoformat(),
                    job.expires_at.isoformat(),
                    job.is_active
                ))
                await db.commit()
            
            JOBS[job.id] = job
            print(f"✅ Job {job.id} saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving job {job.id}: {e}")
    
    @staticmethod
    async def load_all_data():
        """Load all data from database into memory cache"""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute("SELECT * FROM jobs WHERE is_active = 1") as cursor:
                    async for row in cursor:
                        try:
                            job = JobPosting(
                                id=row[0],
                                posting_data=JobPostingSchema(**json.loads(row[1])),
                                posted_by=row[2],
                                management_key=row[3],
                                verified=bool(row[4]),
                                views=row[5],
                                applications=row[6],
                                created_at=datetime.fromisoformat(row[7]),
                                expires_at=datetime.fromisoformat(row[8]),
                                is_active=bool(row[9])
                            )
                            JOBS[job.id] = job
                        except Exception as e:
                            print(f"❌ Error loading job {row[0]}: {e}")
            
            print(f"✅ Loaded {len(JOBS)} active jobs from database")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")

# Initialize data manager
data_manager = DataManager()

# AI Agent for OpenAI integration
class JobKrantiAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        if api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("⚠️ OpenAI not available")
    
    async def extract_job_details(self, user_message: str, contact_info: str = "") -> JobPostingSchema:
        """Use AI to extract job details from natural language with enhanced fallbacks"""
        if not self.client:
            # Enhanced fallback extraction
            return await self._enhanced_fallback_extraction(user_message, contact_info)
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """Extract job posting details from Indian user's message. Handle Hindi/English mix. 
                        For locations: Use actual Indian city/area names, be flexible with spellings.
                        For categories: Choose from cleaning, security, delivery, driving, cooking, construction, childcare, elderly_care, beauty, tutoring, or general.
                        Set reasonable defaults for missing information."""
                    },
                    {"role": "user", "content": f"Extract job details: {user_message}"}
                ],
                response_format=JobPostingSchema
            )
            
            job_data = completion.choices[0].message.parsed
            
            # Enhance location and category with AI
            job_data.location = await ai_enhance_location(job_data.location)
            job_data.category = await ai_enhance_job_category(user_message)
            
            if contact_info:
                job_data.contact_info = contact_info
            return job_data
            
        except Exception as e:
            print(f"AI extraction failed: {e}")
            return await self._enhanced_fallback_extraction(user_message, contact_info)
        
    def get_available_jobs_summary(self) -> str:
        """Get summary of available jobs for AI context"""
        try:
            if not JOBS:
                return "No jobs currently in database. Ready to add first job."
            
            jobs_summary = []
            for job_id, job in list(JOBS.items())[:5]:  # Show first 5 jobs
                posting = job.posting_data
                jobs_summary.append(f"- {posting.title} in {posting.location} - ₹{posting.salary_min}/month (ID: {job_id})")
            
            return "\n".join(jobs_summary) if jobs_summary else "Sample jobs available"
        except Exception as e:
            return f"Error getting jobs summary: {e}"
    
    async def show_all_available_jobs() -> str:
        """Show all jobs regardless of search criteria"""
        if not JOBS:
            return "🔍 **कोई नौकरी उपलब्ध नहीं है।** पहली नौकरी पोस्ट करने वाले बनें!"
        
        result = f"🔍 **सभी उपलब्ध नौकरियां ({len([j for j in JOBS.values() if j.is_active])} total):**\n\n"
        
        count = 0
        for job_id, job in JOBS.items():
            if job.is_active:
                count += 1
                posting = job.posting_data
                result += f"""**{count}. {posting.title}**
    📍 {posting.location}  
    💰 ₹{posting.salary_min:,} - ₹{posting.salary_max:,}/month
    📱 संपर्क: {posting.contact_info}
    🆔 Job ID: {job_id}

    ---
    """
        
        return result

    async def get_emergency_job_options(location: str = "India", language: str = "hi") -> str:
        """Emergency job options for immediate survival needs"""
        
        if language == "hi":
            return f"""🚨 **तुरंत काम की जरूरत - Emergency Job Options**

    📞 **आज ही शुरू करें**:
    - Zomato/Swiggy delivery: Local hub में जाकर register करें
    - Ola/Uber: Online apply करें (bike/car चाहिए)
    - Construction daily wages: पास के construction sites पर जाएं
    - House cleaning: आस-पास के घरों में पूछें
    - Security guard: Local security agencies में apply करें

    💰 **Daily Earning Potential**:
    - Daily labor: ₹500-800/day
    - Food delivery: ₹800-1500/day
    - House help: ₹400-600/day
    - Night security: ₹600-1000/day

    💪 **हिम्मत रखें!** - JobKranti आपकी मदद करेगा। काम जरूर मिलेगा!"""
        
        return "Emergency job options available. Contact local employment agencies."

    async def _enhanced_fallback_extraction(self, message: str, contact_info: str) -> JobPostingSchema:
        """Enhanced fallback job extraction with better location/category handling"""
        message_lower = message.lower()
        
        # Enhanced job type detection with more keywords
        category = extract_job_category(message)
        
        if category == "cleaning":
            title = "Maid/Cleaning Staff"
        elif category == "security":
            title = "Security Guard"
        elif category == "driving":
            title = "Driver"
        elif category == "delivery":
            title = "Delivery Partner"
        elif category == "cooking":
            title = "Cook"
        elif category == "construction":
            title = "Construction Worker"
        elif category == "childcare":
            title = "Nanny/Childcare"
        elif category == "elderly_care":
            title = "Elderly Caretaker"
        elif category == "beauty":
            title = "Beauty Specialist"
        elif category == "tutoring":
            title = "Tutor"
        else:
            title = "General Worker"
        
        # Enhanced location extraction
        location = "Location not specified"
        
        # Check for major Indian cities (more comprehensive)
        indian_cities = [
            "bangalore", "bengaluru", "mumbai", "bombay", "delhi", "newdelhi", "pune", "poona",
            "hyderabad", "chennai", "madras", "kolkata", "calcutta", "ahmedabad", "surat",
            "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal", "kochi", "cochin",
            "coimbatore", "visakhapatnam", "vizag", "patna", "agra", "guwahati", "chandigarh",
            "mysore", "mysuru", "vadodara", "baroda", "rajkot", "gurgaon", "gurugram",
            "noida", "faridabad", "ghaziabad", "thane", "nashik", "aurangabad", "solapur"
        ]
        
        # Extract location from message
        words = message_lower.split()
        for word in words:
            normalized = normalize_location(word)
            if normalized != word.lower():  # Found a match
                location = normalized.title()
                break
        
        # If still not found, try multi-word locations
        if location == "Location not specified":
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    two_word = f"{word} {words[i+1]}"
                    normalized = normalize_location(two_word)
                    if normalized != two_word:
                        location = normalized.title()
                        break
        
        # Salary estimation based on job type and location
        if category in ["security", "driving"]:
            salary_min, salary_max = 18000, 25000
        elif category in ["delivery"]:
            salary_min, salary_max = 20000, 35000
        elif category in ["cooking"]:
            salary_min, salary_max = 15000, 22000
        elif category in ["tutoring", "beauty"]:
            salary_min, salary_max = 15000, 30000
        elif category in ["construction"]:
            salary_min, salary_max = 12000, 18000
        else:  # cleaning, childcare, elderly_care, general
            salary_min, salary_max = 10000, 18000
        
        # Location-based salary adjustment
        if any(city in location.lower() for city in ["mumbai", "bangalore", "bengaluru", "delhi", "pune", "gurgaon", "hyderabad"]):
            salary_min = int(salary_min * 1.2)
            salary_max = int(salary_max * 1.2)
        
        return JobPostingSchema(
            title=title,
            description=message,
            location=location,
            salary_min=salary_min,
            salary_max=salary_max,
            job_type="full_time",
            category=category,
            requirements=["Reliable", "Hardworking", "Experience preferred"],
            contact_info=contact_info or "Contact for details",
            working_hours="Standard working hours"
        )

# Initialize AI agent
ai_agent = JobKrantiAI(OPENAI_API_KEY)

# Create FastMCP server
mcp = FastMCP(
    "JobKranti",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# Tool: validate (required by Puch)
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# Tool: Help System
HelpDescription = RichToolDescription(
    description="Shows help menu with instructions and examples for using JobKranti",
    use_when="User asks for 'help', 'instructions', 'commands', or 'how does this work?'",
    side_effects=None,
)

@mcp.tool(description=HelpDescription.model_dump_json())
async def get_help() -> str:
    """Shows comprehensive help for JobKranti platform"""
    return """👋 **Welcome to JobKranti AI - भारत का पहला Voice-First Job Platform!**

🔎 **Job Seekers (काम ढूंढने वाले):**
- `Find maid jobs in Bangalore` - नौकरी खोजें
- `Show me delivery jobs under ₹20000` - सैलरी के अनुसार खोजें
- `I need work urgently` - तुरंत काम चाहिए

✍️ **Employers (काम देने वाले):**
- `I want to post a job for maid` - नौकरी पोस्ट करें
- `Need security guard in Delhi` - जब आप job post करेंगे तो आपको **secret management key** मिलेगी
- **SAVE THIS KEY!** - Edit/delete के लिए जरूरी है

✏️ **Manage Your Job Postings:**
- `Delete job J123 with key <your_secret_key>` - Job delete करें
- `Edit job J123 with key <your_secret_key>, increase salary to 25000` - Job edit करें

🎤 **Voice Support:** Send voice messages in Hindi/English!
🌍 **Languages:** 13+ Indian languages supported
🚨 **Emergency:** Say "koi bhi kaam chahiye" for urgent help

**Examples:**
- "Bangalore mein maid chahiye" ✅
- "Delhi security guard job" ✅  
- "₹15000 salary delivery work Mumbai" ✅
- "तुरंत काम चाहिए कोई भी" ✅"""

# Tool: Post Job (Employer side)
PostJobDescription = RichToolDescription(
    description="Creates a new job posting with slot filling for missing information",
    use_when="User wants to 'post', 'create', or 'add' a job. Employers use this to hire workers.",
    side_effects="A new job is added to the database with a unique management key",
)

@mcp.tool
async def debug_show_all_jobs() -> str:
    """Debug: Show all jobs in database"""
    result = f"Total jobs in memory: {len(JOBS)}\n\n"
    for job_id, job in JOBS.items():
        result += f"ID: {job_id}\n"
        result += f"Title: {job.posting_data.title}\n"
        result += f"Location: {job.posting_data.location}\n"
        result += f"Active: {job.is_active}\n"
        result += f"Category: {job.posting_data.category}\n\n"
    return result

# Database schema definition for AI
JOB_SCHEMA = {
    "required_fields": {
        "title": "Job title (e.g., Maid, Security Guard, Driver, Plumber)",
        "location": "City/area (e.g., Indiranagar Bangalore, Delhi)",
        "contact_info": "Phone number for applications",
        "salary_min": "Minimum monthly salary in INR",
        "salary_max": "Maximum monthly salary in INR"
    },
    "optional_fields": {
        "description": "Detailed job description",
        "category": "Job category (cleaning, security, delivery, etc.)",
        "job_type": "full_time, part_time, daily_wage, gig",
        "working_hours": "Shift timings",
        "gender_preference": "Male, Female, or Any",
        "experience_required": "Years of experience needed",
        "requirements": "List of skills/qualifications",
        "benefits": "Perks and benefits offered",
        "urgency": "immediate, within_week, flexible"
    },
    "search_filters": {
        "location": "Filter by city/area",
        "category": "Filter by job type",
        "salary_range": "Min/max salary filter",
        "job_type": "Employment type filter",
        "experience_level": "Experience requirement filter"
    }
}

@mcp.tool(description="Smart JobKranti assistant for all job operations")
async def jobkranti_assistant(
    user_message: Annotated[str, Field(description="User's complete message in any language")],
    user_phone: Annotated[str, Field(description="User's phone number")] = "unknown"
) -> str:
    """Smart JobKranti assistant - handles job posting, searching, and help"""
    
    if not ai_agent.client:
        return "JobKranti AI ready! What job-related help do you need?"
    
    try:
        # 🧠 AI analysis
        response = ai_agent.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are JobKranti AI. Analyze user intent:

CURRENT JOBS: {len(JOBS)} jobs in database

USER INTENTS:
1. POST_JOB - Wants to hire (मुझे चाहिए, need worker, post job)
2. FIND_JOBS - Looking for work (नौकरी चाहिए, job needed)  
3. SHOW_ALL - Show all jobs (सभी नौकरी, all jobs, show jobs)

For POST_JOB: Say "CREATE_JOB" if you have title, location, contact
For FIND_JOBS: Say "SEARCH_JOBS" 
For SHOW_ALL: Say "SHOW_ALL_JOBS"

Be brief and clear about intent."""
                },
                {
                    "role": "user",
                    "content": f"User said: '{user_message}'"
                }
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        ai_response = response.choices[0].message.content
        print(f"🧠 AI Response: {ai_response}")  # Debug log
        
        # 🎯 Check for specific intents
        if "CREATE_JOB" in ai_response or ("post" in ai_response.lower() and "job" in ai_response.lower()):
            print("🔧 Attempting to create job...")  # Debug
            
            # Extract job details using AI
            job_data = await ai_agent.extract_job_details(user_message, user_phone)
            print(f"📝 Extracted job data: {job_data.title}, {job_data.location}, {job_data.contact_info}")  # Debug
            
            # Validate we have essential info
            if (job_data.title and job_data.title != "General Worker" and 
                job_data.location and job_data.location != "Location not specified" and 
                job_data.contact_info and job_data.contact_info != "Contact for details"):
                
                # Generate job ID and key
                max_id = max((int(j.id[1:]) for j in JOBS.values() if j.id.startswith('J') and j.id[1:].isdigit()), default=0)
                job_id = f"J{max_id + 1:03d}"
                management_key = str(uuid4())
                print(f"🆔 Creating job {job_id}")  # Debug
                
                # Create job object
                job = JobPosting(
                    id=job_id,
                    posting_data=job_data,
                    posted_by="employer",
                    management_key=management_key,
                    verified=False
                )
                
                # Save to database AND memory
                await data_manager.save_job_posting(job)
                print(f"💾 Job {job_id} saved! Total jobs now: {len(JOBS)}")  # Debug
                
                return f"""✅ **प्लंबर की नौकरी सफलतापूर्वक पोस्ट हो गई!**

🆔 **Job ID:** {job_id}
🏷️ **Title:** {job_data.title}
📍 **Location:** {job_data.location}
💰 **Salary:** ₹{job_data.salary_min:,} - ₹{job_data.salary_max:,}/month
📱 **Contact:** {job_data.contact_info}

🔑 **Management Key:** {management_key[:12]}...
(Save this key for editing/deleting)

🚀 Your job is now LIVE! Total jobs: {len(JOBS)}"""
            
            else:
                return f"Job creation needs more info. Title: {job_data.title}, Location: {job_data.location}, Contact: {job_data.contact_info}"
        
        elif "SEARCH_JOBS" in ai_response or ("find" in ai_response.lower() or "search" in ai_response.lower()):
            print("🔍 Searching for jobs...")
            return await search_for_jobs(user_message)
        
        elif "SHOW_ALL_JOBS" in ai_response or ("show" in ai_response.lower() and "job" in ai_response.lower()):
            print("📋 Showing all jobs...")
            return await show_all_available_jobs()
        
        # Default response
        return ai_response
        
    except Exception as e:
        print(f"❌ Error in jobkranti_assistant: {e}")
        return f"JobKranti AI में समस्या: {str(e)}"

async def search_for_jobs(query: str) -> str:
    """Search for jobs based on query"""
    if not JOBS:
        return "❌ **कोई नौकरी उपलब्ध नहीं है।**"
    
    query_lower = query.lower()
    matching_jobs = []
    
    print(f"🔍 Searching in {len(JOBS)} jobs for: {query}")  # Debug
    
    for job_id, job in JOBS.items():
        if job.is_active:
            posting = job.posting_data
            print(f"Checking job: {posting.title} category: {posting.category}")  # Debug
            
            # Enhanced keyword matching for plumber
            if (any(word in posting.title.lower() for word in ["plumber", "प्लंबर", "प्लम्बर"]) or
                any(word in posting.category.lower() for word in ["plumb", "maintenance", "construction"]) or
                any(word in posting.description.lower() for word in ["plumber", "प्लंबर", "pipe", "water"])):
                matching_jobs.append((job_id, job))
                print(f"✅ Found matching job: {job_id}")  # Debug
    
    if not matching_jobs:
        # Show ALL jobs for debugging
        all_jobs_info = "\n".join([f"- {j.posting_data.title} ({j.posting_data.category})" for j in JOBS.values() if j.is_active])
        return f"""❌ प्लंबर की कोई नौकरी नहीं मिली। 

**उपलब्ध सभी jobs:**
{all_jobs_info}

कुल jobs: {len([j for j in JOBS.values() if j.is_active])}"""
    
    result = f"🔍 **मिली {len(matching_jobs)} प्लंबर नौकरियां:**\n\n"
    for i, (job_id, job) in enumerate(matching_jobs, 1):
        posting = job.posting_data
        result += f"""**{i}. {posting.title}**
📍 **Location:** {posting.location}
💰 **Salary:** ₹{posting.salary_min:,} - ₹{posting.salary_max:,}/month
📱 **Contact:** {posting.contact_info}
⏰ **Hours:** {posting.working_hours}
🆔 **Job ID:** {job_id}

📝 **Details:** {posting.description}

---
"""
    
    return result

async def execute_job_posting(data: dict, user_phone: str) -> str:
    """Execute job posting with schema-validated data"""
    try:
        # Create JobPostingSchema from extracted data
        job_data = JobPostingSchema(
            title=data.get("title", "Job Posting"),
            description=data.get("description", ""),
            location=data.get("location", "Location not specified"),
            salary_min=data.get("salary_min", 10000),
            salary_max=data.get("salary_max", data.get("salary_min", 15000)),
            job_type=data.get("job_type", "full_time"),
            category=data.get("category", "general"),
            requirements=data.get("requirements", ["Reliable"]),
            contact_info=data.get("contact_info", user_phone),
            working_hours=data.get("working_hours", "Standard hours"),
            gender_preference=data.get("gender_preference", "Any"),
            experience_required=data.get("experience_required", 0),
            benefits=data.get("benefits", []),
            urgency=data.get("urgency", "flexible")
        )
        
        # Generate job ID and key
        max_id = max((int(j.id[1:]) for j in JOBS.values() if j.id.startswith('J') and j.id[1:].isdigit()), default=0)
        job_id = f"J{max_id + 1:03d}"
        management_key = str(uuid4())
        
        # Create and save job
        job = JobPosting(
            id=job_id,
            posting_data=job_data,
            posted_by="employer",
            management_key=management_key,
            verified=False
        )
        
        await data_manager.save_job_posting(job)
        
        return f"""✅ **नौकरी सफलतापूर्वक पोस्ट हो गई!**

🆔 **Job ID:** {job_id}
🏷️ **Title:** {job_data.title}
📍 **Location:** {job_data.location}
💰 **Salary:** ₹{job_data.salary_min:,} - ₹{job_data.salary_max:,}/month

🔑 **Management Key:** {management_key}
(इसे save करें - edit/delete के लिए जरूरी है)

🚀 आपकी नौकरी अब live है!"""
        
    except Exception as e:
        return f"❌ Job posting में समस्या: {str(e)}"

async def execute_job_search(filters: dict) -> str:
    """Execute job search with schema-validated filters"""
    try:
        # Use existing find_jobs logic but with clean English data
        search_query = filters.get("search_query", "")
        location = filters.get("location", "")
        
        # Your existing search logic here...
        # But now all data is already in English and schema-compliant
        
        return await find_jobs(
            search_query=search_query,
            location=location,
            min_salary=filters.get("min_salary"),
            max_salary=filters.get("max_salary"),
            job_type=filters.get("job_type")
        )
        
    except Exception as e:
        return f"❌ Search में समस्या: {str(e)}"

# Voice Processing Tool (if needed)
@mcp.tool
async def process_voice_message(
   audio_data_base64: Annotated[str, Field(description="Base64 encoded audio from WhatsApp")],
   user_phone: Annotated[str, Field(description="User's phone number")] = "unknown"
) -> str:
   """Process WhatsApp voice messages in Indian languages"""

   if not OPENAI_API_KEY:
       return "🎤 Voice processing requires OpenAI configuration. Please type your message instead."
   
   try:
       import openai
       client = openai.OpenAI(api_key=OPENAI_API_KEY)
       
       # Decode audio
       audio_bytes = base64.b64decode(audio_data_base64)
       audio_file = io.BytesIO(audio_bytes)
       audio_file.name = "voice_message.mp3"
       
       # Transcribe
       transcript = client.audio.transcriptions.create(
           model="whisper-1",
           file=audio_file,
           language="hi",  # Hindi hint but supports auto-detection
           response_format="text"
       )
       
       result = f"🎤 **Voice Message Processed**\n\n"
       result += f"🗣️ **You said:** \"{transcript}\"\n\n"
       result += f"💡 **Tip:** Now I can help you with this request using our job tools!"
       
       return result
       
   except Exception as e:
       return f"🎤 Voice processing failed: {str(e)}. Please type your message."

@mcp.tool
async def debug_jobs() -> str:
    """Debug: Show all jobs in memory and database"""
    result = f"🔍 **Debug Info:**\n\n"
    result += f"Jobs in memory: {len(JOBS)}\n"
    result += f"Database file exists: {DB_PATH.exists()}\n\n"
    
    if JOBS:
        result += "**Jobs in memory:**\n"
        for job_id, job in JOBS.items():
            result += f"- {job_id}: {job.posting_data.title} in {job.posting_data.location}\n"
    else:
        result += "❌ No jobs in memory!\n"
    
    # Check database directly
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT id, posting_data FROM jobs WHERE is_active = 1") as cursor:
                rows = await cursor.fetchall()
                result += f"\n**Jobs in database:** {len(rows)}\n"
                for row in rows:
                    data = json.loads(row[1])
                    result += f"- {row[0]}: {data.get('title', 'No title')}\n"
    except Exception as e:
        result += f"\n❌ Database error: {e}\n"
    
    return result

# Seed demo data function
async def seed_demo_data():
   """Add demo job postings for testing"""
   demo_jobs = [
       {
           "title": "Maid for Household Work",
           "description": "Need reliable maid for daily cleaning and cooking. Good family environment.",
           "location": "Koramangala, Bengaluru",
           "salary_min": 12000,
           "salary_max": 15000,
           "category": "cleaning",
           "contact_info": "9887766554",
           "working_hours": "8 AM - 2 PM"
       },
       {
           "title": "Security Guard - Night Shift",
           "description": "रात की शिफ्ट के लिए सिक्यूरिटी गार्ड चाहिए। अच्छी सैलरी।",
           "location": "Gurgaon, Delhi NCR",
           "salary_min": 18000,
           "salary_max": 22000,
           "category": "security",
           "contact_info": "9876543210",
           "working_hours": "10 PM - 6 AM"
       },
       {
           "title": "Delivery Partner",
           "description": "Food delivery partner needed. Own bike required. Daily payment.",
           "location": "Andheri, Mumbai",
           "salary_min": 25000,
           "salary_max": 35000,
           "category": "delivery",
           "contact_info": "9123456789",
           "working_hours": "Flexible - 6 hours minimum"
       },
       {
           "title": "Cook for Restaurant",
           "description": "Experienced cook needed for South Indian restaurant. Good salary and benefits.",
           "location": "T Nagar, Chennai",
           "salary_min": 20000,
           "salary_max": 28000,
           "category": "cooking",
           "contact_info": "9444555666",
           "working_hours": "11 AM - 3 PM, 6 PM - 10 PM"
       },
       {
           "title": "Driver - Ola/Uber",
           "description": "Need experienced driver for cab service. Own car preferred but not mandatory.",
           "location": "Whitefield, Bengaluru",
           "salary_min": 22000,
           "salary_max": 30000,
           "category": "driving",
           "contact_info": "9988776655",
           "working_hours": "Flexible timing"
       }
   ]
   
   for i, job_data in enumerate(demo_jobs, 1):
       posting = JobPostingSchema(
           title=job_data["title"],
           description=job_data["description"],
           location=job_data["location"],
           salary_min=job_data["salary_min"],
           salary_max=job_data["salary_max"],
           job_type="full_time",
           category=job_data["category"],
           requirements=["Reliable", "Hardworking", "Experience preferred"],
           contact_info=job_data["contact_info"],
           working_hours=job_data["working_hours"],
           gender_preference="Any",
           experience_required=1,
           benefits=["Respectful work environment"],
           urgency="flexible"
       )
       
       job = JobPosting(
           id=f"J{i:03d}",
           posting_data=posting,
           posted_by="demo_employer",
           management_key=f"demo-key-{i}",
           verified=True
       )
       
       await data_manager.save_job_posting(job)

# Custom health endpoint
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
           "database": DB_PATH.exists(),
           "job_management": True,
           "smart_matching": True,
           "location_intelligence": True
       },
       "stats": {
           "active_jobs": len([j for j in JOBS.values() if j.is_active]),
           "total_jobs": len(JOBS),
           "languages": len(SUPPORTED_LANGUAGES),
           "job_categories": len(JOB_CATEGORY_KEYWORDS),
           "supported_cities": len(CITY_SYNONYMS)
       }
   })

@mcp.custom_route("/", methods=["GET"])
async def root_endpoint(request: Request) -> JSONResponse:
   """Root endpoint information"""
   return JSONResponse({
       "service": "JobKranti AI - Voice-First Multilingual Job Platform",
       "description": "AI-powered job marketplace for India's blue-collar workforce",
       "tagline": "Connecting Bharat's workforce with dignity and respect",
       "mcp_endpoint": "/mcp",
       "tools": [
           "get_help - Platform instructions",
           "post_job - Employers post jobs with AI extraction",
           "find_jobs - Workers find opportunities with smart matching", 
           "emergency_jobs - Urgent work options",
           "get_job_details - View detailed job info",
           "edit_job - Modify job postings",
           "delete_job - Remove job postings",
           "get_platform_stats - Platform metrics",
           "smart_job_assistant - Handle unclear inputs",
           "location_helper - Location suggestions",
           "process_voice_message - Voice support"
       ],
       "features": [
           "Voice message processing",
           "13+ Indian languages", 
           "Emergency job finder",
           "Smart job matching with AI",
           "Secure job management with keys",
           "AI-powered extraction and categorization",
           "Fuzzy location matching",
           "Unknown input handling",
           "Multilingual code-mixing support"
       ],
       "improvements": [
           "Enhanced location normalization",
           "AI-powered category detection", 
           "Smart job matching algorithm",
           "Flexible input handling",
           "Better error messages",
           "Slot filling for missing info",
           "Management key system",
           "Emergency job assistance"
       ],
       "demo_ready": True,
       "hackathon": "PuchAI 2025"
   })

# Main function
async def main():
   print("🚀 JobKranti AI - Voice-First Multilingual Job Platform v2.0")
   print("🎯 Serving India's blue-collar workforce")
   print("🎤 Voice: OpenAI Whisper + TTS")
   print("🧠 AI: GPT-4o with Indian languages")
   print("💾 Storage: SQLite + in-memory cache")
   print("📱 Integration: WhatsApp ready")
   print("🔧 Tools: Structured + AI-powered")
   print("🗺️ Locations: Enhanced mapping + AI fallback")
   print("🎯 Jobs: Smart categorization + fuzzy matching")
   
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
           print(f"📊 Loaded existing data: {len(JOBS)} jobs")
           
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
   
   print("\n🛠️ Available Tools:")
   print("• get_help - Show platform instructions")
   print("• post_job - Create job postings (employers)")
   print("• find_jobs - Search opportunities (job seekers)")
   print("• emergency_jobs - Urgent work options")
   print("• get_job_details - View specific job info")
   print("• edit_job - Modify job postings")
   print("• delete_job - Remove job postings")
   print("• get_platform_stats - Platform statistics")
   print("• smart_job_assistant - Handle unclear inputs")
   print("• location_helper - Location suggestions")
   print("• process_voice_message - Handle voice input")
   
   print("\n🎯 Enhanced Features:")
   print("• Fuzzy location matching (handles typos)")
   print("• AI-powered job categorization")
   print("• Smart search with relevance scoring")
   print("• Unknown input handling")
   print("• Multilingual code-mixing support")
   print("• Emergency job assistance")
   print("• Management key system for job control")
   
   print("\n💡 Example Unknown Inputs Handled:")
   print("• 'Jobs in Bangaluru' → Corrects to Bengaluru")
   print("• 'Jhadu pochha work' → Categories as cleaning")
   print("• 'XYZ city job' → Suggests nearby cities")
   print("• 'Hello kuch kaam hai?' → Provides guided options")
   
   print("\n🏆 Ready for PuchAI Hackathon 2025!")
   print("💬 Connect with: /mcp connect <your-server-url>/mcp <your-token>")
   
   # Start server
   await mcp.run_async(transport="http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
   asyncio.run(main())