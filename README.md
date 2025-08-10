# JobKranti AI - Voice-First Multilingual Job Platform on WhatsApp

> **India's First Voice-First, AI-Powered Job Marketplace for Blue-Collar Workers**

JobKranti AI bridges the employment gap in India by connecting blue-collar workers with employers through voice-powered, multilingual AI assistance. Built for the PuchAI Hackathon 2025, it addresses the critical need for accessible job matching in India's diverse linguistic landscape.

## **The Problem We Solve**

- **450M+ blue-collar workers** in India struggle to find jobs due to language barriers
- **Employers** can't find reliable domestic help (maids, drivers, security guards)
- **Technology gap** - most job platforms are text-based and English-only
- **Emergency situations** - people need immediate work for survival but don't know where to look

## **Our Solution**

### **Voice-First Design**
- **WhatsApp voice messages** in 13+ Indian languages
- **Real-time transcription** using OpenAI Whisper
- **Voice responses** with culturally appropriate AI voices
- **No reading required** - perfect for users with limited literacy

### **AI-Powered Matching**
- **GPT-4o understands colloquial** job descriptions in local languages
- **Smart intent recognition** - knows when someone desperately needs work vs. casual browsing
- **Emergency job finder** - provides immediate survival options
- **Context-aware responses** - understands family situations and urgency

### **Multilingual Support**
- Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu, Odia, Assamese, English
- **Code-mixing support** - understands Hinglish and other mixed languages
- **Regional job categories** - Mumbai finance support, Bangalore tech assistance, Delhi security

## **Technical Architecture**

### **MCP Server Integration**
```
PuchAI ↔ JobKranti MCP Server ↔ OpenAI GPT-4o ↔ Database
                ↕
         Voice Processing (Whisper + TTS)
```

### **Core Components**
- **FastMCP Server** - Model Context Protocol integration for PuchAI
- **OpenAI Integration** - GPT-4o for intelligence, Whisper for voice, TTS for responses
- **SQLite + Cache** - Persistent storage with in-memory performance
- **Pydantic Models** - Structured data extraction from natural language
- **Bearer Authentication** - Secure API access

### **Key Features**
- **Voice Message Processing** - WhatsApp voice → text → AI response → voice
- **Smart Job Matching** - AI understands "maid chahiye" vs "koi bhi kaam chahiye"
- **Emergency Options** - Immediate survival job recommendations
- **Real-time Analytics** - Track user engagement and successful matches
- **Secure** - Bearer token authentication with rate limiting

## **Quick Start**

### **Prerequisites**
- Python 3.11+
- OpenAI API key
- WhatsApp number for validation

### **Installation**
```bash
# Clone repository
git clone <repo_name>
cd <repo_name>

uv run main.py

# Setup environment
cp .env.example .env
# Edit .env with your credentials
```

### **Environment Variables**
```env
# Required
AUTH_TOKEN=your_secret_bearer_token_here
MY_NUMBER=919998881729  # Your WhatsApp number (country_code + number)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional
DATABASE_URL=sqlite:///jobkranti.db
LOG_LEVEL=INFO
```

### **Run Server**
```bash
uv run main.py
```

Server starts at: `http://localhost:8086`
- Health check: `GET /health`
- MCP endpoint: `POST /mcp/`

## 🔧 **PuchAI Integration**

### **Connect to PuchAI**
```bash
/mcp connect https://your-domain.com/mcp/ YOUR_AUTH_TOKEN
```

### **Available Tools**
1. **`validate`** - Server validation (returns phone number)
2. **`jobkranti_ai_assistant`** - Main AI assistant for all job needs
3. **`process_voice_message`** - WhatsApp voice message processing
4. **`emergency_survival_jobs`** - Urgent job options

### **Example Usage in PuchAI**
```
User: "Maid chahiye Bangalore mein"
AI: Uses jobkranti_ai_assistant → Creates job posting + shows available candidates

User: "Koi bhi kaam chahiye, very urgent"
AI: Uses emergency_survival_jobs → Provides immediate survival options

User: [Voice message in Hindi]
AI: Uses process_voice_message → Transcribes + responds with voice
```

## 🎯 **Core Use Cases**

### **1. Employer Hiring**
```
Input: "Need maid for Koramangala, good salary"
Output: Job posting created + matching candidates shown
```

### **2. Job Seeking**
```
Input: "Security guard job Delhi mein"
Output: Available security jobs in Delhi with contact info
```

### **3. Emergency Employment**
```
Input: "Koi bhi kaam chahiye, bachche bhookhe hain"
Output: Immediate survival jobs (delivery, daily labor, etc.)
```

### **4. Voice Processing**
```
Input: [WhatsApp voice in Tamil]
Output: Transcribed → AI response → Tamil voice reply
```

## **Demo Data**

The server includes realistic demo jobs:
- **Maid** - Bangalore, Koramangala (₹12,000-15,000/month)
- **Security Guard** - Delhi NCR (₹18,000-22,000/month)  
- **Delivery Partner** - Mumbai, Andheri (₹25,000-35,000/month)

## **API Documentation**

### **Main AI Assistant**
```python
POST /mcp
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "jobkranti_ai_assistant",
    "arguments": {
      "user_message": "Driver job chahiye Mumbai mein",
      "user_phone": "919876543210"
    }
  }
}
```

### **Voice Processing**
```python
POST /mcp/
{
  "jsonrpc": "2.0", 
  "method": "tools/call",
  "params": {
    "name": "process_voice_message",
    "arguments": {
      "audio_data_base64": "base64_encoded_audio_data",
      "user_phone": "919876543210",
      "language_hint": "hi"
    }
  }
}
```

## **Architecture Decisions**

### **Why Voice-First?**
- **Accessibility** - Many blue-collar workers have limited literacy
- **Convenience** - Voice is faster than typing on mobile
- **Cultural fit** - Voice communication is preferred in Indian context

### **Why MCP + PuchAI?**
- **Rapid deployment** - MCP protocol enables quick integration
- **WhatsApp integration** - Leverages PuchAI's WhatsApp connectivity
- **Scalability** - MCP servers can handle multiple clients

### **Why SQLite + Cache?**
- **Simplicity** - Easy deployment without complex database setup
- **Performance** - In-memory cache for fast responses
- **Persistence** - Data survives server restarts

## **Contact & Support**

- **Demo Server**: `https://puchaimcp.chagadiye.xyz`
- **Health Check**: `https://puchaimcp.chagadiye.xyz/health`


**🇮🇳 Built with ❤️ for Bharat's Workforce**

*"Every voice message could change a life. Every job match could feed a family."*

**#BuildWithPuch #VoiceAI #JobKranti #India #Hackathon2025**