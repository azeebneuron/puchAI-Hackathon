
Great question! Let's test this step by step. Here's your complete testing guide:

## 🧪 **Testing JobKranti MCP Server**

### **Step 1: Local Setup & Testing**

#### **1.1 Environment Setup**
```bash
# Create project directory
mkdir jobkranti-mcp
cd jobkranti-mcp

# Create the main file
nano jobkranti_mcp.py
# Paste the complete code here

# Create environment file
cp .env.example .env
nano .env
```

**Edit `.env` file:**
```bash
AUTH_TOKEN="demo_token_hackathon_2025"
MY_NUMBER="919998881729"
OPENAI_API_KEY="your_openai_key_here"  # Optional
```

#### **1.2 Install Dependencies**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install fastmcp python-dotenv pydantic pillow httpx beautifulsoup4 markdownify readabilipy

# Optional: Install OpenAI for advanced features
pip install openai
```

#### **1.3 Test Server Startup**
```bash
python jobkranti_mcp.py
```

**Expected Output:**
```
🚀 Starting JobKranti AI-Powered MCP server on http://0.0.0.0:8086
🤖 AI Agent initialized with intelligent conversation analysis
🔗 Connect from PuchAI using your auth token
📱 Demo data loaded with sample job postings
✅ OpenAI API connected - Advanced AI features enabled
```

### **Step 2: Basic Server Testing**

#### **2.1 Test Server Health**
Open another terminal:
```bash
# Test if server is running
curl http://localhost:8086/health
# Should return server status

# Test MCP endpoint
curl http://localhost:8086/sse
# Should return SSE connection info
```

#### **2.2 Test with ngrok (Required for PuchAI)**
```bash
# Install ngrok if not already installed
# Download from https://ngrok.com/download

# Start tunnel
ngrok http 8086
```

**Copy the HTTPS URL** (e.g., `https://abc123.ngrok.app`)

### **Step 3: Connect with PuchAI**

#### **3.1 Open PuchAI**
- Go to: https://wa.me/+919998881729
- Or scan QR code if available

#### **3.2 Connect MCP Server**
Send this message to PuchAI:
```
/mcp connect https://abc123.ngrok.app/sse demo_token_hackathon_2025
```

**Expected Response:**
```
✅ MCP server connected successfully!
Available tools: validate, create_user_profile, intelligent_job_search, post_job_intelligently, generate_smart_resume, ai_job_safety_analysis, ai_conversation_analyzer
```

### **Step 4: Test Each MCP Tool**

#### **4.1 Test Profile Creation**
```
Create my profile: My name is Ravi Kumar, I live in Gurgaon Sector 15, I have 3 years security guard experience, and I can also drive. I prefer night shifts.

Phone: 9876543210
```

**Expected:** Structured profile creation with AI analysis

#### **4.2 Test Job Search**
```
I need a security guard job in Delhi or Gurgaon with at least 18000 salary
```

**Expected:** Relevant job listings with safety scores

#### **4.3 Test Job Posting**
```
Post this job: Need reliable house cleaning person in Dwarka, Delhi. 2 hours daily, 8000 per month. Contact 9123456789
```

**Expected:** Structured job posting with candidate matching

#### **4.4 Test Safety Analysis**
```
Check if this job is safe: "Earn 50000 per month from home! Just pay 5000 registration fee. Call 9999999999"
```

**Expected:** HIGH_RISK warning with detailed analysis

#### **4.5 Test Resume Generation**
```
Generate resume for user ID [user_id_from_step_4.1] targeting: Security guard position for office complex, night shift preferred
```

**Expected:** Professional resume image

#### **4.6 Test Salary Insights**
```
What salary can I expect for security guard job in Delhi with 3 years experience?
```

**Expected:** Comprehensive salary analysis

### **Step 5: Debug Common Issues**

#### **5.1 Server Not Starting**
```bash
# Check Python version
python3 --version  # Should be 3.11+

# Check dependencies
pip list | grep fastmcp

# Check .env file
cat .env
```

#### **5.2 PuchAI Connection Failed**
```bash
# Verify ngrok is running
curl https://your-ngrok-url.ngrok.app/sse

# Check auth token
grep AUTH_TOKEN .env

# Test with curl
curl -H "Authorization: Bearer demo_token_hackathon_2025" https://your-ngrok-url.ngrok.app/sse
```

#### **5.3 Tools Not Working**
Send to PuchAI:
```
/mcp tools
```

Should list all 7 tools. If not:
```
/mcp diagnostics-level debug
```

### **Step 6: Demo Testing Script**

Create `test_demo.py`:
```python
#!/usr/bin/env python3
"""Test script for JobKranti demo scenarios"""

import asyncio
import httpx
import json

async def test_demo_scenarios():
    base_url = "http://localhost:8086"
    
    # Test scenarios
    scenarios = [
        {
            "name": "Profile Creation",
            "query": "Main Ravi hun, Gurgaon mein rehta hun, 3 saal security ka experience hai",
            "phone": "9876543210"
        },
        {
            "name": "Job Search", 
            "query": "Security guard job in Delhi"
        },
        {
            "name": "Safety Check",
            "description": "Earn 50000 per month! Pay 5000 registration fee",
            "salary": 50000
        }
    ]
    
    print("🧪 Testing JobKranti Demo Scenarios...")
    
    # Add your test logic here
    print("✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_demo_scenarios())
```

Run tests:
```bash
python test_demo.py
```

### **Step 7: Performance Testing**

#### **7.1 Load Testing**
```bash
# Install hey for load testing
# Linux: sudo apt install hey
# Mac: brew install hey

# Test server performance
hey -n 100 -c 10 http://localhost:8086/sse
```

#### **7.2 Response Time Testing**
```bash
# Test tool response times
time curl -X POST http://localhost:8086/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list"}'
```

### **Step 8: Pre-Demo Checklist**

Before the hackathon demo:

- [ ] ✅ Server starts without errors
- [ ] ✅ ngrok tunnel is stable
- [ ] ✅ PuchAI connects successfully
- [ ] ✅ All 7 tools respond correctly
- [ ] ✅ AI features work (or fallback gracefully)
- [ ] ✅ Demo data is loaded
- [ ] ✅ Safety analysis detects scams
- [ ] ✅ Resume generation works
- [ ] ✅ Multilingual input handled

### **Step 9: Troubleshooting Guide**

**Common Errors & Solutions:**

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: fastmcp` | `pip install fastmcp` |
| `AUTH_TOKEN not set` | Check `.env` file |
| `Connection refused` | Restart server, check port 8086 |
| `OpenAI error` | Check API key or use without OpenAI |
| `Tools not showing` | Send `/mcp tools` to refresh |

### **Step 10: Demo Day Setup**

**30 minutes before demo:**
1. Start server: `python jobkranti_mcp.py`
2. Start ngrok: `ngrok http 8086`
3. Test connection with PuchAI
4. Run through all 6 demo scenarios
5. Have backup screenshots ready

## 🔑 **Auth Token Setup for PuchAI**

### **The Auth Token is YOUR Choice!**

The auth token is **not provided by PuchAI** - **YOU create it yourself**! It's a secret key that you define to secure your MCP server.

### **Step 1: Generate Your Auth Token**

**Option A: Simple Random String**
```bash
# Generate a secure random token
openssl rand -hex 32
# Example output: a1b2c3d4e5f6...
```

**Option B: Use Any Strong Password**
```bash
# Or just create a strong password
AUTH_TOKEN="jobkranti_hackathon_2025_secure_token"
```

**Option C: UUID-based**
```bash
# Generate UUID
python3 -c "import uuid; print(str(uuid.uuid4()).replace('-', ''))"
```

### **Step 2: Set Your Auth Token**

**In your `.env` file:**
```bash
AUTH_TOKEN="your_chosen_secure_token_here"
MY_NUMBER="919876543210"  # Your WhatsApp number
```

**Example:**
```bash
AUTH_TOKEN="jobkranti_puchai_demo_2025_secure"
MY_NUMBER="919998881729"
```

### **Step 3: Understanding the Validation Process**

According to PuchAI docs, your MCP server **MUST have a `validate` tool** that:

1. **Accepts the bearer token** (the one you created)
2. **Returns your phone number** in format `{country_code}{number}`

**Our implementation already does this:**
```python
@mcp.tool
async def validate() -> str:
    """Validate this MCP server for PuchAI integration"""
    return MY_NUMBER  # Returns "919998881729"
```

### **Step 4: Connect to PuchAI**

**When you connect, use YOUR token:**
```
/mcp connect https://your-ngrok-url.ngrok.app/sse your_chosen_secure_token_here
```

**Real example:**
```
/mcp connect https://abc123.ngrok.app/sse jobkranti_puchai_demo_2025_secure
```

### **Step 5: How Authentication Works**

```mermaid
sequenceDiagram
    participant P as PuchAI
    participant M as Your MCP Server
    
    P->>M: /mcp connect <url> <your_token>
    M->>M: Check if token matches AUTH_TOKEN
    P->>M: Call validate() tool
    M->>P: Return MY_NUMBER (919998881729)
    P->>P: Verify phone number matches user
    P->>M: ✅ Connection established
```

### **Step 6: Complete Setup Example**

**Your `.env` file:**
```bash
# Security
AUTH_TOKEN="hackathon_demo_secure_2025"
MY_NUMBER="919998881729"

# Optional AI (enhances features but not required)
OPENAI_API_KEY="sk-your-openai-key-here"
```

**Start your server:**
```bash
python jobkranti_mcp.py
ngrok http 8086
```

**Connect from PuchAI:**
```
/mcp connect https://abc123.ngrok.app/sse hackathon_demo_secure_2025
```

### **Step 7: Security Best Practices**

**For Demo:**
- Use any memorable token: `"puchai_demo_2025"`
- Keep it simple for testing

**For Production:**
- Use cryptographically secure random tokens
- Never share your token publicly
- Use environment variables
- Rotate tokens regularly

### **Step 8: Testing Your Auth**

**Test locally:**
```bash
# Test your validation endpoint
curl -H "Authorization: Bearer hackathon_demo_secure_2025" \
     https://your-ngrok-url.ngrok.app/sse
```

**Check server logs for:**
```
✅ OpenAI API connected - Advanced AI features enabled
🔗 Connect from PuchAI using your auth token
```

### **Step 9: Demo Day Setup**

**For the hackathon demo:**

1. **Choose a simple token:** `"puchai_hackathon_2025"`
2. **Set your phone:** Your actual WhatsApp number
3. **Start server:** `python jobkranti_mcp.py`
4. **Create tunnel:** `ngrok http 8086`
5. **Connect:** `/mcp connect https://xyz.ngrok.app/sse puchai_hackathon_2025`

### **Step 10: Common Auth Issues & Solutions**

| Issue | Cause | Solution |
|-------|-------|----------|
| `Authentication failed` | Token mismatch | Check `.env` AUTH_TOKEN |
| `Validate tool not found` | Missing tool | Ensure `validate()` tool exists |
| `Invalid phone format` | Wrong format | Use `919876543210` format |
| `Connection refused` | Server down | Restart server and ngrok |

## 🎯 **Key Takeaway**

**You are in complete control!** 
- **You create** the auth token
- **You set** your phone number  
- **PuchAI connects** using YOUR credentials

The auth token is like a password for your MCP server - make it secure but memorable for demos! 🔐🚀