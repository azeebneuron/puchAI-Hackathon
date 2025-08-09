

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