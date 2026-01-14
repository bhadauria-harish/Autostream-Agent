# AutoStream AI Agent - Social-to-Lead System

A production-ready conversational AI agent that converts social media conversations into qualified business leads through natural dialogue. Built with LangGraph and Google Gemini.

Check out [Video Demo](https://drive.google.com/file/d/1ZQp6848MR-yDpBNOyjY2BKHuVRr6Ut1r/view?usp=sharing)

## Overview

AutoStream is an intelligent chat bot designed for a fictional SaaS company providing automated video editing tools for content creators. The agent demonstrates:

- **Intent Classification**: Automatically detects user intent (greeting, inquiry, high-intent)
- **RAG-Powered Responses**: Answers questions using a local knowledge base with FAISS
- **Progressive Lead Capture**: Collects user information naturally through conversation
- **Validation Loop**: Confirms data accuracy before submission
- **State Management**: Maintains context across multiple conversation turns

## Key Features

### 1. **Intelligent Intent Detection**
- Classifies user messages into three categories
- Routes conversation appropriately
- No re-classification during lead capture flow

### 2. **Knowledge Retrieval (RAG)**
- FAISS vector store for semantic search
- Answers questions about pricing, features, and policies
- Contextual, accurate responses

### 3. **Natural Lead Capture**
- Asks for information progressively (name → email → platform)
- Validates data with user confirmation
- Allows corrections through conversational interface
- Loops until user confirms accuracy

### 4. **Production-Ready Design**
- Clean, modular code architecture
- Comprehensive error handling
- Human-like conversation flow
- Easy to extend and maintain

## Quick Start

### Prerequisites

- Python 3.9+
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/bhadauria-harish/Autostream-Agent.git
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**

Create a `.env` file:
```
GOOGLE_API_KEY= <your-gemini-api-key-here>
```

4. **Run the agent**
```bash
python my_agent.py
```

## 💬 Usage Example

```
You: Hi there

Agent: Hello! I'm AutoStream's AI assistant. I'm here to help you 
       with our automated video editing tools for content creators. 
       How can I assist you today?

You: What are your pricing plans?

Agent: AutoStream offers two pricing tiers:
       
       Basic Plan ($29/month):
       - 10 videos per month
       - 720p resolution
       
       Pro Plan ($79/month):
       - Unlimited videos
       - 4K resolution
       - AI captions and priority rendering

You: I want to sign up for the Pro plan.

Agent: Great! I'd love to help you get started with AutoStream. 
       To set up your account, I'll need a few details. 
       Could you please provide your full name?

You: YOUR-NAME

Agent: Thanks! What's your email address?

You: your@gmail.com

Agent: Perfect! Which platform do you primarily create content for?

You: platform

Agent: I've collected the following information:
       
        Name:     YOUR-NAME
        Email:    your@example.com
        Platform: platform
       
        Is this information correct?

You: yes

==================================================
Lead Captured Successfully!
==================================================
Name:     YOUR-NAME
Email:    your@example.com
Platform: platform
==================================================

Agent: Excellent! I've registered your interest in AutoStream. 
       Our team will reach out to john@example.com shortly to 
       help you get started. Thank you for choosing AutoStream!
```

## Project Structure

```
autostream-agent/
│
├── my_agent.py           # Main agent implementation
├── knowledge_base.json     # RAG data source
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in repo)
│
├── README.md              # This file
├── ARCHITECTURE.md        # Technical architecture details
```

## Architecture

### LangGraph Workflow

```
                    START
                      │
                      ▼
            ┌──────────────────┐
            │ Classify Intent  │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │Generate Response │
            └────────┬─────────┘
                     │
         ┌───────────┴──────────┐
         │                      │
    [greeting/inquiry]    [high_intent]
         │                      │
         ▼                      ▼
        END              ┌──────────────┐
                         │  Ask Field   │
                         └──────┬───────┘
                                │
                      ┌─────────┴─────────┐
                      │                   │
                [need field]        [all collected]
                      │                   │
                      ▼                   ▼
              ┌──────────────┐    ┌──────────────┐
              │Collect Field │    │Confirm Data  │
              └──────┬───────┘    └──────┬───────┘
                     │                   │
                     └─────────┬─────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                 [confirmed]          [denied]
                     │                   │
                     ▼                   ▼
              ┌──────────────┐    ┌──────────────┐
              │Execute Tool  │    │Identify Wrong│
              └──────┬───────┘    │    Field     │
                     │            └──────┬───────┘
                     │                   │
                     ▼                   │
                    END   ←──────────────┘
```


#### 8 Core Nodes

1. **classify_intent**: Detects user intent (greeting/inquiry/high_intent)
2. **generate_response**: Routes to appropriate handler
3. **ask_field**: Asks for next required field
4. **collect_field**: Stores user's answer
5. **confirm_data**: Shows collected data for confirmation
6. **handle_confirmation**: Processes confirmation response
7. **identify_wrong_field**: Determines which field to recollect
8. **execute_tool**: Calls mock_lead_capture and completes flow

### Validation Loop

```
All 3 fields collected
        ↓
Show data to user
        ↓
"Is this correct?"
        ↓
    ┌───┴───┐
    │       │
   YES     NO
    │       │
    │       └─→ "Which field is wrong?"
    │               ↓
    │          Delete field
    │               ↓
    │          Ask for field again
    │               ↓
    └───────────────┘
    (repeat until YES)
        ↓
Execute tool
```
## Design Decisions

### Why LangGraph?

| Feature | LangGraph | Traditional Approach |
|---------|-----------|---------------------|
| **State Management** | Built-in, immutable | Manual tracking |
| **Flow Control** | Graph-based, visual | Nested if-else |
| **Debugging** | Clear node execution | Complex stack traces |
| **Scalability** | Easy to add nodes | Refactor entire code |
| **Production Ready** | Enterprise-grade | Custom implementation |

### Why Progressive Collection?

**Alternatives Considered:**

1. **All at once**: "Give me name, email, and platform"
   - ❌ Overwhelming for users
   - ❌ Less natural
   - ❌ Higher drop-off rate

2. **Form-based**: Display HTML form
   - ❌ Not conversational
   - ❌ Breaks chat experience
   - ❌ Requires UI change

3. **Progressive (chosen)**: One field at a time
   - ✅ Natural conversation
   - ✅ Higher completion rate
   - ✅ Better user experience
   - ✅ Easy error correction

---

## Configuration

### Knowledge Base

Edit `knowledge_base.json` to customize:

- Pricing plans
- Product features
- Company policies
- Support information

### LLM Model

Change in `my_agent.py`:
```python
llm = ChatGoogleGenerativeAI(
    model='your-model-name',  # Change model here
    temperature=0
)
```

### Embeddings

Change in `my_agent.py`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Change here
)
```

## Testing

### Test Scenarios

1. **Greeting Flow**
   - Input: "Hello", "Hi", "Hey"
   - Expected: Friendly welcome message

2. **Inquiry Flow**
   - Input: "What's your pricing?", "Tell me about features"
   - Expected: RAG-powered response from knowledge base

3. **High-Intent Flow**
   - Input: "I want to buy", "Sign me up", "I want the Pro plan"
   - Expected: Start lead capture process

4. **Correction Flow**
   - During confirmation, say "no"
   - Specify which field is wrong
   - Provide corrected information
   - Confirm again

## Future Development

#### 1. API Wrapper

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/chat")
async def chat(user_id: str, message: str):
    # Load state from Redis
    state = load_state(user_id)
    
    # Add message
    state["messages"].append(HumanMessage(content=message))
    
    # Run agent
    result = agent.invoke(state)
    
    # Save state
    save_state(user_id, result)
    
    # Return response
    return {"response": result["messages"][-1].content}
```


### 2. WhatsApp Integration

```python
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    data = request.json
    
    user_phone = data['from']
    message = data['message']['text']
    
    # Load state
    state = redis.get(f"user:{user_phone}")
    
    # Process message
    state["messages"].append(HumanMessage(content=message))
    result = agent.invoke(state)
    
    # Save state
    redis.setex(f"user:{user_phone}", 3600, result)
    
    # Send response via WhatsApp Business API
    response_text = result["messages"][-1].content
    send_whatsapp_message(user_phone, response_text)
    
    return {"status": "ok"}
```
---

## Technical Stack

- **Framework**: LangGraph (state machine workflow)
- **LLM**: Google Gemini (gemini-2.5-flash)
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Language**: Python 3.9+

## Security Considerations

- API keys stored in `.env` file (never commit to repo)
- Input validation on collected data
- No sensitive data logged
- Secure state management

## Troubleshooting

### Common Issues

**"GOOGLE_API_KEY not set"**
- Solution: Create `.env` file with your API key

**"Module not found: langchain"**
- Solution: `pip install -r requirements.txt`

**Embedding model download slow**
- Solution: First run downloads ~100MB model, subsequent runs are fast

**Agent not responding correctly**
- Solution: Check `knowledge_base.json` format and content

## License

This project is created as an assignment for educational purposes.

## Contact

For questions about this implementation, please refer to the documentation or create an issue in the repository.

---

**Built with ❤️ for ServiceHive Inflx Platform**

Last Updated: January 2026
