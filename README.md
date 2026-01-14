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
🎯 Lead Captured Successfully!
==================================================
Name:     YOUR-NAME
Email:    your@example.com
Platform: platform
==================================================

Agent: Excellent! I've registered your interest in AutoStream. 
       Our team will reach out to john@example.com shortly to 
       help you get started. Thank you for choosing AutoStream!
```

## 📁 Project Structure

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

### Core Components

1. **Intent Classifier**: LLM-based intent detection
2. **RAG System**: FAISS vector store + HuggingFace embeddings
3. **State Machine**: LangGraph workflow with 8 nodes
4. **Lead Capture**: Progressive information collection
5. **Validation**: User confirmation loop

### Conversation Flow

```
User Input → Intent Classification
                ↓
    ┌───────────┴───────────┐
    │                       │
Greeting/Inquiry      High Intent
    │                       │
RAG Response        Lead Capture Flow
    │                       │
Continue              Name → Email → Platform
                            ↓
                      Confirmation
                            ↓
                      Tool Execution
```

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

## 🧪 Testing

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

## Development

### Adding New Intents

1. Update intent classification prompt in `classify_intent()`
2. Add new handler function (e.g., `handle_new_intent()`)
3. Update `generate_response()` routing
4. Add appropriate edges in `create_agent()`

### Adding New Fields

1. Update `ask_next_field()` with new field question
2. Add field to sequence in `collect_field_info()`
3. Update `confirm_data_with_user()` display
4. Update `mock_lead_capture` tool signature

### Extending RAG

1. Add new data to `knowledge_base.json`
2. Update document processing in `rag()` if needed
3. Adjust retrieval parameters (`k` value) for more/fewer results

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

## 📧 Contact

For questions about this implementation, please refer to the documentation or create an issue in the repository.

---

**Built with ❤️ for ServiceHive Inflx Platform**

Last Updated: January 2026
