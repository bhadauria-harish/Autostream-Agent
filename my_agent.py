# Importing necessary libraries
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# Load knowledge base
def load_knowledge_base():
    """Load AutoStream knowledge base from JSON file"""
    with open('knowledge_base.json', 'r') as f:
        return json.load(f)

# Initialize RAG system
def rag():
    """Setup RAG pipeline with FAISS vector store"""
    kb = load_knowledge_base()
    documents = []
    
    # Add pricing information
    for plan_name, plan_info in kb['pricing'].items():
        doc_text = f"Plan: {plan_name}\n"
        doc_text += f"Price: ${plan_info['price']}/month\n"
        doc_text += f"Videos: {plan_info['videos']}\n"
        doc_text += f"Resolution: {plan_info['resolution']}\n"
        if plan_info['features']:
            doc_text += f"Features: {', '.join(plan_info['features'])}\n"
        documents.append(Document(page_content=doc_text, metadata={"type": "pricing", "plan": plan_name}))
    
    # Add company policies
    for policy_name, policy_text in kb['company policies'].items():
        documents.append(Document(page_content=f"{policy_name}: {policy_text}", metadata={"type": "policy", "name": policy_name}))
    
    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 2})

@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Capture lead information when user shows high intent to sign up"""
    print(f"\n{'='*50}")
    print(f"Lead captured successfully!")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Platform: {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"

# Initialize components
retriever = rag()
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    intent: str
    lead_info: dict
    next_step: str
    awaiting_confirmation: bool
    in_lead_flow: bool
    current_field: str
    just_asked_field: bool

# Intent Classification
def classify_intent(state: AgentState) -> AgentState:
    """Classify user intent from the conversation"""
    messages = state["messages"]
    
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier for AutoStream, a SaaS video editing platform.
        Classify the user's intent into ONE of these categories:
        1. "greeting" - casual greetings, general chat, hello, hi, hey
        2. "inquiry" - questions about product, pricing, features, policies
        3. "high_intent" - user wants to sign up, try the product, get started, subscribe, buy
        
        Respond with ONLY the category name in lowercase, nothing else: greeting, inquiry, or high_intent"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = intent_prompt | llm
    response = chain.invoke({"messages": messages})
    intent = response.content.strip().lower()
    
    state["intent"] = intent
    print(f"Detected Intent: {intent}")
    
    return state

def greeting_bot(state: AgentState) -> AgentState:
    """Handle greeting intent"""
    messages = state["messages"]
    
    response = "Hello! I'm AutoStream's AI assistant. \nHow can I assist you today?"
    
    state["messages"] = messages + [AIMessage(content=response)]
    state["next_step"] = "end"
    return state

def inquiry_bot(state: AgentState) -> AgentState:
    """Handle inquiry intent using RAG"""
    messages = state["messages"]
    last_user_message = messages[-1].content
    docs = retriever.invoke(last_user_message)
    context = "\n".join([doc.page_content for doc in docs])
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are AutoStream's helpful AI assistant. Use the following context to answer questions accurately.
        
        Context:
        {context}
        
        Guidelines:
        - Be concise and helpful
        - Use information from the context
        - If asked about pricing, mention both plans
        - Be encouraging but not pushy"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "messages": messages})
    
    state["messages"] = messages + [AIMessage(content=response.content)]
    state["next_step"] = "end"
    return state

def generate_response(state: AgentState) -> AgentState:
    """Route based on intent"""
    intent = state["intent"]
    
    if intent == "greeting":
        return greeting_bot(state)
        
    elif intent == "inquiry":
        return inquiry_bot(state)
        
    elif intent == "high_intent":
        state["in_lead_flow"] = True
        state["current_field"] = "name"
        state["just_asked_field"] = False
        state["next_step"] = "ask_field"
        
    return state

def ask_next_field(state: AgentState) -> AgentState:
    """Ask for the next field in sequence"""
    messages = state["messages"]
    current_field = state["current_field"]
    lead_info = state.get("lead_info", {})
    
    # If all fields are collected, ask for confirmation
    if len(lead_info) >= 3:
        state["next_step"] = "confirm_data"
        return state
    
    # Ask for the current field
    if current_field == "name":
        question = """"Great! I'd love to help you get started with AutoStream. I'll need a few details.
            Could you please provide your full name?"""
    elif current_field == "email":
        question = "Thanks! What's your email address?"
    elif current_field == "platform":
        question = "Perfect! Which platform do you primarily create content for? (YouTube, Instagram, TikTok, etc.)"
    else:
        # All fields collected
        state["next_step"] = "confirm_data"
        return state
    
    state["messages"] = messages + [AIMessage(content=question)]
    state["just_asked_field"] = True
    state["next_step"] = "end"
    
    return state

def collect_field_info(state: AgentState) -> AgentState:
    """Collect the current field information"""
    messages = state["messages"]
    lead_info = state.get("lead_info", {})
    current_field = state["current_field"]

    last_message = messages[-1].content if messages else ""
    
    # Store the user's input for the current field
    lead_info[current_field] = last_message.strip()
    state["lead_info"] = lead_info
    state["just_asked_field"] = False
    
    print(f"Captured: {current_field} = {last_message.strip()}")

    if current_field == "name":
        state["current_field"] = "email"
    elif current_field == "email":
        state["current_field"] = "platform"
    elif current_field == "platform":
        state["current_field"] = "done"
    
    state["next_step"] = "ask_field"
    return state

def confirm_data_with_user(state: AgentState) -> AgentState:
    """Ask user to confirm the collected data"""
    messages = state["messages"]
    lead_info = state.get("lead_info", {})
    
    confirmation_text = "I've collected the following information:\n\n"
    confirmation_text += f"Name: {lead_info.get('name', '')}\n"
    confirmation_text += f"Email: {lead_info.get('email', '')}\n"
    confirmation_text += f"Platform: {lead_info.get('platform', '')}\n"
    confirmation_text += "\nIs this correct? (Please answer yes/no)"
    
    state["messages"] = messages + [AIMessage(content=confirmation_text)]
    state["awaiting_confirmation"] = True
    state["next_step"] = "end"
    
    print(f"Confirmation requested. Awaiting user response...")
    
    return state

def handle_confirmation(state: AgentState) -> AgentState:
    """Handle user confirmation response"""
    messages = state["messages"]
    last_message = messages[-1].content.lower() if messages and messages[-1].content else ""
    
    # Simple confirmation check without LLM

    if not last_message:
        state["next_step"] = "confirm_data"
        return state

    confirm_words = ["yes", "correct", "right", "ok", "okay", "yep", "yeah", "yup", "true", "confirmed"]
    deny_words = ["no", "wrong", "incorrect", "not", "false", "nope", "nah"]
    
    user_confirmed = False
    last_message_lower = last_message.lower()
    
    for word in confirm_words:
        if word in last_message_lower:
            user_confirmed = True
            break

    if not user_confirmed:
        for word in deny_words:
            if word in last_message_lower:
                user_confirmed = False
                break
    
    if user_confirmed:
        state["awaiting_confirmation"] = False
        state["next_step"] = "execute_tool"
        print("User confirmed data!")
    else:
        question = "I understand. Which field is incorrect? Please tell me which one needs to be changed (name, email, or platform)."
        state["messages"] = messages + [AIMessage(content=question)]
        state["awaiting_confirmation"] = False
        state["next_step"] = "identify_wrong_field"
    
    return state

def identify_wrong_field(state: AgentState) -> AgentState:
    """Identify which field user wants to correct"""
    messages = state["messages"]
    last_message = messages[-1].content if messages and messages[-1].content else ""
    
    if not last_message:
        question = "Which field would you like to correct? (name, email, or platform)"
        state["messages"] = messages + [AIMessage(content=question)]
        state["next_step"] = "end"
        return state
    
    # Simple pattern matching for field identification
    last_message_lower = last_message.lower()
    
    if "name" in last_message_lower:
        wrong_field = "name"
    elif "email" in last_message_lower or "mail" in last_message_lower:
        wrong_field = "email"
    elif "platform" in last_message_lower or "youtube" in last_message_lower or "instagram" in last_message_lower or "tiktok" in last_message_lower:
        wrong_field = "platform"
    else:
        # Use LLM as fallback
        identify_prompt = ChatPromptTemplate.from_messages([
            ("system", """The user is indicating which field is wrong in their lead information.
            
            User message: {message}
            
            Determine which field they want to correct: name, email, or platform
            
            Respond with ONLY: "name" or "email" or "platform"
            """),
        ])
        
        chain = identify_prompt | llm
        response = chain.invoke({"message": last_message})
        wrong_field = response.content.strip().lower()
    
    print(f"User wants to correct: {wrong_field}")
    
    # Remove the wrong field
    lead_info = state.get("lead_info", {})
    if wrong_field in lead_info:
        del lead_info[wrong_field]
    
    state["lead_info"] = lead_info
    state["current_field"] = wrong_field
    state["just_asked_field"] = False
    state["next_step"] = "ask_field"
    
    return state

def execute_lead_tool(state: AgentState) -> AgentState:
    """Execute the mock_lead_capture tool"""
    lead_info = state["lead_info"]
    messages = state["messages"]
    
    # Call the tool
    result = mock_lead_capture.invoke({
        "name": lead_info["name"],
        "email": lead_info["email"],
        "platform": lead_info["platform"]
    })
    
    response = f"""Excellent! I've registered your interest in AutoStream.
    Our team will reach out to {lead_info['email']} shortly to help you.\n
     Thank you for choosing AutoStream!"""
    
    state["messages"] = messages + [AIMessage(content=response)]
    state["next_step"] = "end"
    state["in_lead_flow"] = False
    state["lead_capture_complete"] = True
    
    return state

# Router function
def route_next(state: AgentState) -> str:
    """Determine next node based on state"""
    next_step = state.get("next_step", "end")
    return next_step if next_step != "end" else "end"

# Build the graph
def create_agent():
    """Create the LangGraph agent"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("ask_field", ask_next_field)
    workflow.add_node("collect_field", collect_field_info)
    workflow.add_node("confirm_data", confirm_data_with_user)
    workflow.add_node("handle_confirmation", handle_confirmation)
    workflow.add_node("identify_wrong_field", identify_wrong_field)
    workflow.add_node("execute_tool", execute_lead_tool)
    
    # Add edges
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "generate_response")
    
    workflow.add_conditional_edges(
        "generate_response",
        route_next,
        {
            "ask_field": "ask_field",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "ask_field",
        route_next,
        {
            "collect_field": "collect_field",
            "confirm_data": "confirm_data",
            "end": END
        }
    )
    
    workflow.add_edge("collect_field", "ask_field")
    workflow.add_edge("confirm_data", END)
    
    workflow.add_conditional_edges(
        "handle_confirmation",
        route_next,
        {
            "execute_tool": "execute_tool",
            "identify_wrong_field": "identify_wrong_field"
        }
    )
    
    workflow.add_edge("identify_wrong_field", "ask_field")
    workflow.add_edge("execute_tool", END)
    
    return workflow.compile()

# Main conversation loop
def run_agent():
    """Run the conversational agent"""
    agent = create_agent()
    
    print("=" * 70)
    print("AutoStream AI Agent - Type 'quit' to exit")
    print("=" * 70)
    
    state = {
        "messages": [],
        "intent": "",
        "lead_info": {},
        "next_step": "",
        "awaiting_confirmation": False,
        "in_lead_flow": False,
        "current_field": "",
        "just_asked_field": False,
        "lead_capture_complete": False
    }
    
    while True:
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'leave', 'goodbye']:
            print("\nAgent: Thank you for chatting with AutoStream! Have a great day!")
            return
        
        if not user_input:
            continue
        
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            result = agent.invoke(state)
            state = result
            
            # Print agent response
            if state["messages"]:
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage):
                        print(f"\nAgent: {msg.content}")
                        break
            
            # Break after first interaction to enter main loop
            break
            
        except Exception as e:
            print(f"\nError: {e}")
            print("\nAgent: I encountered an error. Let's try again!")
    
    # Main conversation loop after first intent detection
    while True:
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'leave', 'goodbye']:
            print("\nAgent: Thank you for chatting with AutoStream! Have a great day!")
            break
        
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))
        
        # If lead capture is complete, exit
        if state.get("lead_capture_complete"):
            print("\nAgent: Thank you! Your registration is complete. Goodbye!")
            break
        
        try:
            # ALWAYS skip intent classification after first message in lead flow
            if state.get("in_lead_flow"):
                # print("\n....Lead Flow Active....")
                
                if state.get("awaiting_confirmation"):
                    state = handle_confirmation(state)
                    
                    if state["next_step"] == "execute_tool":
                        state = execute_lead_tool(state)
                        # Print final message
                        if state["messages"]:
                            print(f"\nAgent: {state['messages'][-1].content}")
                        # Exit after successful capture
                        break
                    elif state["next_step"] == "identify_wrong_field":
                        # Print the question asking which field is wrong
                        if state["messages"]:
                            print(f"\nAgent: {state['messages'][-1].content}")
                        continue
                
                # If we just asked which field is wrong, identify it
                elif state.get("next_step") == "identify_wrong_field":
                    state = identify_wrong_field(state)
                    # Now go back to ask for that field
                    state = ask_next_field(state)
                    if state["messages"]:
                        print(f"\nAgent: {state['messages'][-1].content}")
                    continue
                
                elif state.get("just_asked_field"):
                    state = collect_field_info(state)
                    
                    # All fields collected, ask for confirmation
                    if len(state.get("lead_info", {})) >= 3:
                        state = confirm_data_with_user(state)
                    else:
                        # Ask for next field
                        state = ask_next_field(state)
                    
                    # Print response
                    if state["messages"]:
                        print(f"\nAgent: {state['messages'][-1].content}")
                    continue
            
            else:
                result = agent.invoke(state)
                state = result
                
                # Print agent response
                if state["messages"]:
                    for msg in reversed(state["messages"]):
                        if isinstance(msg, AIMessage):
                            print(f"\nAgent: {msg.content}")
                            break
        
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print("\nAgent: I encountered an error. Let's try again!")

if __name__ == "__main__":    
    run_agent()