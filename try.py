import streamlit as st
import json
import re
import logging
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Sequence
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bank_rates_data = """
Fixed Deposit Interest Rates (as of 23/4/2025)
📌 PUBLIC SECTOR BANKS
1. Bank of Baroda: All slabs: 2.75%, Above 10Cr: 2.75% to 4.50%
2. Bank of India: All slabs: 2.75%, Above 10Cr: 2.75% to 3.05%
3. Bank of Maharashtra: All slabs: 2.75%
4. Canara Bank: Up to 1Cr: 2.90%, 1Cr to 10Cr: 2.90% to 2.95%, Above 10Cr: 3.05% to 4.00%
5. Central Bank of India: All slabs: 2.80%, Above 10Cr: 3.00% to 4.50%
6. Indian Bank: Up to 10L: 2.75%, 10L to 1Cr: 2.80%, Above 10Cr: 2.80% to 2.90%
7. Indian Overseas Bank: Up to 1Cr: 2.75%, 1Cr to 10Cr: 2.90%, Above 10Cr: 2.90%
8. Punjab & Sind Bank: Up to 1Cr: 2.60%, Above 10Cr: 2.80% to 4.90%
9. Punjab National Bank: Up to 10L: 2.70%, Above 10L: 2.75%, Above 10Cr: 2.75% to 3.00%
10. State Bank of India: All slabs: 2.70%, Above 10Cr: 3.00%
11. UCO Bank: Up to 5L: 2.60%, Above 5L: 2.75%
12. IPPB (India Post Payments Bank): Flat 4.00% across all slabs
13. Union Bank of India: Up to 50L: 2.75%, Above 50L to 1Cr: 2.90%, Above 10Cr: 2.90% to 4.20%
"""

system_message = f"""
You are a financial assistant helping customers choose the best fixed deposit plan from Indian public sector banks.

🎯 Your goal:
- Use ONLY the interest rates provided below. Do NOT assume, guess, or invent any other rates.
- Recommend the TOP 2 BANKS that match the user's needs.
- Justify based on DEPOSIT AMOUNT, DURATION, GOAL (return, safety, or stability), and RISK PREFERENCE.
- Be clear about the interest rate, slab, and unique features.
- If information is missing, ask for it (e.g., 'What is your deposit amount?').

🚫 Never use external or assumed data.
📌 Use only this:

{bank_rates_data}
"""

st.set_page_config(page_title="Fixed Deposit Assistant", layout="centered")
st.title("🏦 Fixed Deposit Assistant")
st.markdown("Ask about fixed deposits, and I'll recommend the best plans from Indian public sector banks based on your preferences! Try something like: 'I want to invest ₹5 lakh for 5 years with high returns.'")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_message}
    ]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"
if "preferences" not in st.session_state:
    st.session_state.preferences = {}
    logger.info("Initialized st.session_state.preferences as {}")

def extract_preferences(message: str) -> dict:
    message_lower = message.lower()
    preferences = {}
    amount_pattern = r"(?:₹|\b)(\d+(?:\.\d+)?)\s*(lakh|cr|thousand|[\d,]+)?"
    amount_match = re.search(amount_pattern, message_lower)
    if amount_match:
        value = float(amount_match.group(1))
        unit = amount_match.group(2) or ""
        if "lakh" in unit:
            preferences["deposit_amount"] = value * 100000
        elif "cr" in unit:
            preferences["deposit_amount"] = value * 10000000
        elif "thousand" in unit:
            preferences["deposit_amount"] = value * 1000
        else:
            preferences["deposit_amount"] = value
    if "highest return" in message_lower or "max return" in message_lower or "high return" in message_lower:
        preferences["goal"] = "Highest Return"
    elif "stable rate" in message_lower or "consistent" in message_lower:
        preferences["goal"] = "Stable Rate"
    elif "safe" in message_lower or "secure" in message_lower:
        preferences["goal"] = "Safe Bank"
    duration_pattern = r"(\d+)\s*(year|years)"
    duration_match = re.search(duration_pattern, message_lower)
    if duration_match:
        preferences["duration"] = f"{duration_match.group(1)} Year{'s' if int(duration_match.group(1)) > 1 else ''}"
    if "low risk" in message_lower or "safe" in message_lower:
        preferences["risk_preference"] = "Low"
    elif "medium risk" in message_lower:
        preferences["risk_preference"] = "Medium"
    elif "high risk" in message_lower or "doesn't matter" in message_lower:
        preferences["risk_preference"] = "Doesn't Matter"
    return preferences

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    ("human", "{user_message}")
])

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key
)

trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=model,
    include_system=False,
    allow_partial=False,
    start_on="human",
)

ALLOWED_TOPICS = ["fixed deposits", "finance", "banks", "interest rates"]
RESTRICTED_TOPICS = ["politics", "religion", "cats", "dogs"]
OFF_TOPIC_RESPONSE = "I'm a financial assistant focused on fixed deposits. Please ask about fixed deposits or banking!"

def is_on_topic(message: str) -> bool:
    message_lower = message.lower()
    for topic in ALLOWED_TOPICS:
        if topic in message_lower:
            return True
    for topic in RESTRICTED_TOPICS:
        if topic in message_lower:
            return False
    return True

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    preferences: dict

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    logger.info("Entering call_model with state: %s", state)
    system_msg = SystemMessage(content=system_message)
    updated_messages = [system_msg] + state["messages"]
    trimmed_messages = trimmer.invoke(updated_messages)
    if not trimmed_messages:
        trimmed_messages = state["messages"]
    human_message = next((msg.content for msg in trimmed_messages if isinstance(msg, HumanMessage)), "")
    if not human_message:
        human_message = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")
    if not human_message:
        raise ValueError("No human message found")
    if not is_on_topic(human_message):
        return {"messages": [AIMessage(content=OFF_TOPIC_RESPONSE)], "preferences": state["preferences"]}
    new_preferences = extract_preferences(human_message)
    current_preferences = state.get("preferences", {}).copy()
    if new_preferences:
        current_preferences.update(new_preferences)
    today = datetime.now().strftime("%A, %B %d, %Y")
    preferences_str = json.dumps(current_preferences) if current_preferences else ""
    if preferences_str:
        human_message = f"Today is {today}.\n{human_message}\n\nUser Preferences: {preferences_str}"
    else:
        human_message = f"Today is {today}.\n{human_message}"
    try:
        prompt = prompt_template.invoke({
            "system_message": system_message,
            "user_message": human_message
        })
        messages = prompt.to_messages() if hasattr(prompt, "to_messages") else prompt
        response = model.invoke(messages)
    except Exception as e:
        raise ValueError(f"Model invocation failed: {e}")
    return {"messages": [response], "preferences": current_preferences}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.preferences:
    st.subheader("Extracted Preferences (JSON)")
    st.json(st.session_state.preferences)

if prompt := st.chat_input("Type your question about fixed deposits (e.g., 'I want to invest ₹5 lakh for 5 years with high returns')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    new_preferences = extract_preferences(prompt)
    if new_preferences:
        st.session_state.preferences.update(new_preferences)
    logger.info("Session state preferences before LangGraph: %s", st.session_state.preferences)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_state = {"messages": [HumanMessage(content=prompt)], "preferences": st.session_state.preferences.copy()}
    with st.spinner("🧠 Analyzing your request..."):
        try:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for state in app.stream(input_state, config, stream_mode="values"):
                    if state.get("messages"):
                        latest_message = state["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            full_response = latest_message.content
                            response_placeholder.markdown(full_response + "▌")
                    if state.get("preferences"):
                        st.session_state.preferences = state["preferences"]
                        logger.info("Updated st.session_state.preferences: %s", st.session_state.preferences)
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
