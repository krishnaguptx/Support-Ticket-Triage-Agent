
import json
import re
import sqlite3
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

load_dotenv()

# 1. LLM: Groq free-tier model
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


# 2. Structured outputs
class Classification(BaseModel):
    category: Literal["billing", "technical", "general"]=Field(description="Classify the support ticket into billing, technical, or general")
    urgency: Literal["low", "medium", "high"]=Field(description="Determine the urgency level of the support ticket")


class Review(BaseModel):
    score: int=Field(description="Rate the drafted reply from 1 to 10 based on accuracy, clarity, helpfulness, and professionalism",
                     ge=1,le=10)
    feedback: str=Field(description="Provide constructive feedback explaining the score and suggesting improvements to the drafted reply")


classifier = llm.with_structured_output(Classification, method="json_schema", strict=True)
reviewer = llm.with_structured_output(
    Review,
    method="function_calling"
)



# 3. State (messages are retained across turns for the same thread_id)
class TicketState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    category: str
    urgency: str
    order_details: str
    draft: str
    feedback: str
    score: int
    attempts: int
    approved: bool
    status: str


def latest_ticket(state: TicketState) -> str:
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            return str(msg.content)

    return ""
    


def conversation(state: TicketState) -> str:
    recent = state["messages"][-8:]
    return "\n".join(f"{msg.type}: {msg.content}" for msg in recent)


# 4. Tool: a small mock order database
@tool
def lookup_order(order_id: str) -> str:
    """Look up one order by its order number in a mock database."""
    orders = {
        "1042": {"status": "paid", "amount": "$49.99", "charges": 2,
                 "refund_status": "not initiated"},
        "1043": {"status": "shipped", "amount": "$25.00", "charges": 1,
                 "refund_status": "not initiated"},
    }
    order = orders.get(order_id)
    return json.dumps({"order_id": order_id, "found": order is not None, "data": order})


# 5. Nodes
def classify_ticket(state: TicketState):
    result = classifier.invoke(
        "Classify this customer's latest support message as billing, technical, or general. "
        "Also mark urgency as low, medium, or high. Use the history only for context.\n\n"
        f"History:\n{conversation(state)}\n\nLatest message: {latest_ticket(state)}"
    )
    return {"category": result.category, "urgency": result.urgency}


def billing_handler(state: TicketState):
    # extract order id 
    # Search for an order number in the newest message first, then previous messages.
    order_id = None

    for msg in reversed(state["messages"]):
        if msg.type == "human":
            match = re.search(
                r"\border\s*(?:id|number|no\.?)?\s*[:#-]?\s*(\d+)\b",
                str(msg.content),
                re.IGNORECASE
            )
            if match:
                order_id = match.group(1)
                break

    # if order id exits then extract the details from mock database 
    if order_id:
        details = lookup_order.invoke({"order_id": order_id})
    else:
        details = "No order number provided."

    reply = llm.invoke(
        "You are a polite billing support agent. Write a short, helpful reply. "
        "Use the mock order record below. Do not claim that any refund or action was "
        "completed. If information is missing(order id), ask for it.\n\n"
        f"Conversation:\n{conversation(state)}\n"
        f"Order lookup: {details}\n"
        f"Reviewer feedback to fix: {state['feedback']}"
    )
    return {"order_details": details, "draft": reply.content,
            "attempts": state["attempts"] + 1}


def technical_handler(state: TicketState):
    reply = llm.invoke(
        "You are a polite technical support agent. Write a short, practical response. "
        "Suggest basic troubleshooting steps and ask for details if necessary. "
        "Do not promise that the issue is fixed.\n\n"
        f"Conversation:\n{conversation(state)}\n"
        f"Reviewer feedback to fix: {state['feedback']}"
    )
    return {"draft": reply.content, "attempts": state["attempts"] + 1}


def general_handler(state: TicketState):
    reply = llm.invoke(
        "You are a polite customer support agent. Write a short, clear reply "
        "to the general question. Do not invent company policies or facts.\n\n"
        f"Conversation:\n{conversation(state)}\n"
        f"Reviewer feedback to fix: {state['feedback']}"
    )
    return {"draft": reply.content, "attempts": state["attempts"] + 1}


def review_draft(state: TicketState):
    result = reviewer.invoke(
        "You are a support reply reviewer. Score the draft from 1 to 10. "
        "A score of 8 or more means the draft is good enough to send. "
        "Check relevance, clarity, empathy, and accuracy. "
        "Reject invented refunds, guarantees, and unsupported claims. "
        "Give specific feedback if improvement is needed.\n\n"
        f"Customer ticket: {latest_ticket(state)}\n"
        f"Category: {state['category']}\n"
        f"Conversation:\n{conversation(state)}\n"
        f"Known order details: {state.get('order_details', '')}\n"
        f"Draft: {state['draft']}"
    )
    return {"score": result.score, "feedback": result.feedback}


def human_approval(state: TicketState):

    # For billing only. The graph pauses here until a person responds.

    decision = interrupt({"question": "Approve, edit, or reject this billing reply?",
                          "draft": state["draft"]})
    
    if decision["action"] == "reject":
        return {"approved": False, "status": "rejected"}

    
    if decision["action"] == "edit" and decision.get("reply", "").strip():
        return {"approved": True, "draft": decision["reply"], "status": "approved"}

    
    return {"approved": True, "status": "approved"}


def send_reply(state: TicketState):

    # Mock send: display the reply and save it to the conversation.
    print("\n--- SENT REPLY (SIMULATED) ---")
    print(state["draft"])
    return {"status": "sent", "messages": [AIMessage(content=state["draft"])]}


# 6. Conditional routing
def route_category(state: TicketState):
    return state["category"]


def route_review(state: TicketState):
    if state["score"] < 8 and state["attempts"] < 3:
        return state["category"]      # Retry the same handler
    if state["category"] == "billing":
        return "human_approval"
    return "send"                     # Pass, or stop retrying after 3 attempts


def route_approval(state: TicketState):
    if state['approved']:
        return "send" 
    else:
        return "end"


# 7. Build and compile the graph
def build_graph(checkpointer):
    builder = StateGraph(TicketState)
    builder.add_node("classify", classify_ticket)
    builder.add_node("billing", billing_handler)
    builder.add_node("technical", technical_handler)
    builder.add_node("general", general_handler)
    builder.add_node("review", review_draft)
    builder.add_node("human_approval", human_approval)
    builder.add_node("send", send_reply)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges("classify", route_category, {
        "billing": "billing",
        "technical": "technical",
          "general": "general"})

    
    builder.add_edge("billing", "review")
    builder.add_edge("technical", "review")
    builder.add_edge("general", "review")

    builder.add_conditional_edges("review", route_review, {
        "billing": "billing", 
        "technical": "technical",
        "general": "general",
        "human_approval": "human_approval",
        "send": "send"})
    builder.add_conditional_edges("human_approval", route_approval,
                                  {"send": "send", "end": END})
    
    builder.add_edge("send", END)

    return builder.compile(checkpointer=checkpointer)


# here we have taken the input and output in the terminal 

def main():
    print("Support Ticket Triage Agent")
    mode = input("Choose memory: 1 = InMemorySaver, 2 = SQLite [2]: ").strip()
    if mode == "1":
        checkpointer = InMemorySaver()
        conn = None
    else:
        conn = sqlite3.connect("tickets.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    app = build_graph(checkpointer)
    try:
        app.get_graph().draw_mermaid_png(output_file_path="ticket_graph.png")
        print("Graph image saved as ticket_graph.png")
    except Exception:
        print("Graph PNG could not be regenerated (Mermaid needs internet). Use the included image.")
    # for edge in app.get_graph().edges:
    #     print(
    #         edge.source,
    #         "->",
    #         edge.target,
    #         "| Conditional:",
    #         edge.conditional
    #     )

    try:
        while True:
            thread_id = input("\nCustomer thread ID (q to quit): ").strip()
            if thread_id.lower() == "q":
                break
            if not thread_id:           # prevent blank id
                continue

            config = {"configurable": {"thread_id": thread_id}}
            snapshot = app.get_state(config)

            if snapshot.next:  # A billing ticket is waiting for human approval.

                print("\nPending billing reply:\n" + snapshot.values["draft"])
                choice = input("Approve (y), edit (e), reject (n): ").strip().lower()
                if choice == "e":
                    new_reply = input("Write the edited reply: ").strip()
                    if not new_reply:
                        print("Empty edit: nothing was sent.")
                        continue
                    decision = {"action": "edit", "reply": new_reply}
                elif choice == "y":
                    decision = {"action": "approve"}
                elif choice == "n":
                    decision = {"action": "reject"}
                else:
                    print("Please enter y, e, or n.")
                    continue
                app.invoke(Command(resume=decision), config=config)
            else:
                ticket = input("Customer message: ").strip()
                if not ticket:
                    continue
                initial_state={
                    "messages": [HumanMessage(content=ticket)],
                    "category": "", "urgency": "", "order_details": "",
                    "draft": "", "feedback": "", "score": 0, "attempts": 0,
                    "approved": False, "status": "new"
                }

                result = app.invoke(initial_state, config=config)

                # if human approval is encountered then we have to re enter the id 
                if "__interrupt__" in result:
                    print("\nBILLING REPLY PAUSED: re-enter this thread ID to approve/edit.")

            saved = app.get_state(config).values
            print(f"Category: {saved.get('category')} | Urgency: {saved.get('urgency')}")
            print(f"Review score: {saved.get('score')}/10 | Attempts: {saved.get('attempts')}")
            print(f"Status: {saved.get('status')}")
            print(f"Messages in memory: {len(saved.get('messages', []))}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()

