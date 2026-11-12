# Support Ticket Triage Agent

A mini project built using **Python, LangGraph and Groq**. The agent handles customer support tickets, remembers earlier messages in the same customer thread, and requires human approval before a billing reply is sent.

## Scenario

Customers may send messages such as:

- **Billing:** `I was charged twice for order #1042.`
- **Technical:** `The app crashes on settings.`
- **General:** `How can I contact customer support?`

## Workflow

1. **Classify:** An LLM classifies the latest customer ticket as `billing`, `technical`, or `general`, and assigns urgency (`low`, `medium`, or `high`).
2. **Route:** Conditional edges send the ticket to the appropriate handler.
3. **Draft:** The handler generates a reply. The billing handler calls a `lookup_order` tool using mock order data (orders `#1042` and `#1043`).
4. **Review:** An LLM reviewer scores the draft out of 10 and provides feedback. If the score is below **8**, the graph loops back to the same handler. The workflow allows **at most 3 drafts total**. After the third draft, it continues even if the score is below 8.
5. **Human approval:** Billing tickets pause using LangGraph `interrupt()`. A person can **approve, edit, or reject** the draft. Rejected drafts are not sent.
6. **Send:** The final reply is printed in the terminal and added to conversation history. **Sending is simulated**; no actual email or customer message is delivered.

## Conversation Persistence

The graph uses `add_messages` and a checkpointer, with `config = {"configurable": {"thread_id": "customer-1"}}` to associate messages with a customer thread.

- **InMemorySaver:** Stores conversation checkpoints during the current Python process; memory is lost after the program exits.
- **SQLite (`SqliteSaver`):** Saves checkpoints to `tickets.db`, allowing a customer to return on the **same thread ID**, including after restarting the program.

The two checkpoint options are demonstrated separately; they are not simultaneously applied to one graph execution. Both implement **thread-level conversation memory**, rather than cross-thread long-term memory.

## LLM Used

- **Provider:** [Groq](https://console.groq.com/)
- **Model:** `openai/gpt-oss-20b` (open-weight)
- **Integration:** `langchain-groq`
- **API key:** Create one at [Groq API Keys](https://console.groq.com/keys). Groq's free tier is subject to rate limits and availability.

## Project Files

```text
support-ticket-triage-agent/
├── MINI_PROJECT.py          # Complete compiled LangGraph agent and terminal runner
├── README.md                # Project explanation and run instructions
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Excludes secrets, local database and virtual environment
├── ticket_graph.png         # Rendered state graph
```

The graph is built with `StateGraph(TicketState)` and compiled with a selected checkpointer in `ticket_agent.py`.

## How to Run (Windows / VS Code)

1. Install Python and open the extracted project folder in VS Code.
2. Create a Groq API key at https://console.groq.com/keys.
3. Copy `.env.example` to `.env` and enter your key:

   ```env
   GROQ_API_KEY=your_actual_groq_key_here
   ```

4. Open a PowerShell terminal in the project folder and run:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe ticket_agent.py
   ```

   This uses the virtual environment without requiring PowerShell script activation.

5. Choose `1` for **InMemorySaver** or `2` for **SQLite** (default).
6. Enter a customer thread ID and message. **Reuse that ID** for follow-up messages.
7. For billing tickets, the program pauses. Enter the **same thread ID** again and choose `y` to approve, `e` to edit, or `n` to reject.

The program attempts to save `ticket_graph.png`. If online Mermaid rendering fails, the ZIP already includes a graph image.

## Test Scenarios

| Scenario | Thread ID | Example message | Expected workflow |
|---|---|---|---|
| Billing | `billing-1` | `I was charged twice for order #1042` | Classify → Billing + order tool → Review → Human approval → Simulated send (if approved) |
| Technical | `technical-1` | `The app crashes on settings` | Classify → Technical → Review → Simulated send |
| General | `general-1` | `How can I contact customer support?` | Classify → General → Review → Simulated send |

For any case, the reviewer may return the draft to its handler if the score is below 8 and fewer than 3 drafts have been made.

**Memory test:** Complete the billing approval for `billing-1`, then enter `billing-1` again and ask `Can you give me an update on that order?`. With SQLite, restart the program and use `billing-1` again to confirm persistence.

**Note:** Real classifications, review scores, replies and iteration counts depend on the live LLM. The included sample screenshots are illustrative; capture **actual run screenshots** for submission.

##  One Thing I Found Tricky

The tricky part was implementing **human approval together with conversation persistence**. `interrupt()` pauses a billing ticket, and `Command(resume=decision)` must be invoked using the **same `thread_id`** so LangGraph can resume the correct saved execution. SQLite also preserves the interrupted state across a program restart.

Additionally, to avoid writing the same graph code twice for InMemorySaver and SqliteSaver, I created a build_graph(checkpointer) function. This function handles graph construction and compilation using the selected checkpointer. In the main() function, I provided the option to choose between the two memory systems. The selected checkpointer is then passed to build_graph(), allowing the same workflow to run with either memory system without duplicating the nodes and edges.

## Deliverables and Submission

1. **Python code:** `ticket_agent.py` contains the compiled LangGraph workflow.
2. **Rendered graph:** `ticket_graph.png` shows the nodes, conditional routing, review loop and approval path.
3. **README:** This file includes run instructions, LLM details and the tricky part.
4. **GitHub:** Push the code and supporting files to a public GitHub repository and share its URL. **Never upload `.env`, your API key, or `tickets.db`.**
5. **Google Drive:** Run the program, capture screenshots for **billing, technical and general** scenarios, and upload them along with the rendered graph screenshot. In Drive sharing, choose **General access → Anyone with the link → Viewer**, then share the folder URL.

## Reference

- [CampusX LangGraph tutorials](https://github.com/campusx-official/langgraph-tutorials) — the project follows the basic `TypedDict`, `StateGraph`, nodes, conditional edges, `compile()` and `invoke()` teaching style.
