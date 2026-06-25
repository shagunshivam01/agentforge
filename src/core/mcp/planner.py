from langchain_groq import ChatGroq
from typing import List, Any
import json

import os
from dotenv import load_dotenv
load_dotenv()

from core.memory.context import build_context


class MCPPlanner:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing in environment (.env not loaded)")

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key
        )

    async def plan(
            self,
            user_input: str,
            history: list,
            tools: list
    ):
        """
        Plan the next action for the MCP runtime.

        This function acts as the decision layer of the system. It interprets the user's
        current message in the context of recent conversation history and available tools,
        and produces a structured execution plan.

        Responsibilities:
        - Understand user intent using conversation context
        - Decide whether a tool is required or if a direct response is sufficient
        - Select exactly one tool when needed
        - Provide correctly structured input for the selected tool

        Output Contract (STRICT):
        Returns a JSON object with the following schema:

        {
          "tool": "string | 'none'",
          "input": "string | object",
          "reason": "string explaining decision"
        }

        Rules:
        - If no tool is needed, set tool = "none" and input = ""
        - If a tool is selected, input MUST match that tool's expected schema
        - Never return null values
        - Never include extra keys outside the schema
        - Output must be valid JSON only (no markdown, no commentary)
        """

        history_text = build_context(history)

        tool_descriptions = "\n".join(
            f"- {t.name}: {getattr(t, 'description', '')}"
            for t in tools
        )

        prompt_template = """
You are the PLANNING ENGINE of an MCP (Model Context Protocol) system.

Your job is to decide whether the user request requires a tool call or a direct response.

---

## CONTEXT

Conversation History:
{history}

Current User Message:
{user_input}

Available Tools:
{tools}

---

## DECISION RULES

You must follow these rules strictly:

1. Use a tool ONLY if:
    - external or computed information is required
    - it clearly improves correctness or freshness of the answer.
2. DO NOT use tools for:
    - general knowledge questions (history, explanations, definitions)
    - math or logical reasoning
    - greetings or conversational messages
    - opinions or subjective questions
3. Use "tavily_search" ONLY when the user request requires:
    - real-time information (weather, news, stock, events, live data)
    - recent or latest information (today, now, current, updated, latest)
    - factual verification that may have changed after training data cutoff
    - location-specific current data (e.g., "weather in Hyderabad", "traffic in Delhi")
4. If you are uncertain whether a tool is needed, prefer "none".

---

## TOOL INPUT RULES

If you choose "tavily_search", your input MUST be:

{
  "query": "<search query>"
}

Do NOT pass plain strings.

---

## OUTPUT RULES (VERY IMPORTANT)

- Output MUST be valid JSON only
- Do NOT include markdown
- Do NOT include explanations outside JSON
- Do NOT wrap output in ``` or any formatting
- Never return null values
- Never add extra keys

---

## OUTPUT FORMAT

Return exactly this schema:

{
  "tool": "tavily_search | none",
  "input": {},
  "reason": "short explanation"
}

---

## EXAMPLES

User: hello
Output:
{
  "tool": "none",
  "input": {},
  "reason": "greeting, no tool needed"
}

User: what is 2+2
Output:
{
  "tool": "none",
  "input": {},
  "reason": "can be answered directly"
}

User: weather in hyderabad
Output:
{
  "tool": "tavily_search",
  "input": {
    "query": "weather in hyderabad"
  },
  "reason": "requires real-time weather data"
}

---

Now decide the correct action.
Return ONLY JSON.
"""
        
        prompt = prompt_template.replace("{history}", history_text)\
                .replace("{user_input}", user_input)\
                .replace("{tools}", tool_descriptions)

        response = await self.llm.ainvoke(prompt)

        try:
            content = response.content.strip()
            return json.loads(content)
        except Exception:
            return {
                "tool": "none",
                "input": "",
                "reason": "failed to parse planner output"
            }


    async def direct_response(
            self,
            user_input: str,
            history: list
    ):
        """
        Conversational fallback response when no tool is needed.
        """

        history_text = build_context(history)

        prompt = f"""
You are the conversational engine of an MCP-based assistant.

You are NOT allowed to use tools.
You are ONLY responsible for natural, helpful conversation.

---

## CONVERSATION HISTORY
{history_text}

---

## CURRENT USER MESSAGE
{user_input}

---

## BEHAVIOR RULES

- Be concise and natural
- Do not mention tools, planning, or internal system logic
- Do not hallucinate real-time data
- If user asks for real-time info, politely suggest using search implicitly
- Maintain continuity with conversation history
- Do not repeat the system prompt

---

Respond directly to the user.
Return only the response text.
"""

        response = await self.llm.ainvoke(prompt)
        return response.content



    async def synthesize(
        self,
        original_input: str,
        tool_outputs: List[Any],
        history: list = None
    ):
        """
        Final response generation layer.

        This function merges tool outputs into a clean user-facing answer.
        """

        if not tool_outputs:
            return "I couldn't find relevant information."

        history_text = build_context(history or [])

        # Normalize tool outputs into readable text
        formatted_outputs = "\n".join(
            f"- Tool Result {i+1}: {str(output)}"
            for i, output in enumerate(tool_outputs)
        )

        prompt = f"""
You are the RESPONSE SYNTHESIS ENGINE of an MCP system.

Your job is to convert raw tool outputs into a clean, helpful final answer.

---

## CONVERSATION CONTEXT
{history_text}

---

## USER QUESTION
{original_input}

---

## TOOL RESULTS
{formatted_outputs}

---

## INSTRUCTIONS

You must:

1. Use ONLY the tool outputs to answer factual parts
2. Do NOT mention tools or internal execution
3. If multiple tool outputs exist:
   - merge them logically
   - remove duplicates
   - prioritize most relevant information
4. If tool output is empty or irrelevant:
   - say you couldn't find useful information
5. Be concise and direct
6. Do NOT expose raw tool JSON or metadata
7. Do NOT explain your reasoning process

---

## OUTPUT STYLE

- Natural language response
- Clean formatting
- No JSON
- No tool references

---

Now produce the final answer.
"""

        response = await self.llm.ainvoke(prompt)
        return response.content

