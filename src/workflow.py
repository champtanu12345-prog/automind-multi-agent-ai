import os
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# 1. Create the AI model

def get_model():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    return ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        ),
        temperature=0.2
    )


# 2. Define shared information
# Every agent can read the information
# produced by the previous agents.

class AgentState(TypedDict, total=False):
    topic: str
    research: str
    analysis: str
    review: str
    final_report: str


# 3. RESEARCH AGENT

def research_agent(state: AgentState):

    llm = get_model()

    prompt = f"""
    You are the Research Agent.

    Research topic: {state['topic']}

    Prepare a structured research brief.

    Include:
    1. Introduction
    2. Background
    3. Important concepts
    4. Key findings
    5. Potential applications

    Do not invent sources or citations.
    Clearly distinguish established knowledge
    from information that needs verification.
    """

    response = llm.invoke(prompt)

    print("\nRESEARCH AGENT COMPLETED")

    return {
        "research": response.content
    }


# 4. ANALYSIS AGENT

def analysis_agent(state: AgentState):

    llm = get_model()

    prompt = f"""
    You are the Analysis Agent.

    Analyse the following research:

    {state['research']}

    Provide:
    1. Main observations
    2. Advantages
    3. Disadvantages
    4. Opportunities
    5. Risks
    6. Practical applications

    Give logical and clear explanations.
    Do not invent numerical statistics.
    """

    response = llm.invoke(prompt)

    print("\nANALYSIS AGENT COMPLETED")

    return {
        "analysis": response.content
    }


# 5. FACT-CHECKING AGENT

def fact_checker_agent(state: AgentState):

    llm = get_model()

    prompt = f"""
    You are the Fact-Checking Agent.

    Review this research:
    {state['research']}

    Review this analysis:
    {state['analysis']}

    Your responsibilities:

    1. Identify unsupported claims.
    2. Find possible contradictions.
    3. Highlight statements requiring sources.
    4. Identify uncertain information.
    5. Suggest corrections where necessary.

    Important:
    You do not have an external verification
    database in this version.

    Never claim that a statement is independently
    verified unless evidence is provided.
    """

    response = llm.invoke(prompt)

    print("\nFACT-CHECKING AGENT COMPLETED")

    return {
        "review": response.content
    }


# 6. REPORT WRITER AGENT

def report_writer_agent(state: AgentState):

    llm = get_model()

    prompt = f"""
    You are the Professional Report Writer Agent.

    Original topic:
    {state['topic']}

    Research:
    {state['research']}

    Analysis:
    {state['analysis']}

    Fact-checking review:
    {state['review']}

    Create a professional report.

    Include:
    1. Title
    2. Executive Summary
    3. Introduction
    4. Research Findings
    5. Analysis
    6. Risks and Limitations
    7. Conclusion
    8. Information Requiring Verification

    Use professional language.
    Do not invent references.
    Clearly mention limitations.
    """

    response = llm.invoke(prompt)

    print("\nREPORT WRITER AGENT COMPLETED")

    return {
        "final_report": response.content
    }


# 7. CONNECT ALL FOUR AGENTS

def build_workflow():

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "researcher",
        research_agent
    )

    workflow.add_node(
        "analyst",
        analysis_agent
    )

    workflow.add_node(
        "fact_checker",
        fact_checker_agent
    )

    workflow.add_node(
        "report_writer",
        report_writer_agent
    )

    workflow.add_edge(START, "researcher")

    workflow.add_edge(
        "researcher",
        "analyst"
    )

    workflow.add_edge(
        "analyst",
        "fact_checker"
    )

    workflow.add_edge(
        "fact_checker",
        "report_writer"
    )

    workflow.add_edge(
        "report_writer",
        END
    )

    return workflow.compile()


# 8. RUN THE COMPLETE SYSTEM

def run_automind(topic: str):

    if not topic.strip():
        raise ValueError("Please enter a topic.")

    app = build_workflow()

    result = app.invoke({
        "topic": topic
    })

    return result

