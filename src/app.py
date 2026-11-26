import streamlit as st

from src.workflow import run_automind


st.set_page_config(
    page_title="AutoMind AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoMind")
st.subheader(
    "Intelligent Multi-Agent AI Automation System"
)

st.write(
    "Four specialized AI agents collaborate "
    "to research, analyse, review and generate reports."
)

st.divider()

topic = st.text_area(
    "Enter your research topic",
    placeholder="Example: Impact of AI in healthcare"
)

if st.button("Start AI Automation", type="primary"):

    if not topic.strip():

        st.warning("Please enter a topic first.")

    else:

        try:

            with st.spinner(
                "AI agents are working together..."
            ):

                result = run_automind(topic)

            st.success("Your report is ready!")

            with st.expander("1. Research Agent"):
                st.markdown(result["research"])

            with st.expander("2. Analysis Agent"):
                st.markdown(result["analysis"])

            with st.expander("3. Fact-Checking Agent"):
                st.markdown(result["review"])

            st.subheader("4. Final AI Report")

            st.markdown(result["final_report"])

            st.download_button(
                label="Download Final Report",
                data=result["final_report"],
                file_name="automind_report.md",
                mime="text/markdown"
            )

        except Exception as e:

            st.error(f"An error occurred: {e}")

st.divider()

st.caption(
    "AutoMind | Built using Python, "
    "LangGraph, LangChain and Groq"
)

