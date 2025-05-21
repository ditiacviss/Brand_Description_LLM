import os
from langchain_groq import ChatGroq
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
import streamlit as st


def get_llm(API_KEY):
    return ChatGroq(
        groq_api_key=API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.2,
        max_tokens=1024
    )



def summarize_url(url: str,llm) -> str:
    """Fetches and summarizes the actual content of a given brand website URL."""

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract visible text
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        text = " ".join(soup.stripped_strings)

        # Keep only first 1000 words for summarization
        content = " ".join(text.split()[:1000])

    except Exception as e:
        return f"Failed to fetch content from URL: {e}"

    prompt = f"""
    You are a website summarization expert.
    Summarize the following website content in 1–2 lines. Keep it concise and clear.
    
    Website content:
    \"\"\"{content}\"\"\"
    """
    return llm.invoke(prompt).content.strip()



def score_similarity(combined_input: str, llm) -> str:
    """
    Scores how closely a brand description matches the website summary.
    Input should be in the format: <<<description>>> ||| <<<summary>>>
    """
    try:
        description, summary = combined_input.split("|||")
    except ValueError:
        return "Invalid input format. Use: <<<description>>> ||| <<<summary>>>"

    prompt = f"""
    You are a brand evaluator. Compare the following two pieces of text:
    
    Brand Description:
    \"\"\"{description.strip()}\"\"\"
    
    Website Summary:
    \"\"\"{summary.strip()}\"\"\"
    
    On a scale of 0 to 100, give a matching score that reflects how well the website matches the description.
    Also provide 1-line reasoning.
    
    Format:
    Score: XX
    Reason: <reason>
    """
    return llm.invoke(prompt).content.strip()


def json_formatter(raw_output: str) -> dict:
    """
    Formats output like:
    Score: XX
    Reason: <reason>
    into JSON: {"score": XX, "reason": "<reason>"}
    """
    try:
        lines = raw_output.strip().splitlines()
        score_line = next(line for line in lines if "Score:" in line)
        reason_line = next(line for line in lines if "Reason:" in line)

        score = int(score_line.replace("Score:", "").strip())
        reason = reason_line.replace("Reason:", "").strip()

        return {"score": score, "reason": reason}
    except Exception as e:
        return {"error": f"Failed to format output: {e}"}


def run_brand_match_agent(description: str, url: str, api_key: str):
    llm = get_llm(api_key)
    summary = summarize_url(url, llm)
    combined_output = score_similarity(description, summary, llm)
    return combined_output, summary


st.title("Brand Description vs Website Match")

url = st.text_input("Enter Brand Website URL")
description = st.text_area("Enter Brand Description")
API_KEY = st.text_input("Enter GROQ API Key", type="password")

if st.button("Enter") and url and description and API_KEY:
    output, summary = run_brand_match_agent(description, url, API_KEY)
    st.markdown("🔍 Website Summary")
    st.write(summary)

    st.markdown("📊 Match Result")
    st.code(output)

    formatted = json_formatter(output)
    st.markdown("✅ Formatted Result")
    st.json(formatted)