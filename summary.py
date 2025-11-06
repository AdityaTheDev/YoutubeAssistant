import os
import gc
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from embeddings import embedding_model

from utils import *

load_dotenv()

def generate_summary(url):
    """Generates a concise and informative summary for a YouTube video."""

    if not youtube_video_exists(url):
        return " The provided YouTube URL is invalid or unavailable."

    video_id = extract_video_id(url)
    if not video_id:
        return " Could not extract a valid YouTube video ID."

    vectorstore_path = os.path.join(DATA_DIR, video_id)

    # --- Load or create vectorstore ---
    if os.path.exists(vectorstore_path):
        print(f"[Cache Hit] Loading FAISS index for summary: {video_id}")
        vector_store = FAISS.load_local(
            vectorstore_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"[Cache Miss] Creating FAISS index for summary: {video_id}")
        vector_store = youtube_transcript_to_vectorstore(url)
        vector_store.save_local(vectorstore_path)

    # --- Retrieve all chunks for full context ---
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    docs = retriever.invoke("Full summary of the video content")
    context_text = format_docs(docs)

    # --- Choose LLM for summarization ---
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

    # --- Prompt for summarization ---
    summary_prompt = PromptTemplate(
        template="""
        You are an expert summarizer. Summarize the following transcript text
        clearly, concisely, and factually. Avoid repetition. Include all key ideas
        and maintain the logical flow of the video. Don't use the word transcript in the summary.
        User should now know that transcript is used to generate the summary.

        If the video includes educational or informative content, highlight
        important takeaways and structure your summary with short paragraphs
        or bullet points for clarity.

        Transcript:
        {context}

        Summary:
        """,
        input_variables=["context"],
    )

    chain = summary_prompt | llm | StrOutputParser()

    try:
        summary = chain.invoke({"context": context_text})
    except Exception as e:
        summary = f"⚠️ An error occurred while generating summary: {e}"

    # --- Cleanup ---
    del vector_store
    gc.collect()

    return summary

