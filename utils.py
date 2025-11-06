import os
import re
import requests
import gc
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from embeddings import embedding_model

load_dotenv()

DATA_DIR = "data/faiss"
os.makedirs(DATA_DIR, exist_ok=True)



def youtube_video_exists(url: str) -> bool:
    api = "https://www.youtube.com/oembed"
    params = {"url": url, "format": "json"}

    try:
        response = requests.get(api, params=params, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def extract_video_id(url: str) -> str:
    """Extracts the YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  #regular YouTube links
        r"youtu\.be\/([0-9A-Za-z_-]{11})",  #shortened links
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None




def youtube_transcript_to_vectorstore(url: str):
    """Loads, translates, splits, and embeds a YouTube transcript."""

    video_id = extract_video_id(url)

    try:
        loader = YoutubeLoader.from_youtube_url(url, language=["en", "hi", "ta", "kn", "ml", "te", "bn", "mr", "gu", "ur", "pa", "ne", "si", "ko", "ja", "zh-Hans"])
        docs = loader.load()

    except (NoTranscriptFound, TranscriptsDisabled) as e:
        print(f"No transcript available for this video ({video_id}). Error: {e}")
        return None

    except Exception as e:
        print(f"Unexpected error while loading transcript: {e}")
        return None
    # loader = YoutubeLoader.from_youtube_url(url, language=["en", "hi", "ta", "kn"])
    # docs = loader.load()

    transcription = docs[0].page_content

    translator = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    translate_template = PromptTemplate(
        template="Translate the following text to English with only necessary details:\n\n{transcription}",
        input_variables=["transcription"],
    )
    parser = StrOutputParser()
    chain = translate_template | translator | parser
    translated_text = chain.invoke({"transcription": transcription})

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.create_documents([translated_text])

    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)
