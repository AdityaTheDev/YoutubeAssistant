import os
import gc
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from embeddings import embedding_model
from utils import *
load_dotenv()


def generate_answer(url, question):
    """Generates an answer for a given YouTube video and question, using cached embeddings if available."""

    if not youtube_video_exists(url):
        return " The provided YouTube URL is invalid or unavailable."

    video_id = extract_video_id(url)
    if not video_id:
        return "⚠️ Could not extract a valid YouTube video ID."

    vectorstore_path = os.path.join(DATA_DIR, video_id)

    # --- Check if cached vectorstore exists ---
    if os.path.exists(vectorstore_path):
        print(f"[Cache Hit] Loading FAISS index for video: {video_id}")
        vector_store = FAISS.load_local(
            vectorstore_path,
            embedding_model,
            allow_dangerous_deserialization=True 
    )
    else:
        print(f"[Cache Miss] Creating FAISS index for video: {video_id}")
        vector_store = youtube_transcript_to_vectorstore(url)
        vector_store.save_local(vectorstore_path)

    # --- Retrieval + Compression setup
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )

    parser = StrOutputParser()
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        For very trivial or obvious questions, answer by understanding the context. 
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel({
        "context": compression_retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    main_chain = parallel_chain | prompt | llm | parser

    try:
        response_str = main_chain.invoke(question)
    except Exception as e:
        response_str = f"An error occurred during generation: {e}"

    # --- Cleanup to prevent memory buildup ---
    del vector_store
    gc.collect()

    return response_str
