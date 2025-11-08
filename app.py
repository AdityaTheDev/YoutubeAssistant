import streamlit as st
from utils import youtube_video_exists
from youtube_rag import generate_answer
from summary import generate_summary
import traceback

# -----------------------
# Streamlit Page Settings
# -----------------------
st.set_page_config(
    page_title="🎥 YouTube Video Assistant",
    page_icon="🎬",
    layout="wide",
)

# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
    <style>
    .main { background-color: #f9fafc; }
    .stButton>button {
        background-color: #2E86DE;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTextInput>div>div>input, .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2E86DE;
        margin-top: 10px;
    }
    .summary-box {
        background-color: #fff7e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #f39c12;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.title("🎬 YouTube Video Assistant")
st.write("Ask questions or get detailed summaries from YouTube videos in **English, Hindi, Tamil, Kannada, Malayalam, Telugu, Bengali, Marathi, "
    "Gujarati, Urdu, Punjabi, Nepali, Sinhalese, Korean and Japanese.**")

# -----------------------
# Tabs for Q&A and Summary
# -----------------------
tab1, tab2 = st.tabs(["💬 Ask Questions", "🧾 Get Summary"])

# -----------------------
# Q&A Tab
# -----------------------
with tab1:
    st.subheader("💬 Ask a Question from a YouTube Video")

    youtube_url = st.text_input("🔗 Enter YouTube URL:", key="qa_url", placeholder="e.g. https://youtu.be/0hDuL9ifaoE")
    question = st.text_area("❓ Your Question:", key="qa_question", placeholder="e.g. What are the health benefits mentioned in this video?", height=100)

    if st.button("Get Answer", key="qa_button"):
        if not youtube_url.strip():
            st.warning("⚠️ Please enter a YouTube URL.")
        elif not question.strip():
            st.warning("⚠️ Please enter a question.")
        elif not youtube_video_exists(youtube_url):
            st.error(" Invalid or unavailable YouTube video. Please check the link.")
        else:
            with st.spinner("⏳ Processing and generating your answer..."):
                try:
                    answer = generate_answer(youtube_url, question)
                    if answer:
                        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                        st.markdown(f"### 🧠 Answer\n\n{answer}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info(" Couldn't generate an answer from the video given.")
                except Exception as e:
                    st.error("🚨 An unexpected error occurred while processing the video.")
                    st.exception(traceback.format_exc())

# -----------------------
# Summary Tab
# -----------------------
with tab2:
    st.subheader("🧾 Generate a Summary of a YouTube Video")

    youtube_url_summary = st.text_input("🔗 Enter YouTube URL:", key="summary_url", placeholder="e.g. https://youtu.be/FPRk7ZNEqpI")

    if st.button("Generate Summary", key="summary_button"):
        if not youtube_url_summary.strip():
            st.warning("⚠️ Please enter a YouTube URL.")
        elif not youtube_video_exists(youtube_url_summary):
            st.error("❌ Invalid or unavailable YouTube video. Please check the link.")
        else:
            with st.spinner("⏳ Generating summary..."):
                try:
                    summary = generate_summary(youtube_url_summary)
                    if summary:
                        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                        st.markdown(f"### 🧾 Summary\n\n{summary}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("😕 Couldn't generate a summary from the video.")
                except Exception as e:
                    st.error(" An unexpected error occurred while summarizing the video.")
                    st.exception(traceback.format_exc())

# -----------------------
# Sidebar Info
# -----------------------
with st.sidebar:
    st.header("💡 What You Can Do")
    st.markdown("""
    ### 💡 Top Use Cases
- Quickly understand long interviews or lectures  
- Extract key insights from educational videos  
- Turn health and nutrition videos into quick Q&As  
- Learn faster from motivational or self-improvement talks  
- Use it as a smart study assistant for any subject  

---

### 🧠 Pro Tips
- Ask focused, short questions for the best results   
- Try questions like *“Summarize in 5 bullet points”*  
- Use this app to revise lecture content quickly  

---

### 😄 Fun Uses
- Ask your favorite YouTuber’s video: *“What’s their main message?”*  
- Summarize stand-up comedy or debates  
- Get review highlights from tech and movie channels  
- Learn from documentaries in your preferred language  

---

### ❤️ Why People Love It
- Saves hours of manual watching  
- Works seamlessly across multiple languages  
- Gives clear, natural answers  
- Helps you learn smarter, not harder 
    """)

st.markdown("""
---
<div style="text-align: center; font-size: 16px; margin-top: 20px;">
    <p>👨‍💻 Created with ❤️ by <strong>Aditya</strong></p>
    <p>
        <a href="https://www.linkedin.com/in/aditya26" target="_blank" style="text-decoration:none;">
            🔗 Connect on LinkedIn
        </a>
    </p>
</div>
""", unsafe_allow_html=True)