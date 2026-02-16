import streamlit as st
import os
import tempfile
import uuid  # <--- Added this
from dotenv import load_dotenv

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from src.pdf_ingestion import PDFIngestionPipeline
from src.vector_store import VectorStoreManager
from src.question_generator import QuestionGenerator
from src.answer_grader import StrictAnswerGrader

load_dotenv()

st.set_page_config(
    page_title="Lexicognition 1.0",
    page_icon="😶‍🌫️",
    layout="wide"
)

# --- Initialize Session ID First ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None

st.title("Lexicognition 1.0")
st.subheader("AI-Powered Research Paper Interviewer")
st.markdown("---")

with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    difficulty = st.selectbox(
        "Difficulty",
        options=["Basic (Noob)", "Intermediate", "Advanced"],
        index=1
    )
    
    if uploaded_file and not st.session_state.pipeline_ready:
        with st.status("Processing Document...", expanded=True) as status:
            
            # Save uploaded file to a temp path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.write("Chunking document...")
            pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)
            chunks = pipeline.ingest_pdf(tmp_path)
            
            st.write("Generating embeddings & updating Vector DB...")
            
            # --- FIX: Pass the session_id to the manager ---
            vector_manager = VectorStoreManager(
                session_id=st.session_state.session_id
            )
            
            st.session_state.retriever = vector_manager.create_vector_store(
                chunks=chunks,
                source_pdf_path=tmp_path
            )
            
            st.session_state.pipeline_ready = True
            status.update(label="Document Ready!", state="complete")
            os.remove(tmp_path) 

    if st.button("Clear Session / Start Fresh"):
        st.session_state.clear()
        st.rerun()

# --- Main Interface ---
if not st.session_state.pipeline_ready:
    st.info("👈 Please upload a technical PDF in the sidebar to begin.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Question")
        if st.button("Generate Next Question"):
            with st.spinner("Analyzing context..."):
                gen = QuestionGenerator()
                # Safe retrieval check
                if st.session_state.retriever:
                    random_docs = st.session_state.retriever.invoke("core concepts")
                    context = random_docs[0].page_content if random_docs else ""
                    
                    st.session_state.current_question = gen.generate_question(
                        context=context, 
                        difficulty=difficulty
                    )
                    st.session_state.last_feedback = None 
                else:
                    st.error("Retriever not initialized.")

        if st.session_state.current_question:
            q_data = st.session_state.current_question
            st.info(f"**Topic:** {q_data.get('topic', 'General')}\n\n**Question:** {q_data.get('question')}")
            with st.expander("Hints"):
                st.write(", ".join(q_data.get('expected_concepts', [])))

    with col2:
        st.header("Your Answer")
        user_answer = st.text_area("Type your explanation:", height=200)
        
        if st.button("Submit Answer") and user_answer:
            if st.session_state.current_question:
                with st.spinner("Grading..."):
                    grader = StrictAnswerGrader()
                    st.session_state.last_feedback = grader.grade_answer(
                        question=st.session_state.current_question.get('question'),
                        user_answer=user_answer,
                        retriever=st.session_state.retriever
                    )
            else:
                st.warning("Generate a question first.")
                
    if st.session_state.last_feedback:
        st.markdown("---")
        fb = st.session_state.last_feedback
        score = fb.get('score', 0)
        
        if score >= 7:
            st.success(f"Score: {score}/10")
        elif score >= 4:
            st.warning(f"Score: {score}/10")
        else:
            st.error(f"Score: {score}/10")
            
        st.markdown(f"**Feedback:** {fb.get('feedback')}")
        with st.expander("Detailed Reasoning"):
            st.write(fb.get('reasoning'))