import streamlit as st
import os
import tempfile
from dotenv import load_dotenv


from src.pdf_ingestion import PDFIngestionPipeline
from src.vector_store import VectorStoreManager
from src.question_generator import QuestionGenerator
from src.answer_grader import StrictAnswerGrader


load_dotenv()

st.set_page_config(
    page_title="Lexicognition 1.0 (better and faster)",
    page_icon="😶‍🌫️",
    layout="wide"
)


if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())


st.title(" Lexicognition 1.0")
st.subheader("AI-Powered Research paper Interviewer")
st.markdown("---")


with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    difficulty = st.selectbox(
        "Difficulty",
        options=["Basic(noob)", "Intermediate", "Advanced"],
        index=1
    )
    
    if uploaded_file and not st.session_state.pipeline_ready:
        with st.status("Processing Document...", expanded=True) as status:
           
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
          
            st.write("Chunking document...")
            pipeline = PDFIngestionPipeline(chunk_size=1000, chunk_overlap=200)
            chunks = pipeline.ingest_pdf(tmp_path)
            
           
            st.write("Generating embeddings & updating Vector DB..")
           
            vector_manager = VectorStoreManager(
                session_id=st.session_state.session_id
            )
            st.session_state.retriever = vector_manager.create_vector_store(
                chunks=chunks,
                source_pdf_path=tmp_path
            )
            
            st.session_state.pipeline_ready = True
            status.update(label="Document Ready!", state="complete")
            os.remove(tmp_path) # Clean up temp file

    if st.button("Clear Session / Start Fresh"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- Main Interview Interface ---
if not st.session_state.pipeline_ready:
    st.info("👈 Please upload a technical PDF in the sidebar to begin the interview.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Question")
        if st.button("Generate Next Question"):
            with st.spinner("Evaluation in progress..."):
               
                gen = QuestionGenerator()
                
                random_docs = st.session_state.retriever.invoke("core concepts")
                context = random_docs[0].page_content if random_docs else ""
                
                st.session_state.current_question = gen.generate_question(
                    context=context, 
                    difficulty=difficulty
                )
                st.session_state.last_feedback = None 

        if st.session_state.current_question:
            q_data = st.session_state.current_question
            st.info(f"**Topic:** {q_data.get('topic', 'General')}\n\n**Question:** {q_data.get('question')}")
            
            with st.expander("Hints(not that helpful tho):"):
                st.write(", ".join(q_data.get('expected_concepts', [])))

    with col2:
        st.header("Your Answer")
        user_answer = st.text_area("Type your explanation here:", height=200)
        
        if st.button("Submit Answer") and user_answer:
            if st.session_state.current_question:
                with st.spinner("Evaluating against source material..."):
                    grader = StrictAnswerGrader()
                    st.session_state.last_feedback = grader.grade_answer(
                        question=st.session_state.current_question.get('question'),
                        user_answer=user_answer,
                        retriever=st.session_state.retriever
                    )
            else:
                st.error("Please generate a question first dumbo!")
    if st.session_state.last_feedback:
        st.markdown("---")
        st.header("Evaluation Results")
        fb = st.session_state.last_feedback
        
        score = fb.get('score', 0)
        if score >= 7:
            st.success(f"Score: {score}/10")
        elif score >= 4:
            st.warning(f"Score: {score}/10")
        else:
            st.error(f"Score: {score}/10")
            
        st.markdown(f"**Feedback:** {fb.get('feedback')}")
        
        with st.expander("View Grading Reasoning (Detailed)"):
            st.write(fb.get('reasoning'))
            if fb.get('is_question_repetition'):
                st.error("⚠️ System detected your answer was too similar to the question.")
        
        if "evidence" in fb:
            with st.expander("View Source Evidence (Citations)"):
                for i, doc in enumerate(fb['evidence']):
                    st.markdown(f"**Source {i+1} (Page {doc['page']}):**")
                    st.caption(doc['content'])