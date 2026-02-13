import json
import re
import os
import logging
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrictAnswerGrader:
    """
    A multi-step grader using Gemma 3 to provide robust, cheat-resistant 
    evaluation of user answers against source PDF content.
    """
    
    def __init__(
        self,
        model_name: str = "gemma-3-27b-it", # Update to gemma3 when available
        temperature: float = 0.1  # after some "intellectual" testing
    ):
        """
        Initializes the Grader using Google's API for LLM tasks and 
        local CPU-based embeddings for similarity guardrails.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found.")
            raise ValueError("GOOGLE_API_KEY is required for StrictAnswerGrader")

        logger.info(f"Initializing StrictAnswerGrader with model: {model_name}")
        
        # 1. Initialize Google LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )

        # 2. Initialize local Embedding Model (Same as VectorStore for consistency)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "..", "persistent_storage", "model_cache")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=cache_dir,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ Grader API client and local embedding model ready.")

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity to detect question rephrasing."""
        try:
            # We use embed_query for single strings
            emb1 = np.array(self.embedding_model.embed_query(text1))
            emb2 = np.array(self.embedding_model.embed_query(text2))
            
            # Reshape for sklearn and calculate
            similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def _clean_json_response(self, content: str) -> str:
        """Helper to strip markdown blocks from API responses."""
        if "```json" in content:
            return content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()

    def _perform_novelty_analysis(self, question: str, user_answer: str, context: str) -> Dict[str, Any]:
        """Step 1: Determine if the answer provides new info or just rephrases the question."""
        logger.info("Step 1: Performing novelty and relevance analysis...")
        
        prompt = [
            SystemMessage(content="You are a strict relevance filter. Your ONLY job is to determine if the user's answer is a DIRECT response using NEW information from the context."),
            HumanMessage(content=f"""
                Evaluate the USER ANSWER based on the provided CONTEXT.
                
                **FAIRNESS RULES:**
                1. **Concept over Keywords:** If the user explains the correct mechanism using their own words, mark as TRUE.
                2. **Direct Response:** Does the answer actually address the question? (TRUE if yes).
                3. **Anti-Cheat:** Only mark as FALSE if the user is literally just repeating the question or providing common knowledge that has nothing to do with the specific context.

                **SOURCE CONTEXT:** {context}
                **QUESTION:** "{question}"
                **USER ANSWER:** "{user_answer}"

                **Output JSON:**
                {{
                    "provides_novel_information": <boolean>,
                    "reasoning": "<Briefly explain if the user caught the main idea.>"
                }}
            """)
        ]
        
        try:
            response = self.llm.invoke(prompt)
            data = json.loads(self._clean_json_response(response.content))
            return data
        except Exception as e:
            logger.error(f"Novelty analysis failed: {e}")
            return {"provides_novel_information": False, "reasoning": "Internal error in analysis."}

    def _perform_final_grading(self, question: str, user_answer: str, novelty_reasoning: str) -> Dict[str, Any]:
        """Step 2: Assign a final score based on accuracy and detail."""
        logger.info("Step 2: Synthesizing final grade...")
        
        prompt = [
            SystemMessage(content="You are a technical professor grading a deep learning exam. Output ONLY valid JSON."),
            HumanMessage(content=f"""
            Grade the student's answer using the Achievement Tiers below.
            
            **Question:** {question}
            **User Answer:** {user_answer}
            **Context Summary:** {novelty_reasoning}

            **ACHIEVEMENT TIERS:**
            - **MASTERED (Score 9-10):** The answer is technically accurate, complete, and clearly explains the 'how' or 'why'.
            - **COMPETENT (Score 6-8):** The answer is correct on the main point but might be a bit brief or miss a secondary detail.
            - **LEARNING (Score 1-5):** The user is on the right track but the explanation is vague or partially incorrect.
            - **RE-ATTEMPT (Score 0):** The answer doesn't address the question at all or is just a rephrase.

           **Output JSON Schema:**
            {{
            "score": <int>,
            "tier": "<Mastered/Competent/Learning/Re-attempt>",
            "feedback": "<Encouraging feedback for the user>",
            "reasoning": "<Explain your internal logic for why you chose this specific tier and score>",
            "is_question_repetition": <bool>
            }}
        """)  
        ]
        
        try:
            response = self.llm.invoke(prompt)
            return json.loads(self._clean_json_response(response.content))
        except Exception as e:
            logger.error(f"Final grading failed: {e}")
            return {"score": 0, "feedback": "Evaluation error.", "reasoning": str(e)}

    def grade_answer(self, question: str, user_answer: str, retriever: VectorStoreRetriever) -> Dict[str, Any]:
        """The main pipeline: Similarity Guardrail -> Context Retrieval -> Novelty Check -> Final Grade."""
        logger.info("=== Starting Answer Grading Pipeline ===")

        # 1. Similarity Guardrail
        sim = self._calculate_semantic_similarity(question, user_answer)
        if sim > 0.88:
            logger.warning(f"Cheat detected: Similarity {sim:.2f}")
            return {
                "score": 0,
                "feedback": "Your answer is too similar to the question. Please provide an explanation using your own words.",
                "is_question_repetition": True,
                "reasoning": f"Cosine similarity of {sim:.2f} exceeded the rephrase threshold."
            }

        # 2. Retrieve Context
        docs = retriever.invoke(question)
        if not docs:
            return {"score": 0, "feedback": "No context found to verify answer."}
        
        context = "\n\n".join([d.page_content for d in docs])
        evidence = [{"page": d.metadata.get("page", "N/A"), "content": d.page_content} for d in docs]

        # 3. Novelty Check
        novelty = self._perform_novelty_analysis(question, user_answer, context)
        if not novelty.get("provides_novel_information", False):
            return {
                "score": 0,
                "feedback": "Your answer does not provide enough new information from the document to satisfy the question.",
                "reasoning": novelty.get("reasoning"),
                "evidence": evidence
            }

        # 4. Final Grade
        result = self._perform_final_grading(question, user_answer, novelty.get("reasoning"))
        result["evidence"] = evidence
        
        logger.info(f"Grading complete. Final Score: {result.get('score')}/10")
        return result