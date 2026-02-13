import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewQuestion(BaseModel):
    question: str = Field(description="The technical interview question generated from the context")
    topic: str = Field(description="The specific technical topic the question covers")
    difficulty: str = Field(description="Difficulty level: Basic, Intermediate, or Advanced")
    expected_concepts: List[str] = Field(description="List of key concepts that should be in a correct answer")

class QuestionGenerator:

    def __init__(
        self, 
        model_name: str = "gemma-3-27b-it", 
        temperature: float = 0.7
    ):

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY is required for QuestionGenerator")

        logger.info(f"Initializing QuestionGenerator with model: {model_name}")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True # Required for some Gemma/Gemini variants
        )
        
        self.parser = PydanticOutputParser(pydantic_object=InterviewQuestion)
        
        logger.info("Gemma API client initialized successfully")

    def _get_system_prompt(self) -> str:
        """Defines the persona and rules for question generation."""
        return """
        You are an expert technical interviewer specializing in Deep Learning and Computer Science.
        Your task is to generate challenging, insightful interview questions based ONLY on the provided context.

        STRICT RULES:
        1. NO ARTIFACT REFERENCES: Never mention "According to the text", "In Figure 1", "As shown in Table 2", or "The author mentions".
        2. NO META-TALK: Do not say "Based on the PDF provided...". Just ask the question directly as if you already know the material.
        3. CONCEPTUAL FOCUS: Focus on "How" and "Why" rather than simple "What" definitions.
        4. STRUCTURE: You must output your response in valid JSON format.
        """

    def generate_question(self, context: str, difficulty: str = "Intermediate") -> Optional[Dict[str, Any]]:
        logger.info(f"Generating {difficulty} question from provided context...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", (
                "CONTEXT FROM TECHNICAL DOCUMENT:\n{context}\n\n"
                "TASK: Generate one {difficulty} level interview question. "
                "Ensure the question is standalone and professional.\n\n"
                "{format_instructions}"
            ))
        ])

        input_messages = prompt.format_messages(
            context=context,
            difficulty=difficulty,
            format_instructions=self.parser.get_format_instructions()
        )

        try:
            response = self.llm.invoke(input_messages)
            
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            parsed_question = json.loads(content)
            logger.info("Successfully generated and parsed question.")
            return parsed_question

        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return None

  
if __name__ == "__main__":
    generator = QuestionGenerator()
    
    test_context = "Transformers use self-attention mechanisms to weigh the significance of different parts of input data."
    question = generator.generate_question(test_context)
    
    if question:
        print(json.dumps(question, indent=2))