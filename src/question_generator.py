import os
import json
import logging
import streamlit as st
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.utils import LLMLoadBalancer

# Load environment variables
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
        self.model_name = model_name
        self.temperature = temperature
        self.parser = PydanticOutputParser(pydantic_object=InterviewQuestion)
        
        # Initialize the Load Balancer
        self.balancer = LLMLoadBalancer()
        logger.info(f"QuestionGenerator initialized with model: {model_name}")

    def _get_system_prompt(self) -> str:
        """Defines the persona and rules for question generation."""
        return """
        You are an expert technical interviewer specializing in Deep Learning and Computer Science.
        Your task is to generate challenging, insightful interview questions based ONLY on the provided context.

        STRICT RULES:
        1. NO ARTIFACT REFERENCES: Never mention "According to the text", "In Figure 1", etc.
        2. NO META-TALK: Do not say "Based on the PDF...". Ask the question directly.
        3. CONCEPTUAL FOCUS: Focus on "How" and "Why" rather than simple definitions.
        4. STRUCTURE: You must output your response in valid JSON format.
        5. NO REPITITION: DO NOT reppeat questions.Generate new questions,each time.
        """

    def generate_question(self, context: str, difficulty: str = "Intermediate") -> Optional[Dict[str, Any]]:
        logger.info(f"Requesting key from balancer for {difficulty} question generation...")
        
        # 1. Dynamically fetch an available key for this specific request
        api_key = self.balancer.get_next_available_key()
        
        # 2. Instantiate the LLM with the selected key
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=self.temperature,
            convert_system_message_to_human=True
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", (
                "CONTEXT FROM TECHNICAL DOCUMENT:\n{context}\n\n"
                "TASK: Generate one {difficulty} level interview question different from previous questions if any. "
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
            response = llm.invoke(input_messages)
            content = response.content
            
            # Extract JSON if wrapped in markdown blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed_question = json.loads(content)
            logger.info("Successfully generated and parsed question.")
            return parsed_question

        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return None

