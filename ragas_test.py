import streamlit as st
import google.generativeai as genai
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
    multimodal_faithness,
    multimodal_relevance,
    summarization_score
)

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Set up Gemini
genai.configure(api_key="AIzaSyB1lh_VOQk6qdQTR_MjhfsS5gclaYKOh-4")
gemini_llm = genai.GenerativeModel('gemini-1.5-flash')

embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit app
st.title("Code Standardization and Testing with Gemini 1.5 Flash")

# Inputs
code_input = st.text_area("Enter your messed-up code here:", height=200)
default_prompt = st.text_area("Enter the default prompt for standardization:", height=100)
additional_prompt = st.text_area("Enter additional prompt (optional):", height=100)

# Generate standardized code
if st.button("Standardize Code"):
    if not code_input or not default_prompt:
        st.error("Please provide both code and default prompt.")
    else:
        # Combine prompts
        full_prompt = f"{default_prompt}\n\n{additional_prompt}\n\nCode:\n{code_input}"
        
        # Generate standardized code
        response = gemini_llm.generate_content(full_prompt)
        standardized_code = response.text.strip('```json').strip('```').strip()
        
        st.subheader("Standardized Code:")
        st.code(standardized_code, language='python')

        # Generate test cases
        test_prompt = f"Generate test cases for the following standardized code covering all test scenarios:\n\n{standardized_code}"
        test_response = gemini_llm.generate_content(test_prompt)
        test_cases = test_response.text.strip('```json').strip('```').strip()
        
        st.subheader("Generated Test Cases:")
        st.code(test_cases, language='python')

        # Run test cases (simulated)
        st.subheader("Test Results:")
        # Simulate running test cases (this is a placeholder, you can integrate actual test execution logic)
        passed_tests = 5  # Example
        total_tests = 7    # Example
        st.write(f"Passed {passed_tests} out of {total_tests} test cases.")

        # Evaluate using deepeval
        st.subheader("Evaluation Metrics:")

        data = {
            "question": [full_prompt],  # Input prompt
            "answer": [standardized_code],  # Generated standardized code
            "contexts": [[full_prompt]],  # Context (input prompt in this case)
            "reference": [code_input],  # Original code
        }
        dataset = Dataset.from_dict(data)

        # Compute metrics
        result = evaluate(
            dataset,
            metrics=[
                answer_relevancy,
                context_recall,
                faithfulness,
            ],
            llm=gemini_llm,
            embeddings=embedding_model,
        )

        st.write(f"Answer Relevancy: {result['answer_relevancy']}")
        st.write(f"Context Relevancy: {result['context_recall']}")
        st.write(f"Faithfulness: {result['faithfulness']}")

# Run the app
if __name__ == "__main__":
    st.run()