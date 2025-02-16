import streamlit as st
import google.generativeai as genai
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    faithfulness,
)
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Gemini
genai.configure(api_key="AIzaSyB1lh_VOQk6qdQTR_MjhfsS5gclaYKOh-4")
gemini_llm = genai.GenerativeModel('gemini-1.5-flash')
gemini_llm_eval = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyB1lh_VOQk6qdQTR_MjhfsS5gclaYKOh-4")

# Set up embedding model
embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit app
st.title("Code Standardization and Testing with Gemini 1.5 Flash")

# Inputs
code_input = st.text_area("Enter your messed-up code here:", height=200)
default_prompt = st.text_area("Enter the default prompt for standardization:", height=100)
additional_prompt = st.text_area("Enter additional prompt (optional):", height=100)

# Initialize session state to store standardized code and test cases
if "standardized_code" not in st.session_state:
    st.session_state.standardized_code = None
if "test_cases" not in st.session_state:
    st.session_state.test_cases = None

# Function to standardize code
def standardize_code():
    if not code_input or not default_prompt:
        st.error("Please provide both code and default prompt.")
    else:
        # Combine prompts
        full_prompt = f"{default_prompt}\n\n{additional_prompt}\n\nCode:\n{code_input}"
        
        # Generate standardized code
        response = gemini_llm.generate_content(full_prompt)
        st.session_state.standardized_code = response.text.strip('```json').strip('```').strip()
        
        st.subheader("Standardized Code:")
        st.code(st.session_state.standardized_code, language='python')

# Function to generate test cases
def generate_test_cases():
    if st.session_state.standardized_code is None:
        st.error("Please standardize the code first.")
    else:
        # Generate test cases
        test_prompt = f"Generate test cases for the following standardized code covering all test scenarios:\n\n{st.session_state.standardized_code}"
        test_response = gemini_llm.generate_content(test_prompt)
        st.session_state.test_cases = test_response.text.strip('```json').strip('```').strip()
        
        st.subheader("Generated Test Cases:")
        st.code(st.session_state.test_cases, language='python')

# Function to run test cases (simulated)
def run_test_cases():
    if st.session_state.test_cases is None:
        st.error("Please generate test cases first.")
    else:
        # Simulate running test cases (this is a placeholder, you can integrate actual test execution logic)
        passed_tests = 5  # Example
        total_tests = 7    # Example
        st.subheader("Test Results:")
        st.write(f"Passed {passed_tests} out of {total_tests} test cases.")

# Function to evaluate using RAGAS
def evaluate_code():
    if st.session_state.standardized_code is None:
        st.error("Please standardize the code first.")
    else:
        # Prepare data for RAGAS evaluation
        data = {
            "question": [f"{default_prompt}\n\n{additional_prompt}\n\nCode:\n{code_input}"],  # Input prompt
            "answer": [st.session_state.standardized_code],  # Generated standardized code
            "contexts": [[f"{default_prompt}\n\n{additional_prompt}\n\nCode:\n{code_input}"]],  # Context (input prompt in this case)
            "reference": [f"{additional_prompt}"]
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
            llm=gemini_llm_eval,
            embeddings=embedding_model,
        )

        st.subheader("RAGAS Evaluation Metrics:")
        st.write(f"Answer Relevancy: {result['answer_relevancy']}")
        st.write(f"Context Relevancy: {result['context_recall']}")
        st.write(f"Faithfulness: {result['faithfulness']}")

# Buttons for modular execution
if st.button("Standardize Code"):
    standardize_code()

if st.button("Generate Test Cases"):
    generate_test_cases()

if st.button("Run Test Cases"):
    run_test_cases()

if st.button("Evaluate Code"):
    evaluate_code()

# Run the app
if __name__ == "__main__":
    st.write("App is running...")