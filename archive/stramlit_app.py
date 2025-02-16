import streamlit as st
import google.generativeai as genai
import subprocess
import sys
from sklearn.metrics.pairwise import cosine_similarity
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

embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

dataset_dict = {
    "question":["What is the capital of France"],
    "answer":["Paris is China"],
    "contexts":[["Paris is the capital and most populous city of France"]],
    "reference":["Paris"],
    "reference_contexts":[["Paris"]]
}

dataset = Dataset.from_dict(dataset_dict)

metrics_all = [answer_relevancy, answer_correctness, answer_similarity, context_entity_recall, context_precision, context_recall, faithfulness, multimodal_faithness, multimodal_relevance, summarization_score]

results = evaluate(dataset, metrics=metrics_all, llm=llm, embeddings=embedding_model)
print(results)

# Set your OpenAI API key here
GEMINI_API_KEY = "AIzaSyB1lh_VOQk6qdQTR_MjhfsS5gclaYKOh-4"

genai.configure(api_key=GEMINI_API_KEY)
gemini_llm = genai.GenerativeModel('gemini-1.5-flash')

# Load a pre-trained sentence transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Default prompt for cleaning code
DEFAULT_PROMPT = """
You are a senior software engineer. Your task is to convert the provided code into clean, standardized, and maintainable code. Follow these guidelines:
1. Use proper naming conventions for variables, functions, and classes.
2. Add error handling where necessary.
3. Ensure the code follows respective programming language standards.
4. Add comments where necessary to explain complex logic.
5. Optimize the code for readability and performance.

Here is the code to clean:
"""

# Default prompt for generating test cases
TEST_CASE_PROMPT = """
You are a senior software engineer. Your task is to generate unit test cases for the provided code. Follow these guidelines:
1. Use the respective programming language testing framework.
2. Cover all edge cases and typical use cases.
3. Ensure the test cases are comprehensive and well-documented.
4. Include assertions to validate the expected output.

Here is the code to generate test cases for:
"""

def clean_code(input_code, additional_prompt="", DEFAULT_PROMPT=DEFAULT_PROMPT, gemini_llm=gemini_llm):
    """
    Sends the input code to the LLM with the default prompt and additional prompt (if any).
    Returns the cleaned code.
    """
    full_prompt = DEFAULT_PROMPT + input_code + "\n\n" + additional_prompt
    try:
        response = gemini_llm.generate_content(full_prompt)
        if response.candidates:
            extracted_text = response.text.strip('```json').strip('```').strip()
            return extracted_text
        else:
            extracted_text = ''
            return extracted_text
    except Exception as e:
        return f"An error occurred: {e}"

def generate_test_cases(input_code, TEST_CASE_PROMPT, gemini_llm):
    """
    Sends the input code to the LLM to generate test cases.
    Returns the generated test cases.
    """
    full_prompt = TEST_CASE_PROMPT + input_code
    try:
        response = gemini_llm.generate_content(full_prompt)
        if response.candidates:
            extracted_text = response.text.strip('```json').strip('```').strip()
            return extracted_text
        else:
            extracted_text = ''
            return extracted_text
    except Exception as e:
        return f"An error occurred: {e}"

def run_test_cases(code, test_cases):
    """
    Executes the provided code and test cases using the `unittest` framework.
    Returns the test results.
    """
    try:
        # Combine the code and test cases into a single script
        script = f"""
{code}

{test_cases}
        """
        # Save the script to a temporary file
        with open("temp_test_script.py", "w") as f:
            f.write(script)
        # Run the script and capture the output
        result = subprocess.run(
            [sys.executable, "temp_test_script.py"],
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception as e:
        return f"An error occurred while running the test cases: {e}"

def calculate_answer_relevancy(input_text, generated_text):
    """
    Calculates the Answer Relevancy metric by computing the cosine similarity
    between the embeddings of the input text and the generated text.
    """
    try:
        # Generate embeddings for the input and generated text
        input_embedding = embedding_model.encode([input_text], convert_to_tensor=True)
        generated_embedding = embedding_model.encode([generated_text], convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = cosine_similarity(input_embedding.cpu(), generated_embedding.cpu())
        return similarity[0][0]  # Return the similarity score
    except Exception as e:
        return f"An error occurred while calculating Answer Relevancy: {e}"

# Streamlit UI
st.title("Code Cleaner and Test Case Generator")
st.write("Paste your code below, and it will be converted into clean, standardized code. You can also generate and run test cases for the provided code.")

# Input for code
input_code = st.text_area("Paste your code here:", height=300)

# Input for additional prompt (optional)
additional_prompt = st.text_input("Any additional instructions for cleaning the code (optional):")

# Button to trigger the cleaning process
if st.button("Clean Code"):
    if input_code.strip() == "":
        st.warning("Please provide some code to clean.")
    else:
        with st.spinner("Cleaning your code..."):
            cleaned_code = clean_code(input_code, additional_prompt, DEFAULT_PROMPT, gemini_llm)
            st.success("Code cleaned successfully!")
            st.code(cleaned_code, language="python")

# Button to generate test cases
if st.button("Generate Test Cases"):
    if input_code.strip() == "":
        st.warning("Please provide some code to generate test cases.")
    else:
        with st.spinner("Generating test cases..."):
            test_cases = generate_test_cases(input_code, TEST_CASE_PROMPT, gemini_llm)
            st.success("Test cases generated successfully!")
            st.code(test_cases, language="python")

# Button to run test cases
if st.button("Run Test Cases"):
    if input_code.strip() == "":
        st.warning("Please provide some code to run test cases.")
    else:
        with st.spinner("Running test cases..."):
            test_cases = generate_test_cases(input_code)
            test_results = run_test_cases(input_code, test_cases)
            st.success("Test cases executed successfully!")
            st.code(test_results, language="plaintext")

# Streamlit UI for Answer Relevancy
st.title("Answer Relevancy Metric")
st.write("Calculate how relevant the generated code/test cases are to the input code.")

# Input for generated text (cleaned code or test cases)
generated_text = st.text_area("Paste the generated code or test cases here:", height=150)

# Button to calculate Answer Relevancy
if st.button("Calculate Answer Relevancy"):
    if input_code.strip() == "" or generated_text.strip() == "":
        st.warning("Please provide both the input code and generated text.")
    else:
        with st.spinner("Calculating Answer Relevancy..."):
            relevancy_score = calculate_answer_relevancy(input_code, generated_text)
            st.success(f"Answer Relevancy Score: {relevancy_score:.4f}")
            st.write("A score closer to 1 indicates higher relevancy.")