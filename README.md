# llm_evaluation
A repo to evaluate LLM responses for the generated code

## Input to the streamlit application
### Input Code
```
def factorial_recursive(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)

num = int(input("Enter a number: "))
print("Factorial (Recursive):", factorial_recursive(num))
```

### Default Prompt
You are a senior software engineer. Your task is to convert the provided code into clean, standardized, and maintainable code. Follow these guidelines:
1. Use proper naming conventions for variables, functions, and classes.
2. Add error handling where necessary.
3. Ensure the code follows respective programming language standards.
4. Add comments where necessary to explain complex logic.
5. Optimize the code for readability and performance.