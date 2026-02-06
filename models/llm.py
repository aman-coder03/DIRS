import ollama

def generate_answer(prompt, model_name="llama3"):
    print("Using LLM:", model_name)
    
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": 150  # limit max tokens
        }
    )
    return response["message"]["content"]