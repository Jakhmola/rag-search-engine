import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    from google import genai

    client = genai.Client(api_key=api_key)

    model = "gemma-3-27b-it"
    content = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

    response = client.models.generate_content(model=model, contents=content)
    print(response.text)
    print(
        f"Prompt tokens: {response.usage_metadata.prompt_token_count}"
        f"Response tokens: {response.usage_metadata.candidates_token_count}")
    
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)