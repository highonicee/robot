import requests

def ask_llm(prompt):
    url = "http://localhost:11434/api/generate"

    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        return "Error in connecting"
