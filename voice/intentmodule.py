"""
voice/intentmodule.py
=====================
FIX 2: Replaced LLM-based intent classification with pure keyword rules.

The original version called ask_llm() just to classify the intent, which
added 30-90 seconds of Ollama latency BEFORE even processing the query.
This version is instant (microseconds) and equally accurate for the
three categories: news, factual, reasoning.
"""

# News keywords
_NEWS_WORDS = [
    "news", "latest", "today", "headline", "update", "current events",
    "what happened", "recently", "breaking", "report", "announced",
]

# Reasoning keywords
_REASONING_WORDS = [
    "why", "how does", "explain", "compare", "difference between",
    "pros and cons", "should i", "what if", "reason", "analyse", "analyze",
    "think", "opinion", "better", "best way", "suggest", "recommend",
]

def classify_query(query: str) -> str:
    """
    Classify the query into: 'news', 'factual', or 'reasoning'.
    Pure keyword matching — no LLM call, instant response.
    """
    q = query.lower().strip()

    for word in _NEWS_WORDS:
        if word in q:
            return "news"

    for word in _REASONING_WORDS:
        if word in q:
            return "reasoning"

    return "factual"


# ---------------------------
# TEST BLOCK
# ---------------------------
if __name__ == "__main__":
    print("=== Intent Module Test ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        intent = classify_query(query)
        print(f"\nDetected Intent: {intent}")
        print("-" * 40)