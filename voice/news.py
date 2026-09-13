import requests

API_KEY = "2fae78f6bb364b1bb8ddb4c6fc69d825"


def get_news(query=None):
    """
    Fetch latest news headlines.
    If query is provided → search news
    If not → top headlines
    """

    if query:
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={API_KEY}"
    else:
        url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"
    try:
        response = requests.get(url).json()

        if response["status"] != "ok":
            return "Unable to fetch news."

        articles = response["articles"][:3]

        if not articles:
            return "No news found."

        results = []
        for i, article in enumerate(articles):
            title = article.get("title", "No title available")
            results.append(f"{i+1}. {title}")

        return "\n".join(results)

    except Exception as e:
        return f"Error fetching news: {str(e)}"


# -------------------------------
# 🔍 TEST BLOCK (RUN THIS FILE)
# -------------------------------

if __name__ == "__main__":

    print("=== News Module Test ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Enter query: ")

        if query.lower() == "exit" or query.lower() == "go back":
            print("Exiting...")
            break

        if query.strip() == "":
            news = get_news()
        else:
            news = get_news(query)

        print("\n News Results:\n")
        print(news)
        print("\n" + "-" * 50 + "\n")