import wikipediaapi


class WikiAssistant:
    def __init__(self, language='en'):
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="MyRobotAssistant/1.0"
        )

    def get_summary(self, query, sentences=3):
        """
        Returns a clean summary suitable for speaking
        """

        # 🔴 EMPTY INPUT FIX
        if not query or query.strip() == "":
            return "EMPTY_QUERY"

        try:
            page = self.wiki.page(query)

            # 🔴 PAGE NOT FOUND
            if not page.exists():
                return "NO_RESULT"

            summary = page.summary

            # 🔴 EMPTY SUMMARY
            if not summary:
                return "NO_RESULT"

            # Limit sentences
            summary_sentences = summary.split('. ')
            short_summary = '. '.join(summary_sentences[:sentences])

            return short_summary.strip()

        except Exception as e:
            print("[Wiki Error]:", e)
            return "WIKI_ERROR"

    def format_for_speech(self, text):
        text = text.replace('\n', ' ')
        text = text.replace('  ', ' ')
        return text


# TEST BLOCK
if __name__ == "__main__":

    assistant = WikiAssistant()

    while True:
        query = input("Enter query: ")

        if query.lower() == "exit":
            break

        result = assistant.get_summary(query)
        print(result)