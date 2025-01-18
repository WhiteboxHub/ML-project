import requests
from bs4 import BeautifulSoup
import os


def scrape_wikipedia_to_pdf(url):
    try:
        # Fetch the content of the Wikipedia page
        response = requests.get(url)
        response.raise_for_status()

        # Parse the page using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the title of the page
        title = soup.find('h1', id='firstHeading').text.strip()
        print(f"Page title: {title}")

        # Extract all text from the page content
        content_div = soup.find('div', id='bodyContent')
        paragraphs = content_div.find_all('p')
        content_text = '\n'.join(paragraph.text for paragraph in paragraphs)

        directory = "../Knowledgebase"
        os.makedirs(directory, exist_ok=True)

        # File path for the text file
        sanitized_title = title.replace(" ", "_").replace("/", "-")
        file_path = f"{directory}/{sanitized_title}.txt"

        with open(file_path,mode='w',encoding='utf-8') as file:
            file.write(content_text)

        return f"./Knowledgebase/{title}.txt created after scrapting the page"
    except Exception as e:
        print(f"An error occurred: {e}")

# URL of the Wikipedia page
# wikipedia_url = "https://en.wikipedia.org/wiki/Llama_(language_model)"  # Example page
# scrape_wikipedia_to_pdf(wikipedia_url)
