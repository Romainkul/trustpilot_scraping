import requests
from bs4 import BeautifulSoup
import json

base_url_nl = "https://nl-be.trustpilot.com/review/www.keytradebank.be?page="
base_url_fr = "https://fr-be.trustpilot.com/review/www.keytradebank.be?page="

reviews_fr = []
notes_fr = []
reviews_nl = []
notes_nl = []

for i in range(1, 30):  # Scrape the first 10 pages of reviews
    url_nl = base_url_nl + str(i)

    try:
        response = requests.get(url_nl)
        response.raise_for_status()  # Raise an exception if the request fails
        soup = BeautifulSoup(response.text, "html.parser")

        review_elements = soup.find_all("div", class_="styles_reviewContent__0Q2Tg")

        for element in review_elements:
            h2 = element.find("h2").text.strip()
            p = element.find("p").text.strip()
            review = f"Titel: {h2} Tekst: {p}"
            reviews_nl.append(review)

        note_elements = soup.find_all("div", class_="styles_reviewHeader__iU9Px")
        notes_nl.extend([int(element.get("data-service-review-rating")) for element in note_elements])
    except Exception as e:
        print(f"Error scraping page {i}: {e}")

for i in range(1, 30):  # Scrape the first 10 pages of reviews
    url_nl = base_url_fr + str(i)

    try:
        response = requests.get(url_nl)
        response.raise_for_status()  # Raise an exception if the request fails
        soup = BeautifulSoup(response.text, "html.parser")

        review_elements = soup.find_all("div", class_="styles_reviewContent__0Q2Tg")

        for element in review_elements:
            h2 = element.find("h2").text.strip()
            p = element.find("p").text.strip()
            review = f"Titel: {h2} Tekst: {p}"
            reviews_fr.append(review)

        note_elements = soup.find_all("div", class_="styles_reviewHeader__iU9Px")
        notes_fr.extend([int(element.get("data-service-review-rating")) for element in note_elements])
    except Exception as e:
        print(f"Error scraping page {i}: {e}")

print(reviews_fr)
print(notes_fr)
print(reviews_nl)
print(notes_nl)

with open("reviews_27_08_fr.json", "w") as f:
    json.dump(reviews_fr, f)

with open("notes_27_08_fr.json", "w") as f:
    json.dump(notes_fr, f)

with open("reviews_27_08_nl.json", "w") as f:
    json.dump(reviews_nl, f)

with open("notes_27_08_nl.json", "w") as f:
    json.dump(notes_nl, f)


