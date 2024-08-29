import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from transformers import pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSequenceClassification
from collections import Counter
import torch
from typing import List
import fire
#from llama import Llama

class SentimentAnalyzerComparison:
    def __init__(self, language="en", multilingual_model=False, llm=False):
        self.language = language
        self.multilingual_model = multilingual_model
        self.api_url="http://192.168.201.104:8080/api/"
        self.llm=llm
        if self.language == "fr":
            self.base_url = "https://fr-be.trustpilot.com/review/www.keytradebank.be?page="
        elif self.language == "nl": 
            self.base_url = "https://nl-be.trustpilot.com/review/www.keytradebank.be?page="
        else:
            self.base_url = "https://www.trustpilot.com/review/www.keytradebank.be?page="

        if self.llm=="llama3":
            self.pipe = pipeline("text-generation", model=r"C:\Users\Romain\meta-llama\Meta-Llama-3.1-8B-Instruct")
        elif self.multilingual_model and self.llm==False:
            tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            self.pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        elif self.language == "fr" and self.multilingual_model==False and self.llm==False:
            self.base_url = "https://fr-be.trustpilot.com/review/www.keytradebank.be?page="
            tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
            model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
            self.pipe = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        elif self.language == "nl" and self.multilingual_model==False and self.llm==False:
            self.base_url = "https://nl-be.trustpilot.com/review/www.keytradebank.be?page="
            tokenizer = AutoTokenizer.from_pretrained("DTAI-KULeuven/robbert-v2-dutch-sentiment", from_pt=True)
            model = TFAutoModelForSequenceClassification.from_pretrained("DTAI-KULeuven/robbert-v2-dutch-sentiment", from_pt=True)
            self.pipe = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        elif self.llm==False and self.multilingual_model==False and self.language!="fr" and self.language!="nl":
            raise ValueError("Invalid language. Please choose 'fr' or 'nl'.")

    def WebScraper(self):
        reviews = []
        notes = []
        for i in range(1, 30):
            url = self.base_url + str(i)
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception if the request fails
                soup = BeautifulSoup(response.text, "html.parser")
                review_elements = soup.find_all("div", class_="styles_reviewContent__0Q2Tg")
                for element in review_elements:
                    h2 = element.find("h2").text.strip() if element.find("h2") else ""
                    p = element.find("p").text.strip() if element.find("p") else ""
                    if self.language == "fr":
                        review = f"Titre: {h2} Texte: {p}"
                    elif self.language == "nl":
                        review = f"Titel: {h2} Tekst: {p}"
                    reviews.append(review)
                note_elements = soup.find_all("div", class_="styles_reviewHeader__iU9Px")
                for element in note_elements:
                    rating = element.get("data-service-review-rating")
                    if rating:
                        notes.append(int(rating))
            except Exception as e:
                print(f"Error scraping page {i}: {e}")
        return reviews, notes

    def SaveData(self):
        reviews, notes = self.WebScraper()
        with open(f"reviews_{self.language}.json", "w") as f:
            json.dump(reviews, f)
        with open(f"notes_{self.language}.json", "w") as f:
            json.dump(notes, f)

    def LoadData(self):
        with open(f"reviews_{self.language}.json") as f:
            reviews = json.load(f)
        with open(f"notes_{self.language}.json") as f:
            notes = json.load(f)
        return reviews, notes

    def MapSentiment(self, value):
        if value in [1, 2]:
            return "NEGATIVE"
        elif value == 3:
            return "NEUTRAL"
        elif value in [4, 5]:
            return "POSITIVE"
        else:
            return "UNKNOWN"

    def CreateDataFrame(self):
        reviews, notes = self.LoadData()
        df = pd.DataFrame({"reviews": reviews, "notes": notes})
        df["notes"] = df["notes"].apply(self.MapSentiment)  # Corrected method call
        return df

    def ChunkText(self, text, max_length=512):
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]

    def AnalyzeAndVote(self, text, model, max_length=512):
        chunks = self.ChunkText(text, max_length)  # Corrected method call
        predictions = []
        for chunk in chunks:
            prediction = model(chunk)[0]['label']
            predictions.append(prediction)
        majority_vote = Counter(predictions).most_common(1)[0][0]
        if self.multilingual_model:
            if isinstance(majority_vote, str):
                majority_vote=self.MapSentiment(int(majority_vote[0]))
            else:
                majority_vote = self.MapSentiment(majority_vote)
        return majority_vote
        
    def get_sentiment_llm(self, text):
        if self.llm=="llama3":
            if self.language=="fr":
                messages = [
                    {"role": "user", "content": f"<<Commentaire: {text}>>\nÀ partir de cette phrase, déterminez et répondez uniquement si elle est NÉGATIF, NEUTRE ou POSITIF. UN SEUL MOT SANS AUCUN SYMBOLE OU CARACTÈRE SUPPLÉMENTAIRE."}
                ]
            elif self.language=="nl":   
                messages = [
                    {"role": "user", "content": f"<<Commentaar: {text}>>\nBepaal het gevoel van de klant tussen de drie woorden: NEGATIEF, NEUTRAAL of POSITIEF is. ALLEEN EEN WOORD ZONDER ENIGE AANVULLENDE SYMBOLEN OF TEKENS. (Only answer in one word)"}
                ]
            else:
                messages = [
                    {"role": "user", "content": f"<<Comment: {text}>>\nFrom this phrase, determine and only answer whether it is NEGATIVE, NEUTRAL, or POSITIVE. ONLY ONE WORD WITHOUT ANY ADDITIONAL SYMBOL OR CHARACTER."}
                ]
            sentiment=self.pipe(messages, max_new_tokens=10)
            print(sentiment)
            sentiment=sentiment[0]['generated_text'][-1]['content']
            print(sentiment)
        elif self.llm!=False:
            if self.language == "fr":
                payload = {
                    "model": self.llm,
                    "prompt": f"<<Commentaire: {text}>>\n À partir de cette phrase, déterminez et répondez uniquement si elle est NÉGATIF, NEUTRE ou POSITIF. UN SEUL MOT SANS AUCUN SYMBOLE OU CARACTÈRE SUPPLÉMENTAIRE.",
                    "stream": False
                }
            elif self.language == "nl":
                payload = {
                    "model": self.llm,
                    "prompt": f"<<Commentaar: {text}>>\nBepaal het gevoel van de klant tussen de drie woorden: NEGATIEF, NEUTRAAL of POSITIEF is. ALLEEN EEN WOORD ZONDER ENIGE AANVULLENDE SYMBOLEN OF TEKENS. (Only answer in one word)",
                    "stream": False
                }
            else:
                payload = {
                    "model": self.llm,
                    "prompt": f"<<Comment: {text}>>\nFrom this phrase, determine and only answer whether it is NEGATIVE, NEUTRAL, or POSITIVE. ONLY ONE WORD WITHOUT ANY ADDITIONAL SYMBOL OR CHARACTER.",
                    "stream": False
                }

            response = requests.post(self.api_url + "generate", json=payload)

            if response.status_code == 200:
                response_json = response.json()
                sentiment = response_json['response'].strip().upper()
            
        sentiment = sentiment.replace("NÉGATIF", "NEGATIVE")
        sentiment = sentiment.replace("NEUTRE", "NEUTRAL")
        sentiment = sentiment.replace("POSITIF", "POSITIVE")
        sentiment = sentiment.replace("NEGATIEF", "NEGATIVE")
        sentiment = sentiment.replace("NEUTRAAL", "NEUTRAL")
        sentiment = sentiment.replace("POSITIEF", "POSITIVE")
        if sentiment not in ('NEGATIVE', 'NEUTRAL', 'POSITIVE'):
            sentiment='NEUTRAL'
        return sentiment

    def PredictSentiments(self, max_length=512):
        df = self.CreateDataFrame()
        if self.llm:
            df['predicted_notes'] = df['reviews'].apply(lambda review: self.get_sentiment_llm(review))
            df.to_excel(f"sentiment_analysis_results_{self.language}_{self.llm}.xlsx", index=False)
        elif self.multilingual_model:
            df['predicted_notes'] = df['reviews'].apply(lambda review: self.AnalyzeAndVote(review, self.pipe, max_length))
            df.to_excel(f"sentiment_analysis_results_{self.language}_multilingual.xlsx", index=False)
        else:
            df['predicted_notes'] = df['reviews'].apply(lambda review: self.AnalyzeAndVote(review, self.pipe, max_length).upper())
            df.to_excel(f"sentiment_analysis_results_{self.language}.xlsx", index=False)
        return df

    def ShowTopMisclassified(self, df):
        # Misclassified as Negative but True Positive
        false_negative = df[(df['predicted_notes'] == 'NEGATIVE') & (df['notes'] == 'POSITIVE')]
        top_false_negative = false_negative.head(5)
        print("\nTop 5 Reviews Predicted as NEGATIVE but are POSITIVE:")
        for i, row in top_false_negative.iterrows():
            print(f"Review: {row['reviews']}\nTrue: {row['notes']} | Predicted: {row['predicted_notes']}\n")

        # Misclassified as Positive but True Negative
        false_positive = df[(df['predicted_notes'] == 'POSITIVE') & (df['notes'] == 'NEGATIVE')]
        top_false_positive = false_positive.head(5)
        print("\nTop 5 Reviews Predicted as POSITIVE but are NEGATIVE:")
        for i, row in top_false_positive.iterrows():
            print(f"Review: {row['reviews']}\nTrue: {row['notes']} | Predicted: {row['predicted_notes']}\n")

    def EvaluatePredictions(self):
        df = self.PredictSentiments()
        y_true = df['notes']
        y_pred = df['predicted_notes']
        cm = confusion_matrix(y_true, y_pred, labels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'],
                    yticklabels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['POSITIVE', 'NEUTRAL', 'NEGATIVE'], zero_division=0))
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {accuracy:.2f}")
        print("\nTop Misclassified Reviews:")
        self.ShowTopMisclassified(df)   
    
    def EvaluatePredictionsOld(self,df):
        y_true = df['notes']
        y_pred = df['predicted_notes']
        cm = confusion_matrix(y_true, y_pred, labels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'],
                    yticklabels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['POSITIVE', 'NEUTRAL', 'NEGATIVE'], zero_division=0))
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {accuracy:.2f}")
        print("\nTop Misclassified Reviews:")
        self.ShowTopMisclassified(df)  

analyzer=SentimentAnalyzerComparison("fr",llm="llama3")
analyzer.EvaluatePredictions()

class MajorityVoteAnalyzer:
    def __init__(self, language):
        self.language = language
        self.model1_path = f"sentiment_analysis_results_{self.language}_ollama.xlsx"
        self.model2_path = f"sentiment_analysis_results_{self.language}_multilingual.xlsx"
        self.model3_path = f"sentiment_analysis_results_{self.language}.xlsx"
        self.output_path = f"sentiment_analysis_results_{self.language}_majority_vote.xlsx"

    def load_predictions(self):
        """Load predictions from the three model Excel files."""
        self.df_model1 = pd.read_excel(self.model1_path)
        self.df_model2 = pd.read_excel(self.model2_path)
        self.df_model3 = pd.read_excel(self.model3_path)

    def get_majority_vote(self, predictions):
        """Return the majority vote from a list of predictions."""
        vote_counts = Counter(predictions)
        # Return the most common prediction; if there is a tie, it returns one of the most common
        majority_vote = vote_counts.most_common(1)[0][0]
        return majority_vote

    def combine_and_vote(self):
        """Combine predictions and determine the majority vote for each instance."""
        combined_df = pd.DataFrame({
            'reviews': self.df_model1['reviews'],
            'notes': self.df_model1['notes'],
            'model1_predicted': self.df_model1['predicted_notes'],
            'model2_predicted': self.df_model2['predicted_notes'],
            'model3_predicted': self.df_model3['predicted_notes']
        })

        # Apply majority voting for each row
        combined_df['predicted_notes'] = combined_df.apply(
            lambda row: self.get_majority_vote([row['model1_predicted'], 
                                                row['model2_predicted'], 
                                                row['model3_predicted']]), 
            axis=1
        )

        return combined_df

    def save_combined_results(self, combined_df):
        """Save the combined DataFrame with majority votes to Excel."""
        combined_df.to_excel(self.output_path, index=False)
        print(f"Combined results with majority vote saved to {self.output_path}")

    def run(self):
        """Run the complete process: load, combine, vote, and save results."""
        self.load_predictions()
        combined_df = self.combine_and_vote()
        self.save_combined_results(combined_df)
        return combined_df

