import string
import os
import re

import asyncio
import aiohttp
import nltk
from bs4 import BeautifulSoup
import pandas as pd

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('cmudict')


cmu_dict = cmudict.dict()

data = [
    ["URL_ID", "URL", "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE", "AVG SENTENCE LENGTH",
     "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX", "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT", "WORD COUNT",
     "SYLLABLE PER WORD", "PERSONAL PRONOUNS"]]


async def fetch_and_save_article(url_id, url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html_content = await response.text()

            soup = BeautifulSoup(html_content, 'html.parser')

            entry_titles = soup.find_all('h1', class_='entry-title') if soup.find_all('h1',
                                                                                      class_='entry-title') else soup.find_all(
                'h1', class_='tdb-title-text')
            article_content = soup.find_all('div', class_='td-post-content') if soup.find_all('div',
                                                                                              class_='td-post-content ') else soup.find_all(
                'div', class_='tdb-block-inner')
            divs = soup.find_all('div', class_='td-post-content tagdiv-type') if soup.find_all('div',
                                                                                               class_='td-post-content ' 'tagdiv-type') else soup.find_all(
                'div',
                class_='td_block_wrap tdb_single_content tdi_130 td-pb-border-top td_block_template_1 td-post-content ' 'tagdiv-type')

            with open(f'ExtractedArticles/{url_id}.txt', 'w', encoding='utf-8') as file:
                for entry_title in entry_titles:
                    file.write(entry_title.text)
                for div_content in divs:
                    text_content = div_content.get_text(separator='')
                    file.write(text_content)

            print(f"{url_id} website completed")


async def extract_articles():
    df = pd.read_excel('Input.xlsx')

    tasks = []
    for index, row in df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        tasks.append(fetch_and_save_article(url_id, url))

    await asyncio.gather(*tasks)


def stopwords_list():
    stopwords_folder_path = "StopWords"
    stop_words = []
    for filename in os.listdir(stopwords_folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(stopwords_folder_path, filename), "r", encoding="ISO-8859-1") as file:
                for line in file:
                    stop_words.append(line.strip())
    return stop_words


def remove_stopwords(tokenized_words, stop_words):
    return [word for word in tokenized_words if word.lower() not in stop_words]


def positive_words_list():
    words = open("MasterDictionary/positive-words.txt", encoding="utf-8").read()
    positive_words = word_tokenize(words)
    return positive_words


def negative_words_list():
    words = open("MasterDictionary/negative-words.txt", encoding="ISO-8859-1").read()
    negative_words = word_tokenize(words)
    return negative_words


def syllable_count(word):
    if word.lower() in cmu_dict:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()]])
    else:
        return 0


def is_complex(word):
    threshold = 2
    return syllable_count(word) >= threshold


def analysis():
    df = pd.read_excel('Input.xlsx')
    for index, row in df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        words_score = {}

        plain_text = open(f"./ExtractedArticles/{url_id}.txt", 'r').read()

        if plain_text == '':
            continue
        upper_text = plain_text.upper()
        cleaned_text = upper_text.translate(str.maketrans('', '', string.punctuation))
        tokenized_words = word_tokenize(cleaned_text)
        stop_words = stopwords_list()
        filtered_words = remove_stopwords(tokenized_words=tokenized_words, stop_words=stop_words)
        positive_words = positive_words_list()
        negative_words = negative_words_list()
        for word in filtered_words:
            if word.lower() in positive_words:
                words_score[word] = 1
            elif word.lower() in negative_words:
                words_score[word] = -1
        positive_score = sum(1 for value in words_score.values() if value == 1)
        negative_score = sum(1 for value in words_score.values() if value == -1)
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        total_words_after_cleaning = len(tokenized_words)
        subjective_score = (positive_score + negative_score) / (total_words_after_cleaning + 0.000001)

        sentences = sent_tokenize(plain_text)

        words = [word.lower() for word in word_tokenize(cleaned_text)]

        average_sentence_length = len(words) / len(sentences)

        # Find complex words
        complex_words = [word for word in words if is_complex(word)]
        syllable_words = sum(syllable_count(word) for word in words)
        syllable_per_word = syllable_words / len(words)
        # complex words count
        complex_word_count = len(complex_words)

        percentage_of_complex_words = complex_word_count / len(words)
        fog_index = 0.4 * (average_sentence_length / percentage_of_complex_words)

        average_number_of_words_per_sentence = len(words)/len(sentences)
        word_count = len(tokenized_words)

        total_characters = sum(len(word) for word in words)

        average_word_length = total_characters / len(words)
        pronoun_pattern = r'\b(?:i|we|my|ours|us)\b'
        pronouns_found = re.findall(pronoun_pattern, cleaned_text.lower())
        personal_pronouns = len(pronouns_found)
        data.append([url_id, url, positive_score, negative_score, polarity_score, subjective_score, average_sentence_length, percentage_of_complex_words, fog_index, average_number_of_words_per_sentence, complex_word_count, word_count, syllable_per_word, personal_pronouns])
        print(f"{url_id}'s analysis is completed")
    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_excel('output.xlsx', index=False)


if __name__ == "__main__":
    import time

    start_time = time.time()
    try:
        print("Extracting data from websites")
        asyncio.run(extract_articles())
        print("Extract data from websites is completed\nAnalysing the data")
        analysis()
        print("\nAnalysis is completed please check out the `output.xlsx` file in root directory\n")
    except Exception as e:
        print("\nsomething went wrong!\n",e)
    end_time = time.time()
    print(f"Executed in {end_time - start_time} s")
