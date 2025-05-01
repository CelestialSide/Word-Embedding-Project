import spacy
import re
import csv
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
nlp.add_pipe("sentencizer")

def CleanUp(text):
        # Load the spaCy English model and process text
    dictionary = set(nlp.vocab.strings)
    doc = nlp(text)

        # Extract lemmatized tokens
    filtered_words = [token.lemma_ for token in doc if not token.is_stop]
    filtered_words = [token for token in filtered_words if len(token) > 1]

        # Removes Non-English words and rejoins into sentence
    sentence = " ".join(w for w in filtered_words if w.lower() in dictionary) + "."
    return sentence.lower()


    # Append is an auxiliary function, please do not call
def append(text, title, outputFile):
    fields = ["Title", "Story"]

    doc = nlp(text)
    sentences = doc.sents

        # Opens output file and appends new data into file
        # (NOTE: FILE NEEDS TO BE CREATED BEFOREHAND AND FIELDS MUST BE PRESET!!!)
    with open(outputFile, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        for sent in tqdm(sentences):
            cleaned_sent = CleanUp(sent.text)
                # Finds how many words make up the sentence and then removes fragments
            fragment = len(cleaned_sent.split())
            if fragment > 2:
                data = {"Title": title, "Story": cleaned_sent}
                writer.writerow(data)
    csvfile.close()

    # txtToCSV opens Input File and removes sentence fragments,
        # words not present in the english language
            # (These can be names, made up words, ect...),
        # lemanizes words, and removes stop-words before appending
        # into new file
def txtToCSV(fileName, title, outputFile):
    story = open(fileName, "r", errors="ignore")
    story = story.read()

    story = re.sub("â€™", "'", story)
    story = re.sub("[^A-Za-z '!?.:]", " ", story)
    story = ' '.join(story.split())
    append(story, title, outputFile)

txtToCSV("Input.txt", "The Eyes of the Panther", "Cleaned_Corpus.csv")