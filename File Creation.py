import csv
import re
from nltk.tokenize import sent_tokenize

    # Append is an auxiliary function, please do not call
def append(story, title, outputFile):
        # Initialize needed variables
    sentences = []
    fields = ["Title", "Story"]

        # Strips out anything that is not an alphabet character or '.'
    story = re.sub("[^A-Za-z. ]", " ", story)
        # Sent_tokenize splits the inputted string into sentences They are then
        #   appended into a list after setting all characters to lower case
    for i in sent_tokenize(story):
        sentences.append(i.lower())

        # Opens output file and appends new data into file
        # (NOTE: FILE NEEDS TO BE CREATED BEFOREHAND AND FIELDS MUST BE PRESET!!!)
    with open(outputFile, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        for sent in sentences:
            data = {"Title": title, "Story": sent}
            writer.writerow(data)
    csvfile.close()

    # csvFix opens CSV File of previous format and recreates
    #   in new format.
def csvFix(fileName, outputFile):
    with open(fileName, mode='r') as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            title = lines.get('Title')
            story = lines.get('Story')
            append(story, title, outputFile)
    file.close()

    # txtToCSV opens the txt file of a story and adds all sentences
    #   to the outputFile specified, (NOTE: Creates run-on sentences
    #   due to stripping punctuation but does not strip Non-English words)
def txtToCSV(fileName, title, outputFile):
    story = open(fileName, "r", errors="ignore")
    story = story.read()
    append(story, title, outputFile)


txtToCSV("Input.txt", "Frankenstein", "Story_Database.csv")
# csvFix("Dupe.csv", "Story_Database.csv")

