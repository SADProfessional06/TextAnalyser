import re
import numpy as np

with open("sample.txt", "r") as f:
    text = f.read()

paragraphs = text.split("\n\n")

# initialize lists to store the statistics for each paragraph
avg_word_lengths = []
avg_sentence_lengths = []
avg_word_counts = []

# loop through each paragraph
for paragraph in paragraphs:
    # split the paragraph into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", paragraph)
    
    # initialize lists to store the statistics for each sentence
    word_lengths = []
    word_counts = []
    sentence_lengths = []
    
    # loop through each sentence
    for sentence in sentences:
        # split the sentence into words
        words = sentence.split()
        
        # calculate the length of each word and store it in word_lengths
        word_lengths += [len(word) for word in words]
        # calculate the number of words in the sentence and store it in word_counts
        word_counts.append(len(words))
        
        # calculate the length of the sentence in characters and store it in sentence_lengths
        sentence_lengths.append(len(sentence))
    # calculate the average word length for the paragraph and store it in avg_word_lengths
    avg_word_lengths.append(np.mean(word_lengths))
    
    # calculate the average sentence length for the paragraph and store it in avg_sentence_lengths
    avg_sentence_lengths.append(np.mean(sentence_lengths))
    
    # calculate the average number of words in a sentence for the paragraph and store it in avg_word_counts
    avg_word_counts.append(np.mean(word_counts))

#Deletes all Nan values in avg_word_length
avg_word_lengths = [x for x in avg_word_lengths if not np.isnan(x)]

# calculate the standard deviation of avg_word_lengths, avg_sentence_lengths, and avg_word_counts
std_word_length = np.std(avg_word_lengths)
std_sentence_length = np.std(avg_sentence_lengths)
std_word_count = np.std(avg_word_counts)


# print the results
print("Average word length:", np.mean(avg_word_lengths))
print("Standard deviation of average word length:", std_word_length)
print("Average words in a sentence:", np.mean(avg_word_counts))
print("Standard deviation of average words in a sentence:", std_word_count)
print("Average words in a paragraph:", np.mean(avg_sentence_lengths))
print("Standard deviation of average words in a paragraph:", std_sentence_length)