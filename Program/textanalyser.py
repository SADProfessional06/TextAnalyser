import os
import re
import numpy as np
import random
import matplotlib.pyplot as plt

#Modules used for Random Forest Classifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Modules used for the generation of a sample Tree
from IPython.display import Image
from pathlib import Path
import graphviz
def text_stats(folder_path):
    # create a list to store the variable for each essay
    stats = []

    # loop through all the files in the folder
    for filename in os.listdir(folder_path):
        # check if the file is a text file
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                text = f.read()

            paragraphs = text.split("\n\n")

            # create lists to store the statistics for each paragraph
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

            # Deletes all Nan values in avg_word_length
            avg_word_lengths = [x for x in avg_word_lengths if not np.isnan(x)]

            # calculate the standard deviation of avg_word_lengths, avg_sentence_lengths, and avg_word_counts
            std_word_length = np.std(avg_word_lengths)
            std_sentence_length = np.std(avg_sentence_lengths)
            std_word_count = np.std(avg_word_counts)

            # count the frequency of function words
            function_words = ["a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "as", "is", "was", "were",
                              "be", "been", "being", "that", "which", "who", "whom", "whose", "this", "these", "those",
                              "such", "like", "about", "after", "before", "from", "through", "until", "unless", "since",
                              "while", "although", "even", "just", "only", "not", "no", "neither", "nor"]
            total_words = sum(avg_word_counts)
            function_word_counts = [text.count (fw)for fw in function_words]
            function_word_frequencies = [fw_count / total_words for fw_count in function_word_counts]
            #Create a list of all the data rounding up the first 6 values to 2 decimal place and the frequency to 6
            #
            output = [
                round(np.mean(avg_word_lengths), 2),
                round(std_word_length, 2),
                round(np.mean(avg_word_counts), 2),
                round(std_word_count, 2),
                round(np.mean(avg_sentence_lengths), 2),
                round(std_sentence_length, 2),
            ]
            for fw_frequency in function_word_frequencies:
                output.append(round(fw_frequency, 6))
            stats.append(output)
    return stats


expectedauthor_data = text_stats("/workspaces/codespaces-jupyter/Author")
expectedauthor_data = np.array(expectedauthor_data)

expectedauthor_labels = np.zeros(len(expectedauthor_data)) # 0 represents author
nonexpectedauthor_data = text_stats("/workspaces/codespaces-jupyter/Non-Author")
nonexpectedauthor_data = np.array(nonexpectedauthor_data)

#This part of the program plots a graph using the two datasets
# Select only  two columns (mean and standard deviation)
mean_lengths1 = expectedauthor_data[:, 2]
std_lengths1 = expectedauthor_data[:, 3]
mean_lengths2 = nonexpectedauthor_data[:, 2]
std_lengths2 = nonexpectedauthor_data[:, 3]

# Create a scatter plot with different colors for the two arrays
plt.scatter(mean_lengths1, std_lengths1, color='blue', label='Author')
plt.scatter(mean_lengths2, std_lengths2, color='red', label='Non-Author')

# Add labels, title, and legend
plt.xlabel('Mean Length')
plt.ylabel('Standard Deviation of Length')
plt.title('Scatter Plot of Sentence Length')
plt.legend()

# Display the plot
plt.show()

#This part of the program generates the random forest classifier model
nonexpectedauthor_labels = np.ones(len(nonexpectedauthor_data)) # 1 represents non-author
# These are sample the data that are not from the same author. The standard deviation 
data = np.concatenate((expectedauthor_data, nonexpectedauthor_data), axis=0)
labels = np.concatenate((expectedauthor_labels, nonexpectedauthor_labels), axis=0)
indices = np.random.permutation(len(data))
data = data[indices]
labels = labels[indices]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
rf = RandomForestClassifier(n_estimators=1500, max_depth=3)

print("Amount of training data:  " + str(len(train_labels)))
print("Amount of testing data:  " + str(len(test_labels)))


rf.fit(train_data, train_labels)

pred_labels = rf.predict(test_data)
label_names = ['Non-Author', 'Author']

pred_labels = np.array([label_names[int(label)] for label in pred_labels])
test_labels = np.array([label_names[int(label)] for label in test_labels])

print("Predicted labels:\n", pred_labels)
print("Actual labels:\n", test_labels)

accuracy = accuracy_score(test_labels, pred_labels)
print(f"Accuracy of the random forest classifier: {accuracy}")

estimator = rf.estimators_[0]
dot_data = export_graphviz(estimator, out_file=None, 
                feature_names=["mean_word_length", "s.d._word_length", "mean_sentence_length", "s.d._sentence_length", "mean_paragraph_length", "s.d._paragraph_length", "Frequency of a", "Frequency of an", "Frequency of the", "Frequency of in", "Frequency of on", "Frequency of at", "Frequency of to", "Frequency of for", "Frequency of of", "Frequency of with", "Frequency of by", "Frequency of as", "Frequency of is", "Frequency of was", "Frequency of were", "Frequency of be", "Frequency of been", "Frequency of being", "Frequency of that", "Frequency of which", "Frequency of who", "Frequency of whom", "Frequency of whose", "Frequency of this", "Frequency of these", "Frequency of those", "Frequency of such", "Frequency of like", "Frequency of about", "Frequency of after", "Frequency of before", "Frequency of from", "Frequency of through", "Frequency of until", "Frequency of unless", "Frequency of since", "Frequency of while", "Frequency of although", "Frequency of even", "Frequency of just", "Frequency of only", "Frequency of not", "Frequency of no", "Frequency of neither", "Frequency of nor"],
                class_names=["Author", "Non-Author"],
                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree")

sample_features = np.array(text_stats("/workspaces/codespaces-jupyter/Unlabelled Data"))
if len(os.listdir("/workspaces/codespaces-jupyter/Unlabelled Data")) != 0: #Check if file is empty
    class_labels=["Author", "Non-Author"]
    predicted_labels = rf.predict(sample_features)
    for predicted_label in predicted_labels:
        class_name = class_labels[int(predicted_label)]
        print(class_name) #This prints the classification made by the model of the text file(s) in the folder


#This shows the features importance of each variable
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names=["mean_wl", "s.d._wl", "mean_sl", "s.d._sl", "mean_pl", "s.d.pl", "freq. of a", "freq. of an", "freq. of the", "freq. of in", "freq. of on", "freq. of at", "freq. of to", "freq. of for", "freq. of of", "freq. of with", "freq. of by", "freq. of as", "freq. of is", "freq. of was", "freq. of were", "freq. of be", "freq. of been", "freq. of being", "freq. of that", "freq. of which", "freq. of who", "freq. of whom", "freq. of whose", "freq. of this", "freq. of these", "freq. of those", "freq. of such", "freq. of like", "freq. of about", "freq. of after", "freq. of before", "freq. of from", "freq. of through", "freq. of until", "freq. of unless", "freq. of since", "freq. of while", "freq. of although", "freq. of even", "freq. of just", "freq. of only", "freq. of not", "freq. of no", "freq. of neither", "freq. of nor"]

n_features = 10 # Number of top features to show
plt.figure(figsize=(12, 6))
plt.title("Top {} Feature Importances".format(n_features))
plt.bar(range(n_features), importances[indices][:n_features], color="r", align="center")
plt.xticks(range(n_features), [feature_names[i] for i in indices][:n_features], fontsize = 20, rotation=90)
plt.xlim([-1, n_features])
plt.tight_layout()
plt.show()