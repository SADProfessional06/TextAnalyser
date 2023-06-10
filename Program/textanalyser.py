import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from IPython.display import Image
from pathlib import Path
from sklearn.tree import export_graphviz

N_ESTIMATORS = 1500
MAX_DEPTH = 3
N_FEATURES = 10

def text_stats(folder_path):
    """This function computes the text statistics."""
    # create a list to store the variable for each essay
    stats = []
    file_names = []

    # loop through all the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            file_names.append(filename)  # store file name
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

                    # skip sentence if it contains no words
                    if not words:
                        continue

                    # calculate the length of each word and store it in word_lengths
                    word_lengths += [len(word) for word in words]
                    # calculate the number of words in the sentence and store it in word_counts
                    word_counts.append(len(words))

                    # calculate the length of the sentence in characters and store it in sentence_lengths
                    sentence_lengths.append(len(sentence))
                
                # skip paragraph if it contains no sentences
                if not word_lengths or not word_counts or not sentence_lengths:
                    continue

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
            function_word_counts = [text.count(fw) for fw in function_words]
            function_word_frequencies = [fw_count / total_words for fw_count in function_word_counts]

            #Create a list of all the data rounding up the first 6 values to 2 decimal place and the frequency to 6
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
    return stats, file_names

def train_random_forest_classifier(expectedauthor_data, nonexpectedauthor_data):
    """This function trains a random forest classifier model."""
    # store file names
    expectedauthor_files = expectedauthor_data[1]
    nonexpectedauthor_files = nonexpectedauthor_data[1]
    # keep only features for data
    expectedauthor_data = expectedauthor_data[0]
    nonexpectedauthor_data = nonexpectedauthor_data[0]
    expectedauthor_labels = np.zeros(len(expectedauthor_data)) # 0 represents author
    nonexpectedauthor_labels = np.ones(len(nonexpectedauthor_data)) # 1 represents non-author
    data = np.concatenate((expectedauthor_data, nonexpectedauthor_data), axis=0)
    labels = np.concatenate((expectedauthor_labels, nonexpectedauthor_labels), axis=0)
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)
    rf.fit(train_data, train_labels)
    all_files = expectedauthor_files + nonexpectedauthor_files
    train_files, test_files = train_test_split(all_files, test_size=0.2)

    return rf, train_data, test_data, test_labels, train_files, test_files

def predict_labels(rf, sample_features):
    """This function predicts the labels for the given features."""
    class_labels=["Author", "Non-Author"]
    predicted_labels = rf.predict(sample_features)
    predicted_labels = [class_labels[int(predicted_label)] for predicted_label in predicted_labels]
    return predicted_labels

def plot_feature_importances(rf, feature_names):
    """This function plots the feature importances."""
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(f"Top {N_FEATURES} Feature Importances")
    plt.bar(range(N_FEATURES), importances[indices][:N_FEATURES], color="r", align="center")
    plt.xticks(range(N_FEATURES), [feature_names[i] for i in indices][:N_FEATURES], fontsize = 20, rotation=90)
    plt.xlim([-1, N_FEATURES])
    plt.tight_layout()
    plt.show()

def evaluate_model(rf, test_data, test_labels):
    """This function evaluates a trained model."""
    pred_labels = rf.predict(test_data)
    label_names = ['Non-Author', 'Author']
    pred_labels = np.array([label_names[int(label)] for label in pred_labels])
    test_labels = np.array([label_names[int(label)] for label in test_labels])
    accuracy = accuracy_score(test_labels, pred_labels)
    return accuracy, pred_labels, test_labels

def print_classification(rf, sample_features):
    """This function prints the classification made by the model."""
    if len(sample_features) != 0: 
        class_labels = predict_labels(rf, sample_features)
        for class_name in class_labels:
            print(class_name)

def main():
    """The main function to run the program."""
    expectedauthor_data = text_stats("/workspaces/TextAnalyzer/Author")    
    nonexpectedauthor_data = text_stats("/workspaces/TextAnalyzer/Non-Author")

    rf, train_data, test_data, test_labels, train_files, test_files = train_random_forest_classifier(expectedauthor_data, nonexpectedauthor_data)

    print("Training data files:")
    for file in train_files:
        print(file)

    print("Testing data files:")
    for file in test_files:
        print(file)


    print(f"Amount of training data: {len(train_data)}")
    print(f"Amount of testing data: {len(test_data)}")

    accuracy, pred_labels, test_labels = evaluate_model(rf, test_data, test_labels)

    print("Test data files with predicted and actual labels:")
    for file, pred, actual in zip(test_files, pred_labels, test_labels):
        print(f"File: {file}, Predicted: {pred}, Actual: {actual}")

    print("Predicted labels:\n", pred_labels)
    print("Actual labels:\n", test_labels)

    print(f"Accuracy of the random forest classifier: {accuracy}")

    sample_features = np.array(text_stats("/workspaces/TextAnalyzer/Unlabelled Data"))
    if len(sample_features[0]) != 0:
        print_classification(rf, sample_features)
    else:
        print("No unlabelled data to classify.")

    feature_names=["mean_wl", "s.d._wl", "mean_sl", "s.d._sl", "mean_pl", "s.d.pl", "freq. of a", "freq. of an", "freq. of the", "freq. of in", "freq. of on", "freq. of at", "freq. of to", "freq. of for", "freq. of of", "freq. of with", "freq. of by", "freq. of as", "freq. of is", "freq. of was", "freq. of were", "freq. of be", "freq. of been", "freq. of being", "freq. of that", "freq. of which", "freq. of who", "freq. of whom", "freq. of whose", "freq. of this", "freq. of these", "freq. of those", "freq. of such", "freq. of like", "freq. of about", "freq. of after", "freq. of before", "freq. of from", "freq. of through", "freq. of until", "freq. of unless", "freq. of since", "freq. of while", "freq. of although", "freq. of even", "freq. of just", "freq. of only", "freq. of not", "freq. of no", "freq. of neither", "freq. of nor"]
    
    # Calling the plot_feature_importances function to plot the graph
    plot_feature_importances(rf, feature_names)

if __name__ == "__main__":
    main()