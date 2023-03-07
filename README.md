# TextAnalyzer
This code analyzes a set of text files to extract statistics about the text. The code uses the extracted data to create a dataset that includes these statistics and whether the text was authored by a specific author. Text files in the Author and Non-Author are classified as Author and Non-Author respecitvely.
A graph showing information about the sentence lengths is plotted.
The dataset is then used to train a Random Forest Classifier model that can determine if a new text file is written by the author analysed or not.

Everytime the program runs, a new tree.pdf is created. This is a sample tree created in the random classification model.
