# The linguistic foundations of Trump's and Hillary's speeches

### Introduction

This is a repository for Statistical Linguistic Analysis of two of the leading nominees for the 2016 Election, Donald Trump and Hillary Clinton. It is by no means limited to just these two candidates, and should someone put in the work to gather data for other candidates, the code can easily be extended. 

### Corpus
At this point, I've gathered a few speech transcripts from Trump and Clinton from the internet. Each speech is copied into a text file, with the first two lines designated for metadata. The format is as follows:
Source:[source url]
Label:[candidate last name]
<body>

### Linguistic Analysis
While there are many different techniques to analyze text and draw conclusions, the most straightforward way to compare two candidates was to use a supervised approach to train a classifier to distinguish between the two candidates. The accuracy of such a classifier offers a simple objective measure of the comparison's efficacy. With a sufficiently predictive model, we could then look inwardly at the models weights to identify the most predictive features. These predictive features provide solid non-subjective accounts of how each candidate distinctly speaks.

Of course, this is heavily biased by the features that we use. If we only use the bigram feature "great again", the trained model becomes largely a tuning of our own biases. Instead, feature engineering should be focused on removing extraneous syntax introduced by each of the transcript writers and the websites serving them. For example, [trump3.txt](trump3.txt) prepends every statement made by Trump with a 'D. TRUMP: '. Training on this data would introduce heavy bias towards that phrase being a predictive feature. 

#### A Sentence-based Classifier
Though I had only gathered a few speeches from each candidate, we can get a small but manageable training and test set by learning from the sentences. In fact, this is more interesting because it allows for fine-grain features such as constitutents, parts of speech, etc. 

The [`get_data.sh`](get_data.sh) bash script takes these files and outputs an annotated set of sentences. It depends on `gnu parallel` for efficiency, but can easily be modified

#### Training and Testing


### My Commentary on the results
