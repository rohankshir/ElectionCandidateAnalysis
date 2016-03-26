# The linguistic foundations of Trump's and Hillary's speeches

### Introduction

This is a repository for Statistical Linguistic Analysis of two of the leading nominees for the 2016 Election, Donald Trump and Hillary Clinton. It is by no means limited to just these two candidates, and should someone put in the work to gather data for other candidates, the code can easily be extended. 

### Corpus
At this point, I've gathered a few speech transcripts from Trump and Clinton from the internet. Each speech is copied into a text file, with the first two lines designated for metadata. The format is as follows:
Source:[source url]
Label:[candidate last name]
<body>


The [`get_data.sh`](get_data.sh) bash script takes these files and outputs an annotated set of sentences. It depends on `gnu parallel` for efficiency, but can easily be modified
