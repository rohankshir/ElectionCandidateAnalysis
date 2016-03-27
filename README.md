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

##### Feature Engineering
* lowercase vs not
* nltk tokenization vs split
* unigrams vs bigrams vs trigrams
* stop words
* default vs stemming vs lemmatization
* train and test set
* Count vectorizer vs tfidf vectorizer
* cleaning up tokens 
** removing repeated signs (--- becomes -)

### My Commentary on the results
#### Lowercase
Hillary                     Trump
```
-1.1429	american       		1.8230	it's
-1.1382	work           		1.6675	-
-1.0505	it’s           		1.6211	we're
-1.0047	new            		1.5885	going
-0.9685	need           		1.1919	love
-0.8472	family         		1.0632	don't
-0.7279	america        		0.9043	can't
-0.6547	i’ve           		0.8727	i'm
-0.6416	–              		0.8562	it.
-0.6214	i’m            		0.7983	we're going
-0.6050	that’s         		0.7570	great
-0.5558	child          		0.7239	they're
-0.5312	growth         		0.7002	just
-0.5309	economy        		0.6506	really
-0.5268	story          		0.6406	wa
-0.5058	job            		0.6123	people
-0.4776	support        		0.5821	that's
-0.4746	community      		0.5769	everybody
-0.4620	i’ll           		0.5505	said,
-0.4556	we’re          		0.5498	win
-0.4503	let’s          		0.5484	...
-0.4346	year           		0.5338	them.
-0.4317	tax            		0.5319	right?
-0.4276	help           		0.5297	it's going
-0.4203	worker         		0.5294	he's
```	
#### Uppercase
Hillary                     Trump
```
-1.0199	work           		1.4873	going
-0.8936	need           		1.4056	-
-0.8508	Americans      		1.3148	it's
-0.7905	family         		1.0264	love
-0.7696	America        		1.0181	great
-0.6528	new            		1.0001	don't
-0.6388	–              		0.9546	We're
-0.6136	President      		0.9063	wa
-0.6107	It’s           		0.8414	we're
-0.6000	help           		0.8312	I'm
-0.5922	economy        		0.8298	So
-0.5888	That’s         		0.7854	can't
-0.5743	pay            		0.7852	It's
-0.5689	job            		0.7796	it.
-0.5421	New            		0.7629	I
-0.5103	child          		0.7029	...
-0.4857	it’s           		0.6664	people
-0.4745	I’ll           		0.6440	right?
-0.4724	support        		0.6269	He
-0.4653	American       		0.6007	win
-0.4602	For            		0.5967	really
-0.4590	worker         		0.5913	We
-0.4585	make           		0.5866	that.
-0.4578	strong         		0.5855	And
-0.4544	u              		0.5469	I love
```
