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

#### Bigrams

```
-0.6696	I believe      		0.9624	We're going
-0.5899	You know,      		0.9579	We love
-0.4404	middle class   		0.8847	I love
-0.4283	We need        		0.7342	we're going
-0.4227	I want         		0.6803	It's going
-0.3908	Thank all.     		0.5944	So I
-0.3685	break barrier  		0.5363	We don't
-0.3677	family issue.  		0.5300	going happen.
-0.3367	come far       		0.5267	I mean
-0.3259	I’ve seen      		0.5244	great again.
-0.3199	small business 		0.5158	know it.
-0.3055	God bless      		0.4813	We won
-0.3036	hard work      		0.4636	We great
-0.2964	I work         		0.4609	I said,
-0.2820	we’re going    		0.4534	I tell
-0.2795	I wish         		0.4269	Thank much.
-0.2603	The middle     		0.4114	Be careful.
-0.2603	class need     		0.4066	I don't
-0.2586	story Flint.   		0.4027	going Nevada.
-0.2440	can’t other.   		0.3938	And that's
-0.2426	expect come    		0.3793	it's going
-0.2397	need raise.    		0.3786	- I
-0.2294	problem solvers,		0.3729	It wa
-0.2294	solvers, deniers.		0.3667	make country
-0.2272	pay off.       		0.3667	America great
```

#### POS Tags

```
0.806748466258
	-1.8281	NNS            		1.7925	VBP VBG
	-1.5770	NN             		1.6088	PRP VBP
	-1.1180	,              		1.5163	PRP
	-1.1043	CC JJ          		1.3601	VBZ
	-1.0756	JJ             		1.2139	PRP VBZ
	-1.0486	IN             		1.1932	VBG TO
	-1.0214	TO             		1.1488	VBG
	-0.9856	NN NN          		1.1247	VBP
	-0.9605	PRP$           		1.0923	CC PRP
	-0.9221	NNS WP         		1.0873	NN .
	-0.9137	NN TO          		1.0785	RB VB
	-0.8717	NN ,           		1.0235	RB JJ
	-0.8408	JJ RB          		0.9439	POS
	-0.8396	WRB            		0.9353	:
	-0.8103	JJR            		0.8795	MD RB
	-0.8087	VBP VBP        		0.8363	VBZ VBG
	-0.8063	IN PRP$        		0.7445	RB PRP
	-0.7982	NN NNS         		0.6739	: DT
	-0.7500	PRP DT         		0.6699	.
	-0.7437	WP             		0.6414	PRP VBD
	-0.7340	RB .           		0.6146	VBZ RB
	-0.6984	IN JJ          		0.5967	NN POS
	-0.6758	RB TO          		0.5658	VB :
	-0.6737	RB IN          		0.5398	VBD DT
	-0.6617	DT NN          		0.5338	VB .
```
	
#### POS Tags bigrams

```
0.812883435583
-1.4478	NN NN          		2.1424	PRP VBP
-1.4178	NN TO          		1.9040	VBP VBG
-1.2650	NNS IN         		1.8008	VBG TO
-1.2515	CC JJ          		1.4969	PRP VBZ
-1.2281	NN NNS         		1.3170	PRP VBD
-1.2238	NNS WP         		1.3144	VBZ VBG
-1.1461	TO VB          		1.2360	NN .
-1.1453	IN PRP$        		1.1813	RB VB
-1.0683	NNS CC         		1.1773	CC PRP
-1.0101	NN CC          		1.1333	MD RB
-0.9828	NNS ,          		1.0472	RB PRP
-0.9785	NN ,           		0.8130	RB JJ
-0.9675	IN NN          		0.8114	: PRP
-0.9639	JJ NNS         		0.8064	VBP RB
-0.8850	CC NN          		0.7879	VBZ RB
-0.7988	JJ NN          		0.7530	VBZ NN
-0.7574	PRP$ NN        		0.7049	: DT
-0.7470	WRB PRP        		0.7033	VBP DT
-0.7429	JJ DT          		0.6761	VB PRP
-0.7424	NNS DT         		0.6481	NN POS
-0.7328	JJ RB          		0.5966	CD IN
-0.7255	NNS VBP        		0.5581	VBN VBG
-0.7103	PRP$ JJ        		0.5556	VBG IN
-0.6986	IN JJ          		0.5551	RB :
-0.6876	VB IN          		0.5371	WP VBZ	
```
	
#### Words and contractions

```
0.886503067485
-1.4171	american       		2.4726	's
-1.3550	,              		1.9331	n't
-1.0977	work           		1.8876	're
-1.0911	america        		1.4828	going
-0.9339	it’s           		1.2829	-
-0.9231	need           		1.1915	love
-0.9222	new            		1.1104	.
-0.9155	family         		0.9541	'm
-0.7457	that’s         		0.8929	people
-0.6570	growth         		0.8911	great
-0.5809	–              		0.8650	?
-0.5482	worker         		0.8409	everybody
-0.5438	i’ve           		0.8347	're going
-0.5328	job            		0.7767	...
-0.5323	economy        		0.7631	've
-0.5269	president      		0.7196	know .
-0.5243	i’m            		0.7148	wa
-0.5149	make           		0.7088	'll
-0.4804	america .      		0.7018	ca n't
-0.4766	i’ll           		0.7018	ca
-0.4672	child          		0.6492	's going
-0.4566	barrier        		0.6369	thing
-0.4546	community      		0.6219	said
-0.4415	support        		0.6210	incredible
-0.4334	ahead          		0.6137	amazing
```
