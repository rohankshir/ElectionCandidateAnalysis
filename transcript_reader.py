#! /Users/rohan/miniconda/bin/python
import fileinput
from nltk.tokenize import sent_tokenize
import pprint
import sys

def read_source(line):
    return line.split(':')[1].strip()

def convert_label(s):
    if s == "Trump":
        return 1
    return 0



reload(sys)
sys.setdefaultencoding('utf-8')

c = 0
source_url = None
sentences = []
for line in fileinput.input():
    line = line.decode('utf-8')
    if c == 0:
        source_url = read_source(line)
    if c == 1:
        label = convert_label(read_source(line))
    else:
        sent_tokenize_list = sent_tokenize(line)
        sentences.extend(sent_tokenize_list)
    c += 1

for sentence in sentences:
    if not sentence:
        pass
    else:
        print str(label) + "\t" + sentence.strip()
