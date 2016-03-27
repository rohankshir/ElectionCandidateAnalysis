#! /bin/bash
parallel -P 0 -k --gnu  "./transcript_reader.py" ::: *.txt  | awk 'NF>=4' | perl -MList::Util -e 'print List::Util::shuffle <>'
