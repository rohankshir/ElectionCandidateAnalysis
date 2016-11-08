import dill as pickle
from feature_extractor import *
import numpy as np
import scipy

class Predictor:
    def __init__(self,
                 model_fname,
                 lexical_vocab_pkl,
                 pos_vocab_pkl):
        self.clf = pickle.load(open(model_fname, 'rb'))
        lex_vocab = pickle.load(open(lexical_vocab_pkl,'rb'))
        pos_vocab = pickle.load(open(pos_vocab_pkl,'rb'))
        lexical_vectorizer = get_lexical_vectorizer(lex_vocab)

        pos_vectorizer = get_pos_vectorizer(pos_vocab)
        arbitrary_vectorizer = ArbitraryFeaturesVectorizer()
        
        self.vectorizers = [lexical_vectorizer, pos_vectorizer, arbitrary_vectorizer]
        self.feature_names = self.vectorizers[0].vocabulary
        self.feature_index = {v: k for k, v in self.feature_names.items()}
        print "initialized"

    def predict(self,
                sentence):
        sentence = [sentence]
        features = []
        for vectorizer in self.vectorizers:
            f = vectorizer.transform(sentence)
            if scipy.sparse.issparse(f):
                f = f.toarray()
            features.append(f)
        x = np.hstack(features)
        y = self.clf.predict(x)[0]
        y_prob = self.clf.predict_proba(x)[0]
        relevant_feats = print_features(x, self.feature_index)
        return y, y_prob, relevant_feats

    def predict_x(self,
                  x):
        return self.clf.predict(x)[0]
            

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('lexical_vocab_file')
    parser.add_argument('pos_vocab_file')
    args = parser.parse_args()
    predictor = Predictor(args.model_file, args.lexical_vocab_file, args.pos_vocab_file)
    print predictor.predict("We are the greatest country in the world")

        
if __name__ == "__main__":
    main()
    
