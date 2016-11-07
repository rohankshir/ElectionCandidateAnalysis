import dill as pickle
from feature_extractor import *
import numpy as np
import scipy

class Predictor:
    def __init__(self,
                 model_fname,
                 feature_vectorizer_fname):
        self.clf = pickle.load(open(model_fname, 'rb'))
        self.vectorizers = pickle.load(open(feature_vectorizer_fname, 'rb'))
        self.feature_names = self.vectorizers[0].vocabulary_
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
    
            

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('featurevecs_file')
    args = parser.parse_args()
    predictor = Predictor(args.model_file, args.featurevecs_file)
    print predictor.predict("We are the greatest country in the world")
        
        
if __name__ == "__main__":
    main()
    
