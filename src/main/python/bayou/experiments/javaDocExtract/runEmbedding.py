from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from javaDocExt import extract_jD
import os
import argparse
import sys
# define training data
log =  "log/"

def extract_evidence(clargs):
    # clargsinputFile = 'DATA-Licensed_test.json'

    sentences = extract_jD(clargs.input_file[0])

    # sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
    # 			['this', 'is', 'the', 'second', 'sentence'],
    # 			['yet', 'another', 'sentence'],
    # 			['one', 'more', 'sentence'],
    # 			['and', 'the', 'final', 'sentence']]

    # train model
    model = Word2Vec(sentences, min_count=5, size=256)
    # fit a 2d PCA model to the vectors
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
    	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()
    pyplot.savefig(os.path.join(os.getcwd(), log + "jDocEmb.jpeg"), bbox_inches='tight')

    model.save(log + 'model.bin')
    # load model
    # new_model = Word2Vec.load(log + 'model.bin')
    # print(new_model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')


    clargs = parser.parse_args()
    #['/home/rm38/Research/Bayou_Code_Search/Corpus/LicensedData/DATA-Licensed_test.json'])

    sys.setrecursionlimit(clargs.python_recursion_limit)
    extract_evidence(clargs)
