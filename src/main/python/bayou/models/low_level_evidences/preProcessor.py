import re;
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords;
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import simplejson as json


STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

ENG_STOPWORDS = set(stopwords.words("english"));
JAVA_STOPWORDS = set([
    'abstract',   'continue',   'for',          'new',         'switch',
    'assert,'     'default',    'if',           'package',     'synchronized',
    'boolean',    'do',         'goto',         'private',     'this',
    'break',      'double',     'implements',   'protected',   'throw',
    'byte',       'else',       'import',       'public',      'throws',
    'case',       'enum',       'instanceof',   'return',      'transient',
    'catch',      'extends',    'int',          'short',       'try',
    'char',       'final',      'interface',    'static',      'void',
    'class',      'finally',    'long',         'strictfp',    'volatile',
    'const',      'float',      'native',       'super',       'while',
    'fixme', 'todo', 'and', 'bitand', 'bitor', 'bool', 'decltype',
    'template', 'typename', 'xor', 'import']);


class preProcessor():


    def __init__(self):
        # allAPIs is a dict that contains name ['name'] and description ['desc'] of each API in the dictionary
        return




    def preProcessing(self, inputString):
        """
        Code From Yanxin; courtesy GrammaTech.

        Preprocess all the terms. This step includes case splitting,
        stemming, lemmatization, etc.. This should return a list of terms
        that can be directly used for calculating TFIDF.
        """

        result_list = inputString.strip().split()

        # replace all non alphabetical char into underscore
        result_list = [re.sub("[^a-zA-Z]", '_', w) for w in result_list]

        # break the terms using underscores
        tmp_list = []
        for t in result_list:
            s = re.split("_+", t)
            tmp_list.extend(s)

        result_list = []
        for x in tmp_list:
            if len(x) > 1:
                result_list.extend(self.camelCaseSplit(x))

        # remove stop words and java keywords
        tmp_list = []
        for t in result_list:
            if t not in ENG_STOPWORDS and t not in JAVA_STOPWORDS:
                s = STEMMER.stem(t)
                # s = LEMMATIZER.lemmatize(s);
                tmp_list.append(s)

        result_list = [x for x in tmp_list if len(x) > 1]

        return result_list



    def camelCaseSplit(self, inputString):
        s = re.sub("(.)([A-Z][a-z]+)", r"\1 \2", inputString);
        s = re.sub("([a-z])([A-Z])", r"\1 \2", s).lower().split();
        return s
    #
    # def queryPreProcess(self, query):
    #     # words = self.preProcessing(query)
    #     qry_dict = defaultdict(int)
    #     for word in query:
    #         qry_dict[word] += 1 # BUG?
    #      return qry_dict
