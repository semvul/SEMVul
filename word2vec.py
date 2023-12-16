import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def my_tokenizer(code):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    ## Remove newlines & tabs
    code = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(/)|(\\r)','',code)
    ## Mix split (characters and words)
    splitter = '\"(.*?)\"| +|(;)|(->)|(&)|(\*)|(\()|(==)|(~)|(!=)|(<=)|(>=)|(!)|(\+\+)|(--)|(\))|(=)|(\+)|(\-)|(\[)|(\])|(<)|(>)|(\.)|({)'
    code = re.split(splitter,code)
    ## Remove None type
    code = list(filter(None, code))
    code = list(filter(str.strip, code))
    # snakecase -> camelcase and split camelcase
    code_1 = []
    for i in code:
        code_1 += convert(i).split('_')
    #filt
    code_2 = []
    for i in code_1:
        if i in ['{', '}', ';', ':']:
            continue
        code_2.append(i)
    return(code_2)


# read corpus
def create_dictionary():
    corpus = []
    with open('sard.txt', 'r', encoding='utf-8') as file:
        corpus = file.readlines()

    special_tokens = ['<pad>', '<unk>']
    corpus.extend(special_tokens)
    # create TfidfVectorizer 
    tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)

    tfidf_matrix = tfidf_vectorizer.fit_transform(tqdm(corpus))

    # get dictionary
    dictionary = tfidf_vectorizer.vocabulary_


    save_path = 'tfidf_dictionary.pkl'
    with open(save_path, 'wb') as file:
        pickle.dump(dictionary, file)

    print("字典已保存到", save_path)
    

def preprocess_sentence(sentence, min_length=10):
    special_tokens = ['<PAD>', '<UNK>']
    tokens = my_tokenizer(sentence)

    tokens_with_special = [token if token in tokens else '<UNK>' for token in tokens]

    if len(tokens_with_special) < min_length:
        tokens_with_special += ['<PAD>'] * (min_length - len(tokens_with_special))
    return tokens_with_special


def word2vec():
    sentence = []
    with open('sard.txt', 'r', encoding='utf-8') as file:
        content = file.readlines()
    for line in tqdm(content):
        line = preprocess_sentence(line)
        sentence.append(line)
        
    wvmodel = Word2Vec(sentence, min_count=1, workers=8, vector_size=50)
    
    for i in tqdm(range(5)):
        wvmodel.train(sentence, total_examples=len(sentence), epochs=1)
        
    print('Embedding Size : ', wvmodel.vector_size)
    wvmodel.save('myword2')
        
def read_dictionary():
    load_path = 'tfidf_dictionary.pkl'
    with open(load_path, 'rb') as file:
        loaded_dictionary = pickle.load(file)
        
    print(len(loaded_dictionary))
    return loaded_dictionary

# if __name__ == '__main__':
#     # dictionary = read_dictionary()
#     # print(dictionary)
    
    # word2vec()

create_dictionary()
word2vec_model = Word2Vec.load('myword2')
print(len(word2vec_model.wv.key_to_index))

