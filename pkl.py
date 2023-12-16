import os
import re
import glob
import pickle
import argparse
import numpy as np
import networkx as nx
from functools import partial
from multiprocessing import Pool
from word2vec import my_tokenizer, word2vec_model


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', default='devign/pdgs/Vul/')
    parser.add_argument('-o', '--out', default='outputs/Vul/')
    parser.add_argument('-c', '--code', default='Vul')
    args = parser.parse_args()
    return args
    

def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph

def preprocess_sentence(sentence, fixed_length=30):
    special_tokens = ['<PAD>', '<UNK>']
    tokens = my_tokenizer(sentence)  

    tokens_with_special = [token if token in tokens else '<UNK>' for token in tokens]

    if len(tokens_with_special) < fixed_length:
        tokens_with_special += ['<PAD>'] * (fixed_length - len(tokens_with_special))
    else:
        tokens_with_special = tokens_with_special[:fixed_length]
    
    tokens_numerical = [word2vec_model.wv.key_to_index.get(token, len(word2vec_model.wv.key_to_index)+2) for token in tokens_with_special]
    
    return tokens_numerical


def modify_array(arr):
    current_shape = arr.shape
    
    if current_shape[0] < 120 or current_shape[1] < 120:
        new_arr = np.full((120, 120), 0)
        new_arr[:current_shape[0], :current_shape[1]] = arr
    else:
        new_arr = arr[:120, :120]

    return new_arr


def image_generation(dot, source):
    numeric_sequences = []
    max_token = 30
    max_line = 120
    assert os.path.basename(dot).split('.')[0] == os.path.basename(source).split('.')[0], 'dont misk file'
    try:
        with open(source) as file:
            lines = file.readlines()
        pdg = graph_extraction(dot)
        line_nums = sorted(set([int(i) for i in list(pdg.nodes)]))
        
        for num in line_nums[:max_line]:
            content = lines[num - 1].strip()
            content = content.replace("static void", "void")
            sentence = preprocess_sentence(content)
            numeric_sequences.append(sentence)

        for idx in range(max_line - len(line_nums)):
            numeric_sequences.append([0] * max_token)
        
        pdg_edage = nx.get_edge_attributes(pdg, 'label')
        cdg = nx.DiGraph()
        cdg.add_nodes_from(pdg.nodes)
        ddg = nx.DiGraph()
        ddg.add_nodes_from(pdg.nodes)

        for each, value in pdg_edage.items():
            if "DDG" in pdg_edage[each] and each[0] != each[1]:
                cdg.add_edge(each[0], each[1])
            elif "CDG" in pdg_edage[each] and each[0] != each[1]:
                ddg.add_edge(each[0], each[1])
                
        cdg_out = np.array(nx.adjacency_matrix(cdg).toarray())
        ddg_out = np.array(nx.adjacency_matrix(ddg).toarray())
        
        for i in range(len(cdg_out)):
            cdg_out[i][i] = 1
        for i in range(len(ddg_out)):
            ddg_out[i][i] = 1
            
        cdg_out = modify_array(cdg_out)
        ddg_out = modify_array(ddg_out)
        
        pdgs = np.array([cdg_out, cdg_out.T, ddg_out, ddg_out.T])
        input = np.array(numeric_sequences)
        
        return [input, pdgs]
    
    except Exception as e:
        print(e)
        return None


def write_to_pkl(dot, code_path, out, existing_files):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    codefile = code_path + dot_name + ".c"
    if dot_name in existing_files:
        return None
    else:
        print(os.path.basename(dot_name), os.path.basename(codefile))
        channels = image_generation(dot, codefile)
        if channels is None:
            return None
        else:
            (input, pdgs) = channels
            out_pkl = out + dot_name + '.pkl'
            data = [input, pdgs]
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)


def main():
    args = parse_options()
    dir_name = args.input
    code_path = args.code
    out_path = args.out
    
    

    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    if code_path[-1] == '/':
        code_path = code_path
    else:
        code_path += "/"
    dotfiles = sorted(glob.glob(dir_name + '/*.dot'))

    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    existing_files = sorted(glob.glob(out_path + "/*.pkl"))
    existing_files = [f.split('.pkl')[0] for f in existing_files]
    pool = Pool(10)
    pool.map(partial(write_to_pkl, code_path=code_path, out=out_path, existing_files=existing_files), dotfiles)

if __name__ == '__main__':
    main()
    # image_generation('func.dot', 'func.c')
