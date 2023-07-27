import networkx as nx
import numpy as np
import argparse
import os
import sent2vec
import pickle
import glob
from multiprocessing import Pool
from functools import partial


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-c', '--code', help='The path of a dir which consists of dot_files\' source code',
                        required=True)
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-m', '--model', help='The path of model.', required=True)
    args = parser.parse_args()
    return args


def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def image_generation(dot, code):
    try:
        pdg = graph_extraction(dot)
        lines = []
        with open(code) as f:
            content = f.readline()
            while content:
                if content.find("static void") != -1:
                    print(content)
                    content = content.replace("static void", "void")
                    print(content)
                lines.append(content)
                content = f.readline()

        labels_dict = nx.get_node_attributes(pdg, 'label')
        vector_code = []
        for label, all_code in labels_dict.items():

            lineNum = int(all_code[all_code.index("<SUB>") + 5:all_code.index("</SUB>")])
            line_vec = sentence_embedding(lines[lineNum - 1].strip())
            line_vec = np.array(line_vec)
            vector_code.append(line_vec)
        pdgEdges = nx.get_edge_attributes(pdg, 'label')
        cdg = nx.DiGraph()
        cdg.add_nodes_from(pdg.nodes())
        ddg = nx.DiGraph()
        ddg.add_nodes_from(pdg.nodes())
        for each in pdgEdges:
            if "CDG" in pdgEdges[each]:
                cdg.add_edge(each[0], each[1])
            elif "DDG" in pdgEdges[each]:
                ddg.add_edge(each[0], each[1])
        cdg_in = np.array(nx.adjacency_matrix(cdg).toarray())
        for i in range(len(cdg_in)):
            cdg_in[i][i] = 1
        ddg_in = np.array(nx.adjacency_matrix(ddg).toarray())
        for i in range(len(ddg_in)):
            ddg_in[i][i] = 1
        return vector_code, cdg_in, ddg_in
    except Exception as e:
        print(str(e))
        return None


def write_to_pkl(dot, code_path, out, existing_files):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    codefile = code_path + dot_name + ".c"
    if dot_name in existing_files:
        return None
    else:
        print(dot_name)
        channels = image_generation(dot, codefile)
        if channels == None:
            return None
        else:
            (vector_code, cdg_in, ddg_in) = channels
            out_pkl = out + dot_name + '.pkl'
            data = [vector_code, cdg_in, ddg_in]
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)


def main():
    args = parse_options()
    dir_name = args.input
    code_path = args.code
    out_path = args.out
    trained_model_path = args.model

    global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(trained_model_path)

    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    if code_path[-1] == '/':
        code_path = code_path
    else:
        code_path += "/"
    dotfiles = glob.glob(dir_name + '*.dot')

    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    existing_files = glob.glob(out_path + "/*.pkl")
    existing_files = [f.split('.pkl')[0] for f in existing_files]
    pool = Pool(10)
    pool.map(partial(write_to_pkl, code_path=code_path, out=out_path, existing_files=existing_files), dotfiles)

    sent2vec_model.release_shared_mem(trained_model_path)


if __name__ == '__main__':
    main()
