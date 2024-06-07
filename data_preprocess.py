import gzip
import random
import re
import os
import html
from tqdm import tqdm
import numpy as np
import json
from gensim.models import word2vec, Word2Vec
from sklearn import decomposition
import jsonlines
from collections import Counter
from itertools import chain


def create_cpg(source_dir,root_dir,data_dir):
    if os.path.exists(data_dir):
        print("data already exists.")
        return
    print("generating data...")
    os.mkdir(data_dir)
    for root, dirs, files in os.walk(source_dir):
        for file in tqdm(files):
            cpg = root_dir+"/"+"cpg"+"/"+file+"/"+'1-cpg.dot'
            code_file = source_dir + "/"+file
            generate_data(cpg,code_file,data_dir+"/"+file.split('.')[0]+'.json')


def generate_data(cpg_path,code_path,output_path):

    with open(cpg_path) as f_g,open(code_path) as f_e,open(output_path, 'w') as w:
        tmp_dirs = {}
        label = code_path.split('/')[-1][-3]
        tmp_dirs["label"] = int(label)
        codes = f_e.readlines()
        code = ""
        for c in codes:
            code += c
        tmp_dirs["code"] = code
        dot_graphs = f_g.readlines()
        cpg_list = []
        dot_list = []
        Title = re.findall("digraph \"(.*)\" \{  ",dot_graphs[0])[0]
        if Title not in codes[0]:
            print("\n",output_path)
            os.remove(output_path)
            return
        for dot_graph in dot_graphs:
            dot_graph = html.unescape(dot_graph)
            side = re.findall("\"(\d+)\" -> \"(\d+)\"", dot_graph)
            if side:
                type = re.findall("label = \"(\w+): ", dot_graph)[0]
                if type == "AST":
                    fintype = 0
                elif type == "CFG":
                    fintype = 1
                elif type == "DDG" or type =="CDG":
                    fintype = 2
                (begin, end, type) = (int(side[0][0]), int(side[0][1]), fintype)
                cpg_list.append((begin,end,type))
            else:
                id = re.findall("\"(\d+)\"", dot_graph)
                if id:
                    id = id[0]
                CodeFeature = re.findall("label = <(\(.+?\)*)<SUB>", dot_graph)
                if CodeFeature:
                    CodeFeature = CodeFeature[0]
                    CodeFeature =CodeFeature[1:-1]
                    listCF = CodeFeature.split(",")
                    if len(listCF) >= 3:
                        tmp_code = CodeFeature.replace(listCF[0] + ',', "")
                        find_or_not = re.search("\S,\S",tmp_code)
                        if find_or_not:
                            fin_Code = tmp_code[:int(find_or_not.span()[0])+1]
                        else:
                            fin_Code = tmp_code
                    else:
                        fin_Code = listCF[1]
                    Feature = listCF[0]
                    code = fin_Code
                    dot_type = 0
                    for c in codes:
                        code_for_compare = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", "", c )
                        code_for_compare1 = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", "", code )
                        if code_for_compare and code_for_compare1:
                            if code_for_compare in code_for_compare1:
                                dot_type = 1
                    dot_list.append((int(id),code,Feature,dot_type))
        tmp_dirs["cpg"] = cpg_list
        tmp_dirs["dot"] = dot_list
        json_str = json.dumps(tmp_dirs,indent=4)
        w.write(json_str)


def preproess(data_dir, output_path,dataset):
    W2v(data_dir, output_path,dataset)
    if not os.path.exists(output_path + "/"+ dataset):
        os.makedirs(output_path + "/"+ dataset)
    CPGgraph = output_path + "/"+ dataset+"/"+"CPG.jsonl"
    if os.path.exists(CPGgraph):
        print("data is ready")
        Split(0.8, 0.1, 0.1, CPGgraph)
    else:
        GetInfor(data_dir,output_path,dataset)
        Split(0.8, 0.1, 0.1, CPGgraph)


def Split(train, valid, test, path):
    print("\nbegin to split files...")
    tem_cpg = []
    with open(path, "r+", encoding="utf8") as f:
        for item in tqdm(jsonlines.Reader(f)):
            tem_cpg.append(item)

    randomtem = np.arange(len(tem_cpg))
    vul_list = []
    nonvul_list = []

    for step, i in enumerate(randomtem):
        ast_and_cdfg = tem_cpg[i]
        if ast_and_cdfg['Property'] == 0:
            nonvul_list.append(ast_and_cdfg)
        else:
            vul_list.append(ast_and_cdfg)
    len_vul = len(vul_list)
    len_nonvul = len(nonvul_list)
    ratio = len_nonvul / len_vul
    print("vul: " + str(len_vul) + " non_vul: " + str(len_nonvul) + " ratio: " + str(ratio))
    vul_partition = int(train * len_vul)
    train_list = vul_list[:vul_partition] + nonvul_list[:vul_partition]
    random.shuffle(train_list)
    non_vul_valid = int(int(0.1 * len_vul) * ratio)
    valid_list = vul_list[vul_partition:vul_partition + int(0.1 * len_vul)] + nonvul_list[
                                                                              vul_partition:vul_partition + non_vul_valid]
    random.shuffle(valid_list)
    test_list = vul_list[vul_partition + int(0.1 * len_vul):] + nonvul_list[
                                                                vul_partition + non_vul_valid:vul_partition + non_vul_valid + non_vul_valid]

    print(
        "train: " + str(len(train_list)) + "\tvalid:" + str(len(valid_list)) + "\ttest: " + str(len(test_list)) + "\t")
    path_tem = os.path.split(path)[0] + '/temp.jsonl'
    for i in tqdm(train_list):
        with jsonlines.open(path_tem,
                            mode='a') as writer:
            writer.write(i)
    f_in = open(path_tem, 'rb')
    f_out = gzip.open(os.path.split(path)[0] + '/train.jsonl.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    os.remove(path_tem)
    print("train ready")

    for i in tqdm(valid_list):
        with jsonlines.open(path_tem,
                            mode='a') as writer:
            writer.write(i)
    f_in = open(path_tem, 'rb')
    f_out = gzip.open(os.path.split(path)[0] + '/valid.jsonl.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()


    os.remove(path_tem)
    print("valid ready")

    for i in tqdm(test_list):
        with jsonlines.open(path_tem,
                            mode='a') as writer:
            writer.write(i)

    # zip
    f_in = open(path_tem, 'rb')
    f_out = gzip.open(os.path.split(path)[0]  + '/test.jsonl.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    os.remove(path_tem)
    print("test ready")


def GetInfor(filename,output_path,dataset):
    print("\ngenerating vectors files...")
    for root, dirs, files in os.walk(filename):
        np.random.shuffle(files)
        model = word2vec.Word2Vec.load(output_path + "/word2vec_"+dataset+"_code" + '.pkl')
        model_1 = word2vec.Word2Vec.load(output_path + "/word2vec_"+dataset+"_Feature" + '.pkl')
        for file in tqdm(files):
            with open(root + "/" + file, "r") as f:
                content = f.read()
                data = json.loads(content)
                Property = data["label"]
                nodes = data["dot"]
                if len(nodes) < 1:
                    continue
                bias = data["dot"][0][0]
                vectors_nodes = []
                adjacency_lists = []
                nodes_class = []
                for edge in data["cpg"]:
                    adjacency_lists.append([edge[0]-bias,edge[1]-bias,edge[2]])
                for node in nodes:
                    node_id = node[0]
                    node_code = node[1]
                    node_feature = node[2]
                    nodes_class.append(node[3])
                    try:
                        vector = model.wv[node_code]
                        vector_1 = model_1.wv[node_feature]
                    except:
                        vector = np.zeros(100)
                        vector_1 = np.zeros(100)
                    if np.size(vector) > 100:
                        vector = vector.T
                        kpca = decomposition.KernelPCA(kernel='rbf', gamma=10, n_components=1)
                        vector = kpca.fit_transform(vector).T
                        vector_1 = kpca.fit_transform(vector).T
                    if len(vector) != 100:
                        print(file)
                    if len(vector) == 0:
                        print(file)
                        print(node_id)
                    feature_list  = vector.tolist()+vector_1.tolist()
                    vectors_nodes.append(feature_list)
            flag = 0
            for edge in adjacency_lists:
                if edge[0]>(len(vectors_nodes)-1) or edge[1]>(len(vectors_nodes)-1):
                    flag = 1
                    break
            if flag == 1:
                continue
            with jsonlines.open(output_path +"/"+dataset+'/CPG.jsonl',mode='a') as writer:
                writer.write(
                        {"Property": Property,
                         "graph_cdfg": {"node_features": vectors_nodes,"adjacency_lists": adjacency_lists,"nodes_class": nodes_class}})

def W2v(data_dir,output_path,dataset):
    if os.path.exists(output_path + "/" + "word2vec_"+dataset+"_code"+ ".pkl") and os.path.exists(output_path + "/" + "word2vec_"+dataset+"_Feature"+ ".pkl"):
        print("Word2Vec already exists.")
        return
    # if not,...
    if not os.path.exists(output_path + "/" + "word2vec_"+dataset+"_code"+ ".pkl"):
        print("generating word2vec_"+dataset+"_code...")
        words = []
        for root, dirs, files in os.walk(data_dir):
            # shuffle files randomly
            np.random.shuffle(files)
            for file in tqdm(files):
                with open(root+"/"+file,"r") as f:
                    content = f.read()
                    data = json.loads(content)
                    code_word = collect_code_data(data)
                    words.append(code_word)
        print("The vocablary is ready")
        model = Word2Vec(words, min_count=1, vector_size=100, sg=1, window=5,
                         negative=3, sample=0.001, hs=1, workers=4)
        model.save(output_path + "/" + "word2vec_"+dataset+"_code" + ".pkl")
        print("word2vec_"+dataset+"_code is ready")
    if not os.path.exists(output_path + "/" + "word2vec_"+dataset+"_feature"+ ".pkl"):
        print("generating word2vec_"+dataset+"_Feature...")
        Feature = []
        for root, dirs, files in os.walk(data_dir):
            # shuffle files randomly
            np.random.shuffle(files)
            for file in tqdm(files):
                with open(root+"/"+file,"r") as f:
                    content = f.read()
                    data = json.loads(content)
                    Feature_word = collect_Feature_data(data)
                    Feature.append(Feature_word)
        print("The vocablary is ready")
        list_Feature = list(chain.from_iterable(Feature))
        print("The num of Feature is ",len(Counter(list_Feature)))
        model = Word2Vec(Feature, min_count=1, vector_size=100, sg=1, window=5,
                         negative=3, sample=0.001, hs=1, workers=4)
        model.save(output_path + "/" + "word2vec_"+dataset+"_Feature" + ".pkl")
        print("word2vec_"+dataset+"_Feature is ready")

def collect_code_data(data):
    big_code = []
    for dot in data["dot"]:
        big_code.append(dot[1])
    return big_code


def collect_Feature_data(data):
    big_Feature = []
    for dot in data["dot"]:
        big_Feature.append(dot[2])
    return big_Feature

if __name__ == "__main__":
    dataset = "Reveal" # Reveal,FFmp,Fan
    source_dir = 'data/raw_code_'+dataset
    root_dir = 'data/parse_code_'+dataset
    data_dir = 'data/data_'+dataset
    output_path = "data/fin_code_"+dataset
    create_cpg(source_dir,root_dir,data_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    preproess(data_dir, output_path,dataset)




