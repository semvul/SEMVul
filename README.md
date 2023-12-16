## the implementation of semvul



## Dataset
1. sard
2. nvd
3. devign

## Source Code

#### Step 1: Code normalization
Normalize the code with normalization.py 
```
python ./normalization.py -i ./data+ 'your dataset'
```
#### Step 2: Generate pdgs with the help of joern
Prepare the environment refering to: [joern](https://github.com/joernio/joern) you can try the version between 1.1.995 to 1.1.1125
```
# first generate .bin files
python joern_graph_gen.py  -i ./data/'your dataset'/Vul -o ./data/'your dataset'/bins/Vul -t parse
python joern_graph_gen.py  -i ./data/'your dataset'/No-Vul -o ./data/'your dataset'/bins/No-Vul -t parse


# then generate pdgs (.dot files)
python joern_graph_gen.py  -i ./data/'your dataset'/bins/Vul -o ./data/'your dataset'/pdgs/Vul -t export -r pdg
python joern_graph_gen.py  -i ./data/'your dataset'/bins/Vul -o ./data/'your dataset'/pdgs/No-Vul -t export -r pdg
```

#### Step 3: modify dot file
、、、
python changeline.py 
、、、

#### Step 4: get a pre-trained word2vec
##### step4.1 extract the corpus from the source code files.
、、、
python read_file.py
、、、
##### step4.2 you need to use the create_dictionary() function in word2vec.py while commenting out the other functions.
、、、
python word2vec.py
、、、

##### step 4.3 you need to use the  Word2Vec.load() function in word2vec.py while commenting out the other functions.
、、、
python word2vec.py
、、、

#### Step5: Generate pkl file for each dot file
、、、
python pkl.py -i ./data/'your dataset'/pdgs/No-Vul -o ./data/'your dataset'/outputs/No-Vul -c ./data/'your dataset'/No-Vul
python pkl.py -i ./data/'your dataset'/pdgs/Vul -o ./data/'your dataset'/outputs/Vul -c ./data/'your dataset'/Vul
、、、

#### Step 6: split train pkl and test pkl
```
python split_train_test.py -i ./data/'your dataset'/pkl/ -o ./data/'you dataset'/pkl
```
#### Step 6: train the model
```
# n denotes the number of kfold, i.e., n=10 then the training set and test set are divided according to 9:1 and 10 sets of experiments will be performed
python generate_train_test_data.py -i ./data/sard/outputs -o ./data/sard/pkl -n 5
```

# We will proceed with further updates, which include providing the code link, along with details regarding file execution.
