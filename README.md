This is the execution code for SemVulCNN. We will further maintain it once our work is accepted.

## Dataset
The complete SARD dataset is referenced from Vulcnn: https://github.com/CGCL-codes/VulCNN.

## Source Code

#### Step 1: Code normalization
Normalize the code with normalization.py (This operation will overwrite the data file, please make a backup)
```
python ./normalization.py -i ./data/sard
```
#### Step 2: Generate pdgs with the help of joern
Prepare the environment refering to: [joern](https://github.com/joernio/joern) you can try the version between 1.1.995 to 1.1.1125
```
# first generate .bin files
python joern_graph_gen.py  -i ./data/sard/Vul -o ./data/sard/bins/Vul -t parse
python joern_graph_gen.py  -i ./data/sard/No-Vul -o ./data/sard/bins/No-Vul -t parse


# then generate pdgs (.dot files)
python joern_graph_gen.py  -i ./data/sard/bins/Vul -o ./data/sard/pdgs/Vul -t export -r pdg
python joern_graph_gen.py  -i ./data/sard/bins/Vul -o ./data/sard/pdgs/No-Vul -t export -r pdg
```
#### Step 3: Train a sent2vec model
Refer to [sent2vec](https://github.com/epfml/sent2vec#train-a-new-sent2vec-model)
```
./fasttext sent2vec -input ./data/data.txt -output ./data/data_model -minCount 8 -dim 128 -epoch 9 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000 -maxVocabSize 750000 -numCheckPoints 10
```
(For convenience, we share a simple sent2vec model [here|baidu](https://pan.baidu.com/s/1i4TQP8gSk5_0WlD34yDHwg?pwd=6666) or [here|google](https://drive.google.com/file/d/1p4X4PH9tqFbKByTHGnUiIwtjvmYL8VsL/view?usp=share_link) trained by using our sard dataset. If you want to achieve better performance of VulCNN, you'd better train a new sent2vec by using larger dataset such as Linux Kernel.)

#### Step 4: Generate images from the pdgs
Generate Images from the pdgs with ImageGeneration.py, this step will output a .pkl file for each .dot file.
```
python ImageGeneration.py -i ./data/sard/pdgs/Vul -o ./data/sard/outputs/Vul -m ./data/data_model.bin
python ImageGeneration.py -i ./data/sard/pdgs/No-Vul -o ./data/sard/outputs/No-Vul  -m ./data/data_model.bin
```
#### Step 5: Integrate the data and divide the training and testing datasets
Integrate the data and divide the training and testing datasets with generate_train_test_data.py, this step will output a train.pkl and a test.pkl file.
```
# n denotes the number of kfold, i.e., n=10 then the training set and test set are divided according to 9:1 and 10 sets of experiments will be performed
python generate_train_test_data.py -i ./data/sard/outputs -o ./data/sard/pkl -n 5
```
#### Step 6: Train with CNN
```
python VulCNN.py -i ./data/sard/pkl
```

## Publication
Yueming Wu, Deqing Zou, Shihan Dou, Wei Yang, Duo Xu, and Hai Jin.
2022. VulCNN: An Image-inspired Scalable Vulnerability Detection System.
In 44th International Conference on Software Engineering (ICSE ’22), May
21–29, 2022, Pittsburgh, PA, USA. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3510003.3510229

If you use our dataset or source code, please kindly cite our paper:
```
@INPROCEEDINGS{vulcnn2022,
  author={Wu, Yueming and Zou, Deqing and Dou, Shihan and Yang, Wei and Xu, Duo and Jin, Hai},
  booktitle={2022 IEEE/ACM 44th International Conference on Software Engineering (ICSE)}, 
  title={VulCNN: An Image-inspired Scalable Vulnerability Detection System}, 
  year={2022},
  pages={2365-2376},
  doi={10.1145/3510003.3510229}}
```
