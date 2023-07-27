# SEMVul

Now, let's proceed with the steps to achieve the desired outcomes:

## Prerequisites

Before getting started, ensure you have the required data and models ready:
- data normalization
- PDG Data: Generate PDGs for both vulnerable and non-vulnerable code.
- Pre-trained Sent2Vec Model
For guidance on data normalization, PDG generation, and Sent2Vec training, you can refer to the VulCNN project at the following link: [VulCNN on GitHub](https://github.com/CGCL-codes/VulCNN/tree/main).

## Step 1: Generating Images from PDG

To generate images from PDG, execute the following Python script:

```bash
python myimage.py -i ./data/sard/pdgs/Vul -o ./data/sard/outputs/Vul -m ./data/data_model.bin
python myimage.py -i ./data/sard/pdgs/No-Vul -o ./data/sard/outputs/No-Vul -m ./data/data_model.bin
```

## Step 2: Data Set Splitting

Split the data set into training and testing sets using the following Python script:

```bash
python split_train_test_data.py -i ./data/sard/outputs -o ./data/sard/pkl -n 5
```

## Step 3: Model Training

Train the model using the provided Python script:

```bash
python VulCNN.py -i ./data/three.pkl
```

## Optional Step: Visualization

If desired, visualize the results after training with the following additional steps:

1. Uncomment line 38 in `mydata.py`.
2. Uncomment lines 76 and 78, and comment line 77 in `train_test.py`.
3. Uncomment lines 106 and 107 in `mymodel.py`.
4. Uncomment line 30 and comment line 31
5. now, you can:
```bash
python VulCNN.py -i ./data/three.pkl
```

## Conclusion

By following these steps and referring to the VulCNN project, you will be able to perform data normalization, generate PDGs, and train the Sent2Vec model effectively. For more details and customization, please explore the VulCNN project on GitHub.

## License

This project is licensed under the [License Name](LICENSE).
