import argparse
from train_test import load_data, TRAIN_TEST


def parse_options():
    parser = argparse.ArgumentParser(description='VulCNN training.')
    parser.add_argument('-i', '--input', help='The dir path of train.pkl and test.pkl', type=str, required=True)
    args = parser.parse_args()
    return args


def get_kfold_data(pathname="", item_num=0):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + "train.pkl")[item_num]
    eval_df = load_data(pathname + "test.pkl")[item_num]

    return train_df, eval_df


def main():
    args = parse_options()
    hidden_size = 200  # the dim of sent2vec
    data_path = args.input
    for item_num in range(1):
        train_df, eval_df = get_kfold_data(pathname=data_path, item_num=item_num)

        classifier = TRAIN_TEST(item_num=item_num, epochs=100, hidden_size=hidden_size,
                                data_list=[train_df['data'], train_df['label'], eval_df['data'], eval_df['label']])

        classifier.train(itemnum=item_num, savepath='./model_save')
        # classifier.eval(loadfile='')  # test only

