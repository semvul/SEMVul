import os
import torch
from torch import nn
from tqdm import tqdm
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from transformers import AdamW, get_linear_schedule_with_warmup
from mymodel import DFMCNN
from mydata import load_for_model
from tools import *


class TRAIN_TEST:
    def __init__(self, max_len=125, n_classes=2, epochs=100, batch_size=64, data_list=[], learning_rate=2e-4,
                 item_num=0, hidden_size=128):
        self.item_num = item_num
        self.model = DFMCNN(max_len, n_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        self.hidden_size = hidden_size
        self.train_loader, self.valid_loader = load_for_model(data_list)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=len(self.train_loader) * self.epochs)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()

            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))
            labels += list(np.array(targets.cpu()))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for _, data in progress_bar:
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                # index = data['index']
                outputs = self.model(vectors)
                # outputs = self.model(vectors, index)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))

                losses.append(loss.item())
                progress_bar.set_description(
                    f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.valid_set)
        print("val_acc : ", val_acc)
        score_dict = get_MCM_score(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict

    def train(self, loadfile=None, savepath=None):
        if loadfile is not None:
            self.model.load_state_dict(torch.load(loadfile))
        learning_record_dict = {}
        train_table = PrettyTable(['typ', 'epo', 'loss', 'recall', 'precision', 'f1', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'recall', 'precision', 'f1', 'ACC'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(
                ["tra", str(epoch + 1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if
                                                                      j != "MCM"])
            print(train_table)

            val_loss, val_score = self.eval()
            test_table.add_row(
                ["val", str(epoch + 1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
            print(test_table)
            print("\n")
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss,
                                           "train_score": train_score, "val_score": val_score}
            print("\n")
            if savepath is not None:
                if epoch % 1 == 0:
                    savefilename = savepath + "/model_parameter_no" + str(self.itemnum) + "_epoch" + str(epoch) + ".pkl"
                    torch.save(self.model.state_dict(), savefilename)
