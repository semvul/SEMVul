import os
import torch
import numpy as np
from tqdm import tqdm
from model2 import combined
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from data import TraditionalDataset, single_dataset
from torch.utils.data import DataLoader
from tools import  get_MCM_score, sava_data
from transformers import AdamW, get_linear_schedule_with_warmup


class CNN_Classifier:
    def __init__(self, max_len=100, n_classes=2, epochs=200, batch_size=128, learning_rate=1e-4,
                 result_save_path="", item_num=0, hidden_size=128):
        self.model = combined
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        self.hidden_size = hidden_size
        result_save_path = result_save_path + "/" if result_save_path[-1] != "/" else result_save_path
        if not os.path.exists(result_save_path): os.makedirs(result_save_path)
        self.result_save_path = result_save_path + str(item_num) + "_epo" + str(epochs) + "_bat" + str(
            batch_size) + ".result"

    def preparation(self, x_train,  y_train, x_valid, y_valid):
        # create datasets
        self.train_set = TraditionalDataset(x_train, y_train)
        self.valid_set = TraditionalDataset(x_valid, y_valid)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            input = data["input"].to(self.device)
            pdg = data['pdg'].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(input, pdg)
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
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        parameter = torch.load('model/model_parameter_no_epoch22.pkl')
        self.model = combined
        self.model.load_state_dict(parameter)
        self.model = self.model.cuda()
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for _, data in progress_bar:
                pdg = data["pdg"].to(self.device)
                input = data['input'].to(self.device)
                targets = data["targets"].to(self.device)
                outputs = self.model(input, pdg, train=False)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                
                losses.append(loss.item())
                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.valid_set)
        print("val_acc : ",val_acc)
        score_dict = get_MCM_score(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict

    
    def train(self):
        learning_record_dict = {}
        train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if j != "MCM"])
            print(train_table)

            val_loss, val_score = self.eval()
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
            print(test_table)
            print("\n")
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                    "train_score": train_score, "val_score": val_score}
            if epoch % 1 == 0:
                savefilename = 'model' + "/model_parameter_no" + "_epoch" + str(epoch) + ".pkl"
                torch.save(self.model.state_dict(), savefilename)
                
                
                
                
                
                
                
class single_classifer:
    def __init__(self):
        self.model = combined
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def preparation(self, data):
        # create datasets
        self.valid_set = single_dataset(data)

        # create data loaders
        self.valid_loader = DataLoader(self.valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            input = data["input"].to(self.device)
            pdg = data['pdg'].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(input, pdg)
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
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def eval(self, data, line_nums, line_count):
        print("start evaluating...")
        parameter = torch.load('model/model_parameter_no_epoch29.pkl')
        self.model = combined
        self.model.load_state_dict(parameter)
        self.model = self.model.cuda()
        self.model = self.model.eval()
        input = torch.from_numpy(data[0]).cuda()
        pdg = torch.from_numpy(data[1]).cuda()
        input = input.unsqueeze(0)
        pdg = pdg.unsqueeze(0)
        index = [0 for i in range(len(line_nums))]
        
        with torch.no_grad():

            outputs = self.model(input, pdg, train=False)
            outputs = outputs[:, :, :len(line_nums), :]
            max_index = torch.argmax(outputs, dim=2).cpu().numpy().squeeze()
            for i in max_index:
                index[i-1] += 1
        result = dict(zip(line_nums, index))
        print(result)
        print('finished')
    
