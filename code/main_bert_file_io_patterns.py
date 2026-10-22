import json, numpy as np
from random import randint as r
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, AutoTokenizer
import torch, pickle
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

## randonly select samples for training

with open("./file_gadgets.json", "r",encoding="ISO-8859-1") as read_file:
    dataset = json.load(read_file)

data = dataset

def read_custom_dataset(dataset):
    text = []
    label = []
    for egd in dataset:
        try:
            text.append(dataset[egd]['text'])
            label.append(int(dataset[egd]['label']))
        except:
            print(egd)
    return text, label


texts, labels = read_custom_dataset(data)

train_ratio = 0.5
validation_ratio = 0.1
test_ratio = 0.4

# train is now 75% of the entire data sety
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size= 1 - train_ratio, random_state=55)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio))

# with open("./test_1_10.json", "r",encoding="ISO-8859-1") as read_file:
#     testdataset = json.load(read_file)
#
# x_test, y_test = read_custom_dataset(testdataset)

print(len(x_train), len(x_val), len(x_test))

tokenizer = AutoTokenizer.from_pretrained('../model/')

train_encodings = tokenizer(x_train, truncation=True, padding=True)
val_encodings = tokenizer(x_val, truncation=True, padding=True)
test_encodings = tokenizer(x_test, truncation=True, padding=True)


class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = AmazonDataset(train_encodings, y_train)
val_dataset = AmazonDataset(val_encodings, y_val)
test_dataset = AmazonDataset(test_encodings, y_test)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


args = TrainingArguments(
    output_dir="test-amazon",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

model = DistilBertForSequenceClassification.from_pretrained("../model/")


trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=args,  # training arguments, defined above
    train_dataset=train_dataset,  # trai+ning dataset
    eval_dataset=val_dataset,  # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train("../model/")

pred = trainer.predict(test_dataset)
print(pred.metrics)

actual = pred.label_ids
preds = pred.predictions.argmax(-1)
[tn, fp, fn, tp] = confusion_matrix(actual,preds).ravel()
print([tn, fp, fn, tp])
print([tp/(tp+fn)*100, fp/(fp+tn)*100, fn/(fn+tp)*100, tn/(tn+fp)*100])


# with open("./unseen_1_10.json", "r", encoding="ISO-8859-1") as read_file:
#     dataset = json.load(read_file)
# # sel_recs = [ (r(0,len(dataset)-1)) for i in range(ds_size)]
# # testdataset = []
# # for line in sel_recs:
# #     testdataset.append(dataset[line])
#
# testdataset = dataset
# mytestset, mytestlabels = read_custom_dataset(testdataset)
# mytest_encodings = tokenizer(mytestset, truncation=True, padding=True)
# mytest_dataset = AmazonDataset(mytest_encodings, mytestlabels)
#
# pred = trainer.predict(mytest_dataset)
# print(pred.metrics)


