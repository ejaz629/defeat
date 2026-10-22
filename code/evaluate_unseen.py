import json
import numpy as np
from numpy import array
from random import randint as r
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, AutoTokenizer
import torch
import pickle
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix

with open("malware_test_file_gadgets.json", "r", encoding="ISO-8859-1") as read_file:
    dataset = json.load(read_file)

# for line in range(ds_size):
#     data.append(dataset[line])

data = dataset

def read_custom_dataset(dataset):
    text = []
    label = []
    id = []
    no = []
    for egd in dataset:
        try:
            text.append(dataset[egd]['text'])
            label.append(int(dataset[egd]['label']))
        except:
            print(egd)
    return text, label


x_test, y_test = read_custom_dataset(data)

print(len(x_test))

tokenizer = AutoTokenizer.from_pretrained('../model/')
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

model = DistilBertForSequenceClassification.from_pretrained("./test-amazon/checkpoint-33542") #/test-amazon/checkpoint-33542")

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    compute_metrics=compute_metrics
)

pred = trainer.predict(test_dataset)
print(pred.metrics)

with open('pred.pickle', 'wb') as handle:
    pickle.dump(pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

# a = array(pred.label_ids)
# b = array(pred.predictions.argmax(-1))
# print(pred.predictions.argmax(-1))
# x = zip(id, no, a, b)
#np.savetxt("unseen_scores.csv", tuple(x), delimiter=",", fmt='%s')

actual = pred.label_ids
preds = pred.predictions.argmax(-1)
[tn, fp, fn, tp] = confusion_matrix(actual, preds).ravel()
print([tn, fp, fn, tp])

print([tp / (tp + fn) * 100, fp / (fp + tn) * 100, fn / (fn + tp) * 100, tn / (tn + fp) * 100])
