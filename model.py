import re
from torchtext.data import Pipeline, Field, Dataset, TabularDataset, Iterator, BucketIterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import tqdm
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import timedelta
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from torch.autograd import Variable


seed = 1234
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

start_time = time.monotonic()

cuda =torch.cuda.is_available()
TRAIN=True

data_name = "electronics"

TRAIN_DATA = data_name + "_traindata.csv"
TEST_DATA = data_name + "_testdata.csv"
VALID_DATA = data_name + "_validdata.csv"
HUMAN_ANNOT = "electronics.human.csv"

if cuda:
    plt.switch_backend('agg')
    torch.cuda.set_device(0)
    device = "cuda:0"
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Data  batch size
    batch_size = 256
    # data path
    data_path = "/projdata3/info_fil/olatunji/NLP/Dataset/ARD Amazon/Processed data/"
    data_path2 = "/projdata3/info_fil/olatunji/NLP/Dataset/ARD Amazon/Human annotation/"
    log_interval = 10  # 'how many steps to wait before logging training status [default: 1]')
    test_interval = 100  # 'how many steps to wait before testing [default: 100]')
else:
    device = "cpu"
    # Data  batch size
    batch_size =64  # 'batch size for training [default: 64]')
    data_path = "C:/Users/hpuser/Documents/Python Sandbox/NLP/New folder/"
    data_path2 = "C:/Users/hpuser/Documents/Python Sandbox/NLP/New folder/"
    log_interval = 1  # 'how many steps to wait before logging training status [default: 1]')
    test_interval = 2  # 'how many steps to wait before testing [default: 100]')


tokenize = lambda x:x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float32)

tv_datafields = [("helpful", None),
                 ("xofyHelpfulScore", LABEL), ("overall", None),
                 ("reviewText", TEXT)]
trn, vld = TabularDataset.splits(
               path=data_path, # the root directory where the data lies
               train='validdatagg.csv', validation="validdatagg.csv",
               format='csv',
               skip_header=True,
               fields=tv_datafields)

# print(trn[0].__dict__.keys())
# print(trn[0].reviewText[:3])

# tst_datafields = [("helpful", None),
#                  ("xofyHelpfulScore", None), ("overall", None),
#                  ("reviewText", TEXT)]
# tst = TabularDataset(
#            path=data_path +"validdatagg.csv",
#            format='csv',
#            skip_header=True,
#            fields=tst_datafields)

# print(tst[0].__dict__.keys())
# print(tst[0].reviewText[:3])

# build vocab
TEXT.build_vocab(trn, vectors="glove.6B.100d")
# print(TEXT.vocab.freqs.most_common(10))
vocab_length = len(TEXT.vocab)
# print("vocab length ="); print(vocab_length)
pretrained_embeddings = TEXT.vocab.vectors

# Constructing data iterator
train_iter, val_iter = BucketIterator.splits((trn, vld), batch_sizes=(batch_size, batch_size), device=device,
                                             sort_key=lambda x:len(x.reviewText), sort_within_batch=False,
                                             repeat=False)
test_iter = Iterator(tst, batch_size=batch_size, device=device, sort=False, sort_within_batch=False, repeat=False)



# Embedding
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

# The positional encoding

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        if cuda:
            x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        else:
            x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x




# PeCNN instead of feedforward
class Pe_CNN(nn.Module):
    # embed_dim = d_model
    def __init__(self, embed_num, embed_dim,label_num, kernel_num,kernel_sizes, dropout):
        super(Pe_CNN, self).__init__()
        self.embed_dim = embed_dim
        input_channel = 1
        self.embed =nn.Embedding(embed_num, embed_dim)
        # self.pe = PositionalEncoder(embed_dim)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(input_channel, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.norm_1 = Norm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, label_num)
        self.regression_layer = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        emb_x = self.embed(x)  # (N, W, D) sequnce no, no of words, dim
        # print("embx1 is", emb_x.size())

        pe = PositionalEncoder(self.embed_dim, emb_x.size(1))
        emb_x = pe(emb_x)
        x2 = self.norm_1(emb_x)
        # PeCNN
        x = x2
        # print("x is", x.size())

        x = x.unsqueeze(1) # (N, Ci, W, D)
        conv_outputs = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        conv_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_outputs]  # [(N, Co), ...]*len(Ks)

        feats = torch.cat(conv_outputs, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        feats_dropout = self.dropout(feats)  # (N, len(Ks)*Co)
        logit = self.fc1(feats_dropout)  # (N, C)
        # Regression layer
        # output = self.regression_layer(logit)
        return logit


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm





# 00000000000000000000000000000000000000000000000000000000000000000000000

# Training the model method
def train(train_iter, model):
    if cuda:
        if torch.cuda.device_count() > 1:
            print("let's use", torch.cuda.device_count(), " GPUs")
            # use parallel GPU
            model = nn.DataParallel(model)
        model.to(device)
        # model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCEWithLogitsLoss()
    steps = 0
    best_acc = 0
    last_step = 0
    epoch_loss_list = []
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        print("==" * 20)
        print("Epoch {} of {}".format(epoch, epochs))
        print("==" * 20)
        batch_no = 0
        for batch in train_iter:
            feature, target = batch.reviewText, batch.xofyHelpfulScore
            # print("target shaoe:", target.shape)
            feature.data.t_()  # ,  target.data.sub_(1)  # batch first, index align
            if cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            #print('logit vector', logit.size())
            #print('target vector', target.size())
            # print("pred",logit, "Target",target.unsqueeze(1))
            loss = loss_func(logit, target.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # print(loss)
            running_loss += loss.item() * feature.size(0)
            # print("Running: ",running_loss)
            steps += 1
            batch_no += 1
            if steps % log_interval == 0:
                # corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                # accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{} of {}] - Training loss: {:.6f}'.format(batch_no, total_batch, loss.item()))

            if cuda:
                # release GPU memory cache after every batch
                # it saves it from accumulating
                torch.cuda.empty_cache()


            # Validation
            # validation_loss = 0.0
            if steps % test_interval == 0:
                dev_acc = eval(val_iter, model)
                # validation_loss +=dev_acc
                # print(dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if save_best:
                        save(model, save_dir, 'best', best_acc, steps)
            elif steps % save_interval == 0:
                save(model, save_dir, 'snapshot', best_acc, steps)

        # print("trn length = ",len(trn))
        epoch_loss = running_loss / len(trn)
        epoch_loss_list.append((epoch_loss,epoch))
        # valid_loss = validation_loss / test_interval

        print("*-*"*10)
        print("Epoch {} loss = {}".format(epoch, epoch_loss))
        print("Validation loss = {}".format(valid_loss))
        print("*-*" * 10)

    # save model
    # torch.save(model, save_dir+"/test")
    save(model, save_dir, 'Final', 0.0, "final")
    print("="*20)
    return epoch_loss_list


def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    loss_func = nn.BCEWithLogitsLoss()
    for batch in data_iter:
        feature, target = batch.reviewText, batch.xofyHelpfulScore
        feature.data.t_() #, target.data.sub_(1)  # batch first, index align
        if cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = loss_func(logit, target.unsqueeze(1))

        avg_loss += loss.item() * feature.size(0)

        # corrects += (torch.max(logit, 1)
        #              [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    # print("vld size ", len(vld))
    # print("size is: ", size)
    avg_loss /= size
    # accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f} \n'.format(avg_loss))
    return avg_loss


def save(model, save_dir, save_prefix,loss, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    loss = round(loss,2)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}_loss{}.pt'.format(save_prefix,loss, steps)
    torch.save(model.state_dict(), save_path)


def single_text_predict(text, model, text_field):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    # x = x.data.t_()  # No need to transpose

    x = autograd.Variable(x)
    if TRAIN:
        if cuda:
            x = x.cuda()
    # print(x)
    output = model(x)
    if cuda:
        preds = output.data.cpu().numpy()
    else:
        preds = output.data.numpy()

    # _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0]+1]
    # print("This is ", output)
    return output


def measure_correlation(predictions, test_data):
    df = pd.read_csv(test_data)

    pred_df = pd.DataFrame(predictions)
    df["predHelpfulnessScore"] = round(pred_df, 2)
    # print(df["predHelpfulnessScore"], df["xofyHelpfulScore"])
    # print()

    # define measurement
    ground_truth = df["xofyHelpfulScore"]
    predicted_score = df["predHelpfulnessScore"]

    corr, p_value = pearsonr(ground_truth, predicted_score)
    corr, p_value = round(corr, 2), round(p_value,2)

    # Numpy
    correlation_coeff = np.corrcoef(ground_truth, predicted_score)
    print()
    print("Np corr coeff:")
    print(correlation_coeff)

    # To write to disk
    file = data_path + "output.csv"
    df = df[["xofyHelpfulScore", "predHelpfulnessScore", "reviewText"]]
    df.to_csv(file, encoding='utf-8', index=False)
    return corr, p_value


def plot_loss(train_model):
    x_plot = []
    y_plot = []
    for epoch_loss, epoch_num in train_model:
        x_plot.append(epoch_num)
        y_plot.append(epoch_loss)

    plt.plot(x_plot, y_plot)
    plt.xlabel("Epochs")
    plt.ylabel("Epoch Training loss")
    plt.show()


# 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# Parameters
lr = 0.001  # learning rate
epochs = 50  # number of epochs for train default [1]
save_interval = 100  # steps to wait before saving
save_dir = data_path + 'snapshot' 
save_best = True 

total_batch = math.ceil(len(trn)/batch_size)
# total_test_batch = math.ceil(len(tst)/batch_size)
num_of_labels = 1
embed_num = vocab_length
# model
dropout = 0.5 
if cuda:
    embed_dim = 100  # embedding dimension
else:
    embed_dim = 200  # embedding dimension cpu
kernel_num = 100  # no of each kind of kernel
kernel_sizes = [3,4,5] 

# Initializing model
model = Pe_CNN(embed_num, embed_dim, num_of_labels, kernel_num, kernel_sizes, dropout)
model.embed.weight.data.copy_(pretrained_embeddings)
if cuda:
    start.record()

print(model)

if TRAIN:
    # train model
    train_model = train(train_iter,model)
else:
    # load trained model for evaluation
    model.load_state_dict(torch.load(data_path + "snapshot/Final_steps_0.0_lossfinal.pt"))

# print(train_model)


# prediction

print("Testing 2 (Human annotattion) in progress--------------")
# test_data = data_path+"testdataNew.csv"
test_data = data_path2 + HUMAN_ANNOT
predictions = []
df = pd.read_csv(test_data)
total_test_data = len(df)
steps = 0
for idx, text in enumerate(df["reviewText"]):
    # print(text)
    steps += 1
    if steps % 10 == 0:
        sys.stdout.write(
            '\rData[{} of {}]'.format(idx, total_test_data))
    predict_fn = single_text_predict(text, model, TEXT)
    # print(predict_fn)
    predict_fn = round(predict_fn.item(), 4)
    predictions.append(predict_fn)

# print(predictions)

correlation, p_value = measure_correlation(predictions, test_data)

print("correlation = ", correlation)
print("p-value = ", p_value)
print()


print("Testing 1 (Normal) in progress--------------")
# test_data = data_path+"testdataNew.csv"
test_data = data_path+ TEST_DATA
predictions = []
df = pd.read_csv(test_data)
total_test_data = len(df)
steps = 0
for idx, text in enumerate(df["reviewText"]):
    # print(text)
    steps += 1
    if steps % 10 == 0:
        sys.stdout.write(
            '\rData[{} of {}]'.format(idx, total_test_data))
    predict_fn = single_text_predict(text, model, TEXT)
    # print(predict_fn)
    predict_fn = round(predict_fn.item(), 4)
    predictions.append(predict_fn)

# print(predictions)

correlation, p_value = measure_correlation(predictions, test_data)

print("correlation = ", correlation)
print("p-value = ", p_value)
print()

# test single prediction

text = """I use this Apple USB Ethernet Adapter with my MacBook Air to connect to the internet when WiFi is not
available (in certain hotels, caf&eacute;s, and libraries).  It works about 50% of the time.
I'm unsure whether or not that is the fault of the adapter or perhaps the establishment's ethernet.  When it does work
though, it works well.  It's a bit slower than if you had an ethernet port directly in your computer but I'm
still able to send/receive emails (with attachments!) and visit different websites.  One of my complaints is that the
cord on this adapter is VERY short- maybe 4 inches which means you really need to have a long ethernet cord.  If not,
you might accidentally pull the adapter out which has happened to me on several occasions. The adapter is very small
 and light weight so it can be easily transported with your laptop.  Just know that this may not work every
 single time.
"""

print()

predict_fn = single_text_predict(text, model, TEXT)
print("Testing single text prediction")
print(predict_fn)

print("x"*20)

if cuda:
    end.record()
    torch.cuda.synchronize()
    print("Elapsed Time GPU: ", start.elapsed_time(end))


end_time = time.monotonic()
# print(start_time)
# print(end_time)
print("Total Elapsed time is: ", timedelta(seconds=round(end_time - start_time)))

# plot loss
if TRAIN:
    plot_loss(train_model)

