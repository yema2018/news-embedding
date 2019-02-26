import pandas as pd
import numpy as np
import json
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelBinarizer,scale
from datetime import datetime


class data_tool:
    def __init__(self, stock_price_path='sz50.csv', embedding_path='emd_10_all.emd', event_date='date_index.json',
                 windowsize=6):
        print("initialization")
        self.stock = pd.read_csv(stock_price_path, encoding='gbk')
        embedding = list(open(embedding_path, 'r'))

        # event embedding
        corpus_size, embedding_size = map(int, embedding[0].split())
        self.embedding = np.array([row.split() for row in embedding[1:]], dtype=np.float32)
        self.corpus = self.build_corpus([str(int(i)) for i in self.embedding[:, 0]])  # assign index to vertex
        # vertex embedding corresponding to vertex at by index
        self.embedding_matrix = np.concatenate([np.zeros([1, embedding_size]), self.embedding[:, 1:]], axis=0)

        # event date
        with open(event_date, 'r') as f:
            date_index = [json.loads(i) for i in f.readlines()]
        ## convert ids to text
        max_document_length, date_index = self.build_texts(date_index)
        date_index = pd.DataFrame(date_index)
        self.date_index = date_index.set_index(0)

        # merge two datatables
        self.stock = self.stock.set_index('日期')[['涨跌幅']]
        self.stock.index = [datetime.strftime(datetime.strptime(i, "%Y-%m-%d"), "%Y%m%d")
                            for i in self.stock.index]
        self.stock = self.stock.join(self.date_index, how='inner')
        # self.stock['涨跌幅'] = scale(self.stock['涨跌幅'].copy())
        # self.stock = self.stock.iloc[:-1]
        self.stock.iloc[:, 0] = scale(self.stock.iloc[:, 0].astype('float'))

        # build vocabulary processor
        self.processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length,
                                                                 tokenizer_fn=self.tokenize_fn,#split words
                                                                 vocabulary=self.corpus)
        # construct time-series data
        self.text_data = np.array(list(self.processor.transform(self.stock.iloc[:, 1])))
        self.stock_data = self.stock.iloc[:, 0].values.reshape(-1, 1)

        n = self.stock.shape[0]
        stock_, text_ = [], []
        for i in range(0, n - windowsize):
            stock_.append(self.stock_data[i:i+windowsize, :])
            text_.append(self.text_data[i:i+windowsize, :])
        self.stock_ = np.array(stock_)
        self.text_ = np.array(text_)

        # extract stock data and vertex data
        self.stock_x = self.stock_[:, :-1, :]
        self.stock_y = (self.stock_[:, -1, :] > 0).astype('int')
        label_encoder = LabelBinarizer().fit([-1, 0, 1])
        self.stock_y = label_encoder.transform(self.stock_y)[:, 1:]

        self.text_x = self.text_[:, :-1, :]

        # split train/test at 4:1
        train_test_split = int(self.stock_x.shape[0] * 0.8)
        self.stock_x_train, self.stock_x_test = self.stock_x[:train_test_split], self.stock_x[train_test_split:]
        self.stock_y_train, self.stock_y_test = self.stock_y[:train_test_split], self.stock_y[train_test_split:]
        self.text_x_train, self.text_x_test = self.text_x[:train_test_split], self.text_x[train_test_split:]

    def build_corpus(self, corpus):
        return dict([(token, index+1) for index, token in enumerate(corpus)])

    def build_texts(self, json):
        maxLeng = 0
        lis = []
        for i, dic in enumerate(json):
            item_len = 0
            for key in dic:
                item_len = len(dic[key])
                dic[key] = ' '.join([str(i) for i in dic[key]])
            if item_len == 0:
                continue
            maxLeng = max(maxLeng, item_len)
            lis.append(dic.popitem())
        return maxLeng, lis

    def tokenize_fn(self, iterator):
        for i in iterator:
            yield i.split(' ')

    def batches_generate(self, stock_x, stock_y, text_x, epoch_size=3, batch_size=32, shuffle=True):
        """
        generate training batches
        """
        data_size = len(stock_y)
        num_batches = data_size // batch_size + 1

        for i in range(epoch_size):
            if shuffle:
                np.random.seed(1000)
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffle_data_x, shuffle_data_y, shuffle_text= \
                    stock_x[shuffle_indices], stock_y[shuffle_indices], text_x[shuffle_indices]
            else:
                shuffle_data_x, shuffle_data_y, shuffle_text = stock_x, stock_y, text_x

            for j in range(num_batches):
                start_index = j * batch_size
                end_index = min((j + 1) * batch_size, data_size)
                batch_x = shuffle_data_x[start_index: end_index]
                batch_y = shuffle_data_y[start_index: end_index]
                batch_text = shuffle_text[start_index: end_index]
                yield batch_x, batch_y, batch_text

if __name__ == '__main__':
    test = data_tool()
