import pandas as pd
import nltk
from InferSent.models import InferSent
import torch

def clean(csv_file_names):
	nltk.download('punkt')
	V = 2
	MODEL_PATH = 'InferSent/encoder/infersent%s.pickle' % V
	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
			'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

	infersent = InferSent(params_model)
	infersent.cuda()
	infersent.load_state_dict(torch.load(MODEL_PATH))

	W2V_PATH = 'InferSent/dataset/fastText/crawl-300d-2M-subword.vec'
	infersent.set_w2v_path(W2V_PATH)
	
	cols = ["index", "filename", "url", "title", "author", "class", "class_name"]		
	datasets = [pd.read_csv(file_name, header = None, names = cols, encoding = "ISO-8859-1") for file_name in csv_file_names]	

	def filter(title):
		return infersent.test_word_in_db(title)

	datasets = infersent.filter_dataset(datasets, 'title')
	
	for dataset, file_name in zip(datasets, csv_file_names):
		file_names = file_name.split(".")
		new_file_name = file_names[0] + "_cleaned." + file_names[1]
		dataset.to_csv(new_file_name, index = False, encoding = "ISO-8859-1", header = False)

if __name__ == "__main__":
	clean(["dataset/book30-listing-train.csv",
		   "dataset/validation_set.csv",
		   "dataset/book30-listing-test.csv",
		   "dataset/train_set.csv"])