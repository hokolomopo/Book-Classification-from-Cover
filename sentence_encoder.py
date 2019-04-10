import nltk
nltk.download('punkt')
from models import InferSent
import pandas as pd

class BookDataset(Dataset):
    """Book dataset."""

    def __init__(self, csv_file, image_dir, datasetTransform = None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        cols = ["index", "filename", "url", "title", "author", "class", "class_name"]
        self.dataset = pd.read_csv(csv_file, header = None, names = cols, encoding = "ISO-8859-1")
        # self.dataset = self.dataset.head(128)
        self.dataset = datasetTransform[self.dataset]


        self.image_dir = image_dir
        self.transform = transform

        #Create list of classes
        df = self.dataset.reset_index().drop_duplicates(subset='class', keep='last').set_index('index')
        df = df.sort_values(by=['class'])
        self.classes = df['class_name'].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = self.dataset.iloc[idx]
        title = line["title"]
        label = line["class"]
        # sample = {'cover': cover, 'title' : title, 'class' : id}
        # sample = {'cover': cover, 'class' : label}

        if self.transform:
            cover = self.transform(cover)

        return (title, label)

def SentenceEmbedding():

	def __init__(self, csvFiles):
		"""
		csvFiles: A list of csv files containing the datasets used.
		"""
		V = 2
		MODEL_PATH = 'encoder/infersent%s.pkl' % V
		params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
		infersent = InferSent(params_model)
		infersent.load_state_dict(torch.load(MODEL_PATH))

		W2V_PATH = 'fastText/crawl-300d-2M.vec'
		infersent.set_w2v_path(W2V_PATH)

		sentences = []
		for file in csvFiles:
			dataset = pd.read_csv(csv_file, header = None, names = cols, encoding = "ISO-8859-1")
			titles = dataset['title'].tolist()

			sentences.extent(titles)

		infersent.build_vocab(sentences, tokenize=True)

	def transform_dataset(self, dataset):
		dataset['title'] = infersent.encode(dataset['title'], tokenize=True)

		return dataset

def create_text_data_loaders():
	datasetTransform = SentenceEmbedding([train_csv_file, test_csv_file])

	train_set = BookDataset(train_csv_file, image_dir, datasetTransform = datasetTransform)
    test_set = BookDataset(test_csv_file, image_dir, datasetTransform = datasetTransform)

    data_loaders = {
        "train": DataLoader(train_set, batch_size = batch_size, shuffle = True,
                            num_workers = num_workers),
        "test": DataLoader(test_set, batch_size = batch_size, shuffle = True, 
                           num_workers = num_workers)
    }

    return data_loaders
