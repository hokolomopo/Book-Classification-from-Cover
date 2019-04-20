class BookDataset(Dataset):
    """Book dataset."""

    def __init__(self, csv_file, image_dir, transform=None):
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


        self.image_dir = image_dir
        self.transform = transform

        #Create list of classes
        df = self.dataset.reset_index().drop_duplicates(subset='class', keep='last').set_index('index')
        df = df.sort_values(by=['class'])
        self.classes = df['class_name'].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = self.image_dir + '/' +  self.dataset.iloc[idx, 1]
        cover = Image.open(img_name)
        line = self.dataset.iloc[idx]
        title = line["title"]
        label = line["class"]
        # sample = {'cover': cover, 'title' : title, 'class' : id}
        # sample = {'cover': cover, 'class' : label}

        if self.transform:
            cover = self.transform(cover)

        return (cover, label)