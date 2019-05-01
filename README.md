# Book-Classification-from-Cover

To run text models, Intructions provided at https://github.com/facebookresearch/InferSent must first be done. We tested both Glove and FastText word embedding but the final model was built using FastText. Don't pull their git repository and use the one provided in the archive as it has been slightly modified!

The datasets are to be dowloaded at https://github.com/uchidalab/book-dataset/tree/master/Task1. Then the clean_dataset, split_dataset and 10_classes_dataset scripts must be run in order to generate the used dataset.

Some dataloaders have to be generate beforehands using the sentence_encoder and combined_dataloaders scripts.

Before running the combined model, the separate models have to be trained as the combined model relies on pretrained separate models.

All files with cover in the name refers to the model using only the cover. the ones with text in the name refers to the InferSent + MLP model. The ones with cnn_text refers to the convolutionnal network for text classification. The ones with combined are for the model using the combination of cover and title as input. The cover generation are done in the AC_GAN script.