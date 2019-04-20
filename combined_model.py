from testmodel import load_resnet
import torch.nn as nn
import torch
from text_model import create_model_3

class CombinedModel(nn.Module):
	def __init__(self):
		resnet = 18
	    trained_layers = 10 
	    n_outputs = 30

		self.image_model = load_resnet(resnet)
    	self.image_model = change_model(self.image_model, trained_layers, n_outputs)

    	self.text_model = create_model_3(4096, n_outputs)
    	self.text_model.load_state_dict(torch.load("textmodels/final_text_model.pt"))
    	removed = list(self.model.children())[:-2]
    	self.text_model = nn.Sequential(*removed)

    	self.join_layer = nn.Linear(1256, n_outputs)

    def forward(self, inputs):
    	cover = inputs[0]
    	title_emb = inputs[1]

    	cover_output = self.image_model(cover)
    	title_output = self.text_model(title_emb)