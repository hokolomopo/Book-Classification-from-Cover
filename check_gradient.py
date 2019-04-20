from text_model import *
import torch

def show_grad(model):
	for p in model.parameters():
		print('===========\ngradient:\n----------\n{}'.format(p.grad))

if __name__ == "__main__":
	NB_INPUTS = 4096
	NB_OUTPUTS = 30

	model = create_model_4(NB_INPUTS, NB_OUTPUTS)
	model.load_state_dict(torch.load("text_models/text_model4.pt"))

	show_grad(model)
