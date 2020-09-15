
# backup
import random
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss as Loss, BCELoss as Logit_loss
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from network import CNN__, LogisticRegression
from convert_to_mnist_format import format_image
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# # Parse the Arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--save_model", type=int, default=-1)
# ap.add_argument("-l", "--load_model", type=int, default=-1)
# ap.add_argument("-w", "--save_weights", type=str)
# args = vars(ap.parse_args())
def load_mnist(batch_size=64):
	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.5,), (0.5,)),
									])

	print('Loading MNIST Dataset...')
	trainset = datasets.MNIST('dataset/mnist/train/', download=False, train=True, transform=transform)
	valset = datasets.MNIST('dataset/mnist/validation/', download=False, train=False, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
	valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
	return trainloader, valloader, valset, trainset


def train(device, epochs=2):

	### define temperature scale value refer to out-of-distribution paper for more detail
	### create model
	model = CNN__(num_classes=10)
	## load mnist data
	train_loader, validation_loader, _, _ = load_mnist(batch_size=64)



	criterion = Loss()
	optimizer = optim.Adam(model.parameters(), lr=0.003)
	model.to(device)
	for epoch in range(epochs):
		print('Runing epoch: {}'.format(epoch))
		for local_batch, local_labels in train_loader:
			local_batch = local_batch.view(local_batch.shape[0], -1)
			optimizer.zero_grad()
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)
			dense_outputs, softmax_outputs = model(local_batch)
			loss = criterion(softmax_outputs, local_labels.long())
			loss.backward()
			optimizer.step()
			print('loss: {}'.format(loss))
	import os
	os.makedirs('trained_model', exist_ok=True)
	torch.save(model.state_dict(),'trained_model/model_state_dict_{}.pt'.format(epochs))

def turning_params_with_out_of_distribution_dataset(device, model, in_images=None, noise=0.014 , temperature=1000):
	# print('Tuning and saving log to file ... (out of distribution')

	logit_model = LogisticRegression(input_dim=10, output_dim=1)
	logit_model.to(device)
	if in_images is None:
		_, in_images, _, _ = load_mnist(batch_size=1)
	out_images = format_image('D:\locs\data\images\Images\\1\\')
	out_images = torch.Tensor(out_images)
	out_images = TensorDataset(out_images)
	out_images = DataLoader(out_images, batch_size=1)
	criterion1 = Loss()
	criterion2 = Logit_loss(reduce=False)
	# optimizer1 = optim.Adam(model.parameters(), lr=0.003)
	optimizer2 = optim.Adam(logit_model.parameters(), lr=0.003)
	scores=[]
	stop=0
	for j, (in_image, out_image) in enumerate(zip(in_images,out_images)):
		stop +=1
		if stop >=3000:
			break

		optimizer2.zero_grad()
		# optimizer1.zero_grad()
		rand = int(random.random()*1000%2)
		if rand==1:
			images = in_image[0]
		else:
			images = out_image[0]

		images = images.view(images.shape[0], -1)
		images= (images).to(device)
		images.requires_grad_(True)
		### make input store gradient

		### get original model result:
		original_dense_outputs, original_softmax_output = model(images)

		### get predicted label
		predicted_labels = torch.argmax(original_softmax_output).view(1)
		### scaling outputs with temperature and update gradient
		scaled_output=original_dense_outputs/temperature

		#### calculate log(softmax) of dense output
		numpy_dense_outputs = original_dense_outputs.data.cpu().numpy()
		numpy_dense_outputs = numpy_dense_outputs[0] ## because batch size is 1
		numpy_dense_outputs = numpy_dense_outputs - np.max(numpy_dense_outputs)
		softmax_dense_output_scores = np.exp(numpy_dense_outputs)/np.sum(np.exp(numpy_dense_outputs))
		max_score = np.max(softmax_dense_output_scores)
		if rand==1:
			with open("softmax_scores/original_model_in_of_distribution.txt", 'a+') as f:
				f.write(("{}\n".format(max_score)))
		else:
			with open("softmax_scores/original_model_out_of_distribution.txt", 'a+') as f:
				f.write(("{}\n".format(max_score)))

		### calculate loss and update gradient
		loss1 = criterion1(scaled_output, predicted_labels.long())
		loss1.backward()
		### turning gradient
		gradient = torch.ge(images.grad.data, 0)
		### get gradient sign
		gradient = (gradient.float() - 0.5) * 2

		# gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
		# gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
		# gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
		# new_input = torch.add(images.data, -noise * torch.sign(gradient * log_softmax_dense_output))

		new_input = torch.add(images.data, -noise * gradient)
		### predict
		new_dense_outputs, new_softmax_output=model(new_input)
		### scale new_output with temperature
		new_dense_outputs = new_dense_outputs/temperature
		normlized_outputs = new_dense_outputs.data.cpu().numpy()
		normlized_outputs = normlized_outputs[0]
		normlized_outputs = normlized_outputs- np.max(normlized_outputs)
		softmax_scores = np.exp(normlized_outputs) / np.sum(np.exp(normlized_outputs))
		### take the max softmax score
		max_score = np.max(softmax_scores)
		scores.append(max_score)



		### train logit model
		softmax_scores = torch.Tensor(softmax_scores)
		softmax_scores.requires_grad_(True)
		pred_in_out = logit_model(softmax_scores.to(device))
		if rand==1:
			labels = torch.Tensor([1]).to(device)
		else:
			labels = torch.Tensor([0]).to(device)
		logit_loss = criterion2(pred_in_out, labels)
		logit_loss.backward()

		print('pred: {} -- real {} -- loss {}'.format(pred_in_out.data.cpu().numpy(), labels.data.cpu().numpy(), logit_loss))



		if rand==1:
			with open("softmax_scores/modified_model_in_of_distribution.txt", 'a+') as f:
				f.write(("{}\n".format(max_score)))
		else:
			with open("softmax_scores/modified_model_out_of_distribution.txt", 'a+') as f:
				f.write(("{}\n".format(max_score)))
		### update logit model gradient
		optimizer2.step()
		# optimizer1.step()
	torch.save(logit_model.state_dict(), 'trained_model/logit_model.pt')
	return scores

		# import matplotlib.pyplot as plt
		# plt.imshow(new_input.data.cpu().numpy()[0].reshape(28,28))
		# plt.figure()
		# plt.imshow(images.data.cpu().numpy()[0].reshape(28,28))
		# with open('softmax_scores/confidence_Our_In.txt', 'w') as file:
		# 	file.write("temperature: {}, noise magnitude: {}, max score: {}\n".format(temperature, noise, np.max(softmax_scores)))
		# 	file.close()
#


def turning_params_with_in_of_distribution_dataset(device, model, images=None, noise=0.014 , temperature=1000):
	# print('Tuning and saving log to file ... (in of distribution')
	if images is None:
		_, images, _, _ = load_mnist(batch_size=1)

	#
	# images = format_image('D:\locs\data\VN-celeb_images\VN-celeb\\169\\')
	# images = torch.Tensor(images)
	# images = TensorDataset(images)
	# images = DataLoader(images, batch_size=1)
	criterion=Loss()
	i = 0

	score=[]
	for j, data in enumerate(images):
		i +=1
		images = data[0]
		images = images.view(images.shape[0], -1)
		images= (images).to(device)
		images.requires_grad_(True)
		### make input store gradient

		### get original model result:
		original_dense_outputs, original_softmax_output = model(images)

		### get predicted label
		predicted_labels = torch.argmax(original_softmax_output).view(1)
		### scaling outputs with temperature and update gradient
		scaled_output=original_dense_outputs/temperature

		#### calculate log(softmax) of dense output
		numpy_dense_outputs = original_dense_outputs.data.cpu().numpy()
		numpy_dense_outputs = numpy_dense_outputs[0] ## because batch size is 1
		numpy_dense_outputs = numpy_dense_outputs - np.max(numpy_dense_outputs)
		softmax_dense_output_scores = np.exp(numpy_dense_outputs)/np.sum(np.exp(numpy_dense_outputs))
		max_score = np.max(softmax_dense_output_scores)
		with open("softmax_scores/original_model_in_of_distribution.txt", 'a+') as f:
			f.write(("{}\n".format(max_score)))


		### calculate loss and update gradient
		loss = criterion(scaled_output, predicted_labels.long())
		loss.backward()
		### turning gradient
		gradient = torch.ge(images.grad.data, 0)
		### get gradient sign
		gradient = (gradient.float() - 0.5) * 2

		# gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
		# gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
		# gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
		# new_input = torch.add(images.data, -noise * torch.sign(gradient * log_softmax_dense_output))

		new_input = torch.add(images.data, -noise * gradient)
		### predict
		new_dense_outputs, new_softmax_output=model(new_input)
		### scale new_output with temperature
		new_dense_outputs = new_dense_outputs/temperature

		normlized_outputs = new_dense_outputs.data.cpu().numpy()
		normlized_outputs = normlized_outputs[0]
		normlized_outputs = normlized_outputs- np.max(normlized_outputs)
		softmax_scores = np.exp(normlized_outputs) / np.sum(np.exp(normlized_outputs))
		max_score = np.max(softmax_scores)
		std_score = np.std(softmax_scores)

		score.append(max_score)
		with open("softmax_scores/modified_model_in_of_distribution.txt", 'a+') as f:
			f.write(("{}\n".format(max_score)))
		### in the paper, they use only 1000 images for tuning, we use 1200
		if i>=1200:
			break
	return score



def inference(device, model, logit_model, noise=0.0014 , temperature=1000, threshold=0.5):

	# ood_images =  format_image('D:\locs\data\images\Images\\1\\')
	ood_images =  format_image('D:\locs\data\images\VN-celeb_images\\1\\')
	ood_images = torch.Tensor(ood_images)
	ood_images = TensorDataset(ood_images)
	# ood_images = DataLoader(ood_images, batch_size=1)

	_, _,iod_images, _ = load_mnist(batch_size=1)
	iod_images = Subset(iod_images, range(8000))
	### merge in of distribution and out of distribution dataset for testing
	images = ConcatDataset([ood_images, iod_images])
	images = DataLoader(images, batch_size=1, shuffle=True)
	optimizer = optim.Adam(model.parameters(), lr=0.003)
	#
	# images = format_image('D:\locs\data\VN-celeb_images\VN-celeb\\169\\')
	# images = torch.Tensor(images)
	# images = TensorDataset(images)
	# images = DataLoader(images, batch_size=1)
	criterion=Loss()
	i = 0
	for j, data in enumerate(images):
		i +=1
		if i>=100:
			break
		images = data[0]
		images = images.view(images.shape[0], -1)
		images= (images).to(device)
		images.requires_grad_(True)
		### make input store gradient

		### get original model result:
		original_dense_outputs, original_softmax_output = model(images)

		### get predicted label
		predicted_labels = torch.argmax(original_softmax_output).view(1)
		#
		# import matplotlib.pyplot as plt
		# plt.imshow(images.data.cpu().numpy().reshape(28, 28))
		# plt.title('Original model predicted label: {}'.format(predicted_labels.data.cpu().numpy()))
		# plt.savefig('images\\original\\{}.png'.format(time.time()))
		# plt.close('all')


		### scaling outputs with temperature and update gradient
		scaled_output=original_dense_outputs/temperature

		### calculate loss and update gradient
		loss = criterion(scaled_output, predicted_labels.long())
		loss.backward()
		### turning gradient
		gradient = torch.ge(images.grad.data, 0)
		### get gradient sign
		gradient = (gradient.float() - 0.5) * 2
		#
		# gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
		# gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
		# gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
		# new_input = torch.add(images.data, -noise * torch.sign(gradient * log_softmax_dense_output))

		new_input = torch.add(images.data, -noise * gradient)
		### predict
		new_dense_outputs, new_softmax_output=model(new_input)
		### scale new_output with temperature
		new_dense_outputs = new_dense_outputs/temperature


		normlized_outputs = new_dense_outputs.data.cpu().numpy()
		normlized_outputs = normlized_outputs[0]
		normlized_outputs = normlized_outputs- np.max(normlized_outputs)
		softmax_scores = np.exp(normlized_outputs) / np.sum(np.exp(normlized_outputs))
		# np.log(softmax_scores)
		# np.exp(softmax_scores)
		# std = np.std(softmax_scores)
		max_score = np.max(softmax_scores)

		softmax_scores = torch.Tensor(softmax_scores)
		softmax_scores.requires_grad_(True)
		in_out = logit_model(softmax_scores.to(device))

		softmax_scores=softmax_scores.data.cpu().numpy()
		aaa = sorted(softmax_scores, reverse=True)
		mean_ = np.mean(softmax_scores)
		bbb = (aaa[0] - aaa[1])

		if (max_score >= threshold):
			# predicted_label = np.argmax(softmax_scores)
			if bbb > 0.02:
				predicted_label = np.argmax(softmax_scores)
			else:
				predicted_label = 'unknown'
		else:
			predicted_label = 'unknown'
		import matplotlib.pyplot as plt
		fig, (ax1, ax2) = plt.subplots(2)
		ax1.imshow(new_input.data.cpu().numpy().reshape(28,28))
		ax2.plot(softmax_scores)

		plt.title('predicted label: {}, score {}, in_out {}'.format(predicted_label,max_score, in_out.data.cpu().numpy()))
		plt.savefig('images\\modified\\{}.png'.format(time.time()))
		plt.close('all')
		if i>=100:
			break



def tpr95(iod_scores,ood_scores):
	Y1 = ood_scores
	X1 = iod_scores
	start=np.min(iod_scores)
	end=np.max(iod_scores)
	gap = (end - start) / 10000
	# f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
	fprNew=0
	total = 0.0
	fpr = 0.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		if tpr <= 0.9505 and tpr >= 0.9495:
			fpr += error2
			# print(fpr)
			total += 1
	if total > 0:
		fprNew = fpr / total
	return fprNew



if __name__=='__main__':
	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	is_train=False

	if is_train:
		train(device, epochs=20)

	# #### tuning and inference
	model = CNN__(num_classes=10)
	model.load_state_dict(torch.load('trained_model/model_state_dict_20.pt'))
	model.eval()
	model.to(device)

	is_tuning=True
	if is_tuning:
		noise=0.0014
		temperature=1000
		### clear logfile content before tuning
		open('softmax_scores/modified_model_in_of_distribution.txt', 'r+').truncate()
		open('softmax_scores/modified_model_out_of_distribution.txt', 'r+').truncate()
		open('softmax_scores/original_model_in_of_distribution.txt', 'r+').truncate()
		open('softmax_scores/original_model_out_of_distribution.txt', 'r+').truncate()
		iod_scores = turning_params_with_out_of_distribution_dataset(device, model=model, in_images=None, noise=noise, temperature=temperature)
		#
		# temperature=201
		# while True:
		# 	temperature = temperature+200
		# 	noise = 0
		# 	while True:
		# 		noise= noise + (0.004/21)
		# 		iod_scores = turning_params_with_in_of_distribution_dataset(device, model=model, images=images, noise=noise, temperature=temperature)
		# 		# ood_score = turning_params_with_out_of_distribution_dataset(device, model=model, noise=noise, temperature=temperature)
		# 		log = sorted(ood_score, reverse=True)
		# 		_95_confidence_idx = int(0.95 * len(log))
		# 		print('temper {}, noise {}, score {}, fpr(tpr95) {}'.format(temperature, noise, ood_score[_95_confidence_idx], tpr95(iod_scores, ood_score)))
		# 		if noise>=0.004:
		# 			break
		# 	if temperature>=1200:
		# 		break
	is_infer=True

	if is_infer:
		logit_model = LogisticRegression(input_dim=10, output_dim=1)
		logit_model.load_state_dict(torch.load('trained_model/logit_model.pt'))
		logit_model.eval()
		logit_model.to(device)
		# with open('softmax_scores/modified_model_in_of_distribution.txt', 'r+') as f:
		# 	log = f.readlines()
		# 	### get 95% TPR index (
		# 	_95_confidence_idx = int(0.95 * len(log))
		# 	threshold = float(sorted(log, reverse=True)[_95_confidence_idx])
		# print(threshold)
		threshold=0.113
		noise = 0.0014
		temperature = 1000
		inference(device, model=model, logit_model=logit_model, noise=noise, temperature=temperature, threshold=threshold)
