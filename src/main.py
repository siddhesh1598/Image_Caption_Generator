
from pickle import load
from os import listdir
from numpy import argmax
from numpy import argmin
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

from tokenizer import *
#from model import *
from generateFeaturesDescriptions import *
from modelNew import *

# path to files

# PC
pathFlickerDataset = '../data/Flicker8k_Dataset/'
pathFlickerText = '../data/Flickr8k_text/'
pathTrainImages = pathFlickerText + 'Flickr_8k.trainImages.txt'
pathDevImages = pathFlickerText + 'Flickr_8k.devImages.txt'
pathTokenTxt = pathFlickerText + 'Flickr8k.token.txt'
pathInfer = '../data/example.jpg'

pathDescriptions = '../data/descriptions.txt'
pathFeatures = '../data/features.pkl'
pathTokenizer = '../data/tokenizer.pkl'
pathModel = '../model/model-ep004-loss3.652-val_loss3.985.h5'

'''
# GC
pathFlickerDataset = '/content/drive/My Drive/Colab Notebooks/CaptionGenerator/Flicker8k_Dataset'
pathFlickerText = '/content/drive/My Drive/Colab Notebooks/CaptionGenerator/Flickr8k_text/'
pathTrainImages = pathFlickerText + 'Flickr_8k.trainImages.txt'
pathDevImages = pathFlickerText + 'Flickr_8k.devImages.txt'
pathTokenTxt = pathFlickerText + 'Flickr8k.token.txt'
pathInfer = '/content/drive/My Drive/Colab Notebooks/CaptionGenerator/example.jpg'

pathDescriptions = ''
pathFeatures = ''
pathTokenizer = ''
pathModel = ''

pathDescriptions = '/content/drive/My Drive/Colab Notebooks/CaptionGenerator/descriptions.txt'
pathFeatures = '/content/drive/My Drive/Colab Notebooks/CaptionGenerator/features.pkl'
pathTokenizer = '/content/drive/My Drive/Colab Notebooks/CaptionGenerator/tokenizer.pkl'
pathModel = '/content/model_19.h5'
'''

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
  # seed the generation process
  in_text = 'startseq'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]

    # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
    # convert probability to integer
    yhat = argmax(yhat)
    # map integer to word
    word = word_for_id(yhat, tokenizer)
    # stop if we cannot map the word
    if word is None:
      break
    # append as input for generating the next word
    in_text += ' ' + word
    # stop if we predict the end of the sequence
    if word == 'endseq':
      break
  return in_text

if not pathFeatures:
  pathFeatures = getFeatures(pathFlickerDataset)
  
if not pathDescriptions:
  pathDescriptions = getDescriptions(pathTokenTxt)

# load the tokenizer
if not pathTokenizer:
  pathTokenizer = generateToken(pathTrainImages, pathDescriptions)

tokenizer = load(open(pathTokenizer, 'rb'))

# pre-define the max sequence length (from training)
max_length = 34

# load the model
if not pathModel:
  model(pathTrainImages, pathDevImages, pathDescriptions, pathFeatures)
  models = []
  loss = []
  for saved in listdir('/content/data'):
    x = saved.split('-')
    if x[0] == 'model':
      loss.append(x[3][8:13])
      models.append(saved)
  pathModel = '/content/data/' + str(models[argmin(loss)])

model = load_model(pathModel)
print(pathModel)

# load and prepare the photograph
photo = extract_features(pathInfer)

# generate description
description = generate_desc(model, tokenizer, photo, max_length)
description = list(description.split(" "))
description = description[1:-1]
description = ' '.join(description)
print("\nCaption: " + str(description))
