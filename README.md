# Image_Caption_Generator

![alt text](https://github.com/siddhesh1598/Image_Caption_Generator/blob/master/thumbnail.png?raw=true)

Generating caption for images using the [Flickr8k](https://forms.illinois.edu/sec/1713398) dataset. The dataset contains 8092 photographs in jpeg format along with a file which containes a number of files containing different sources of descriptions for the photographs. The code uses **VGG16** model to estract features of these photographs along with **RNN** network to generate captions for new images.

## Technical Concepts

**VGG16:** The paper can be found [here](https://arxiv.org/pdf/1409.1556)

**RNN:** Recurrent Neural Network is a generalization of feedforward neural network that has an internal memory. RNN is recurrent in nature as it performs the same function for every input of data while the output of the current input depends on the past one computation. After producing the output, it is copied and sent back into the recurrent network. For making a decision, it considers the current input and the output that it has learned from the previous input. 

![alt text](https://miro.medium.com/max/1254/1*go8PHsPNbbV6qRiwpUQ5BQ.png?raw=true)

**LSTM (Long Short Term Memory):** Long Short-Term Memory (LSTM) networks are a modified version of recurrent neural networks, which makes it easier to remember past data in memory. The vanishing gradient problem of RNN is resolved here. LSTM is well-suited to classify, process and predict time series given time lags of unknown duration. It trains the model by using back-propagation.

![alt text](https://miro.medium.com/max/1400/1*MwU5yk8f9d6IcLybvGgNxA.jpeg?raw=true)

More information can be found [here](https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e)

## Getting Started

Clone the project repository to your local machine, then follow up with the steps as required.

### Requirements

After cloning the repository, install the necessary requirements for the project.
```
pip install -r requirements.txt
```

The *main.py* file in the *src* folder is used to generate captions for the images. It contains paths to various files required to generate captions. These files can be generated my the *main.py* file using extensions from *generateFeaturesDescriptions.py* , *model.py* , *tokenizer.py* files.

The *main.py* requires the following files:
1. Descriptions of the training images <br>
2. Features of the training images captured by the VGG16 model <br>
3. Tokenizer used to generate tokens on the vocabulary created <br>
4. The saved model (if available)

If any of these files are not available then the code generates these files. In order to do so, it requires:
1. path to the Flickr8k images dataset
2. path to the Flickr8k text files

Execute the *main.py* file to generate captions
```
python main.py
```


## Authors

* **Siddhesh Shinde** - *Initial work* - [SiddheshShinde](https://github.com/siddhesh1598)
