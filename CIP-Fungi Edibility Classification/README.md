# CIP

CS6611 Project - Fungi Edibility Classification - ResNet

A LocalHost version, based on Flask allowing the user to upload an image, which will then be sent to the trained ResNet model for edibility prediction.
The model returns Edible/Poisonous classification which is then displayed on the webpage. 


## Team Members 
- Sai Krishna P
- Surya Charan P
- Anas M

## Outcome 
The development of an accurate and reliable image classification model to distinguish between edible and non-edible mushrooms could have a significant impact on public health by helping to prevent mushroom poisoning. Each year, many cases of mushroom poisoning are reported globally, which can lead to severe health complications and even death. By using a well-designed convolutional neural network model to classify mushroom images, we can provide an efficient and effective tool for identifying which mushrooms are safe to consume and which are potentially harmful. This model could be used by individuals, food inspectors, and healthcare professionals to determine the edibility of mushrooms quickly and accurately, thereby reducing the risk of mushroom poisoning and improving public health outcomes. By providing a web implementation, we’ve paved the way for a easy access anywhere with internet connection.  

## About this Repository
The Jupyter Notebook contains code covering dataset pre-processing, augmentation and related activies, then model definition, compilation and training. Then, a function to use the stored model to make predictions based on the uploaded image. 
The dataset is from [Kaggle](https://www.kaggle.com/datasets/marcosvolpato/edible-and-poisonous-fungi).

The Flask application uses localhost and the HTML page to display a webpasge for the user to upload a image which is then processed and evaluated by the model. The model responds with the classification and is displayed on the webpage accordingly.
