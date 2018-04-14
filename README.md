# flower-classification

**Disclaimer: This project reflects the views of the author and should not be construed to represent the U.S. Food and Drug Administration's views or policies.**

This repository contains a brief overview of convolutional neural networks (CNNs) and transfer learning, and an application to the classification of flower images. We use the dataset from the TensorFlow website:   

https://www.tensorflow.org/tutorials/image_retraining

All the images in the flower dataset are licensed under the Creative Commons By-Attribution License, available at:

https://creativecommons.org/licenses/by/2.0/

### Convolutional Neural Networks

CNNs are currently the dominant approach to computer vision tasks such as object recognition and detection (Lecun et al., 2015).  State of the art (SOTA) CNN architectures are generally deep, consisting of several neural network layers.  They are trained using very large image datasets and very powerful computers.  As an example, ResNet (He et al., 2015) consists of 152 layers and was trained on 1.28 million images from ImageNet (Fei-Fei and Russakovsky, 2013).

### Transfer Learning

SOTA architectures are created and developed by advanced deep learning researchers (e.g. Google, Facebook, and Microsoft) with access to very powerful computers to enable training on very large datasets.  For those with limited computing resources and amounts of training data but would like to perform specific computer vision tasks for their own particular needs, there is a technique called transfer learning that allows one to “reuse” or “repurpose” SOTA architectures.  The idea is that, the features that were already learned from previous training, do not have to be completely relearned.  With only minor modifications, a SOTA architecture can be adapted to new tasks.  Transfer learning is said to be effective on new tasks that are not much different from the original task.  For example, the ImageNet dataset consists of images of everyday objects and animals (airplane, automobile, bird, cat, dog, truck, etc.).  Transfer learning using architectures trained on ImageNet can be effective in new similar tasks such as classifying cars vs trucks or animals vs not animals.  However, the tasks do not necessarily have to be “too similar.” Architectures trained on ImageNet have been used in medical applications such as skin cancer (Esteva et al, 2017) and breast mass (Levy and Jain, 2016) classification. 

Chollet (2017) describes two general ways to apply transfer learning: feature selection (with and without data augmentation) and fine tuning.  In feature selection, the final layer is replaced with a new layer to fit the specific task.  For example, architectures that were trained on ImageNet classified images into 1,000 classes.  To use these architectures for classifying skin cancer as benign or malignant, the final layer is replaced with a binary classifier.  The remaining layers are “frozen” (i.e. the original weights are kept) and the new binary classifier is trained using the dataset of lesion images.  Fine tuning goes further by allowing a few more selected deeper layers to also be trained.  The idea is to retrain the last few selected layers (i.e. update their weights slightly) in order to better learn new high-level features that may be specific to the new task. 

### Flower Dataset

The flower dataset consists of 3,670 heterogeneous (in size and appearance) pictures of daisy (633), dandelion (898), roses (641), sunflowers (699), and tulips (799).  We use this dataset to illustrate that classification accuracies of over 90% can be obtained by transfer learning which cannot be achieved by a shallow architecture.  We will use two pre-trained architectures: VGG16 (Simonyan and Zisserman, 2014) and ResNet50 (He et al., 2015).  These were chosen because they are previous winners of the ImageNet Challenge (Russakovsky et al., 2015) and are often cited in the literature. 

### Methods

1. CNN Architectures

	We utilize three architectures: a shallow CNN model (baseline), VGG16, and ResNet50.  

	The shallow baseline model takes a 150 x 150 x 3 image as input (an arbitrary choice) and consists of four ConvD (3x3) - ReLu - Max Pooling (2x2) layers with filter sizes of 32, 64, 128, and 256, plus a fully connected (FC) layer of size 512 units.  It uses dropout (with a rate of 0.5) for regularization and Adam optimizer with a learning rate of 1e-4.  We 	refer to this baseline model as YapNet.  This architecture was based on Chollet (2017) with the main difference that we use 256 instead of 128 	filters.  

	For VGG16 and ResNet50, we use the same base architecture except for the output layer which uses softmax for five classes of flowers instead of 1,000 as used in ImageNet.      

2. Transfer Learning

	As mentioned earlier, Chollet (2017) describes two approaches to using pre-trained networks: feature selection (with and without data augmentation) and fine tuning.  We focus on the latter.  Fine tuning involves training a few top layers plus the fully connected classifier layer (leaving the other lower layers frozen).  

	The following layers in the respective SOTA architectures were fine-tuned:
	
			ARCHITECTURE	FINE-TUNED LAYERS
			VGG16		block5_conv1
			ResNet50	res5c_branch2a

	Thus, we have SOTA architectures that were pre-trained on ImageNet datasets and we fine-tune them on the flowers dataset.  Note that pictures in ImageNet included animal and everyday objects, therefore, the classification tasks are not too different.  We expect these architectures to perform well. We use Adam optimizer with a learning rate of 1e-5. 

3. Input Sizes

	VGG16 and ResNet50 take as input 224x224 RBG images whereas the baseline model takes as input 150x150 RBG images.  It is possible to use 150x150 input images also for the deep architectures but based on this author’s trials, the performance is better with the larger input sizes.  A possible reason for this is that the models were originally trained on 224x224 	images. 

4. Regularization

	We employ two regularization techniques: data augmentation and drop-out.
	
	Data augmentation is a technique of generating more data from training data via random transformations like rotation or translation.  During training, the models will learn from augmented images of the original set of training examples.  This will help prevent overfitting and enable models to generalize to new data.  Examples of augmented flower pictures are  	included in the jupyter notebook, yapnet.ipynb. 
	
	Drop-out (Srivastava et al., 2014) is a regularization technique whereby units or neurons in neural networks are randomly dropped from the network during training.  We include drop-out in the YapNet model only.  

5. Implementation

	The models were were trained on GPU's in Amazon Web Services (AWS) EC2: 
	
	https://aws.amazon.com/ec2/
	
	The jupyter notebook codes were largely based on Chapter 5 of this book:

	https://www.manning.com/books/deep-learning-with-python
	
### Results

Below is a summary of the results of this project.  See the jupyter notebooks for more details.

1. Shallow Architecture (YapNet)

	After 100 epochs, YapNet produced training and validation accuracies of 0.911 and 0.826, respectively.  The training accuracy indicate that the architecture does not model the data quite well.  In general, image classification tasks are best handled by deeper CNNs.  The large difference between the training and validation accuracies indicate overfitting, likely 	because the amount of data is still not large enough even when employing data augmentation.  

2. SOTA Architectures (VGG16 and ResNet50)

	Both SOTA architectures attained almost 100% accuracy when trained for 100 epochs.  ResNet50 attained 0.90 training and validation accuracies after only 12 epochs.  The validation accuracies are comparable between the two models and both test accuracies were over .90.  ResNet50 performed better than VGG16 on the test set.  

     					ACCURACY
				Training	Validation	Test
		VGG16		0.997		0.922		0.907
		ResNet50	0.993		0.923		0.939

   See the smoothed loss and accuracy functions for both architectures included in the jupyter notebooks.  

### Final Remarks

Transfer learning of SOTA architectures can be used for image classification when there is a limited amount of training images available.  By fine-tuning SOTA architectures and using data augmentation for regularization, test accuracies of over 90% can be achieved. 
