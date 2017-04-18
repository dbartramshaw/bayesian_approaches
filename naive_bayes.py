"""
##########################
Naive Bayes
##########################
The Naive Bayes algorithm is an intuitive method that uses the probabilities of each attribute belonging to each class to make a prediction.
It is the supervised learning approach you would come up with if you wanted to model a predictive modeling problem probabilistically.

Naive bayes simplifies the calculation of probabilities by assuming that the probability of each attribute belonging to a given class value is independent of all other attributes. This is a strong assumption but results in a fast and effective method.

The probability of a class value given a value of an attribute is called the conditional probability. By multiplying the conditional probabilities together for each attribute for a given class value, we have a probability of a data instance belonging to that class.

To make a prediction we can calculate probabilities of the instance belonging to each class and select the class value with the highest probability.

Naive bases is often described using categorical data because it is easy to describe and calculate using ratios. A more useful version of the algorithm for our purposes supports numeric attributes and assumes the values of each numerical attribute are normally distributed (fall somewhere on a bell curve). Again, this is a strong assumption, but still gives robust results.

DBS Notes
Essentially it looks at the probability of each attribute being classified as each class.
The combines the attribute probabilities to give a final prob for each class (combine by  multiplication)

Note that Naive Bayes makes the assumption of conditional independence of the attributes (features)

Real life example
	- Combining age, gender, parent_yn ect ect
"""

import math
import csv
import random


"""
#####################
1. Load Data
#####################
"""
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

filename = '/Users/bartramshawd/Documents/datasets/pima-indians-diabetes.data.csv'
dataset = loadCsv(filename)
print('Loaded data file {0} with {1} rows').format(filename, len(dataset))


# Split a loaded dataset into a train and test datasetsPython
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

# Test this
dataset1 = [[1], [2], [3], [4], [5]]
splitRatio = 0.67
train, test = splitDataset(dataset1, splitRatio)
print('Split {0} rows into train with {1} and test with {2}').format(len(dataset1), train, test)

# Split the actual data
train, test = splitDataset(dataset, splitRatio)

#df = pd.read_csv('/Users/bartramshawd/Documents/datasets/pima-indians-diabetes.data.csv',header=None)



"""
#####################
2. Summarize Data
#####################
The naive bayes model is comprised of a summary of the data in the training dataset. This summary is then used when making predictions.

The summary of the training data collected involves the mean and the standard deviation for each attribute, by class value. For example, if there are two class values and 7 numerical attributes, then we need a mean and standard deviation for each attribute (7) and class value (2) combination, that is 14 attribute summaries.

These are required when making predictions to calculate the probability of specific attribute values belonging to each class value.

We can break the preparation of this summary data down into the following sub-tasks:

1. Separate Data By Class
2. Calculate Mean
3. Calculate Standard Deviation
4. Summarize Dataset
5. Summarize Attributes By Class
"""

def separateByClass(dataset):
	"""
	The first task is to separate the training dataset instances by class value so that we can calculate statistics for each class.
	We can do that by creating a map of each class value to a list of instances that belong to that class and sort the entire dataset of instances into the appropriate lists.
	This is done based on the last col = classification
	"""
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

# Test this
dataset2 = [[1,20,1], [2,21,0], [3,22,1]]
separated = separateByClass(dataset2)
print('Separated instances: {0}').format(separated)

dataset
# Seperate the actual data
separated = separateByClass(dataset)


def mean(numbers):
	"""
	We need to calculate the mean of each attribute for a class value.
	The mean is the central middle or central tendency of the data, and we will use it as the middle of our gaussian distribution when calculating probabilities.
	"""
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	"""
	We also need to calculate the standard deviation of each attribute for a class value.
	The standard deviation describes the variation of spread of the data, and we will use it to characterize the expected spread of each attribute in our Gaussian distribution when calculating probabilities.
	The standard deviation is calculated as the square root of the variance.
	The variance is calculated as the average of the squared differences for each attribute value from the mean.
	Note we are using the N-1 method, which subtracts 1 from the number of attribute values when calculating the variance.
	"""
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

#Test it
numbers = [1,2,3,4,5]
print('Summary of {0}: mean={1}, stdev={2}').format(numbers, mean(numbers), stdev(numbers))


def summarize(dataset):
	"""
	Now we have the tools to summarize a dataset.
	For a given list of instances (for a class value) we can calculate the mean and the standard deviation for each attribute.
	The zip function groups the values for each attribute across our data instances into their own lists so that we can compute the mean and standard deviation values for the attribute.
	"""
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

# Test it
dataset3 = [[1,20,0], [2,21,1], [3,22,0]]
summary = summarize(dataset3)
print('Attribute summaries: {0}').format(summary)

# Run it
summary = summarize(dataset)


def summarizeByClass(dataset):
	"""
	We can pull it all together by first separating our training dataset into instances grouped by class.
	Then calculate the summaries for each attribute.
	"""
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

# Test it
dataset4 = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
summary = summarizeByClass(dataset4)
print('Summary by class value: {0}').format(summary)

# run it
summary = summarizeByClass(dataset)




"""
#####################
3. Make Predictions
#####################
We are now ready to make predictions using the summaries prepared from our training data.
Making predictions involves calculating the probability that a given data instance belongs to each class, then selecting the class with the largest probability as the prediction.

We can divide this part into the following tasks:

1) Calculate Gaussian Probability Density Function
2) Calculate Class Probabilities
3) Make a Prediction
4) Estimate Accuracy

"""


def calculateProbability(x, mean, stdev):
	"""
	Calculate Gaussian Probability Density Function (Normal)

	We can use a Gaussian function to estimate the probability of a given attribute value, given the known mean and standard deviation for the attribute estimated from the training data.
	Given that the attribute summaries where prepared for each attribute and class value, the result is the conditional probability of a given attribute value given a class value.
	See the references for the details of this equation for the Gaussian probability density function.
	In summary we are plugging our known details into the Gaussian (attribute value, mean and standard deviation) and reading off the likelihood that our attribute value belongs to the class.
	In the calculateProbability() function we calculate the exponent first, then calculate the main division. This lets us fit the equation nicely on two lines.
	"""
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#Test it
x = 71.5
mean = 73
stdev = 6.2
probability = calculateProbability(x, mean, stdev)
print('Probability of X belonging to this class: {0}').format(probability)


def calculateClassProbabilities(summaries, inputVector):
	"""
	Calculate Class Probabilities

	Now that we can calculate the probability of an attribute belonging to a class,
	we can combine the probabilities of all of the attribute values for a data instance and come up with a probability of the entire data instance belonging to the class.
	We combine probabilities together by multiplying them. In the calculateClassProbabilities() below,
	the probability of a given data instance is calculated by multiplying together the attribute probabilities for each class.
	The result is a map of class values to probabilities.
	"""
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

# Test it
summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
inputVector = [1.1, '?']
probabilities = calculateClassProbabilities(summaries, inputVector)
print('Probabilities for each class: {0}').format(probabilities)



def predict(summaries, inputVector):
	"""
	Now that can calculate the probability of a data instance belonging to each class value,
	we can look for the largest probability and return the associated class.
	The predict() function belong does just that.
	"""
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


# Test it
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
inputVector = [1.1, '?']
result = predict(summaries, inputVector)
print('Prediction: {0}').format(result)


"""
#####################
4. Make Predictions
#####################

Finally, we can estimate the accuracy of the model by making predictions for each data instance in our test dataset.
The getPredictions() will do this and return a list of predictions for each test instance.
"""

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

#Test it
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
testSet = [[1.1, '?'], [19.1, '?']]
predictions = getPredictions(summaries, testSet)
print('Predictions: {0}').format(predictions)


"""
#####################
5. Get Accuracy
#####################

The predictions can be compared to the class values in the test dataset and a classification accuracy can be calculated as an accuracy ratio between 0& and 100%.
The getAccuracy() will calculate this accuracy ratio.

"""

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# Test it
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}').format(accuracy)






"""
#####################
FINAL NAIVE BAYES
#####################
"""

# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	filename = 'pima-indians-diabetes.data.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()


"""
##########################################
Implementation Extensions
##########################################
This section provides you with ideas for extensions that you could apply and investigate with the Python code you have implemented as part of this tutorial.
You have implemented your own version of Gaussian Naive Bayes in python from scratch.
You can extend the implementation further.


Calculate Class Probabilities:
Update the example to summarize the probabilities of a data instance belonging to each class as a ratio. This can be calculated as the probability of a data instance belonging to one class, divided by the sum of the probabilities of the data instance belonging to each class. For example an instance had the probability of 0.02 for class A and 0.001 for class B, the likelihood of the instance belonging to class A is (0.02/(0.02+0.001))*100 which is about 95.23%.

Log Probabilities:
The conditional probabilities for each class given an attribute value are small. When they are multiplied together they result in very small values, which can lead to floating point underflow (numbers too small to represent in Python). A common fix for this is to combine the log of the probabilities together. Research and implement this improvement.

Nominal Attributes:
Update the implementation to support nominal attributes. This is much similar and the summary information you can collect for each attribute is the ratio of category values for each class. Dive into the references for more information.

Different Density Function (bernoulli or multinomial):
We have looked at Gaussian Naive Bayes, but you can also look at other distributions. Implement a different distribution such as multinomial, bernoulli or kernel naive bayes that make different assumptions about the distribution of attribute values and/or their relationship with the class value.












""
