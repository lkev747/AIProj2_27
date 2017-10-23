# perceptron.py
# -------------

# Perceptron implementation
import util
#from aifc import data
PRINT = True

import random

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    
    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING

    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      
      ## - Ely, Kevin
      guesses_label = PerceptronClassifier.classify(self, trainingData)
      ## - End Ely, Kevin
      
      print guesses_label
      for i in range(len(trainingData)):
          "*** YOUR CODE HERE ***"
          ## - Ely, Kevin: Update Weights          
          if (guesses_label[i] == trainingLabels[i]):
            pass
          elif (guesses_label[i] != trainingLabels[i]):
            self.weights[guesses_label[i]] = self.weights[guesses_label[i]] - trainingData[i]
            self.weights[trainingLabels[i]] = self.weights[trainingLabels[i]] + trainingData[i]
          ## - End Ely, Kevin
          
    ## Finished running through max_iterations. 
    ## NOW we need to use these weights on the validation set
    
    print "Training Completed"
    
    correct = 0
    val_guesses_label = PerceptronClassifier.classify(self, validationData)
    for i in range(len(validationData)):
      print "Testing: ", i
      if(val_guesses_label[i] == validationLabels[i]):
        correct += 1
        print "Good Job"
    accuracy = float(correct)/float(len(validationLabels))
    print correct
    print accuracy
    
    
          ##util.raiseNotDefined()

    
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    #for datum in data:
    for i in range(0, len(data)):
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * data[i]
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    "*** YOUR CODE HERE ***"
    ## - Kevin
    for labels in self.legalLabels:
      weight_container = self.weights[labels]
      featuresWeights.append(weight_container.sortedKeys()[0:100])
    ## - End Kevin
    util.raiseNotDefined()

    return featuresWeights

## Take and Make Input Flat
from samples import loadDataFile
from samples import loadLabelsFile
from samples import Datum
def FlatInput(n, items):
  flat_items = [[] for _ in range(n)]
  for m in range(0, n): 
    for i in range(0, 28):
      for j in range(0, 28):
        flat_items[m].append(items[m].getPixel(i, j))
  return flat_items
       
def callData():
    n = 100
    #items = loadDataFile("data/digitdata/trainingimages", n,28,28)
    #for item in items:
      #items = util.Counter()
    
    items = loadDataFile("data/digitdata/trainingimages", n,28,28)
    flat_item = FlatInput(n, items)

    
    trainingData = {}
    for i in range(len(flat_item)):
      trainingData[i] = util.Counter()
      for j in range(len(flat_item[i])):
        trainingData[i][j] = flat_item[i][j]
        
    
    labels = loadLabelsFile("data/digitdata/traininglabels", n)
    
    val_items = loadDataFile("data/digitdata/validationimages", n,28,28)
    flat_val = FlatInput(n, val_items)
    
    validationData = {}
    for i in range(len(flat_val)):
      validationData[i] = util.Counter()
      for j in range(len(flat_val[i])):
        validationData[i][j] = flat_val[i][j]
    
    val_labels = loadLabelsFile("data/digitdata/validationlabels", n)
    data = PerceptronClassifier( [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10)
    
    weights = {}
    for w in range(0, 10): 
      weights[w] = util.Counter()
      for i in range(0,784):
        weights[w][i] = random.random()
            
    data.setWeights(weights)
    data.train(trainingData, labels, validationData, val_labels)

## Unit Test
callData()