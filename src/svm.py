# svm.py
# -------------

# svm implementation
import util
PRINT = True

from sklearn import svm


class SVMClassifier:
  """
  svm classifier
  """
  def __init__( self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "svm"
    self.clf = svm.LinearSVC()
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    
    ## Added Iterations Loop
    for iteration in range(0, 1):
      print "Starting iteration ", iteration, "..."
      
      ## --- Kevin, Ely : Train SVM on trainingData --- ##
      self.clf.fit(trainingData, trainingLabels)
      
      test_output = SVMClassifier.classify(self, trainingData)
      print "SVM Trained: ", test_output
      
      ## --- End Kevin, Ely --- ##
            
      
      ## --- Kevin, Ely : Do Validation Test --- ##
      val_guesses = SVMClassifier.classify(self, validationData)
      
      print "Validation Guesses: ", val_guesses
      
      correct = 0
      for i in range(0, len(val_guesses)):
        if val_guesses[i] == validationLabels[i]:
          correct += 1
      
      print "Number Correct:", correct
      ## --- End Kevin, Ely --- ##
      
      
      
      
      for i in range(len(trainingData)): 
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
    
  def classify(self, data ):
    guesses = []
    for datum in data:
      "*** YOUR CODE HERE ***"
      ## Kevin, Ely ##
      guesses.append(self.clf.predict([datum]))
      ## End Kevin, Ely ##
      #util.raiseNotDefined()
      
    return guesses

## ----- End of Class Definition -----##

## Take and Make Input Flat
from samples import loadDataFile
from samples import loadLabelsFile



def FlatInput(n, items):
  flat_items = [[] for _ in range(n)]
  for m in range(0, n): 
    for i in range(0, 28):
      for j in range(0, 28):
        flat_items[m].append(items[m].getPixel(i, j))
  return flat_items
       
def callData():
    n = 1000
    m = 1000
    
    items = loadDataFile("data/digitdata/trainingimages", n,28,28)
    trainingData = FlatInput(n, items)
      
    
    labels = loadLabelsFile("data/digitdata/traininglabels", n)
    
    val_items = loadDataFile("data/digitdata/testimages", m,28,28)
    validationData = FlatInput(m, val_items)
    
    
    val_labels = loadLabelsFile("data/digitdata/testlabels", m)
    data = SVMClassifier([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    data.train(trainingData, labels, validationData, val_labels)

## Unit Test
callData()

