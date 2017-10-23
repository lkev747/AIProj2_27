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
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      
      ## --- Kevin, Ely : Train SVM on trainingData --- ##
      self.clf.fit(trainingData, trainingLabels)
      ## --- End Kevin, Ely --- ##
      
      ## --- Kevin, Ely : Do Validation Test --- ##
      val_guesses = SVMClassifier.classify(self, validationData)
      
      correct = 0
      for i in range(0, len(val_guesses)):
        if val_guesses[i] == validationLabels[i]:
          correct += 1
      
      print "Number Correct:", correct
      ## --- End Kevin, Ely --- ##
      
      for i in range(len(trainingData)): 
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
    
  def classify(self, data ):
    guesses = []
    for datum in data:
      "*** YOUR CODE HERE ***"
      ## Kevin, Ely ##
      guesses.append(self.clf.predict(datum))
      ## End Kevin, Ely ##
      util.raiseNotDefined()
      
    return guesses

