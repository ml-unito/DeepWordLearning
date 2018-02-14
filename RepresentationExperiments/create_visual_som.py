from models.som.SOM import SOM
from models.som.SOMTest import showSom
import numpy as np
from utils.constants import Constants
import os
import logging

fInput = os.path.join(Constants.DATA_FOLDER, 'VisualInputTrainingSet.csv')
N = 1000
lenExample = 2048
NumXClass = 10

if __name__ == '__main__':
  #read the inputs from the file fInput and show the SOM with the BMUs of each input
  inputs = np.zeros(shape=(N,lenExample))
  nameInputs = list()
  with open(fInput, 'r') as inp:
      i = 0
      for line in inp:
        if len(line)>2:
          inputs[i] = (np.array(line.split(',')[1:])).astype(np.float)
          nameInputs.append((line.split(',')[0]).split('/')[6])
          i = i+1

  print(nameInputs[0])

  #get the 20x30 SOM or train a new one (if the folder does not contain the model)
  som = SOM(20, 30, lenExample,
        checkpoint_dir=os.path.join(Constants.DATA_FOLDER, 'VisualModel10classes/'),
        n_iterations=20,sigma=4.0)

  loaded = som.restore_trained()
  if not loaded:
    logging.info('Training SOM')
    som.train(inputs)

  for k in range(len(nameInputs)):
    nameInputs[k] = nameInputs[k].split('_')[0]

  #shows the SOM
  showSom(som,inputs,nameInputs,1,'Visual map')
