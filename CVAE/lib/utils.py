import os
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy

# Paths
DATA_PATH = './data/'
MODELS_PATH = './models/'

def mat2Dto2Dfull(data, lat = 73, lon = 85):

    '''
    Transform 2D matrix to 2D with NAs (sea points)
    '''

    # Load NaN positions
    base = importr('base')
    base.load(DATA_PATH + 'ind_nonNaN.RData')
    ind_nonNaN = np.array(robjects.r['ind_nonNaN']) - 1

    if (len(data.shape) == 1):
        nObs = 1
    else:
        nObs = data.shape[0]

    # Generate 2D array from values
    newData = np.empty((nObs, lat * lon))
    newData[:] = np.nan
    newData[:, ind_nonNaN] = data

    return newData


def mat2Dto3D(data, lat = 73, lon = 85):

    '''
    Transform 2D matrix to 3D for visualization
    '''

    if (len(data.shape) == 1):
        nObs = 1
    else:
        nObs = data.shape[0]

    newData = mat2Dto2Dfull(data = data)

    # Reshape it to 3D
    newData = np.reshape(newData, (nObs, lat, lon), order = 'F')
    # newData = np.flip(newData, axis = 1)

    return newData

def predict_cVAE_fullSet(X, model, modelName):

    '''
    Compute the prediction given a X
    '''

    # Load model state dict
    model.load_state_dict(torch.load(os.path.expanduser(MODELS_PATH +
                                                        modelName + '.pt')))
    model = model.cuda()

    # Subset Y
    X = torch.from_numpy(X).float()

    yPred = model.predictX(X.cuda())

    yPred = yPred.detach().cpu().numpy()

    return yPred


def predict_cVAE(X, index, model, modelName):

    '''
    Compute the prediction given some X (return random conditioned samples)
    '''

    # Load model state dict
    model.load_state_dict(torch.load(os.path.expanduser(MODELS_PATH +
                                                        modelName + '.pt')))
    model = model.cuda()

    # Subset Y
    X = torch.from_numpy(X).float()
    X = X[index, :].unsqueeze(0)

    numSamples = 9

    yPred = model.predictX(X.cuda())

    for i in range(numSamples-1):
        yPrime =  model.predictX(X.cuda())
        yPred = torch.cat((yPred, yPrime), 0)

    yPred = yPred.detach().cpu().numpy()

    return yPred

def plotPrediction_cVAE(X, Y, index, model, modelName):

    '''
    Plot generated prediction (3x3) for a specific X
    '''

    yGen = predict_cVAE(X, index, model, modelName)
    yGen = mat2Dto3D(yGen)

    # Y = mat2Dto3D(Y)

    # Plot
    fig, ax = plt.subplots(nrows = 4, ncols = 3, figsize = (100, 100))
    fig.delaxes(ax[0, 0])
    fig.delaxes(ax[0, 2])

    minValue = np.nanmin(Y[index, :, :])
    maxValue = np.nanmax(Y[index, :, :])

    if (index == 1819):
        fig.suptitle('Prediction for 25-12-2007', fontsize = 150)
    elif (index == 1936):
        fig.suptitle('Prediction for 20-04-2008', fontsize = 150)
    elif (index == 2023):
        fig.suptitle('Prediction for 16-07-2008', fontsize = 150)

    ax[0, 1].imshow(np.flip(np.flip(Y[index, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[0, 1].set_title('Y test', fontsize = 90)

    ax[1, 0].imshow(np.flip(np.flip(yGen[0, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[1, 1].imshow(np.flip(np.flip(yGen[1, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[1, 2].imshow(np.flip(np.flip(yGen[2, :, :]), 1),
                    vmin = minValue, vmax = maxValue)

    ax[2, 0].imshow(np.flip(np.flip(yGen[3, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[2, 1].imshow(np.flip(np.flip(yGen[4, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[2, 2].imshow(np.flip(np.flip(yGen[5, :, :]), 1),
                    vmin = minValue, vmax = maxValue)

    ax[3, 0].imshow(np.flip(np.flip(yGen[6, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[3, 1].imshow(np.flip(np.flip(yGen[7, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[3, 2].imshow(np.flip(np.flip(yGen[8, :, :]), 1),
                    vmin = minValue, vmax = maxValue)

    plt.savefig('./figures/prediction_' + str(index) + '.png')
    plt.close()


def computeRData(x_test, y_test,
                 model, modelName):

    '''
    Compute predictions and save them as RData following the grids architecture
    of climate4R
    '''

    # Compute predictions
    nGenerations = 8
    for i in range(1, nGenerations+1):
        y_pred = predict_cVAE_fullSet(X = x_test,
                                      model = model,
                                      modelName = modelName)

        y_pred = mat2Dto3D(y_pred)
        np.save(DATA_PATH + 'yPred_' + str(i) + '.npy', y_pred)
