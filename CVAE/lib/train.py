import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

# Paths
DATA_PATH = './data/'
MODELS_PATH = './models/'

def train_model(model, modelName, loss_function, optimizer, num_epochs, patience,
                trainData, validData, scheduler = None):

    # Load model into GPU
    model = model.cuda()

    # Epoch counter for early stopping
    epoch_early_stopping = 1

    # Save losses during training
    avg_train_loss_list = []
    avg_val_loss_list = []

    # Earlystopping
    best_model = copy.deepcopy(model.state_dict())
    lowest_loss = 10000

    # Iterate over epochs
    for epoch in range(1, num_epochs + 1):

        # Training step
        model.train()
        loss_train_list = [] # loss per batch (temp)

        # Iterate over batches
        for X, Y in trainData:

            X = X.cuda()
            Y = Y.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            Y_prime, mu, logvar  = model(X, Y)
            loss = loss_function(Y, Y_prime, mu, logvar, epoch)
            loss_train_list.append(loss.item())

            # Compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Train loss on epoch
        avg_train_loss_list.append(np.mean(loss_train_list))

        # Validation step
        model.eval()
        loss_val_list = []

        for X_val, Y_val in validData:

            X_val = X_val.cuda()
            Y_val = Y_val.cuda()

            # Forward pass
            Y_prime, mu, logvar = model(X_val, Y_val)

            # Compute loss
            loss = loss_function(Y_val, Y_prime, mu, logvar, epoch)
            loss_val_list.append(loss.item())

        # Validation loss on epoch
        avg_val_loss_list.append(np.mean(loss_val_list))

        # Epoch log
        log = 'Epoch {} | Training Loss {:.5f} | Validation Loss {:.5f}'.format(epoch,
                                                                         avg_train_loss_list[-1],
                                                                         avg_val_loss_list[-1])
        if scheduler is not None:
            scheduler.step(avg_val_loss_list[-1])

        # Earlystopping checking
        if avg_val_loss_list[-1] < lowest_loss:

            # Update the lowest loss
            lowest_loss = avg_val_loss_list[-1]

            # Restart the epoch counter
            epoch_early_stopping = 1

            # Save model
            torch.save(model.state_dict(),
                       os.path.expanduser(MODELS_PATH + modelName + '.pt'))

            log = log + ' (New best model found)'

        # If patience is surpassed
        elif epoch_early_stopping > patience:

            print(log)
            print('')
            print('{} epochs without loss reduction (Finishing training...)'.format(epoch_early_stopping - 1))
            print('Final loss: {:.5f}'.format(lowest_loss))
            print('')

            # Break training loop
            break

        print(log)
        epoch_early_stopping += 1

        # torch.save(model.state_dict(),
        #            os.path.expanduser(MODELS_PATH + modelName + '.pt'))
