import torch
import os
import numpy as np
import pickle
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from structurednets.asset_helpers import get_animal_classes_filepath
from structurednets.extract_features import get_required_indices
from structurednets.models.visionmodel import VisionModel
from structurednets.models.alexnet import AlexNet
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.models.vgg16 import VGG16

#import parts for SSS matrices
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.approximation import Approximation
import tvsclib.math as math

#for file storage
from time import gmtime, strftime

def print_accuracies(y,y_pred):
    start_idx = 0
    curr_class = y[start_idx]
    end_idx = 0
    accuracies = []
    for y_i, y_value in enumerate(y):
        end_idx = y_i
        if y_value != curr_class:
            accuracies.append(float(np.sum(y[start_idx:end_idx] == y_pred[start_idx:end_idx])) / len(y[start_idx:end_idx]))
            print("Accuracy for class " + str(curr_class) + ": " + str(accuracies[-1]))

            start_idx = y_i
            curr_class = y_value

    accuracies.append(float(np.sum(y[start_idx:end_idx] == y_pred[start_idx:end_idx])) / len(y[start_idx:end_idx]))
    print("Accuracy for class " + str(curr_class) + ": " + str(accuracies[-1]))

    print("---------------")
    print("Min accuracy: " + str(np.min(accuracies)))
    print("Max accuracy: " + str(np.max(accuracies)))
    print("Mean Accuracy: " + str(np.mean(accuracies)) + "+-" + str(np.std(accuracies)))
    print("Median Accuracy: " + str(np.median(accuracies)))


def calc_accuracy(feature_filepath: str, model_class: VisionModel, label_filepath: str, \
    N = 10,stages=20):

    required_indices = get_required_indices(label_filepath)
    model = model_class(required_indices)
    output_mat = model.get_optimization_matrix().detach().numpy()

    X, y, y_pred = pickle.load(open(feature_filepath, "rb"))
    X = np.squeeze(X)
    y = np.squeeze(y)
    y_pred = np.squeeze(y_pred)

    y_featured_pred = model.features_and_optim_mat_to_prediction_with_argmax(torch.tensor(X), torch.tensor(output_mat)).cpu().detach().numpy()
    assert y_featured_pred.shape == y_pred.shape, "dims of y do not match: reference:"+str(y_pred.shape)+" featured:"+str(y_featured_pred.shape)
    assert np.array_equal(y_pred, y_featured_pred), "Features times output matrix do not match the predictions"


    #set the dims
    d_in = output_mat.shape[1]
    boundaries = d_in/stages*np.arange(stages+1)
    boundaries = np.round(boundaries).astype(int)
    dims_in = boundaries[1:]-boundaries[:-1]

    d_out = output_mat.shape[0]
    boundaries = d_out/stages*np.arange(stages+1)
    boundaries = np.round(boundaries).astype(int)
    dims_out = boundaries[1:]-boundaries[:-1]

    assert sum(dims_in)==d_in and sum(dims_out)==d_out

    T = ToeplitzOperator(output_mat, dims_in,dims_out)
    S = SystemIdentificationSVD(T,epsilon=1e-16)

    system = MixedSystem(S)
    approx =Approximation(system)

    assert system.to_matrix().shape == output_mat.shape, "System has incorrect dims"

    max_diff = np.max(np.abs(system.to_matrix()-output_mat))
    print("max difference of system: ",)
    assert max_diff < 1e-5, "System does not represnt the matrix"

    sigma_max = max(\
    max([np.max(approx.sigmas_anticausal[i]) for i in range(len(approx.sigmas_anticausal))]),
    max([np.max(approx.sigmas_causal[i]) for i in range(len(approx.sigmas_anticausal))]))

    epsilons= np.zeros(N+1)
    norms_f = np.zeros(N+1)
    norms_h = np.zeros(N+1)
    accuracies = np.zeros(N+1)
    costs   = np.zeros(N+1)

    info = "Model:"+str(model_class.__name__)

    for i,alpha in enumerate(np.hstack((np.array([-1]),np.linspace(0,1,N)))):
        if alpha <0:
            matrix_approx=output_mat
            cost = output_mat.size
        else:
            approx_system=approx.get_approxiamtion(alpha*sigma_max)
            matrix_approx = approx_system.to_matrix()
            cost = approx_system.cost()

        epsilon = alpha*sigma_max

        norm_f = np.linalg.norm(output_mat-matrix_approx)
        norm_h = math.hankelnorm(output_mat-matrix_approx,dims_in,dims_out)

        #run the model for the approximated matrix
        y_approx= model.features_and_optim_mat_to_prediction_with_argmax(torch.tensor(X), torch.tensor(matrix_approx.astype(np.float32))).cpu().detach().numpy()
        #accuracy = np.count_nonzero(y_approx==y)/len(y)
        assert y_approx.shape == y_pred.shape, "dims of y do not match: reference:"+str(y_pred.shape)+" approx:"+str(y_approx.shape)
        accuracy = float(np.sum(y == y_approx)) / len(y)

        print_accuracies(y,y_approx)

        print("-------------------------")
        print("alpha =    "+str(alpha))
        print("cost =     "+str(cost))
        print("||A||_F =  "+str(norm_f))
        print("||A||_H =  "+str(norm_h))
        print("Accuracy = "+str(accuracy))

        #store it in vectors
        norms_f[i] = norm_f
        norms_h[i] = norm_h
        accuracies[i] = accuracy
        costs[i] = cost
        epsilons[i]=epsilon

    return {'costs':costs,'epsilons':epsilons,'norms_f':norms_f,'norms_h':norms_h,'accuracies':accuracies,'dims_in':dims_in,'dims_out':dims_out,'info':info}

if __name__ == "__main__":
    storage_filepath = "/home/ga87sar/lrz-nashome/"
    feature_filepath = "/home/ga87sar/lrz-nashome/Imagenet"+"/features/AlexNet_animal_features.p"
    model_class = AlexNet
    label_filepath = get_animal_classes_filepath()
    modelname = str(model_class.__name__)

    results = calc_accuracy(feature_filepath=feature_filepath, model_class=model_class, label_filepath=label_filepath)
    print(results)

    np.savez(storage_filepath+"Test_Accuracies_approxiamtion_"+modelname+"_"+strftime("%U%w%H%M", gmtime()),**results)
