import torch
import numpy as np
import time

def prediction_and_label(preds_labels):
    preds,labels =[],[]
    for item in preds_labels:
        preds.append((torch.sigmoid(item[0][0])).detach().numpy().tolist())
        labels.append(item[1][0].detach().numpy().tolist())
    return preds.t(), labels.t()



"""
This function was written previouly. Check this one if you want to

#function to determine in which bin the predictions belongs to
def bin_index(num_bins, predictions):
    bin_idx= torch.zeros((predictions.shape[0], predictions.shape[1]))
    for i in range(0, predictions.shape[0]):
        bins= torch.linspace(0,1, num_bins+1)

        predictions_contiguous= predictions[i,:].contiguous()
        bin_idx[i,:]= torch.bucketize(predictions_contiguous, bins[:-1] )
    return bin_idx

"""




#The following function is the new one. A bit faster than the above but still very slow when compared to numpy
#This function here torch.bucketize
#########################################################################################
def bin_index(num_bins, predictions):
    bins= torch.linspace(0,1, num_bins+1).to(predictions.device)
    predictions_flat = predictions.contiguous().view(-1)
    bins_flat = bins.contiguous().view(-1)
    bin_indices_flat = torch.bucketize(predictions_flat, bins_flat[:-1])
    bin_idx = bin_indices_flat.view(predictions.shape[0], predictions.shape[1])
    
    
    return bin_idx
#########################################################################################


    


#values after calibration by validation data(num_classses*num_bins matrix)
def predictions_by_bin(predictions, num_bins, labels):
    pred_by_bin= torch.zeros((predictions.shape[0], num_bins))
    bin_idx= bin_index(num_bins, predictions)
    for k in range(0, predictions.shape[0]):
        for i in range(1, num_bins+1):
            nu=0
            de=0
            for j in range(0,bin_idx.shape[1]):
                if bin_idx[k,:][j]== i and labels[k,:][j]==1:
                    nu=nu+1
                if bin_idx[k,:][j]==i:
                    de=de+1
            if de==0:
                pred_by_bin[k][i-1]=0
            else:
                pred_by_bin[k][i-1]= nu/de
    
    return pred_by_bin

#for assigning calibrated values to the predicted test probabilities(the output will only be related to test data)
def new_predictions_test_data(val_predictions,val_labels, test_predictions, num_bins):
    val_pred_by_bin= predictions_by_bin(val_predictions,num_bins, val_labels) 
    new_test_pred= torch.zeros((test_predictions.shape[0], test_predictions.shape[1]))
    bins= torch.linspace(0,1, num_bins+1)
    for i in range(0, test_predictions.shape[0]):
        for j in range(0, test_predictions.shape[1]):
            bin_number= torch.bucketize(test_predictions[i][j], bins=bins[:-1])
            new_test_pred[i][j]= val_pred_by_bin[i][bin_number-1]
    return new_test_pred

#To check the number of positive and total instances per bin in each and every bins in any data(train, validation or predict)
def positive_and_total_instances( labels, num_bins, predictions):
    
    bin_index_set= bin_index(num_bins, predictions)
    pos_vals = bin_index_set * labels
    positive = torch.vstack([torch.sum(pos_vals == i, -1) for i in range(1, num_bins+1)]).transpose(1,0)
    total = torch.vstack([torch.sum(bin_index_set == i, -1) for i in range(1, num_bins+1)]).transpose(1,0)
    return positive, total

#the fraction of positive instances in each bin
def fraction_of_positive_instances(labels, num_bins, predictions):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(labels, num_bins, predictions)
    fraction_positive = positive_inst_perbin / (total_inst_perbin + 1e-7)
    return fraction_positive

#weighted average ECE for calibrated predictions 
def avg_ECE(validation_predictions, num_bins, validation_labels, test_predictions, test_labels, data_size):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(test_labels, num_bins, test_predictions)
    weights= positive_inst_perbin.sum(axis=1)
    ece=torch.zeros(14)
    confidence_matrix= predictions_by_bin(predictions=validation_predictions, num_bins=num_bins, labels= validation_labels)
    positive_fractions_of_test_data= fraction_of_positive_instances(test_labels, num_bins, test_predictions)
    for i in range(0, positive_fractions_of_test_data.shape[0]):
        for j in range(0, num_bins):
            if total_inst_perbin[i][j]!=0:
                ece[i]= ece[i]+ (total_inst_perbin[i][j]/test_predictions.shape[1])* abs(positive_fractions_of_test_data[i][j]- confidence_matrix[i][j])
    
    return torch.sum(weights*ece)/data_size

def avg_bin_conf(predictions, num_bins):
    bin_idx= bin_index(num_bins= num_bins, predictions= predictions )
    avg_pred= torch.zeros((bin_idx.shape[0], num_bins ))
    
    cp = torch.vstack([torch.sum(predictions * (bin_idx == i), -1) for i in range(1, num_bins+1)]).transpose(1,0)
    ct = torch.vstack([torch.sum(bin_idx == i, -1) for i in range(1, num_bins+1)]).transpose(1,0)
    avg_pred = cp / (ct + 1e-7)
    return avg_pred

#weighted average ECE for uncalibrated predictions 
def avg_uncal_ECE(labels, predictions, num_bins, data_size):
    #t= time.time()
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(labels, num_bins, predictions)
    weights= positive_inst_perbin.sum(axis=1)
    ece=torch.zeros(predictions.shape[0])
    avg_conf= avg_bin_conf(predictions= predictions, num_bins= num_bins)
    positive_frac= fraction_of_positive_instances(labels, num_bins, predictions)
    
    ece = torch.sum((total_inst_perbin/predictions.shape[1])*torch.abs(avg_conf- positive_frac), -1)
    #print(time.time()-t)
    #asd
    return torch.sum(weights*ece)/data_size


#Total ECE after calibration
def tot_ECE(validation_predictions, num_bins, validation_labels, test_predictions, test_labels):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(test_labels, num_bins, test_predictions)
    #weights= positive_inst_perbin.sum(axis=1)
    ece=torch.zeros(14)
    confidence_matrix= predictions_by_bin(predictions=validation_predictions, num_bins=num_bins, labels= validation_labels)
    positive_fractions_of_test_data= fraction_of_positive_instances(test_labels, num_bins, test_predictions)
    for i in range(0, positive_fractions_of_test_data.shape[0]):
        for j in range(0, num_bins):
            if total_inst_perbin[i][j]!=0:
                ece[i]= ece[i]+ (total_inst_perbin[i][j]/test_predictions.shape[1])* abs(positive_fractions_of_test_data[i][j]- confidence_matrix[i][j])
    
    return ece.sum()

#Total ECE before calibration
def tot_uncal_ECE(labels, predictions, num_bins):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(labels, num_bins, predictions)
    #weights= positive_inst_perbin.sum(axis=1)
    ece=torch.zeros(14)
    avg_conf= avg_bin_conf(predictions= predictions, num_bins= num_bins)
    positive_frac= fraction_of_positive_instances(labels, num_bins, predictions)
    for i in range(0, avg_conf.shape[0]):
        for j in range(0, avg_conf.shape[1]):
            if total_inst_perbin[i][j]!=0:
                ece[i]= ece[i]+ (total_inst_perbin[i][j]/predictions.shape[1])*abs(avg_conf[i][j]- positive_frac[i][j])
    return ece.sum()

def predictions(data):
    return torch.round(data)

def custom_pred(data, threshold):
    preds=[]
    for i in range(0,len(threshold)):
         preds.append(torch.where(data>= threshold[i], torch.ceil(data), torch.floor(data)))
    return preds

def canonical_ECE(labels, predictions, num_bins):
    positive_inst_perbin, total_inst_perbin= positive_and_total_instances(labels, num_bins, predictions)
    avg_conf= avg_bin_conf(predictions= predictions, num_bins= num_bins)
    positive_frac= fraction_of_positive_instances(labels, num_bins, predictions)
    ece = torch.sum((total_inst_perbin/predictions.shape[1])*torch.abs(avg_conf- positive_frac), -1)
    return torch.mean(ece) # torch.mean(ece**2)
    
    
# preds = torch.from_numpy(np.load('test_384_pred_s42_FL.npy')).transpose(1,0)
# print(preds.shape)
# gts = torch.from_numpy(np.load('test_384_gt_s42_FL.npy')).transpose(1,0)
# preds = preds.sigmoid()
# eceval = avg_uncal_ECE(gts, preds, 10, preds.shape[1])
# eceval = avg_uncal_ECE(gts, preds, 10, preds.shape[1])
# print(eceval)
   
