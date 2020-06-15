import torch
import torch.nn as nn
import numpy

y_pred = torch.FloatTensor(numpy.random.randint(0,2,(2,23)))
y_true = torch.FloatTensor(numpy.random.randint(0,2,(2,29)))
loss_used = torch.nn.BCELoss()

# FULL FINE : 29 classes
# FINE : 23 classes
# COARSE : 8 classes

class Masked_loss(nn.Module):
   """ Pytorch compatible implementation of DCASE 2020 task 5 masked loss (for fine)
   Args:
      loss_used (torch.nn.Module or derivated like _Loss): The loss used after the mask
   """
   def __init__(self, loss_used):
      super().__init__()
      self.full_coarse_to_fine_terminal_idxs = numpy.array([ 4,  9, 10, 14, 19, 23, 28, 29])
      self.incomplete_fine_subidxs = [3, 4, None, 3, 4, 3, 4, None]
      self.coarse_to_fine_end_idxs = numpy.array([ 3,  7,  8, 11, 15, 18, 22, 23])
      self.loss_used = loss_used

   def forward(self,y_pred, y_true):
      loss = None

      for coarse_idx in range(8):
         true_terminal_idx = self.full_coarse_to_fine_terminal_idxs[coarse_idx]
         true_incomplete_subidx = self.incomplete_fine_subidxs[coarse_idx]
         pred_end_idx = self.coarse_to_fine_end_idxs[coarse_idx]

         if coarse_idx != 0:
            true_start_idx = self.full_coarse_to_fine_terminal_idxs[coarse_idx - 1]
            pred_start_idx = self.coarse_to_fine_end_idxs[coarse_idx - 1]
         else:
            true_start_idx = 0
            pred_start_idx = 0

         if true_incomplete_subidx is None:
            true_end_idx = true_terminal_idx

            sub_true = y_true[:, true_start_idx:true_end_idx]
            sub_pred = y_pred[:, pred_start_idx:pred_end_idx]

         else:
            # Don't include incomplete label
            true_end_idx = true_terminal_idx - 1
            true_incomplete_idx = true_incomplete_subidx + true_start_idx
            assert true_end_idx - true_start_idx == pred_end_idx - pred_start_idx
            assert true_incomplete_idx == true_end_idx

            # 1 if not incomplete, 0 if incomplete
            mask =(1-y_true[:, true_incomplete_idx]).unsqueeze(-1)
            # Mask the target and predictions. If the mask is 0,
            # all entries will be 0 and the BCE will be 0.
            # This has the effect of masking the BCE for each fine
            # label within a coarse label if an incomplete label exists
            sub_true = y_true[:, true_start_idx:true_end_idx] * mask
            sub_pred = y_pred[:, pred_start_idx:pred_end_idx] * mask
         if loss is not None:
            loss = torch.cat((loss,self.loss_used(sub_pred, sub_true).sum(0)),-1)
         else:
            loss = self.loss_used(sub_pred, sub_true).sum(0)
      return loss