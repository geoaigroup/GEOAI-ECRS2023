from math import ceil

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate


from .estimators import JansenEstimator
from .sampling import ScipySobolSequence
from .torch_perturbations import inpainting
from .utils import resize


class SobolAttributionMethod:
    """
    Sobol' Attribution Method.

    Once the explainer is initialized, you can call it with an array of inputs and labels (int) 
    to get the STi.

    Parameters
    ----------
    grid_size: int, optional
        Cut the image in a grid of grid_size*grid_size to estimate an indice per cell.
    nb_design: int, optional
        Must be a power of two. Number of design, the number of forward will be nb_design(grid_size**2+2).
    sampler : Sampler, optional
        Sampler used to generate the (quasi-)monte carlo samples.
    estimator: Estimator, optional
        Estimator used to compute the total order sobol' indices.
    perturbation_function: function, optional
        Function to call to apply the perturbation on the input.
    batch_size: int, optional,
        Batch size to use for the forwards.
    """

    def __init__(
        self,
        model,
        target_mask,
        grid_size=8,
        nb_design=64,
        sampler=ScipySobolSequence(),
        estimator=JansenEstimator(),
        perturbation_function=inpainting,
        batch_size=256
    ):

        assert (nb_design & (nb_design-1) == 0) and nb_design != 0,\
            "The number of design must be a power of two."

        self.model = model
        self.target_mask =  target_mask
        self.grid_size = grid_size
        self.nb_design = nb_design
        self.perturbation_function = perturbation_function

        self.sampler = sampler
        self.estimator = estimator

        self.batch_size = batch_size

        masks = sampler(grid_size**2, nb_design).reshape((-1, 1, grid_size, grid_size))
        # print("masks: ",masks.shape)
        self.masks = torch.Tensor(masks).cuda()

    def __call__(self, inputs):
        """
        Explain a particular prediction

        Parameters
        ----------
        inputs: ndarray or tf.Tensor [Nb_samples, Width, Height, Channels]
            Images to explain.
        labels: list of int,
            Label of the class to explain.
        """
        input_shape = inputs.shape[2:]
        explanations = []

        # for input, label in zip(inputs, labels):
        for input in inputs:

            perturbator = self.perturbation_function(input)

            y = np.zeros((len(self.masks)))
            nb_batch = ceil(len(self.masks) / self.batch_size)


            for batch_index in range(nb_batch):
                # retrieve masks of the current batch
                start_index = batch_index * self.batch_size
                end_index = min(len(self.masks), (batch_index+1)*self.batch_size)
                batch_masks = self.masks[start_index:end_index]

                # apply perturbation to the input and forward
                batch_y = SobolAttributionMethod._batch_forward(self.model, input, batch_masks,
                                                                perturbator, input_shape, self.target_mask)

                # store the results
                y[start_index:end_index] = batch_y

            # get the total sobol indices
            sti = self.estimator(self.masks, y, self.nb_design)
            # sti = resize(sti[0], input_shape)
            sti = resize(sti[0], (512,512))

            explanations.append(sti)

        return explanations

    @staticmethod
    def _batch_forward(model, input, masks, perturbator, input_shape, target_mask):
        upsampled_masks = interpolate(masks, input_shape)
        perturbated_inputs = perturbator(upsampled_masks)
        
        # print("input shape", input.shape)
        # print("target_mask", target_mask.shape)
        # print("noise_mask", masks.shape);  
        # print("upsampled_masks : ",upsampled_masks.shape)
        # print("perturbated inputs: ",perturbated_inputs.shape)

        

        # ##visualize perturbated inputs##
        # # print("ddd",perturbated_inputs.shape)
        # visualize_perturbated_inputs = np.transpose(perturbated_inputs.squeeze(0).cpu().detach().numpy(),(1,2,0))
        # #print("sss",visualize_perturbated_inputs.shape)
        # visualize_perturbated_inputs = (visualize_perturbated_inputs - np.min(visualize_perturbated_inputs)) / (np.max(visualize_perturbated_inputs) - np.min(visualize_perturbated_inputs))
        # plt.imshow(visualize_perturbated_inputs)
        # plt.show()
        # ##################
        with torch.no_grad():
            logits = model(perturbated_inputs)
        logits = logits[:,0:1,:,:].cpu().detach().numpy()

        # print("logits shape: ",logits.shape)

        # ##visualize outputs ##
        # visualize_perturbated_outputs = logits.squeeze(0).squeeze(0)
        # visualize_perturbated_outputs = (visualize_perturbated_outputs - np.min(visualize_perturbated_outputs)) / (np.max(visualize_perturbated_outputs) - np.min(visualize_perturbated_outputs))
        # plt.imshow(visualize_perturbated_outputs)
        # plt.show()
        # ##################
        

        masked_logits = logits* target_mask
        # print("masked_logits shape: ",masked_logits.shape)

        # ##visualize masked outputs ##
        # visualize_perturbated_masked_outputs = masked_logits.squeeze(0).squeeze(0)
        # visualize_perturbated_masked_outputs = (visualize_perturbated_masked_outputs - np.min(visualize_perturbated_masked_outputs)) / (np.max(visualize_perturbated_masked_outputs) - np.min(visualize_perturbated_masked_outputs))
        # plt.imshow(visualize_perturbated_masked_outputs)
        # plt.show()
        # ##################
        
        outputs = np.sum(masked_logits.squeeze(1), axis=(1, 2))
        
        # print("outputs shape: ",outputs.shape)
     
        return outputs
