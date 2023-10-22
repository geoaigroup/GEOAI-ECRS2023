import segmentation_models_pytorch as smp
import sys
from tqdm import tqdm
import os
import cv2
import torch
# import lookup_table as lut


class Epoch:
    def __init__(
        self,
        model,
        utility_model,
        loss,
        optimizer = None,
        s_phase = "test",
        p_dir_export = None,
        device = "cpu",
        verbose = True,
        writer = None,
    ):
        self.model = model
        self.utility = utility_model
        self.loss = loss
        self.optimizer = optimizer
        if s_phase not in ["training", "validation", "test"]:
            raise ValueError(
                f'Incorrect value for s_phase: "{s_phase}"\n'
                f'Please use one of: "training", "validation", "test"'
            )
        self.s_phase = s_phase
        self.p_dir_export = p_dir_export
        self.device = device
        self.verbose = verbose
        self.writer = writer
        self._to_device()
    
    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
    
    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s
    
    def batch_update(self, image, target = None):
        image = image.to(torch.float32)
        image -= (0.5 * 255.0)
        image /= (0.5 * 255.0)

        if self.s_phase == "training":
            # perform inference, optimization and loss evaluation
            self.optimizer.zero_grad()
            self.utility.eval()
            target = target.to(torch.float32)
            
            B = self.model.forward(image)
            pred = B 
            B = torch.sigmoid(B)
            normal_distribution = torch.distributions.normal.Normal(0, 1)
            epsilon = normal_distribution.sample(B.shape).type_as(B)
            # noise = epsilon * (B * 5.0)
            
            noise = epsilon * B
            noised_image = image + noise 

            
            utility_pred_mask = (self.utility(noised_image))
            utility_pred_mask=utility_pred_mask[:,0,:,:]
            # print("check these: ", utility_pred_mask.shape,utility_pred_mask.min(),utility_pred_mask.max(),  target.shape,target.min(), target.max())

            loss = self.loss(utility_pred_mask, target)- 0.001* torch.mean(B.log())

    
            loss_components = [self.loss(utility_pred_mask, target),- 0.01* torch.mean((B).log())]

            
            loss.backward()
            self.optimizer.step()
            
            
            return loss, pred , utility_pred_mask.unsqueeze(1), loss_components
        
        elif self.s_phase == "validation":
            # perform inference and loss evaluation
            with torch.inference_mode():
                self.utility.eval()
                
                target = target.to(torch.float32)
                
                B = self.model.forward(image)# (1,1,512,512)
                pred = B
                B = torch.sigmoid(B)
                
                normal_distribution = torch.distributions.normal.Normal(0, 1)
                epsilon = normal_distribution.sample(B.shape).type_as(B)
                # noise = epsilon * (B * 5.0)
                noise = epsilon * B
                noised_image = image + noise
                
                utility_pred_mask = (self.utility(noised_image))
                utility_pred_mask = utility_pred_mask[:,0,:,:]
            
                loss = self.loss(utility_pred_mask, target)- 0.01* torch.mean((B).log())
                
                loss_components = [self.loss(utility_pred_mask, target),- 0.01* torch.mean((B).log())]

            return loss, pred , utility_pred_mask.unsqueeze(1) , loss_components
        
        else:  # assume "test"
            # perform inference only
            with torch.inference_mode():
                prediction = self.model.forward(image)
            return None, prediction
    
    def on_epoch_start(self):
        if self.s_phase == "training":
            self.model.train()
        else:  # assume "validation " or "test"
            self.model.eval()
    
    def run(self, dataloader, i_epoch = -1):
        self.on_epoch_start()
        logs = {}
        n_iteration_sum = 0
        l_loss_sum = 0
        BCE_sum = 0
        B_sum = 0
        n_iteration_sum = 0
        d_confusion = {
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
        }
        with tqdm(
            dataloader,
            desc = self.s_phase,
            file = sys.stdout,
            disable = not (self.verbose)
        ) as iterator:
            # image/target tensors have shape [n_patch, n_chan., height, width]
            # l_p_image/l_p_target tensors have shape: [n_patch] (~ lists)
            # and contain the paths to the image and target files, respectively
            for image, target, l_p_image, l_p_target in iterator:
                n_iteration_sum += 1
                image = image.to(self.device)
                if self.s_phase != "test":  # if valid target is available
                    target = target.unsqueeze(dim = 1).to(dataloader.dataset.device)
                    target.squeeze_(dim = 1)
                    target= target/255
                    target = target.long()
                    target = target.to(self.device)
                    loss, logits , util_logits, loss_components = self.batch_update(image, target)
                    
                    prediction = (torch.where(util_logits > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))).to(torch.long)
                    # print("down prediction shape: ", prediction.shape)
                    # print("down target shape: ", target.shape)
                    
                    # update loss logs
                    loss_components= loss_components
                    loss_value = loss.cpu().detach().numpy()
                    
                    BCE_sum += loss_components[0].cpu().detach().numpy()
                    B_sum += loss_components[1].cpu().detach().numpy()
                    l_loss_sum += loss_value
                    
                    loss_logs = {
                        self.loss.__name__: l_loss_sum / n_iteration_sum
                    }
                    BCE_logs = {
                        "BCE_only": BCE_sum / n_iteration_sum
                    }
                    B_logs = {
                        "B": B_sum / n_iteration_sum
                    }
                    
                    logs.update(loss_logs)
                    logs.update(BCE_logs)
                    logs.update(B_logs)
                    
                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)
                    # update confusion matrix
             
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        prediction,
                        target.unsqueeze(1),
                        mode = "binary",

                    )
                    # print("check these: ", prediction.shape, prediction.min(), prediction.max(), target.shape, target.min(), target.max())
                    

                    # sum statistics over image dimension and update
                    if d_confusion["tp"] is None:
                        d_confusion["tp"] = tp.sum(dim = 0, keepdim = True)
                    else:
                        d_confusion["tp"] += tp.sum(dim = 0, keepdim = True)
                    if d_confusion["fp"] is None:
                        d_confusion["fp"] = fp.sum(dim = 0, keepdim = True)
                    else:
                        d_confusion["fp"] += fp.sum(dim =0, keepdim = True)
                    if d_confusion["fn"] is None:
                        d_confusion["fn"] = fn.sum(dim = 0, keepdim = True)
                    else:
                        d_confusion["fn"] += fn.sum(dim = 0, keepdim = True)
                    if d_confusion["tn"] is None:
                        d_confusion["tn"] = tn.sum(dim = 0, keepdim = True)
                    else:
                        d_confusion["tn"] += tn.sum(dim = 0, keepdim = True)

                else:  # in test phase, no target available
                    _, logits, _, _ = self.batch_update(image)
                    prediction = logits.argmax(axis = 1, keepdim = True) # it is useless for now, but needs to be changed
                # export individual predictions as images
                # (skip if no export path was given [default])
                if self.p_dir_export is None:
                    continue
                if prediction.device != dataloader.dataset.device:
                    prediction = prediction.to(dataloader.dataset.device)
                # for i_image, p_target in enumerate(l_p_target):
                    # get target filename
                    # fn_target = os.path.basename(p_target)
                    # fn_prediction_id = fn_target.replace(
                    #     "_gtFine_labelIds.png",
                    #     "_gtFine_predictionIds.png",
                    # )
                    # fn_prediction_color = fn_target.replace(
                    #     "_gtFine_labelIds.png",
                    #     "_gtFine_color.png",
                    # )
                    # p_export_prediction_id = os.path.join(
                    #     self.p_dir_export,
                    #     fn_prediction_id,
                    # )
                    # p_export_prediction_color = os.path.join(
                    #     self.p_dir_export,
                    #     fn_prediction_color,
                    # )
                    # convert prediction values from train_id to id
                    # prediction_id = lut.lookup_chw(
                    #     td_u_input = prediction[i_image].byte(),
                    #     td_i_lut = dataloader.dataset.th_i_lut_trainid2id,
                    # ).permute((1, 2, 0))
                    # ar_f_prediction_id = prediction_id.detach().cpu().numpy()
                    # convert prediction values from train_id to color
                    # prediction_color = lut.lookup_chw(
                    #     td_u_input = prediction[i_image].byte(),
                    #     td_i_lut = dataloader.dataset.th_i_lut_trainid2color,
                    # ).permute((1, 2, 0))
                    # ar_f_prediction_color = prediction_color.detach().cpu().numpy()
                    # # convert from RGB to BGR
                    # ar_f_prediction_color = cv2.cvtColor(
                    #     ar_f_prediction_color,
                    #     cv2.COLOR_RGB2BGR,
                    # )
                    # save prediction image
                    # cv2.imwrite(p_export_prediction_id, ar_f_prediction_id)
                    # cv2.imwrite(p_export_prediction_color, ar_f_prediction_color)
        # compute metrics
        if self.s_phase != "test":  # if valid target is available
            logs["iou_score"] = smp.metrics.functional.iou_score(
                tp = d_confusion["tp"],
                fp = d_confusion["fp"],
                fn = d_confusion["fn"],
                tn = d_confusion["tn"],
                reduction = "micro"
            ).detach().cpu().numpy()
            
            logs["iou_score_macro"] = smp.metrics.functional.iou_score(
                tp = d_confusion["tp"],
                fp = d_confusion["fp"],
                fn = d_confusion["fn"],
                tn = d_confusion["tn"],
                reduction = "macro"
            ).detach().cpu().numpy()
            
            
        # write logs to Tensorboard
        # if self.writer is not None:
        #     self.writer.add_scalar(
        #         f"Losses/{self.loss.__name__}",
        #         logs[self.loss.__name__],
        #         i_epoch,
        #     )
        #     self.writer.add_scalar(
        #         "Metrics/IoU",
        #         logs["iou_score"],
        #         i_epoch,
        #     )
        #     # use last computed batch for generating image logs (max. 4 images)
        #     self.writer.add_images(
        #         "Predictions/Color",
        #         # img_tensor = lut.lookup_nchw(
        #         #     td_u_input = prediction[:4].byte(),
        #         #     td_i_lut = dataloader.dataset.th_i_lut_trainid2color,
        #         # ),
        #         global_step = i_epoch,
        #     )
        #     self.writer.add_images(
        #         "Targets/Color",
        #         # img_tensor = lut.lookup_nchw(
        #         #     td_u_input = target[:4].unsqueeze(dim = 1).byte(),
        #         #     td_i_lut = dataloader.dataset.th_i_lut_trainid2color,
        #         # ),
        #         global_step = i_epoch,
        #     )
        #     self.writer.add_images(
        #         "Images/Color",
        #         # approximate de-normalization
        #         img_tensor = ((image[:4] + 2) * 64).round().clamp(0, 255).byte(),
        #         global_step = i_epoch,
        #     )
        #     self.writer.flush()
        return logs
