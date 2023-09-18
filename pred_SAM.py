import sys
import gc
class SAM:
   
    def __init__(self):
        import sys
        sys.path.append("..")
        from segment_anything import sam_model_registry
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

    def predictSAM(self,x,image,input_point=None,input_label=None,input_boxes=None,mask_input=None,flag=0):
        if flag==1:
                output = self.sam.forward(
                batched_input=[
                    {
                        "image": x,
                        "original_size": image.shape[:2],
                        #'point_coords':input_point,
                        #'point_labels':input_label,
                        'boxes':input_boxes,
                        #'mask_inputs':mask_input
                    }
                ],
                multimask_output=False,
                )[0]
        else :
            output = self.sam.forward(
                batched_input=[
                    {
                        "image": x,
                        "original_size": image.shape[:2],
                        #'point_coords':input_point,
                        #'point_labels':input_label,
                        'boxes':input_boxes,
                        #'mask_inputs':mask_input
                    }
                ],
                multimask_output=False,
            )[0]

        ##Uncomment to select the mask of max iou in case of multimask output is true
        # pred_mask=[]
        # for mask,score in zip(output["masks"],output["iou_predictions"]):
        #   max_score=torch.argmax(score)
        #   pred_mask.append(mask[max_score])
        # pred_mask=torch.stack(pred_mask)
        # pred_mask=pred_mask.unsqueeze(1).cuda()

        ##Uncomment for multimask output which take the last mask
        # pred_mask=[]
        # for mask in output["masks"]:
        #   pred_mask.append(mask[2])
        # pred_mask=torch.stack(pred_mask)
        # pred_mask=pred_mask.unsqueeze(1).cuda()

        pred_mask = output["masks"]
        gc.collect()
        del output
        return pred_mask