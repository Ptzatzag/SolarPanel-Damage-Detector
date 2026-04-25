import torch 
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import sys
from pycocotools import mask as pycoco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import io



def evaluate(model, dataset_val, device, annotation_dir):
    print(f"Evaluate step | Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB, "
        f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    model.eval()
    data_loader = DataLoader(dataset_val,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)))

    predictions = []


    image_ids_val = sorted(dataset_val.image_ids) # Check the attributes name
    #print("Validation image IDs: ", image_ids_val)
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
     #       print(f"Processing image {i+1} of {len(data_loader)}")
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
              outputs = model(images)
            # Move outputs to CPU
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]

########### ADD AUTOCAST HERE
            # Process outputs and convert to COCO format
            for img_idx, output in enumerate(outputs):
                image_id = targets[img_idx]['image_id'].item()


                # Check if model predicted any instances
                if len(output['boxes']) == 0:
                    continue

                # Filter out predictions with low confidence scores (e.g., < 0.05 or 0.1)
                # This helps in mAP calculation by reducing many low-quality FPs
                score_threshold = 0.5 # You can tune this threshold
                keep = output['scores'] > score_threshold

                boxes = output['boxes'][keep].numpy()
                labels = output['labels'][keep].numpy()
                scores = output['scores'][keep].numpy()
                masks = output['masks'][keep].numpy()


                # Post-process masks: Threshold and convert to binary (if not already)
                # and ensure it's (N, H, W) binary mask if not already
                # A common threshold for Mask R-CNN outputs is 0.5
                masks = (masks > 0.5).astype(np.uint8)

                # Iterate through each detected instance
                for j in range(len(boxes)):
                    #print(f"every detected box: {boxes[j]}")
                    bbox = boxes[j]
                    label = labels[j]
                    score = scores[j]
                    mask = masks[j][0]# if masks[j].ndim == 4 else masks[j] # Remove channel dim if present

                    # Convert bounding box from [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height]
                    bbox_coco = [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2] - bbox[0]),
                        float(bbox[3] - bbox[1])
                    ]

                    #mask = (mask > 0.5).astype(np.uint8)
                    # Ensure mask is contiguous for RLE encoding
                    mask = np.asfortranarray(mask)
                    # Convert binary mask to RLE format
                    rle = pycoco_mask.encode(mask)
                    # Convert RLE counts to string for JSON serialization
                    rle['counts'] = rle['counts'].decode('utf-8')
                  #  print("CATEGORY ID: ", int(label))
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label), # Ensure category_id is int
                        "bbox": bbox_coco,
                        "score": float(score),
                        "segmentation": rle
                    })

    # --- COCO Evaluation ---
    # Load ground truth annotations
    coco_gt = COCO(annotation_dir)
    if not predictions:
        print("No predictions generated for evaluation. mAP will be 0.")
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
        return 0.0, 0.0 # Return 0 for both bbox and mask mAP if no predictions

    coco_dt = coco_gt.loadRes(predictions)

    # --- Evaluate bounding box detections ---
    # Redirect stdout to capture pycocotools print statements
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    print("\n--- Evaluating Bounding Boxes ---")
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    # Filter to only evaluate images that were actually in the validation dataset
    coco_eval_bbox.params.imgIds = image_ids_val
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()   #  final summary statistics
    # Extract mAP (IoU=.50:.05:.95) for bounding boxes
    mAP_bbox = coco_eval_bbox.stats[0]   # .stats[0] (mAP@50:95) | .stats[1] (mAP@50) | .stats[2] (mAP@75)

    print("\n--- Evaluating Masks ---")
    coco_eval_mask = COCOeval(coco_gt, coco_dt, 'segm') # Use 'segm' for mask evaluation
    coco_eval_mask.params.imgIds = image_ids_val
    coco_eval_mask.evaluate()
    coco_eval_mask.accumulate()
    coco_eval_mask.summarize()
    # Extract mAP (IoU=.50:.05:.95) for masks
    mAP_mask = coco_eval_mask.stats[0]

    # Restore stdout
    captured_output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    #print(captured_output) # Print captured pycocotools output
    torch.cuda.empty_cache()

    return mAP_bbox, mAP_mask
    