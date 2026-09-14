import torch 
from torch.utils.data import DataLoader
import torchvision.transforms as T
import math
from configs.configs import SolarConfig

config = SolarConfig()  

def calc_validation_loss(model, dataset_val, device):
    model.train()   # Mask RCNN returns list of detections in the eval mode, we need loss dict
    amp_enabled = device.type == "cuda"

    # Hack for simulating eval mode, by switching Batch norm and dropout layers to eval mode
    for module in model.modules():
      if isinstance(module, torch.nn.modules.BatchNorm2d):
        module.eval()
      if isinstance(module, torch.nn.modules.Dropout):
        module.eval()
    
    data_loader = DataLoader(dataset_val,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))
    val_loss = 0.0
    # with torch.no_grad():   # issues with no tracking the gradient while being on train mode
    with torch.set_grad_enabled(False):
      for images, targets in data_loader:
          images = [img.to(device) for img in images]
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
          #########
          with torch.autocast(
              device_type=device.type,
              dtype=torch.float16,
              enabled=amp_enabled,
          ):
            loss_dict = model(images, targets)
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
          #print(f"loss in the eval: {losses.item()}")
          val_loss += losses.item()
      # cleanup to avoid memory accumulation
      del loss_dict, losses, images, targets
      torch.cuda.empty_cache()

    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss
        
        
def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < config.warmup_steps:
            return config.max_lr * (it+1) / (config.warmup_steps)
        # 2) in between, use cosine decay down to min learning rate
        # Clamp decay_ratio to [0, 1] to prevent assertion errors in case of misaligned inputs
        decay_ratio = min(1.0, max(0.0, (it - config.warmup_steps) / (config.num_epochs - config.warmup_steps)))
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # coeff starts at 1 and goes to 0
        return config.min_lr + coeff * (config.max_lr - config.min_lr)
