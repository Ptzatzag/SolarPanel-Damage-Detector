import torch 
import os 
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from utils.utils import calc_validation_loss, get_lr
from configs.configs import SolarConfig
from evaluate.evaluate import evaluate



def train(model, dataset_train, dataset_val, device, activate_l4, activate_l3, activate_l2):

  ############# Every 50 epochs make a prediction #############
    data_loader = DataLoader(dataset_train,
                             batch_size=3,
                             shuffle=True,
                             collate_fn=lambda x: tuple(zip(*x)))


    for param in model.backbone.body.parameters():   # freeze only resnet body
      param.requires_grad = False


    for param in model.backbone.fpn.parameters():   # keep FPN trainable, better for performance
        param.requires_grad = True


    optimizer = optim.AdamW(model.parameters(), lr=SolarConfig.LEARNING_RATE, weight_decay=SolarConfig.WEIGHT_DECAY)

    best_avg_val_loss = float('inf')
    patience = 20
    epochs_no_improve = 0
    checkpoint_path = os.path.join(SolarConfig.LOGS, f"Best_Model_Clean_CC.pth")
#################### NO IDEA ABOUT SCALER ####################
    scaler = torch.amp.GradScaler('cuda')    # Gradient scaler, because we use low percision float16 and the grad could underflow
#################### NO IDEA ABOUT SCALER ####################

    accumulation_steps = 16  # effective batch size
    for epoch in range(SolarConfig.num_epochs):
        # print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB, "
        # f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

        if epoch == activate_l4:
          print(f"Finetune, activate backbone at epoch {epoch}")
          # for param in model.backbone.parameters():
          #     param.requires_grad = True
          for name, param in model.backbone.named_parameters():
            if "layer4" in name:
              param.requires_grad = True
        if epoch == activate_l3:
          print(f"Activate Layer 3 at epoch {epoch}")
          for name, param in model.backbone.named_parameters():
            if "layer3" in name:
              param.requires_grad = True
        if epoch == activate_l2:
          print(f"Activate Layer 2 at epoch {epoch}")
          for name, param in model.backbone.named_parameters():
            if "layer2" in name: #or "layer1" in name:
              param.requires_grad = True
        # if epoch == activate_l1:   # epoch num 250, and full backbone activation with remaining 10 epochs
        #   print(f"Active Layer 1 at epoch {epoch}")
        #   for name, param in model.backbone.named_parameters():
        #       if "layer1" in name:
        #         param.requires_grad = True

        model.train()
        running_loss = 0.0


        # Run through our batch
        for step, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Use Automatic Mixed Precision (AMP) to reduce the overhead
            with torch.autocast(device_type='cuda', dtype=torch.float16):
              loss_dict = model(images, targets)
              loss = sum(loss for loss in loss_dict.values())
              loss = loss / accumulation_steps   # scale for accumulation

            scaler.scale(loss).backward()   # scale the loss
            #optimizer.zero_grad()
            #losses.backward()
            #optimizer.step()
            if (step + 1) % (accumulation_steps) == 0:
              scaler.step(optimizer)    # unscale the gradients before update
              scaler.update()           # update the scale for the next iteration
              optimizer.zero_grad()


            running_loss += loss.item() * accumulation_steps

        avg_train_loss = running_loss / len(data_loader)
        # Clean up memory
        torch.cuda.empty_cache()
        avg_val_loss = calc_validation_loss(model, dataset_val, device)

        # # Update the learning rate per epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(epoch)

        current_lr = optimizer.param_groups[0]['lr']

        log_data = {
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr,
        }

        # Evaluate 3rd epoch
        if (epoch + 1) % 3 == 0:
            mAP_bbox, mAP_mask = evaluate(model, dataset_val, device, SolarConfig.ANNOTATION_JSON_PATH)
            log_data['mAP_bbox'] = mAP_bbox
            log_data['mAP_mask'] = mAP_mask

        # Log everything for the current epoch in a single call
        wandb.log(log_data)

        # Save the best model
        if avg_val_loss < best_avg_val_loss:
          best_avg_val_loss = avg_val_loss
          epochs_no_improve = 0
          torch.save(model.state_dict(), checkpoint_path)

          # checkpoint_path = os.path.join(args.logs, f"best_model_{epoch}.pth")
          # torch.save(model.state_dict(), checkpoint_path)
          #wandb.save(checkpoint_path)
          print(f"Saved best model at epoch {epoch+1} with val loss: {avg_val_loss:.4f}")
        else:
          epochs_no_improve += 1

        # Add early stopping
        if epochs_no_improve > patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement")
            break

        print(f"Epoch [{epoch+1}/{SolarConfig.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


    # Log model to wandb
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    
