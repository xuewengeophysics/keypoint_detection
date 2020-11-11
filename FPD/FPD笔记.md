# KeypointDetection笔记——FPD

# Fast Human Pose Estimation  

## 1、训练

### 1.1、关键点

```
t_checkpoints = cfg.KD.TEACHER #注意是在student配置文件中修改
train_type = cfg.KD.TRAIN_TYPE #注意是在student配置文件中修改
train_type = get_train_type(train_type, t_checkpoints)
def get_train_type(train_type, checkpoint):
    exist_status = checkpoint and os.path.exists(checkpoint)
    if train_type == 'NORMAL': # NORMAL train, just return
        return train_type
    if train_type == 'FPD' and exist_status: # FPD and existed
        return 'FPD'
    if train_type == 'FPD' and not exist_status: # FPD and not existed, exit
        exit('ERROR: teacher checkpoint is not existed.')
    else: # train type error
        exit('ERROR: please change train type {} to NORMAL or FPD.'.format(train_type))
```

+ **student配置文件中需要修改的地方**：

```yaml
KD:
  TRAIN_TYPE: 'KPD'
  TEACHER: 'models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth'
TRAIN:
  CHECKPOINT: 'models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
```

+ **cfg.TRAIN.CHECKPOINT需要修改的地方**：

```
if cfg.TRAIN.CHECKPOINT: #注意cfg.TRAIN.CHECKPOINT要与学生网络匹配
	load_checkpoint(cfg.TRAIN.CHECKPOINT, model, strict=True, model_info='student_'+cfg.MODEL.NAME)
```

+ **模型训练命令**：

```
python tools/fpd_train.py \
	--tcfg experiments/fpd_coco/hrnet/teacher_w48_256x192_adam_lr1e-3.yaml \
	--cfg experiments/fpd_coco/hrnet/student_w32_256x192_adam_lr1e-3.yaml
```



## 2、FPD模型

![image-20201111093831506](C:\Users\86138\AppData\Roaming\Typora\typora-user-images\image-20201111093831506.png) 

```
def fpd_train(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA

    # s_model switch to train mode and t_model switch to evaluate mode
    model.train()
    tmodel.eval()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        toutput = tmodel(input)
        if isinstance(toutput, list):
            toutput = toutput[-1]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[0], target, target_weight)
            kd_pose_loss = kd_pose_criterion(outputs[0], toutput, target_weight)

            for output in outputs[1:]:
                pose_loss += pose_criterion(output, target, target_weight)
                kd_pose_loss += kd_pose_criterion(output, toutput, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target, target_weight)
            kd_pose_loss = kd_pose_criterion(output, toutput, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(pose_loss.item(), input.size(0))
        kd_pose_losses.update(kd_pose_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
```

