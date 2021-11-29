import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
from sklearn import decomposition

from torch.cuda.amp import autocast
import torch.distributed as dist
import sys

sys.path.append('/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/src/')
from training.zero_shot import zero_shot_eval

import pdb
import wandb

import logging


def is_master(args):
    return (not args.distributed) or args.gpu == 0


def get_weights(labels, class_weights):
    weights = torch.ones(labels.shape[0])
    for i in range(labels.shape[0]):
        sample_label = torch.where(labels[i])[0]
        sample_weights = []
        for class_label in sample_label:
            sample_weights.append(class_weights[class_label.item()])
        weights[i] = max(sample_weights)
    return weights


def get_loss(model, images, loss_img, loss_txt, class_weights, texts, labels, args):
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        gathered_labels = [
            torch.zeros_like(labels) for _ in range(world_size)
        ]

        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gathered_labels, labels)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1:]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1:]
        )
        labels = torch.cat(
            [labels]
            + gathered_labels[:rank]
            + gathered_labels[rank + 1:]
        )
        if args.new_model:
            gathered_texts = [torch.zeros_like(texts['input_ids']) for _ in range(world_size)]
            dist.all_gather(gathered_texts, texts['input_ids'])
            texts = torch.cat(
                [texts['input_ids']]
                + gathered_texts[:rank]
                + gathered_texts[rank + 1:]
            )
        else:
            gathered_texts = [torch.zeros_like(texts) for _ in range(world_size)]
            dist.all_gather(gathered_texts, texts)
            texts = torch.cat(
                [texts]
                + gathered_texts[:rank]
                + gathered_texts[rank + 1:]
            )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    if args.Label_grouped: # Basically supervised
        ground_truth = torch.zeros(logits_per_image.shape).float()
        for i in range(len(logits_per_image)):
            mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
            ground_truth[i][mask_same] = 1
    elif args.Healthy_grouped:
        ground_truth = torch.eye(len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
        for i in range(len(logits_per_image)):
            # instead of an eye matrix we have 1 on the diagonal and 1 if the sample from this column belongs to the healthy class
            if labels[i][0] == 1:
                mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
                ground_truth[i][mask_same] = 1
    elif args.Healthy_Caption_grouped:
        ground_truth = torch.eye(len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
        for i in range(len(logits_per_image)):
            if labels[i][0] == 1:
                # replace 0 with 1 if the sample from this column belongs the healthy class
                mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
                ground_truth[i][mask_same] = 1
            else:
                # replace 0 with 1 if the sample from this column belongs the same deseased class and have the same caption
                mask_same = [j for j in range(len(logits_per_image)) if torch.equal(texts[i], texts[j])]
                ground_truth[i][mask_same] = 1
    elif args.Caption_grouped:
        ground_truth = torch.eye(len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
        for i in range(len(logits_per_image)):
            # replace 0 with 1 if the sample from this column belongs the same class and have the same caption
            mask_same = [j for j in range(len(logits_per_image)) if torch.equal(texts[i], texts[j])]
            ground_truth[i][mask_same] = 1
    else:  # Default Clip loss
        ground_truth = torch.arange(len(logits_per_image)).long()

    weights = get_weights(labels, class_weights)

    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        weights = weights.cuda(args.gpu, non_blocking=True)

    loss_vision = loss_img(logits_per_image, ground_truth)
    loss_vision = (loss_vision * weights).mean()
    loss_text = loss_txt(logits_per_text, ground_truth)
    loss_text = (loss_text * weights).mean()

    total_loss = (loss_vision + loss_text) / 2

    return total_loss


def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)

    model.train()

    dataloader, sampler = data['train'].dataloader, data['train'].sampler

    if args.default_loss:
        loss_img = nn.CrossEntropyLoss(reduction='none')
        loss_txt = nn.CrossEntropyLoss(reduction='none')
    else:
        loss_img = nn.BCEWithLogitsLoss(reduction='none')
        loss_txt = nn.BCEWithLogitsLoss(reduction='none')

    if args.use_weights_1:
        # class weights where the weight of a class is: 1 - (class_count / total_count)
        class_weights = {0: 0.5, 1: 0.995, 2: 0.927, 3: 0.964, 4: 0.989, 5: 0.994, 6: 0.993, 7: 0.997,
                   8: 0.856, 9: 0.903, 10: 0.998, 11: 0.879, 12: 0.9984, 13: 0.972, 14: 0.988}
    elif args.use_weights_2:
        # class weights where the weight of a class is: total_count - (num_of_classes / class_count)
        class_weights = {0: 0.133, 1: 14.129, 2: 0.913, 3: 1.868, 4: 6.191, 5: 10.805, 6: 9.501, 7: 26.24,
                   8: 0.461, 9: 0.685, 10: 32.415, 11: 0.552, 12: 30.61, 13: 2.35, 14: 5.681}
    else:
        class_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
                   8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0}

    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, texts, labels = batch

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            if args.new_model:
                for key in texts:
                    texts[key] = texts[key].cuda(args.gpu, non_blocking=True)
            else:
                texts = texts.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss(model, images, loss_img, loss_txt, class_weights, texts, labels, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss(model, images, loss_img, loss_txt, class_weights, texts, labels, args)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale": m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def evaluate(model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return

    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    dataloader = data['val'].dataloader

    if args.default_loss:
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
    else:
        loss_img = nn.BCEWithLogitsLoss()
        loss_txt = nn.BCEWithLogitsLoss()

    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features, all_labels, all_texts = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            images, texts, labels = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                if args.new_model:
                    for key in texts:
                        texts[key] = texts[key].cuda(args.gpu, non_blocking=True)
                else:
                    texts = texts.cuda(args.gpu, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)

            if args.new_model:
                texts = texts['input_ids']

            all_image_features.append(image_features)
            all_text_features.append(text_features)
            all_labels.append(labels)
            all_texts.append(texts)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            if args.Label_grouped:
                ground_truth = torch.zeros(logits_per_image.shape).float()
                for i in range(len(logits_per_image)):
                    mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
                    ground_truth[i][mask_same] = 1
            elif args.Healthy_grouped:
                ground_truth = torch.eye(
                    len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
                for i in range(len(logits_per_image)):
                    # instead of an eye matrix we have 1 on the diagonal and 1 if the sample from this column belongs to the healthy class
                    if labels[i][0] == 1:
                        mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
                        ground_truth[i][mask_same] = 1
            elif args.Healthy_Caption_grouped:
                ground_truth = torch.eye(
                    len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
                for i in range(len(logits_per_image)):
                    if labels[i][0] == 1:
                        # replace 0 with 1 if the sample from this column belongs the healthy class
                        mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
                        ground_truth[i][mask_same] = 1
                    else:
                        # replace 0 with 1 if the sample from this column belongs the same deseased class and have the same caption
                        mask_same = [j for j in range(len(logits_per_image)) if torch.equal(texts[i], texts[j])]
                        ground_truth[i][mask_same] = 1
            elif args.Caption_grouped:
                ground_truth = torch.eye(
                    len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
                for i in range(len(logits_per_image)):
                    # replace 0 with 1 if the sample from this column belongs the same class and have the same caption
                    mask_same = [j for j in range(len(logits_per_image)) if torch.equal(texts[i], texts[j])]
                    ground_truth[i][mask_same] = 1
            else:
                ground_truth = torch.arange(len(logits_per_image)).long()

            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

            total_loss = (
                                 loss_img(logits_per_image, ground_truth)
                                 + loss_txt(logits_per_text, ground_truth)
                         ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        if args.custom_eval:
            metrics = get_metrics_custom(torch.cat(all_image_features),
                                        torch.cat(all_text_features), torch.cat(all_labels), torch.cat(all_texts))
        elif args.custom_eval_no_healthy:
            metrics = get_metrics_custom_no_healthy(torch.cat(all_image_features),torch.cat(all_text_features), torch.cat(all_labels), torch.cat(all_texts))
        else:
            metrics = get_metrics(torch.cat(all_image_features), torch.cat(all_text_features))

        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )
        metrics.update(zero_shot_metrics)

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            if tb_writer is not None:
                for name, val in metrics.items():
                    tb_writer.add_scalar(f"val/{name}", val, epoch)
                if args.t_sne and epoch % 10 == 0:
                    all_labels_onehot = torch.cat(all_labels)
                    all_labels_int = []
                    for index in range(all_labels_onehot.shape[0]):
                        all_labels_int.append(onehot_to_int(all_labels_onehot[index]))
                    all_image_features = torch.cat(all_image_features).cpu().detach().numpy()
                    all_text_features = torch.cat(all_text_features).cpu().detach().numpy()
                    pca = decomposition.PCA(n_components=36)
                    pca.fit(all_image_features)
                    all_image_features = pca.transform(all_image_features)
                    pca.fit(all_text_features)
                    all_text_features = pca.transform(all_text_features)
                    tb_writer.add_embedding(mat=all_image_features, metadata=all_labels_int,
                                            global_step=epoch, tag='val_image_features')
                    tb_writer.add_embedding(mat=all_text_features, metadata=all_labels_int,
                                            global_step=epoch, tag='val_text_features')
        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def evaluate_train(model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return

    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    dataloader = data['train'].dataloader

    if args.default_loss:
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
    else:
        loss_img = nn.BCEWithLogitsLoss()
        loss_txt = nn.BCEWithLogitsLoss()

    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features, all_labels, all_texts = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            images, texts, labels = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                if args.new_model:
                    for key in texts:
                        texts[key] = texts[key].cuda(args.gpu, non_blocking=True)
                else:
                    texts = texts.cuda(args.gpu, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)

            if args.new_model:
                texts = texts['input_ids']

            all_image_features.append(image_features)
            all_text_features.append(text_features)
            all_labels.append(labels)
            all_texts.append(texts)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            if args.Label_grouped:
                ground_truth = torch.zeros(logits_per_image.shape).float()
                for i in range(len(logits_per_image)):
                    mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
                    ground_truth[i][mask_same] = 1
            elif args.Healthy_grouped:
                ground_truth = torch.eye(len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
                for i in range(len(logits_per_image)):
                    # instead of an eye matrix we have 1 on the diagonal and 1 if the sample from this column belongs to the healthy class
                    if labels[i][0] == 1:
                        mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
                        ground_truth[i][mask_same] = 1
            elif args.Healthy_Caption_grouped:
                ground_truth = torch.eye(len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
                for i in range(len(logits_per_image)):
                    if labels[i][0] == 1:
                        #replace 0 with 1 if the sample from this column belongs the healthy class
                        mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
                        ground_truth[i][mask_same] = 1
                    else:
                        # replace 0 with 1 if the sample from this column belongs the same deseased class and have the same caption
                        mask_same = [j for j in range(len(logits_per_image)) if torch.equal(texts[i], texts[j])]
                        ground_truth[i][mask_same] = 1
            elif args.Caption_grouped:
                ground_truth = torch.eye(len(logits_per_image)).float()  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
                for i in range(len(logits_per_image)):
                    # replace 0 with 1 if the sample from this column belongs the same class and have the same caption
                    mask_same = [j for j in range(len(logits_per_image)) if torch.equal(texts[i], texts[j])]
                    ground_truth[i][mask_same] = 1
            else:
                ground_truth = torch.arange(len(logits_per_image)).long()

            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

            total_loss = (
                                 loss_img(logits_per_image, ground_truth)
                                 + loss_txt(logits_per_text, ground_truth)
                         ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        if args.custom_eval:
            metrics = get_metrics_custom(torch.cat(all_image_features),
                                         torch.cat(all_text_features), torch.cat(all_labels), torch.cat(all_texts))
        elif args.custom_eval_no_healthy:
            metrics = get_metrics_custom_no_healthy(torch.cat(all_image_features),torch.cat(all_text_features), torch.cat(all_labels), torch.cat(all_texts))
        else:
            metrics = get_metrics(torch.cat(all_image_features), torch.cat(all_text_features))

        loss = cumulative_loss / num_elements
        metrics.update(
            **{"train_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )
        metrics.update(zero_shot_metrics)

        logging.info(
            f"Eval Train Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            if tb_writer is not None:
                for name, val in metrics.items():
                    tb_writer.add_scalar(f"train_eval/{name}", val, epoch)
                if args.t_sne and epoch % 10 == 0:
                    all_labels_onehot = torch.cat(all_labels)
                    all_labels_int = []
                    for index in range(all_labels_onehot.shape[0]):
                        all_labels_int.append(onehot_to_int(all_labels_onehot[index]))
                    all_image_features = torch.cat(all_image_features).cpu().detach().numpy()
                    all_text_features = torch.cat(all_text_features).cpu().detach().numpy()
                    pca = decomposition.PCA(n_components=36)
                    pca.fit(all_image_features)
                    all_image_features = pca.transform(all_image_features)
                    pca.fit(all_text_features)
                    all_text_features = pca.transform(all_text_features)
                    tb_writer.add_embedding(mat=all_image_features, metadata=all_labels_int,
                                            global_step=epoch, tag='train_image_features')
                    tb_writer.add_embedding(mat=all_text_features, metadata=all_labels_int,
                                            global_step=epoch, tag='train_text_features')

        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"train_eval/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "train_results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features):
    metrics = {}
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = (
        torch.arange(len(text_features)).view(-1, 1).to(logits_per_image.device)
    )

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def get_metrics_custom(image_features, text_features, labels, texts):
    metrics = {}
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.eye(
        len(logits_per_text)).float().to(logits_per_image.device)  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
    for i in range(len(logits_per_text)):
        if labels[i][0] == 1:
            # replace 0 with 1 if the sample from this column belongs the healthy class
            mask_same = [j for j in range(len(logits_per_image)) if torch.equal(labels[i], labels[j])]
            ground_truth[i][mask_same] = 1
        else:
            # replace 0 with 1 if the sample from this column belongs the same deseased class and have the same caption
            mask_same = [j for j in range(len(logits_per_image)) if torch.equal(texts[i], texts[j])]
            ground_truth[i][mask_same] = 1

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True).to(logits_per_image.device)
        preds = torch.zeros(len(logits_per_text)).to(logits_per_image.device)
        for j in range(len(logits_per_text)):
            ground_truth_sample = torch.where(ground_truth[j])[0].view(-1, 1).to(logits_per_image.device)
            preds[j] = torch.min(torch.where(ranking[j] == ground_truth_sample)[1])

        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def get_metrics_custom_no_healthy(image_features, text_features, labels, texts):
    metrics = {}
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.eye(
        len(logits_per_text)).float().to(logits_per_image.device)  # logits_per_image.shape = logits_per_text.shape = ground_truth.shape = batchsize x batchsize
    for i in range(len(logits_per_text)):
            mask_same = [j for j in range(len(logits_per_image)) if torch.equal(texts[i], texts[j])]
            ground_truth[i][mask_same] = 1

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True).to(logits_per_image.device)
        preds = torch.zeros(len(logits_per_text)).to(logits_per_image.device)
        for j in range(len(logits_per_text)):
                ground_truth_sample = torch.where(ground_truth[j])[0].view(-1, 1).to(logits_per_image.device)
                preds[j] = torch.min(torch.where(ranking[j] == ground_truth_sample)[1])

        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def onehot_to_int(lst):
    return [i for i, x in enumerate(lst) if x > 0]
