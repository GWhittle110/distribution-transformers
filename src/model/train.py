"""
Main training routine for transformer models
"""
from tqdm import tqdm
import time
import itertools

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel

from model.transformers import TransformerModel
from distributions.distributions import CompleteDistribution
from model.utils import get_openai_lr, get_cosine_schedule_with_warmup, torch_nanmean


def train(model: TransformerModel, complete_distribution: CompleteDistribution, epochs: int = 100,
          warmup_epochs: int = 10, steps_per_epoch: int = 100, batch_size: int = 1000, lr: float = None,
          weight_decay: float = 0.01, scheduler: type[LRScheduler] = None, gpu_device: str = "cuda:0",
          compute_prior_loss: bool = False, verbose: bool = False, progress_bar: bool = False) -> TransformerModel:
    """
    Training routine for variational transformers. Note that a validation set is not necessary here as the model will
    see each datapoint only once, so overfitting is impossible and a reduction in train error corresponds to an
    improvement to the model.

    Args:
        model: Model to train.
        complete_distribution: Complete data distribution to sample from.
        epochs: Number of epochs.
            Defaults to 100.
        warmup_epochs: Number of epochs for warmup phase of lr scheduler.
            Defaults to 10.
        steps_per_epoch: Number of batches per epoch.
            Defaults to 100.
        batch_size: Number of samples per batch.
            Defaults to 200.
        lr: Learning rate.
            Defaults to the OpenAI method of determining learning rate.
        weight_decay: Weight decay.
            Defaults to 0.01.
        scheduler: Learning rate scheduler.
            Defaults to cosine annealing with warmup.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        compute_prior_loss: Whether to compute the training loss on the prior distribution for reference.
            Defaults to False.
        verbose: Whether to display metrics during training.
            Defaults to False.
        progress_bar: Whether to display a progress bar during training.
            Defaults to False.

    Returns:
        Trained model.

    """
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    device = torch.device(device)
    model.to(device)

    if lr is None:
        lr = get_openai_lr(model) if lr is None else lr
        print(f"Using OpenAI max lr of {lr}")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs - warmup_epochs)\
        if scheduler is None else scheduler(optimizer)

    def train_epoch(mean_prior_loss: float = None):
        model.train()
        total_loss = 0.
        total_prior_loss = 0.
        total_forward_time = 0.
        total_step_time = 0.
        nan_steps = 0.
        tqdm_iter = tqdm(range(steps_per_epoch), desc='Training Epoch') if progress_bar else None

        before_get_batch = time.time()
        complete_data_sample = complete_distribution.sample((steps_per_epoch, batch_size))
        time_to_get_epoch = time.time() - before_get_batch

        for batch, full_data in enumerate(complete_data_sample):
            tqdm_iter.update() if tqdm_iter is not None else None
            full_data_decoded = complete_distribution.decode_sample(full_data)
            data = torch.cat([full_data_decoded["phi"], full_data_decoded["z"]], dim=-1).to(device)
            targets = full_data_decoded["x"].to(device)
            before_forward = time.time()
            phi_out = model(data)
            forward_time = time.time() - before_forward
            phi_out_decoded = complete_distribution.meta_prior.decode_sample(phi_out)
            losses = -complete_distribution.meta_prior.prior(**phi_out_decoded).log_prob(targets)
            loss, nan_share = torch_nanmean(losses, return_nanshare=True)
            if compute_prior_loss and mean_prior_loss is None:
                with torch.no_grad():
                    prior_losses = -complete_distribution.meta_prior.prior(**complete_distribution.meta_prior.decode_sample(
                        full_data_decoded["phi"].to(device)
                    )).log_prob(targets)
                    prior_loss = torch_nanmean(prior_losses, return_nanshare=False)
                    total_prior_loss += prior_loss.cpu().item()
            nan_steps += nan_share.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_time = time.time() - before_forward

            total_loss += loss.cpu().item()
            total_forward_time += forward_time
            total_step_time += step_time

            if tqdm_iter:
                postfix_dict = {'step time': step_time, 'mean loss': total_loss / (batch + 1)}
                if compute_prior_loss:
                    if mean_prior_loss is None:
                        mean_prior_loss = total_prior_loss / (batch + 1)
                    postfix_dict.update({'mean prior loss': mean_prior_loss})
                tqdm_iter.set_postfix(postfix_dict)

        return {
            "mean_loss": total_loss / steps_per_epoch,
            "mean_prior_loss": total_prior_loss / steps_per_epoch,
            "epoch_load_time": time_to_get_epoch,
            "epoch_time": time.time() - before_get_batch,
            "mean_forward_time": total_forward_time / steps_per_epoch,
            "mean_step_time": total_step_time / steps_per_epoch,
            "nan_share": nan_steps / steps_per_epoch,
        }, mean_prior_loss

    mean_prior_loss = None
    for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
        epoch_start_time = time.time()
        try:
            with sdpa_kernel(SDPBackend.MATH):
                epoch_metrics, mean_prior_loss = train_epoch(mean_prior_loss)
        except Exception as e:
            print("Invalid epoch encountered, skipping...")
            raise e

        if verbose:
            print('\n' + '-' * (161 + 26 * compute_prior_loss))
            print(
                f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s '
                f'| nan share {epoch_metrics["nan_share"]:5.2f} | lr {scheduler.get_last_lr()[0]:5.6f} '
                f'| data time {epoch_metrics["epoch_load_time"]:5.2f} | epoch time {epoch_metrics["epoch_time"]:5.2f} '
                f'| step time {epoch_metrics["mean_step_time"]:5.2f} '
                f'| forward time {epoch_metrics["mean_forward_time"]:5.5f} '
                f'| mean loss {epoch_metrics["mean_loss"]:5.2f} ' +
                f'| mean prior loss {epoch_metrics["mean_prior_loss"]:5.2f} |' if compute_prior_loss else '|')
            print('-' * (161 + 26 * compute_prior_loss))

        scheduler.step()

    model.eval()
    return model.cpu()
