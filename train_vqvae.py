import os
import torch
from torch import nn
from tqdm import tqdm
from vq_vae import VQVAE
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as Compose
from torchvision.utils import make_grid, save_image

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist

def save_model(autoencoder: VQVAE
               ) -> None:
    data = {
        "encoder_state_dict": autoencoder.encoder.state_dict(),
        "decoder_state_dict": autoencoder.decoder.state_dict(),
        "vector_q_state_dict": autoencoder.vector_q.state_dict()
    }

    torch.save(data, "vqvae_weights.pt")

def load_model(model: VQVAE,
               state_dict_path: str = "vqvae_weights.pt") -> VQVAE:
    sd = torch.load(state_dict_path, map_location="cpu",
                    weights_only=True)
    model.encoder.load_state_dict(sd["encoder_state_dict"])
    model.decoder.load_state_dict(sd["decoder_state_dict"])
    model.vector_q.load_state_dict(sd['vector_q_state_dict'])

    return model

@torch.no_grad()
def log_vae_visualizations(model: VQVAE,
                           val_batch: torch.Tensor,
                           epoch: int,
                           epoch_step: int,
                           save_dir: str = "samples",
                           device: str | torch.device = "cpu"
                           ) -> None:
    """
    Saves a comparison of original vs reconstructed images and novel generated samples.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Ground Truth vs Reconstruction
    # Take up to the first 8 images from the batch
    x = val_batch.to(device)[:64]
    B, C, H, W = x.shape
    x_hat, *_ = model(x)

    # Interleave original and reconstructed images side-by-side
    #[orig_1, recon_1, orig_2, recon_2, ...]
    comparison = torch.empty((x.size(0) * 2, C, H, W),
                             dtype=x.dtype,
                             device=device
                             )
    comparison[0::2] = x
    comparison[1::2] = x_hat

    # 2 rows of pairs
    recon_grid = make_grid(comparison,
                           nrow=8,
                           padding=2)
    save_image(recon_grid,
               os.path.join(save_dir,
    f"epoch_{epoch}_step_{epoch_step:03d}_reconstruction.png"),
               )


    B, H, W = 64, 8, 8
    rand_indices = torch.randint(
        0, model.num_embeddings, (B, H, W), device=device
    )

    # embedding.weight has shape (K, D) -> z_q has shape (B, H, W, D)
    z_q = model.vector_q.embedding.weight[rand_indices]
    # Permute to BCHW: (B, D, H, W)
    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    samples = model.decode(z_q)

    gen_grid = make_grid(samples,
                         nrow=8,
                         padding=2
                         )
    save_image(gen_grid,
               os.path.join(save_dir,
                            f"epoch_{epoch}_step_{epoch_step:03d}_generated.png")
               )
    model.train()

def train(lr              : float=3e-4,
          epochs          : int  = 1,
          batch_size      : int  = 64,
          resume_training : bool = False,
          checkpoint_path : str = "vqvae_weights.pt",
          weight_decay    : float=1e-5,
          commitment_cost: float = 0.25
          ):
    device = "cpu"
    BATCH_SIZE = batch_size
    EPOCHS = epochs

    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        assert torch.cuda.is_available(), "DDP is only supported in cuda mode. cuda not available"

        init_process_group(backend="nccl") # nvidia cuda communications library
        ddp_rank       = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device         = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0

        if master_process:
            print(f"[INFO]: DDP run: ddp_rank: {ddp_rank} ddp_local_rank: {ddp_local_rank} World Size: {ddp_world_size}")
    else:
        # Running on 1 GPU
        ddp_rank       = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        if torch.cuda.is_available():
            device     = 'cuda'

    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    torch.manual_seed(42)
    torch.cuda.manual_seed(42) # ignored if we do not have a cuda GPU

    model = VQVAE().to(device=device)
    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    model = torch.compile(model)
    def get_raw_model(model):                                                                                                       
        # 1. Unwrap torch.compile if present                                                                                        
        if hasattr(model, "_orig_mod"):                                                                                             
            model = model._orig_mod                                                                                                 
        # 2. Unwrap DDP if present                                                                                                  
        if hasattr(model, "module"):                                                                                                
            model = model.module                                                                                                    
        return model                                                                                                                

    raw_model = get_raw_model(model)
    optimizer = Adam(lr=lr,
                     params=model.parameters(),
                     weight_decay=weight_decay
                     )

    transforms = Compose.Compose([Compose.ToImage(),
                                  Compose.RandomHorizontalFlip(p=0.5),
                                 Compose.ToDtype(torch.float32, scale=True)
                                  ])
    train_dataset = torchvision.datasets.CIFAR10(root="data",
                                           train=True,
                                           download=True,
                                           transform=transforms)
    test_dataset = torchvision.datasets.CIFAR10(root="data",
                                                train=False,
                                                download=True,
                                                transform=transforms)


    dataset = train_dataset + test_dataset
    sampler = DistributedSampler(dataset=dataset,
                                 num_replicas=ddp_world_size,rank=ddp_rank,
                                 shuffle=True
                                 ) if is_ddp else None
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler
    )

    print("Starting training run...")
    BEST_LOSS = float("inf")

    # grab a fixed batch of data for validation before training starts
    val_batch = next(iter(data_loader))
    if resume_training:
        if master_process:
            print("[INFO]: Resuming training from previous checkpoint.")
        model = load_model(raw_model, checkpoint_path)

    rec_loss   = []
    q_losses   = []
    c_losses   = []
    t_losses   = []
    perplexs   = []
    for epoch in range(1, EPOCHS + 1):
        if is_ddp and sampler is not None:
            sampler.set_epoch(epoch)

        pbar = tqdm(
            data_loader,
            desc=f"Epoch: {epoch:02d}/{EPOCHS:02d}",
            unit=" batch",
            disable=not master_process,
            ncols=100,
            mininterval=2.0
        )

        i = 0
        for x, y in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, q_loss, c_loss, perplexity = model(x)

            vq_loss = q_loss + commitment_cost * c_loss
            
            # x shape: (B, C, H, W)
            mse_loss   = F.mse_loss(x_hat, x, reduction="none").sum(dim=[1, 2, 3]).mean()
            l1_loss    = F.l1_loss(x_hat, x,
                                   reduction='none').sum(dim=[1, 2, 3]).mean()
            recon_loss = mse_loss + 0.25 * l1_loss

            total_loss = recon_loss + vq_loss
            total_loss.backward()
            optimizer.step()


            if is_ddp:
                loss_to_log = total_loss.detach()
                recon_loss_log = recon_loss.detach()
                q_loss_log = q_loss.detach()
                c_loss_log = c_loss.detach()
                dist.all_reduce(perplexity,
                                op=dist.ReduceOp.AVG)
                dist.all_reduce(loss_to_log,
                                op=dist.ReduceOp.AVG)
                dist.all_reduce(recon_loss_log,
                                op=dist.ReduceOp.AVG)
                dist.all_reduce(q_loss_log,
                                op=dist.ReduceOp.AVG)
                dist.all_reduce(c_loss_log,
                                op=dist.ReduceOp.AVG)
            else:
                loss_to_log   = total_loss.detach()
                recon_loss_log=recon_loss.detach()
                q_loss_log    = q_loss.detach()
                c_loss_log    = c_loss.detach()   

            rec_loss.append(recon_loss_log.cpu())
            q_losses.append(q_loss_log.cpu())
            c_losses.append(c_loss_log.cpu())
            t_losses.append(loss_to_log.cpu())
            perplexs.append(perplexity.cpu())
            if master_process:
                if i % 100 == 0:
                    log_vae_visualizations(raw_model,
                                       val_batch[0],
                                           epoch,
                                       i,
                                       save_dir="samples",
                                       device=device)
                if loss_to_log < BEST_LOSS:
                    tqdm.write(f"New best loss: {BEST_LOSS} -> {total_loss:.4f}")
                    BEST_LOSS = loss_to_log
                    save_model(raw_model)

                tqdm.write(f"Total Loss: {loss_to_log:.4f} Q Loss: {q_loss_log:.4f} Reconstruction Loss: {recon_loss_log:.4f} perplexity: {perplexity.item():.1f} C Loss: {c_loss_log.item():.4f}")
                #      flush=True, sep="")
                pbar.set_postfix({
                    "total"     : f"{loss_to_log:.4f}",
                    "recon"     : f"{recon_loss_log:.4f}",
                    "q"         : f"{q_loss_log:.4f}",
                    "c"         : f"{c_loss_log:.4f}",
                    "perplexity": f"{perplexity:.1f}"
                })
                i += 1


    if is_ddp:
        destroy_process_group()

    from matplotlib import pyplot as plt
    t_losses = torch.tensor(t_losses).tolist()
    c_losses= torch.tensor(c_losses).tolist()
    q_losses= torch.tensor(q_losses).tolist()
    rec_loss = torch.tensor(rec_loss).tolist()
    perplexs = torch.tensor(perplexs).tolist()
    plt.plot(range(len(t_losses)), t_losses, label="Total losses")
    plt.plot(range(len(perplexs)), perplexs, label="perplexity")
    plt.plot(range(len(q_losses)), q_losses, label="q_losses")
    plt.plot(range(len(c_losses)), c_losses, label="c_losses")
    plt.plot(range(len(rec_loss)), rec_loss, label="Reconstruction loss")
    plt.legend()
    plt.savefig("losses")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr",
                    type=float,
                    default=3e-4,
                    help="Learning rate. Defaut: 3e-4"
                    )
    ap.add_argument("--epochs",
                    type=int,
                    default=1,
                    help="How many epochs to train the model on. Defaut: 1"
                    )
    ap.add_argument("--batch-size",
                    type=int,
                    default=64,
                    help="Mini-batch size. Defaut: 64"
                    )
    ap.add_argument("--resume-training",
                    action="store_true",
                    help="Whether to resume training from previous checkpoint. Defaut: False"
                    )
    ap.add_argument("--weight-decay",
                    type=float,
                    default=1e-5,
                    help="Weight decay"
                    )
    ap.add_argument("--commitment-cost",
                    type=float,
                    default=0.25,
                    help="Weighting for the commitment loss"
                    )
    ap.add_argument("--checkpoint-path",
                    type=str,
                    default="vqvae_weights.pt",
                    help="If resume_training is passed, weights will be loaded from this chckpoint file"
                    )

    args = ap.parse_args()
    train(lr              = args.lr,
          epochs          = args.epochs,
          batch_size      = args.batch_size,
          resume_training = args.resume_training,
          weight_decay    = args.weight_decay,
          commitment_cost = args.commitment_cost,
          checkpoint_path = args.checkpoint_path
          )
    #torchrun --standalone --nproc_per_node=8 train_vae.py
