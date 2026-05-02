import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

now_dir = Path.cwd()
sys.path.append(str(now_dir))

import datetime

from infer.lib.train import utils
from loguru import logger
from lib.accelerate_utils import get_accelerator, use_half_precision

hps = utils.get_hparams()
import torch
from torch import nn

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

from torch.nn import functional as F
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    TextAudioCollateMultiNSFsid,
    TextAudioLoaderMultiNSFsid,
)

if hps.version not in {"v2", "v3"} or int(hps.if_f0) != 1:
    raise ValueError("Training only supports v2/v3 models with f0 enabled.")

from infer.lib.infer_pack.models import (
    MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    SynthesizerTrnMs768NSFsid as RVC_Model_f0,
)

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee

global_step = 0


class FrozenDacCodec(nn.Module):
    def __init__(self, sample_rate: int, model_id: str = "descript/dac_16khz") -> None:
        super().__init__()
        try:
            from transformers import DacModel
        except ImportError as exc:
            raise RuntimeError(
                "V3 training requires Hugging Face Transformers with DacModel support. "
                "Install it, then rerun training."
            ) from exc

        self.sample_rate = sample_rate
        self.codec = DacModel.from_pretrained(model_id)
        self.codec_sample_rate = int(getattr(self.codec.config, "sampling_rate", 16000))
        self.hop_length = int(getattr(self.codec.config, "hop_length", 512))
        self.codec.eval()
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)

    def _resample_for_codec(self, audio: torch.Tensor) -> torch.Tensor:
        if self.sample_rate == self.codec_sample_rate:
            return audio
        target_length = max(1, round(audio.shape[-1] * self.codec_sample_rate / self.sample_rate))
        return F.interpolate(audio, size=target_length, mode="linear", align_corners=False)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        audio = self._resample_for_codec(audio.clamp(-1.0, 1.0))
        outputs = cast(Any, self.codec.encode(audio))
        return cast(torch.Tensor, outputs.quantized_representation)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        outputs = cast(Any, self.codec.decode(latents))
        audio = cast(torch.Tensor, outputs.audio_values)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio

    def to_audio_device(self, device: torch.device) -> None:
        if next(self.codec.parameters()).device != device:
            self.codec.to(device)


class TransformerDacGenerator(nn.Module):
    def __init__(
        self,
        phone_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        spk_embed_dim: int,
        gin_channels: int,
        dac_latent_dim: int,
        codec: FrozenDacCodec,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.phone_proj = nn.Linear(phone_channels, hidden_channels)
        self.pitch_embed = nn.Embedding(256, hidden_channels)
        self.spk_embed = nn.Embedding(spk_embed_dim, hidden_channels)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=n_heads,
            dim_feedforward=filter_channels,
            dropout=float(p_dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_channels)
        self.proj = nn.Linear(hidden_channels, dac_latent_dim)

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        sid: torch.Tensor,
        ids_slice: torch.Tensor | None = None,
        segment_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.phone_proj(phone) + self.pitch_embed(pitch)
        x = x + self.spk_embed(sid).unsqueeze(1)
        max_len = phone.shape[1]
        padding_mask = torch.arange(max_len, device=phone.device).unsqueeze(0) >= phone_lengths.unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        latents = self.proj(self.norm(x)).transpose(1, 2)
        latents = latents.masked_fill(padding_mask.unsqueeze(1), 0.0)
        if ids_slice is not None and segment_frames is not None:
            latents = commons.slice_segments(latents, ids_slice, segment_frames)
        self.codec.to_audio_device(latents.device)
        y_hat = self.codec.decode(latents)
        return y_hat, latents, phone_lengths


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    training_logger = utils.get_logger(hps.model_dir, stdout=True)
    training_logger.bind(
        event="ui_progress",
        detail_event="train_started",
        stage="train",
        current=0,
        total=max(hps.total_epoch, 1),
        fraction=0.0,
        message=f"Starting training 0/{hps.total_epoch} epochs",
    ).info("Starting training")
    run(hps, training_logger)


def run(hps, training_logger):
    global global_step
    training_logger.bind(
        event="train_hparams",
        hparams=utils.hparams_to_dict(hps),
    ).info("Loaded training configuration")
    training_logger.bind(
        event="ui_progress",
        detail_event="train_setup",
        stage="train",
        current=0,
        total=max(hps.total_epoch, 1),
        fraction=0.0,
        message="Preparing training data and models...",
    ).info("Preparing training setup")
    utils.check_git_hash(hps.model_dir)
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hps.train.seed)
    accelerator = get_accelerator()
    use_fp16 = use_half_precision()
    training_logger.info(f"Accelerate selected device: {accelerator.device}")
    if hps.version == "v3" and (hps.pretrainG or hps.pretrainD):
        raise ValueError("V3 has no base model. Do not pass pretrained G/D paths.")

    train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=True,
        pin_memory=accelerator.device.type != "cpu",
        collate_fn=collate_fn,
        batch_size=hps.train.batch_size,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.version == "v3":
        codec = FrozenDacCodec(hps.data.sampling_rate, hps.model.dac_model_type)
        net_g = TransformerDacGenerator(
            768,
            hps.model.hidden_channels,
            hps.model.transformer_ffn_channels,
            hps.model.n_heads,
            hps.model.transformer_layers,
            hps.model.p_dropout,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.model.dac_latent_dim,
            codec,
        )
    else:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            is_half=use_fp16,
            sr=hps.sample_rate,
        )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    try:  # If it can load, automatically resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D mostly loads fine
        training_logger.info("Loaded discriminator checkpoint")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # If it can't load the first time, load pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            training_logger.info(f"Loading pretrained generator from {hps.pretrainG}")
            training_logger.info(
                net_g.load_state_dict(
                    torch.load(hps.pretrainG, map_location="cpu", weights_only=False)[
                        "model"
                    ]
                )
            )
        if hps.pretrainD != "":
            training_logger.info(f"Loading pretrained discriminator from {hps.pretrainD}")
            training_logger.info(
                net_d.load_state_dict(
                    torch.load(hps.pretrainD, map_location="cpu", weights_only=False)[
                        "model"
                    ]
                )
            )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    (
        net_g,
        net_d,
        optim_g,
        optim_d,
        train_loader,
        scheduler_g,
        scheduler_d,
    ) = accelerator.prepare(
        net_g,
        net_d,
        optim_g,
        optim_d,
        train_loader,
        scheduler_g,
        scheduler_d,
    )

    target_total_epoch = int(hps.total_epoch)
    if epoch_str > target_total_epoch:
        training_logger.warning(
            f"Latest checkpoint starts at epoch {epoch_str}, which is beyond requested total_epoch {target_total_epoch}. Nothing to train."
        )
        return

    for epoch in range(epoch_str, target_total_epoch + 1):
        train_and_evaluate(
            0,
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            accelerator,
            [train_loader, None],
            training_logger,
            None,
        )


def train_and_evaluate(
    rank,
    epoch: int,
    hps,
    nets,
    optims,
    schedulers,
    accelerator,
    loaders,
    logger,
    dac_loss,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    # if writers is not None:
    #     writer, writer_eval = writers

    if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator
    data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            spec,
            spec_lengths,
            wave,
            wave_lengths,
            sid,
        ) = info
        dac_latents: torch.Tensor | None = None
        target_latents: torch.Tensor | None = None
        y_mel: torch.Tensor | None = None
        y_hat_mel: torch.Tensor | None = None
        z_p: torch.Tensor | None = None
        logs_q: torch.Tensor | None = None
        m_p: torch.Tensor | None = None
        logs_p: torch.Tensor | None = None
        z_mask: torch.Tensor | None = None

        # Calculate
        with accelerator.autocast():
            if hps.version == "v3":
                codec = net_g.module.codec if hasattr(net_g, "module") else net_g.codec
                segment_size = round(
                    hps.train.segment_size * codec.codec_sample_rate / hps.data.sampling_rate
                )
                segment_frames = max(1, segment_size // codec.hop_length)
                max_start = torch.clamp(phone_lengths - segment_frames, min=0)
                ids_slice = (
                    torch.rand(phone_lengths.shape[0], device=phone_lengths.device)
                    * (max_start + 1).to(torch.float32)
                ).to(torch.long)
                y_hat, dac_latents, _ = net_g(
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    sid,
                    ids_slice,
                    segment_frames,
                )
                wave = codec._resample_for_codec(wave.float())
                wave = commons.slice_segments(wave, ids_slice * codec.hop_length, y_hat.shape[-1])
                y_hat = y_hat[..., : wave.shape[-1]]
                with torch.no_grad():
                    target_latents = codec.encode(wave)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                wave = commons.slice_segments(
                    wave, ids_slice * hps.data.hop_length, hps.train.segment_size
                )  # slice
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )
                with nullcontext():
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )
                if use_half_precision():
                    y_hat_mel = y_hat_mel.half()

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with nullcontext():
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        accelerator.backward(loss_disc)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

        with accelerator.autocast():
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with nullcontext():
                if hps.version == "v3":
                    assert dac_latents is not None
                    assert target_latents is not None
                    length = min(dac_latents.shape[-1], target_latents.shape[-1])
                    loss_recon = (
                        F.l1_loss(
                            dac_latents[..., :length],
                            target_latents[..., :length],
                        )
                        * hps.train.c_dac
                    )
                    loss_mel = loss_recon
                    recon_name = "dac"
                    loss_kl = y_hat.new_zeros(())
                else:
                    assert y_mel is not None
                    assert y_hat_mel is not None
                    loss_recon = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_mel = loss_recon
                    recon_name = "mel"
                    loss_kl = kl_loss(
                        cast(torch.Tensor, z_p),
                        cast(torch.Tensor, logs_q),
                        cast(torch.Tensor, m_p),
                        cast(torch.Tensor, logs_p),
                        cast(torch.Tensor, z_mask),
                    ) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_recon + loss_kl
        optim_g.zero_grad()
        accelerator.backward(loss_gen_all)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        optim_g.step()

        schedulers[1].step()
        schedulers[0].step()

        if global_step % hps.train.log_interval == 0:
            lr = float(optim_g.param_groups[0]["lr"])
            loss_recon_value = min(float(loss_mel), 75.0)
            loss_kl_value = min(float(loss_kl), 9.0)
            total_batches = len(train_loader)
            progress_current = ((epoch - 1) * total_batches) + batch_idx + 1
            progress_total = max(hps.total_epoch * total_batches, 1)
            logger.bind(
                event="ui_progress",
                detail_event="train_progress",
                stage="train",
                epoch=epoch,
                total_epoch=hps.total_epoch,
                batch=batch_idx + 1,
                total_batches=total_batches,
                current=progress_current,
                total=progress_total,
                fraction=progress_current / progress_total,
                message=(
                    f"Epoch {epoch}/{hps.total_epoch}, batch {batch_idx + 1}/{total_batches}, "
                    f"lr {lr:.6f}, {recon_name} loss {loss_recon_value:.3f}"
                ),
                global_step=global_step,
                learning_rate=lr,
                loss_disc=round(float(loss_disc), 4),
                loss_gen=round(float(loss_gen), 4),
                loss_fm=round(float(loss_fm), 4),
                loss_mel=round(loss_recon_value, 4),
                loss_dac=round(loss_recon_value, 4) if hps.version == "v3" else None,
                loss_kl=round(loss_kl_value, 4),
            ).info(
                f"Epoch {epoch}/{hps.total_epoch} batch {batch_idx + 1}/{total_batches} "
                f"lr={lr:.6f} loss_{recon_name}={loss_recon_value:.3f} loss_kl={loss_kl_value:.3f}"
            )
                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(
                #         y_mel[0].data.cpu().numpy()
                #     ),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                #         y_hat_mel[0].data.cpu().numpy()
                #     ),
                #     "all/mel": utils.plot_spectrogram_to_numpy(
                #         mel[0].data.cpu().numpy()
                #     ),
                # }
                # utils.summarize(
                #     writer=writer,
                #     global_step=global_step,
                #     images=image_dict,
                #     scalars=scalar_dict,
                # )
        global_step += 1
    # /Run steps

    if epoch % hps.save_every_epoch == 0:
        model_dir = hps.model_dir
        unwrapped_net_g = accelerator.unwrap_model(net_g)
        unwrapped_net_d = accelerator.unwrap_model(net_d)
        if hps.if_latest == 0:
            utils.save_checkpoint(
                unwrapped_net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                model_dir / f"G_{global_step}.pth",
            )
            utils.save_checkpoint(
                unwrapped_net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                model_dir / f"D_{global_step}.pth",
            )
        else:
            utils.save_checkpoint(
                unwrapped_net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                model_dir / f"G_{2333333}.pth",
            )
            utils.save_checkpoint(
                unwrapped_net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                model_dir / f"D_{2333333}.pth",
            )
        if hps.save_every_weights == "1":
            ckpt = unwrapped_net_g.state_dict()
            saved_path = savee(
                ckpt,
                hps.sample_rate,
                hps.if_f0,
                f"{hps.name}_e{epoch}_s{global_step}",
                epoch,
                hps.version,
                hps,
            )
            logger.info(f"Saved intermediate checkpoint {hps.name}_e{epoch}:{saved_path}")

    logger.bind(
        event="ui_progress",
        detail_event="train_epoch_complete",
        stage="train",
        epoch=epoch,
        total_epoch=hps.total_epoch,
        current=epoch,
        total=hps.total_epoch,
        fraction=epoch / max(hps.total_epoch, 1),
        message=f"Finished epoch {epoch}/{hps.total_epoch}",
        elapsed=epoch_recorder.record(),
    ).info(f"Finished epoch {epoch}/{hps.total_epoch}")
    if epoch >= hps.total_epoch:
        logger.info("Training is done. The program is closed.")

        ckpt = accelerator.unwrap_model(net_g).state_dict()
        final_path = savee(
            ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
        )
        logger.bind(event="train_finished", epoch=epoch, total_epoch=hps.total_epoch).info(
            f"Saved final checkpoint: {final_path}"
        )
        sleep(1)
        return


if __name__ == "__main__":
    main()
