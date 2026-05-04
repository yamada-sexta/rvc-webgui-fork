import dataclasses
import datetime
import json
import os
import pathlib
import shlex
import shutil
import subprocess
from random import shuffle
from collections.abc import Generator
from time import sleep
from typing import Literal

import gradio as gr
import numpy as np
from loguru import logger
from pydantic import ValidationError
from scipy.io.wavfile import write as write_wav
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

import shared
from configs.v3_config import get_v3_training_config
from lib.json_validation import (
    JsonLogPayload,
    LogEventName,
    ModelVersion,
    SampleRateName,
)
from shared import i18n

ProgressComponent = gr.Progress

SampleRate = SampleRateName
PitchExtractionMethod = Literal["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"]


def read_json_log_records(log_path: Path) -> list[JsonLogPayload]:
    if not log_path.exists():
        return []
    records: list[JsonLogPayload] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = JsonLogPayload.model_validate_json(line)
        except ValidationError:
            continue
        records.append(payload)
    return records


def format_log_messages(records: list[JsonLogPayload]) -> str:
    return "\n".join(
        record.record.message for record in records if record.record.message
    )


def get_latest_event(
    records: list[JsonLogPayload], event_name: LogEventName
) -> JsonLogPayload | None:
    for record in reversed(records):
        if record.record.extra.event == event_name:
            return record
    return None


def parse_json_log_line(line: str) -> JsonLogPayload | None:
    try:
        return JsonLogPayload.model_validate_json(line)
    except ValidationError:
        return None


def get_latest_ui_progress(records: list[JsonLogPayload]) -> tuple[float, str]:
    latest = get_latest_event(records, "ui_progress")
    if latest is None:
        return 0.0, "Starting..."
    extra = latest.record.extra
    fraction = extra.fraction
    message = extra.message
    if message:
        return fraction, message
    stage = extra.stage
    return fraction, str(stage) if stage is not None else "Working..."


def is_skip_update(value: object) -> bool:
    return value == {"__type__": "update"}


def preprocess_dataset(
    audio_dir: Path,
    exp_dir: Path,
    sr: SampleRate,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    log_dir = pathlib.Path(shared.now_dir) / "logs" / exp_dir
    log_path = log_dir / "preprocess.log"
    preprocess_script = pathlib.Path("infer/modules/train/preprocess.py")

    # 1. Validate audio_dir and count files
    if not audio_dir.is_dir():
        error_msg = (
            f"Error: Audio directory '{audio_dir}' not found or is not a directory."
        )
        logger.error(error_msg)
        yield error_msg
        return

    actual_file_count = 0
    try:
        # List all entries in the directory and filter for files
        file_names = [path.name for path in audio_dir.iterdir() if path.is_file()]
        actual_file_count = len(file_names)
        info_msg = f"Found {actual_file_count} files in audio directory: {audio_dir}"
        logger.info(info_msg)
        # yield info_msg # Optionally yield this information to the UI

        if actual_file_count == 0:
            warning_msg = f"Warning: No files found in '{audio_dir}'. Preprocessing script will run, but may not find items to process."
            logger.warning(warning_msg)
            yield warning_msg
            # Update progress to indicate nothing to process, but the step is "complete"
            if progress:
                progress(
                    1.0,
                    desc=f"No files found in {audio_dir}. Preprocessing step initiated.",
                )
    except OSError as e:
        error_msg = (
            f"Error: Could not access audio directory '{audio_dir}' to count files: {e}"
        )
        logger.error(error_msg)
        yield error_msg
        return
    sr_hz = shared.sr_dict[sr]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    cmd = [
        shared.config.python_cmd,
        str(preprocess_script),
        "--inp_root",
        str(audio_dir),
        "--sr",
        str(sr_hz),
        "--n_p",
        str(shared.config.n_cpu),
        "--exp_dir",
        str(log_dir),
        "--per",
        f"{shared.config.preprocess_per:.1f}",
    ]
    if shared.config.noparallel:
        cmd.append("--noparallel")
    logger.info(f"Execute: {shlex.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=shared.now_dir)
    while True:
        records = read_json_log_records(log_path)
        fraction, description = get_latest_ui_progress(records)
        progress(fraction, desc=description)
        sleep(0.5)
        if p.poll() is not None:
            break
    records = read_json_log_records(log_path)
    log = format_log_messages(records)
    logger.info(f"Preprocess stage completed for {exp_dir}")
    yield log


def preprocess_meta(
    experiment_name: str,
    audio_dir: Path,
    audio_files: list[Path] | None,
    sr: SampleRate,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    save_dir = audio_dir / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if audio_files is not None:
        for idx, audio_file in enumerate(audio_files, start=1):
            audio_file_path = pathlib.Path(audio_file)
            shutil.copy(audio_file_path, save_dir / audio_file_path.name)
            progress(idx / max(len(audio_files), 1), "Copying files...")

    for update in preprocess_dataset(
        audio_dir=save_dir,
        exp_dir=pathlib.Path(experiment_name),
        sr=sr,
        progress=progress,
    ):
        yield update


def extract_f0_feature(
    f0method: PitchExtractionMethod,
    exp_dir: Path,
    version: ModelVersion = "v2",
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "extract_f0_feature.log"
    log_path.write_text("")
    if f0method == "rmvpe_gpu":
        cmd = [
            shared.config.python_cmd,
            "infer/modules/train/extract/extract_f0_rmvpe.py",
            "--exp_dir",
            str(log_dir),
        ]
    else:
        cmd = [
            shared.config.python_cmd,
            "infer/modules/train/extract/extract_f0_print.py",
            "--exp_dir",
            str(log_dir),
            "--n_p",
            str(shared.config.n_cpu),
            "--f0method",
            f0method,
        ]
    logger.info(f"Execute: {shlex.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=shared.now_dir)
    while True:
        records = read_json_log_records(log_path)
        fraction, description = get_latest_ui_progress(records)
        progress(fraction, desc=description)
        sleep(0.2)
        if p.poll() is not None:
            break
    if p.wait() != 0:
        yield "F0 extraction failed."
        return
    cmd = [
        shared.config.python_cmd,
        "infer/modules/train/extract_feature_print.py",
        "--exp_dir",
        str(log_dir),
        "--version",
        version,
    ]
    logger.info(f"Execute: {shlex.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=shared.now_dir)
    while True:
        records = read_json_log_records(log_path)
        fraction, description = get_latest_ui_progress(records)
        progress(fraction, desc=description)
        sleep(0.2)
        if p.poll() is not None:
            break
    records = read_json_log_records(log_path)
    log = format_log_messages(records)
    if p.wait() != 0:
        yield (
            "Feature extraction failed.\n" + log
            if log
            else "Feature extraction failed."
        )
        return
    logger.info(f"Feature extraction stage completed for {exp_dir}")
    yield log


def get_pretrained_models(path: Path, f0_str: str, sr2: SampleRate) -> tuple[str, str]:
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            f"assets/pretrained{path_str}/{f0_str}G{sr2}.pth does not exist, so the pretrained generator will not be used"
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            f"assets/pretrained{path_str}/{f0_str}D{sr2}.pth does not exist, so the pretrained discriminator will not be used"
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2: SampleRate) -> tuple[str, str]:
    return get_pretrained_models("_v2", "f0", sr2)


def change_version_and_sr(
    version: ModelVersion, sr2: SampleRate
) -> tuple[dict[str, object], str, str]:
    if version == "v3":
        return (
            {"choices": ["44k"], "value": "44k", "__type__": "update"},
            "",
            "",
        )
    pretrained_g, pretrained_d = change_sr2(sr2 if sr2 != "44k" else "48k")
    return (
        {
            "choices": ["32k", "48k"],
            "value": "48k" if sr2 == "44k" else sr2,
            "__type__": "update",
        },
        pretrained_g,
        pretrained_d,
    )


def change_pretrained_inputs(version: ModelVersion, sr2: SampleRate) -> tuple[str, str]:
    if version == "v3":
        return "", ""
    return change_sr2(sr2)


def ensure_mute_assets(version: ModelVersion, sr2: SampleRate) -> None:
    mute_root = pathlib.Path(shared.now_dir) / "logs" / "mute"
    gt_dir = mute_root / "0_gt_wavs"
    feature_dir = mute_root / "3_feature768"
    f0_dir = mute_root / "2a_f0"
    f0nsf_dir = mute_root / "2b-f0nsf"
    for path in (gt_dir, feature_dir, f0_dir, f0nsf_dir):
        path.mkdir(parents=True, exist_ok=True)

    if version == "v3":
        config = get_v3_training_config()
        sample_rate = config.data.sampling_rate
        hop_length = config.data.hop_length
    else:
        sample_rate = shared.sr_dict[sr2]
        config_key = f"v2/{sr2}.json"
        config = shared.config.json_config[config_key]
        hop_length = config.data.hop_length

    frames = 100
    wav_path = gt_dir / f"mute{sr2}.wav"
    phone_path = feature_dir / "mute.npy"
    pitch_path = f0_dir / "mute.wav.npy"
    pitchf_path = f0nsf_dir / "mute.wav.npy"

    if not wav_path.exists():
        silence = np.zeros(frames * hop_length, dtype=np.int16)
        write_wav(wav_path, sample_rate, silence)
    if not phone_path.exists():
        phone = np.zeros((frames // 2, 768), dtype=np.float32)
        np.save(phone_path, phone)
    if not pitch_path.exists():
        pitch = np.zeros(frames, dtype=np.int64)
        np.save(pitch_path, pitch)
    if not pitchf_path.exists():
        pitchf = np.zeros(frames, dtype=np.float32)
        np.save(pitchf_path, pitchf)


def click_train(
    exp_dir1: str,
    sr2: SampleRate,
    version19: ModelVersion,
    spk_id5: int,
    save_epoch10: int,
    total_epoch11: int,
    batch_size12: int,
    if_save_latest13: str,
    pretrained_G14: str,
    pretrained_D15: str,
    if_save_every_weights18: str,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    if version19 == "v3" and sr2 != "44k":
        yield "v3 currently supports only 44k."
        return
    if version19 == "v2" and sr2 == "44k":
        yield "44k is currently reserved for v3."
        return
    ensure_mute_assets(version19, sr2)
    # Generating file list
    exp_dir = pathlib.Path(shared.now_dir) / "logs" / exp_dir1
    exp_dir.mkdir(parents=True, exist_ok=True)
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / "3_feature768"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"
    missing_dirs = [
        str(path)
        for path in (gt_wavs_dir, feature_dir, f0_dir, f0nsf_dir)
        if not path.exists()
    ]
    if missing_dirs:
        yield (
            "Training data is incomplete. Missing required directories:\n"
            + "\n".join(missing_dirs)
            + "\nRun preprocessing and feature extraction first, then retry training."
        )
        return
    names = (
        {name.split(".")[0] for name in os.listdir(gt_wavs_dir)}
        & {name.split(".")[0] for name in os.listdir(feature_dir)}
        & {name.split(".")[0] for name in os.listdir(f0_dir)}
        & {name.split(".")[0] for name in os.listdir(f0nsf_dir)}
    )
    if not names:
        yield (
            "Training data is incomplete. No matching items were found across "
            "`0_gt_wavs`, `3_feature768`, `2a_f0`, and `2b-f0nsf`."
        )
        return
    opt = []
    for name in names:
        opt.append(
            "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
            % (
                str(gt_wavs_dir).replace("\\", "\\\\"),
                name,
                str(feature_dir).replace("\\", "\\\\"),
                name,
                str(f0_dir).replace("\\", "\\\\"),
                name,
                str(f0nsf_dir).replace("\\", "\\\\"),
                name,
                spk_id5,
            )
        )
    for _ in range(2):
        opt.append(
            "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature768/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
            % (
                shared.now_dir,
                sr2,
                shared.now_dir,
                shared.now_dir,
                shared.now_dir,
                spk_id5,
            )
        )
    shuffle(opt)
    with open(exp_dir / "filelist.txt", "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Training device is managed by Hugging Face Accelerate")
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    config_save_path = exp_dir / "config.json"
    if version19 == "v2" and not config_save_path.exists():
        config_path = f"{version19}/{sr2}.json"
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                dataclasses.asdict(shared.config.json_config[config_path]),
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    cmd = [
        shared.config.python_cmd,
        "infer/modules/train/train.py",
        "-e",
        exp_dir1,
        "-sr",
        sr2,
        "-f0",
        "1",
        "-bs",
        str(batch_size12),
        "-te",
        str(total_epoch11),
        "-se",
        str(save_epoch10),
        "-l",
        str(1 if if_save_latest13 == i18n("Yes") else 0),
        "-sw",
        str(1 if if_save_every_weights18 == i18n("Yes") else 0),
        "-v",
        version19,
    ]
    if pretrained_G14 != "":
        cmd.extend(["-pg", pretrained_G14])
    if pretrained_D15 != "":
        cmd.extend(["-pd", pretrained_D15])
    logger.info(f"Execute: {shlex.join(cmd)}")
    train_log_path = exp_dir / "train.log"
    p = subprocess.Popen(cmd, cwd=shared.now_dir, stdout=subprocess.PIPE, text=True)
    if p.stdout is None:
        raise RuntimeError("Training process stdout was not captured")
    while True:
        line = p.stdout.readline()
        if not line:
            break
        payload = parse_json_log_line(line)
        if payload is None:
            continue
        extra = payload.record.extra
        event = extra.event
        if event == "ui_progress":
            fraction = extra.fraction
            description = extra.message or "Training..."
            progress(fraction, desc=description)

    return_code = p.wait()
    train_records = read_json_log_records(train_log_path)
    summary = format_log_messages(train_records)
    yield f"Training finished with exit code {return_code}.\n{summary}".strip()


def create_train_tab() -> None:

    with gr.TabItem(i18n("Train")):
        with gr.Group():
            gr.Markdown(value=i18n("## Experiment Config"))
            with gr.Row():
                current_date = datetime.date.today()
                formatted_date = current_date.strftime("%Y-%m-%d")
                experiment_name = gr.Textbox(
                    label=i18n("Experiment Name"), value=f"experiment_{formatted_date}"
                )
                target_sr = gr.Radio(
                    label=i18n("Target Sample Rate"),
                    choices=["32k", "48k"],
                    value="48k",
                    interactive=True,
                )
                model_version = gr.Radio(
                    label=i18n("Model Version"),
                    choices=["v2", "v3"],
                    value="v2",
                    interactive=True,
                )

        with gr.Group():
            gr.Markdown(value=i18n("## Preprocess"))
            spk_id = gr.Slider(
                minimum=0,
                maximum=4,
                step=1,
                label=i18n("Speaker ID"),
                value=0,
                interactive=True,
                visible=False,
            )

            with gr.Row():
                with gr.Column():
                    audio_data_root = gr.Textbox(
                        label=i18n("Audio Directory"),
                        value=i18n("./datasets"),
                    )
                    audio_files = gr.Files(
                        type="filepath", label=i18n("Audio Files"), file_types=["audio"]
                    )
                with gr.Column():
                    preprocessing_btn = gr.Button(i18n("Preprocess"), variant="primary")
                    info1 = gr.Textbox(label=i18n("Info"), value="")
                    preprocessing_btn.click(
                        preprocess_meta,
                        [
                            experiment_name,
                            audio_data_root,
                            audio_files,
                            target_sr,
                        ],
                        [info1],
                        api_name="train_preprocess",
                    )
        with gr.Group():
            gr.Markdown(value=i18n("## Extract Pitch"))
            with gr.Row():
                with gr.Column():
                    f0method8 = gr.Radio(
                        label=i18n("Method"),
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                        value="rmvpe_gpu",
                        interactive=True,
                    )
                with gr.Column():
                    extract_f0_btn = gr.Button(i18n("Extract"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Info"), value="", max_lines=8)
                    # progress = gr.Progress()
                    extract_f0_btn.click(
                        extract_f0_feature,
                        [
                            f0method8,
                            experiment_name,
                            model_version,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
        with gr.Group():
            gr.Markdown(value=i18n("## Training Config"))
            with gr.Row():
                with gr.Column():
                    save_epoch = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("Save Frequency"),
                        value=5,
                        interactive=True,
                    )
                    total_epoch = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("Total Epochs"),
                        value=20,
                        interactive=True,
                    )
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("Batch Size"),
                        value=shared.default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n("Only Save Latest Model"),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n("Save Finalized Model Every Time"),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    pretrained_G14 = gr.Textbox(
                        label=i18n("Base Model G"),
                        value="assets/pretrained_v2/f0G48k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=i18n("Base Model D"),
                        value="assets/pretrained_v2/f0D48k.pth",
                        interactive=True,
                    )
                with gr.Column():
                    train_btn = gr.Button(i18n("Train"), variant="primary")
                    training_info = gr.Textbox(
                        label=i18n("Info"), value="", max_lines=10
                    )

            target_sr.change(
                change_pretrained_inputs,
                [model_version, target_sr],
                [pretrained_G14, pretrained_D15],
            )
            model_version.change(
                change_version_and_sr,
                [model_version, target_sr],
                [target_sr, pretrained_G14, pretrained_D15],
            )
            train_btn.click(
                click_train,
                [
                    experiment_name,
                    target_sr,
                    model_version,
                    spk_id,
                    save_epoch,
                    total_epoch,
                    batch_size,
                    if_save_latest13,
                    pretrained_G14,
                    pretrained_D15,
                    if_save_every_weights18,
                ],
                training_info,
                api_name="train_start",
            )
