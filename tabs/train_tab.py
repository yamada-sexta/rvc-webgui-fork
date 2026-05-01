import datetime
import json
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import traceback
from random import shuffle
from collections.abc import Generator
from time import sleep
from typing import Any, Literal

import faiss
import gradio as gr
import numpy as np
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

import shared
from shared import i18n

ProgressComponent = gr.Progress

F0GPUVisible = True
SampleRate = Literal["32k", "40k", "48k"]


def change_f0_method(f0method8: str):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


def read_json_log_records(log_path: pathlib.Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def get_log_record(payload: dict[str, Any]) -> dict[str, Any]:
    record = payload.get("record")
    return record if isinstance(record, dict) else {}


def get_log_extra(payload: dict[str, Any]) -> dict[str, Any]:
    extra = get_log_record(payload).get("extra")
    return extra if isinstance(extra, dict) else {}


def get_log_message(payload: dict[str, Any]) -> str:
    message = get_log_record(payload).get("message")
    return message if isinstance(message, str) else ""


def format_log_messages(records: list[dict[str, Any]]) -> str:
    return "\n".join(
        message for message in (get_log_message(record) for record in records) if message
    )


def get_latest_event(
    records: list[dict[str, Any]], event_name: str
) -> dict[str, Any] | None:
    for record in reversed(records):
        if get_log_extra(record).get("event") == event_name:
            return record
    return None


def parse_json_log_line(line: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_latest_ui_progress(records: list[dict[str, Any]]) -> tuple[float, str]:
    latest = get_latest_event(records, "ui_progress")
    if latest is None:
        return 0.0, "Starting..."
    extra = get_log_extra(latest)
    fraction = parse_float(extra.get("fraction"), 0.0)
    message = extra.get("message")
    if isinstance(message, str) and message:
        return fraction, message
    stage = extra.get("stage")
    return fraction, str(stage) if stage is not None else "Working..."


def is_skip_update(value: object) -> bool:
    return value == {"__type__": "update"}


def preprocess_dataset(
    audio_dir: str | pathlib.Path,
    exp_dir: str,
    sr: SampleRate,
    n_p: int,
    progress=gr.Progress(),
) -> Generator[str, None, None]:
    audio_dir_path = pathlib.Path(audio_dir)
    log_dir = pathlib.Path(shared.now_dir) / "logs" / exp_dir
    log_path = log_dir / "preprocess.log"
    preprocess_script = pathlib.Path("infer/modules/train/preprocess.py")

    # 1. Validate audio_dir and count files
    if not audio_dir_path.is_dir():
        error_msg = (
            f"Error: Audio directory '{audio_dir}' not found or is not a directory."
        )
        logger.error(error_msg)
        yield error_msg
        return

    actual_file_count = 0
    try:
        # List all entries in the directory and filter for files
        file_names = [path.name for path in audio_dir_path.iterdir() if path.is_file()]
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
        str(audio_dir_path),
        str(sr_hz),
        str(n_p),
        str(log_dir),
        str(shared.config.noparallel),
        f"{shared.config.preprocess_per:.1f}",
    ]
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
    audio_dir: str | pathlib.Path,
    audio_files: list[str | pathlib.Path] | None,
    sr: SampleRate,
    n_p: int,
    progress=gr.Progress(),
):
    save_dir = pathlib.Path(audio_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if audio_files is not None:
        for idx, audio_file in enumerate(audio_files, start=1):
            audio_file_path = pathlib.Path(audio_file)
            shutil.copy(audio_file_path, save_dir / audio_file_path.name)
            progress(idx / max(len(audio_files), 1), "Copying files...")

    for update in preprocess_dataset(
        audio_dir=save_dir,
        exp_dir=experiment_name,
        sr=sr,
        n_p=n_p,
        progress=progress,
    ):
        yield update


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(
    gpus: str,
    n_p: int,
    f0method: str,
    if_f0: bool,
    exp_dir: str,
    version19: str,
    gpus_rmvpe: str,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    gpu_ids = gpus.split("-")
    log_dir = pathlib.Path(shared.now_dir) / "logs" / exp_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "extract_f0_feature.log"
    log_path.write_text("")
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = [
                shared.config.python_cmd,
                "infer/modules/train/extract/extract_f0_print.py",
                str(log_dir),
                str(n_p),
                f0method,
            ]
        else:
            if gpus_rmvpe != "-":
                selected_gpu = gpus_rmvpe.split("-")[0]
                cmd = [
                    shared.config.python_cmd,
                    "infer/modules/train/extract/extract_f0_rmvpe.py",
                    selected_gpu,
                    str(log_dir),
                    str(shared.config.is_half),
                ]
            else:
                warning = "RMVPE GPU extraction was selected without a GPU id."
                logger.warning(warning)
                yield warning
                return
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
    # Open multiple processes for different parts
    if len(gpu_ids) > 1:
        logger.warning(
            f"Multiple GPU ids were provided for feature extraction ({gpus}); using only {gpu_ids[0]} to avoid log races."
        )
    selected_gpu = gpu_ids[0] if gpu_ids and gpu_ids[0] else ""
    if selected_gpu:
        cmd = [
            shared.config.python_cmd,
            "infer/modules/train/extract_feature_print.py",
            shared.config.device,
            selected_gpu,
            str(log_dir),
            version19,
            str(shared.config.is_half),
        ]
    else:
        cmd = [
            shared.config.python_cmd,
            "infer/modules/train/extract_feature_print.py",
            shared.config.device,
            str(log_dir),
            version19,
            str(shared.config.is_half),
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
    logger.info(f"Feature extraction stage completed for {exp_dir}")
    yield log


def get_pretrained_models(path_str: str, f0_str: str, sr2: SampleRate):
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


def change_sr2(sr2: SampleRate, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2: SampleRate, if_f0_3: bool, version19: str):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3: bool, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


def click_train(
    exp_dir1: str,
    sr2: SampleRate,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13: str,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    progress=gr.Progress(),
):
    # Generating file list
    exp_dir = "%s/logs/%s" % (shared.now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    f0_dir = ""
    f0nsf_dir = ""
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (
                    shared.now_dir,
                    sr2,
                    shared.now_dir,
                    fea_dim,
                    shared.now_dir,
                    shared.now_dir,
                    spk_id5,
                )
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (shared.now_dir, sr2, shared.now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # Generate config# No need to generate config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info(f"Using GPU setting: {gpus16}")
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                getattr(shared.config, config_path.replace('/', '_').replace('.json', '')).model_dump(exclude_none=True),
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    selected_gpu = gpus16.split("-")[0] if gpus16 else ""
    if gpus16 and "-" in gpus16:
        logger.warning(
            f"Multiple GPU ids were provided for training ({gpus16}); using only {selected_gpu} to avoid subprocess races."
        )
    cmd = [
        shared.config.python_cmd,
        "infer/modules/train/train.py",
        "-e",
        exp_dir1,
        "-sr",
        sr2,
        "-f0",
        str(1 if if_f0_3 else 0),
        "-bs",
        str(batch_size12),
        "-te",
        str(total_epoch11),
        "-se",
        str(save_epoch10),
        "-l",
        str(1 if if_save_latest13 == i18n("Yes") else 0),
        "-c",
        str(1 if if_cache_gpu17 == i18n("Yes") else 0),
        "-sw",
        str(1 if if_save_every_weights18 == i18n("Yes") else 0),
        "-v",
        version19,
    ]
    if selected_gpu:
        cmd.extend(["-g", selected_gpu])
    if pretrained_G14 != "":
        cmd.extend(["-pg", pretrained_G14])
    if pretrained_D15 != "":
        cmd.extend(["-pd", pretrained_D15])
    logger.info(f"Execute: {shlex.join(cmd)}")
    train_log_path = pathlib.Path(exp_dir) / "train.log"
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
        extra = get_log_extra(payload)
        event = extra.get("event")
        if event == "ui_progress":
            fraction = parse_float(extra.get("fraction"), 0.0)
            description = str(extra.get("message", "Training..."))
            progress(fraction, desc=description)

    return_code = p.wait()
    train_records = read_json_log_records(train_log_path)
    summary = format_log_messages(train_records)
    yield f"Training finished with exit code {return_code}.\n{summary}".strip()


def train_index(exp_dir1: str, version19: str, progress=gr.Progress()):
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "Please perform feature extraction first!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform feature extraction first!"

    progress(0.05, desc="Loading features...")  # Initial progress update
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append(
            "Trying to perform KMeans on %s samples to 10k centers." % big_npy.shape[0]
        )
        # yield "\n".join(infos)
        progress(0.2, desc="Performing KMeans...")  # Progress update for KMeans
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * shared.config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    # yield "\n".join(infos)
    progress(0.5, desc="Training FAISS index...")  # Progress update for training
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    # yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    progress(0.7, desc="Adding vectors to index...")
    infos.append("Adding vectors to index...")
    # yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "Successfully built index: added_IVF%s_Flat_nprobe_%s_%s_%s.index"  # Original: "Successfully built index added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(
            "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
            "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (
                shared.outside_index_root,
                exp_dir1,
                n_ivf,
                index_ivf.nprobe,
                exp_dir1,
                version19,
            ),
        )
        infos.append(
            "Linked index to external directory: %s" % (shared.outside_index_root)
        )  # Original: "Linked index to external - %s"
    except:
        infos.append(
            "Failed to link index to external directory: %s"
            % (shared.outside_index_root)
        )  # Original: "Failed to link index to external - %s"
    progress(1.0, desc="Indexing complete!")  # Final progress update
    yield "\n".join(infos)


def one_click_training(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    final_sections: list[str] = []

    # step1: Process data
    progress = gr.Progress()
    progress(0.0, desc=shared.i18n("step1: processing data..."))
    for update in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7):
        if not is_skip_update(update):
            final_sections.append(str(update))

    # step2a: Extract pitch
    progress(0.0, desc=shared.i18n("step2: extracting feature & pitch"))
    for update in extract_f0_feature(
        gpus16,
        np7,
        f0method8,
        if_f0_3,
        exp_dir1,
        version19,
        gpus_rmvpe,
    ):
        if not is_skip_update(update):
            final_sections.append(str(update))

    # step3a: Train model
    progress(0.0, desc=shared.i18n("step3a: Training model"))
    for update in click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
    ):
        if not is_skip_update(update):
            final_sections.append(str(update))
    final_sections.append(
        i18n("Training finished, you can view the console training log or train.log in the experiment folder")
    )

    # step3b: Train index
    progress(0.0, desc=i18n("Training index..."))
    for update in train_index(exp_dir1, version19):
        final_sections.append(update)
    final_sections.append(i18n("Full process completed!"))
    yield "\n\n".join(section for section in final_sections if section).strip()


def create_train_tab():

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
                    choices=["40k", "48k"],
                    value="48k",
                    interactive=True,
                )
                use_f0 = gr.Radio(
                    label=i18n("Pitch Guidance"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                model_version = gr.Radio(
                    label=i18n("Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                    visible=True,
                )
                cpu_count = gr.Slider(
                    minimum=0,
                    maximum=shared.config.n_cpu,
                    step=1,
                    label=i18n("CPU Process Count"),
                    value=int(np.ceil(shared.config.n_cpu / 1.5)),
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
                            cpu_count,
                        ],
                        [info1],
                        api_name="train_preprocess",
                    )
        with gr.Group():
            gr.Markdown(value=i18n("## Extract Pitch"))
            with gr.Row():
                with gr.Column():
                    gpus6 = gr.Textbox(
                        label=i18n(
                            "Enter card numbers separated by '-', e.g., 0-1-2 to use card 0, card 1, and card 2"
                        ),
                        value=shared.gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    gr.Textbox(
                        label=i18n("GPU Info"),
                        value=shared.gpu_info,
                        visible=F0GPUVisible,
                    )
                with gr.Column():
                    gr.Markdown(value=i18n("""### Select pitch extraction algorithm:
                              
                - PM speeds up vocal input.
                
                - DIO speeds up high-quality speech on weaker CPUs.
                
                - Harvest is higher quality but slower.
                
                - RMVPE is the best and slightly CPU/GPU-intensive."""))
                    f0method8 = gr.Radio(
                        label="Method",
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                        value="rmvpe_gpu",
                        interactive=True,
                    )
                    gpus_rmvpe = gr.Textbox(
                        label=i18n(
                            "rmvpe card number config: Enter different process card numbers separated by '-', e.g., 0-0-1 uses 2 processes on card 0 and 1 process on card 1"
                        ),
                        value="%s-%s" % (shared.gpus, shared.gpus),
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                with gr.Column():
                    extract_f0_btn = gr.Button(i18n("Extract"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Info"), value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    # progress = gr.Progress()
                    extract_f0_btn.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            cpu_count,
                            f0method8,
                            use_f0,
                            experiment_name,
                            model_version,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
        with gr.Group():
            gr.Markdown(value=i18n("## Training Config"))
            with gr.Row():
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
                    label=i18n("Batch Size per GPU"),
                    value=shared.default_batch_size,
                    interactive=True,
                )
                if_save_latest13 = gr.Radio(
                    label=i18n("Only Save Latest Model"),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("No"),
                    interactive=True,
                )
                if_cache_gpu17 = gr.Radio(
                    label=i18n("Cache Data to GPU (Recommend for Data < 10 mins)"),
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
            with gr.Row():
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
                target_sr.change(
                    change_sr2,
                    [target_sr, use_f0, model_version],
                    [pretrained_G14, pretrained_D15],
                )
                model_version.change(
                    change_version19,
                    [target_sr, use_f0, model_version],
                    [pretrained_G14, pretrained_D15, target_sr],
                )
                use_f0.change(
                    change_f0,
                    [use_f0, target_sr, model_version],
                    [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                )
                gpus16 = gr.Textbox(
                    label=i18n(
                        "Enter card numbers separated by '-', e.g., 0-1-2 to use card 0, card 1, and card 2"
                    ),
                    value=shared.gpus,
                    interactive=True,
                )
                train_btn = gr.Button(i18n("Train"), variant="primary")
                index_btn = gr.Button(i18n("Extra Feature Index"), variant="primary")
                one_click_btn = gr.Button(i18n("Train Everything"), variant="primary")

                training_info = gr.Textbox(label=i18n("Info"), value="", max_lines=10)
                train_btn.click(
                    click_train,
                    [
                        experiment_name,
                        target_sr,
                        use_f0,
                        spk_id,
                        save_epoch,
                        total_epoch,
                        batch_size,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        model_version,
                    ],
                    training_info,
                    api_name="train_start",
                )
                index_btn.click(
                    train_index, [experiment_name, model_version], training_info
                )
                one_click_btn.click(
                    one_click_training,
                    [
                        experiment_name,
                        target_sr,
                        use_f0,
                        audio_data_root,
                        spk_id,
                        cpu_count,
                        f0method8,
                        save_epoch,
                        total_epoch,
                        batch_size,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        model_version,
                        gpus_rmvpe,
                    ],
                    training_info,
                    api_name="train_start_all",
                )
