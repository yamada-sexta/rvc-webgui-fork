import traceback
from pathlib import Path
import gradio as gr
from pydantic import ValidationError

from lib.json_validation import JsonLogPayload

# from shared import i18n
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)


def change_info_(ckpt_path: str | Path):
    train_log = Path(ckpt_path).parent / "train.log"
    if not train_log.exists():
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(train_log, "r") as f:
            first_line = next(
                (line for line in f.read().splitlines() if line.strip()), ""
            )
            payload = JsonLogPayload.model_validate_json(first_line)
            sr = payload.record.extra.hparams.sample_rate
            if sr is None:
                raise ValueError("Missing sample_rate in train.log hparams")
            return sr, "1", "v2"
    except (ValueError, ValidationError):
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


def create_ckpt_processing_tab():
    with gr.TabItem("ckpt processing"):
        with gr.Group():
            gr.Markdown(value="Model fusion, can be used to test timbre fusion")
            with gr.Row():
                ckpt_a = gr.Textbox(label="AModel path", value="", interactive=True)
                ckpt_b = gr.Textbox(label="BModel path", value="", interactive=True)
                alpha_a = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Model A weight",
                    value=0.5,
                    interactive=True,
                )
            with gr.Row():
                sr_ = gr.Radio(
                    label="Target sample rate",
                    choices=["32k", "48k"],
                    value="48k",
                    interactive=True,
                )
                if_f0_ = gr.Radio(
                    label="Does the model have pitch guidance?",
                    choices=["Yes"],
                    value="Yes",
                    interactive=False,
                )
                info__ = gr.Textbox(
                    label="Model info to insert",
                    value="",
                    max_lines=8,
                    interactive=True,
                )
                name_to_save0 = gr.Textbox(
                    label="Saved model name without extension",
                    value="",
                    max_lines=1,
                    interactive=True,
                )
                version_2 = gr.Radio(
                    label="Model version type",
                    choices=["v2"],
                    value="v2",
                    interactive=False,
                )
            with gr.Row():
                but6 = gr.Button("Fuse", variant="primary")
                info4 = gr.Textbox(label="Output info", value="", max_lines=8)
            but6.click(
                merge,
                [
                    ckpt_a,
                    ckpt_b,
                    alpha_a,
                    sr_,
                    if_f0_,
                    info__,
                    name_to_save0,
                    version_2,
                ],
                info4,
                api_name="ckpt_merge",
            )  # def merge(path1,path2,alpha1,sr,f0,info):
        with gr.Group():
            gr.Markdown(
                value="Modify model info (only supports small model files extracted under the weights folder)"
            )
            with gr.Row():
                ckpt_path0 = gr.Textbox(label="Model path", value="", interactive=True)
                info_ = gr.Textbox(
                    label="Model info to modify",
                    value="",
                    max_lines=8,
                    interactive=True,
                )
                name_to_save1 = gr.Textbox(
                    label="Saved filename, empty defaults to the same name as the source file",
                    value="",
                    max_lines=8,
                    interactive=True,
                )
            with gr.Row():
                but7 = gr.Button("Modify", variant="primary")
                info5 = gr.Textbox(label="Output info", value="", max_lines=8)
            but7.click(
                change_info,
                [ckpt_path0, info_, name_to_save1],
                info5,
                api_name="ckpt_modify",
            )
        with gr.Group():
            gr.Markdown(
                value="View model info (only supports small model files extracted under the weights folder)"
            )
            with gr.Row():
                ckpt_path1 = gr.Textbox(label="Model path", value="", interactive=True)
                but8 = gr.Button("View", variant="primary")
                info6 = gr.Textbox(label="Output info", value="", max_lines=8)
            but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
        with gr.Group():
            gr.Markdown(
                value="Model Extract (input large Model path under logs folder), useful when you stop training halfway and the small model wasn't automatically saved, or for testing intermediate models"
            )
            with gr.Row():
                ckpt_path2 = gr.Textbox(
                    label="Model path",
                    value="",
                    interactive=True,
                )
                save_name = gr.Textbox(label="Save name", value="", interactive=True)
                sr__ = gr.Radio(
                    label="Target sample rate",
                    choices=["32k", "48k"],
                    value="48k",
                    interactive=True,
                )
                if_f0__ = gr.Radio(
                    label="Does the model have pitch guidance? 1 for yes, 0 for no",
                    choices=["1"],
                    value="1",
                    interactive=False,
                )
                version_1 = gr.Radio(
                    label="Model version type",
                    choices=["v2"],
                    value="v2",
                    interactive=False,
                )
                info___ = gr.Textbox(
                    label="Model info to insert",
                    value="",
                    max_lines=8,
                    interactive=True,
                )
                but9 = gr.Button("Extract", variant="primary")
                info7 = gr.Textbox(label="Output info", value="", max_lines=8)
                ckpt_path2.change(
                    change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                )
            but9.click(
                extract_small_model,
                [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                info7,
                api_name="ckpt_extract",
            )
