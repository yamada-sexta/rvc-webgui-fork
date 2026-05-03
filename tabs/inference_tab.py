import gradio as gr

import shared
from lib.f0 import PITCH_METHODS, PitchMethod
from loguru import logger
from shared import i18n


def clean() -> dict[str, object]:
    return {"value": "", "__type__": "update"}


def change_choices() -> dict[str, object]:
    names = []
    for entry in shared.weight_root.iterdir():
        if entry.suffix == ".pth":
            names.append(entry.name)
    return {"choices": sorted(names), "__type__": "update"}


def get_pitch_methods() -> list[PitchMethod]:
    return PITCH_METHODS


def get_model_list() -> list[str]:
    print(f"Models: {shared.names}")
    return sorted(shared.names)




def create_inference_tab(app: gr.Blocks) -> None:

    with gr.TabItem(i18n("Inference")):
        gr.api(get_pitch_methods, api_name="get_pitch_methods")
        with gr.Row():
            with gr.Column():
                model_list = sorted(shared.names)
                if not model_list:
                    gr.Textbox(
                        label=i18n("Model"),
                        value=i18n("No models found."),
                        interactive=False,
                        visible=True,
                    )
                    model_dropdown = gr.Dropdown(
                        label=i18n("Model"), choices=[], visible=False
                    )
                else:
                    model_dropdown = gr.Dropdown(
                        label=i18n("Model"), choices=model_list, visible=True
                    )

                refresh_btn = gr.Button(i18n("Refresh"), variant="primary")

                with gr.Group():
                    gr.Markdown(f"### {i18n('Basic')}")
                    audio_input = gr.Audio(
                        label=i18n("Input Audio"),
                        type="numpy",
                    )
                    convert_btn = gr.Button(i18n("Convert"), variant="primary")
                    autoplay_checkbox = gr.Checkbox(label=i18n("Autoplay"), value=False)

                    vc_file_output = gr.Audio(
                        label=i18n("Output Audio"),
                    )

                    def set_autoplay(x: bool) -> dict[str, object]:
                        print(f"Set auto play: {x}")
                        return {"autoplay": x, "__type__": "update"}

                    autoplay_checkbox.input(
                        set_autoplay,
                        [autoplay_checkbox],
                        [vc_file_output],
                    )

            with gr.Column():
                pitch_offset = gr.Slider(
                    label="Pitch Offset",
                    minimum=-24,
                    maximum=24,
                    step=1,
                    value=0,
                )
                resample_sr0 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label=i18n("Resample Rate (Skip if it is 0)"),
                    value=0,
                    step=1,
                    interactive=True,
                )
                rms_mix_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("RMS Mix Rate"),
                    value=0.25,
                    interactive=True,
                )
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protect 0 (Reduce Artifact)"),
                    value=0.33,
                    step=0.01,
                    interactive=True,
                )
                protect0.change(
                    fn=lambda x: logger.info(f"Protect 0 value changed to: {x}"),
                    inputs=[protect0],
                    outputs=[],
                )
            with gr.Column():
                f0method0 = gr.Radio(
                    label=i18n("Pitch Method"),
                    choices=get_pitch_methods(),
                    value="rmvpe",
                    interactive=True,
                )
                vc_log_output = gr.Textbox(label=i18n("Log info"))

        convert_btn.click(
            shared.vc.vc_single,
            [
                audio_input,
                pitch_offset,
                f0method0,
                resample_sr0,
                rms_mix_rate0,
                protect0,
            ],
            [vc_log_output, vc_file_output],
            api_name="infer_convert",
        )
        refresh_btn.click(
            fn=change_choices,
            inputs=[],
            outputs=[model_dropdown],
            api_name="infer_refresh",
        )
        model_dropdown.change(
            fn=shared.vc.get_vc,
            inputs=[
                model_dropdown,
                protect0,
            ],
            outputs=[protect0],
            api_name="infer_change_voice",
        )
        app.load(
            fn=shared.vc.get_vc,
            inputs=[
                model_dropdown,
                protect0,
            ],
            outputs=[protect0],
        )
