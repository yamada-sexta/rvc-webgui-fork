import { Slider } from "@ark-ui/solid";
import { Accessor, For, Setter } from "solid-js";
import { Divider } from "./divider";
export interface SliderRowProps {
    value: Accessor<number>,
    setValue: Setter<number>,
    min: number,
    max: number,
    step: number,
    label: string,
}
export function SliderRow(props: SliderRowProps) {
    return (
        <div
            style={{
                width: "100%",
                padding: "12px 24px",
                "box-sizing": "border-box",
                "display": "flex",
                "flex-direction": "column",
                "align-items": "center"
            }}
        >
            <Slider.Root
                defaultValue={[props.value()]}
                onValueChange={(details) => {
                    props.setValue(details.value[0] || 0)
                }}
                min={props.min} max={props.max} step={props.step}
            >
                <Slider.Label style={{ "text-wrap": "nowrap" }}>{props.label}: {props.value()}</Slider.Label>
                <div
                    style={{
                        width: "300px",

                        "box-sizing": "border-box",
                        "display": "flex",
                    }}
                >
                    <Slider.Control>
                        <Slider.Track>
                            <Slider.Range />
                        </Slider.Track>
                        <Slider.Thumb index={0}>
                            <Slider.HiddenInput />
                        </Slider.Thumb>
                    </Slider.Control>
                </div>

            </Slider.Root>
        </div>

    )
}



export function SliderPanel(props: { title: string, sliders: SliderRowProps[] }) {
    return (
        <div
            style={{
                "max-width": "550px",
                "display": "flex",
                "flex-direction": "column",
                "gap": "12px"
            }}
        >
            <div style={{
                "font-weight": "600",
                "font-size": "12pt"
            }}>{props.title}</div>
            <div
                style={{
                    "width": "100%",
                    "background-color": "white",
                    "box-shadow": "0 1px 5px rgba(0, 0, 0, 0.1)",
                    "border-radius": "16px",
                }}
            >
                <For each={props.sliders}>
                    {
                        (props, i) =>
                            i() === 0 ?
                                (
                                    <SliderRow
                                        value={props.value}
                                        setValue={props.setValue}
                                        min={props.min}
                                        max={props.max}
                                        step={props.step}
                                        label={props.label}
                                    />
                                ) :
                                (
                                    <>
                                        <Divider></Divider>
                                        <SliderRow
                                            value={props.value}
                                            setValue={props.setValue}
                                            min={props.min}
                                            max={props.max}
                                            step={props.step}
                                            label={props.label}
                                        />
                                    </>

                                )
                    }
                </For>
            </div>
        </div>
    )
}