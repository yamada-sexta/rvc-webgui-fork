import { Slider } from "@ark-ui/solid";
import { Accessor, Component, createEffect, createSignal, JSX, onCleanup, onMount, Setter } from "solid-js";

import playIcon from "../assets/play-symbolic.svg?url"
import pauseIcon from "../assets/pause-symbolic.svg?url"
import downloadIcon from "../assets/folder-download-symbolic.svg?url"
type AudioPlayerProps = JSX.AudioHTMLAttributes<HTMLAudioElement> & {
    saveAs?: Accessor<string>,
    playControls?: [Accessor<boolean>, Setter<boolean>],
};

export const AudioPlayer: Component<AudioPlayerProps> = (props) => {
    let audioRef: HTMLAudioElement | undefined;
    const [isPlaying, setIsPlaying] = props.playControls || createSignal(false);

    const [currentTime, setCurrentTime] = createSignal(0);
    const [duration, setDuration] = createSignal(0);

    const togglePlay = () => {
        if (!audioRef) return;
        if (audioRef.paused) {
            audioRef.play();
            setIsPlaying(true);
        } else {
            audioRef.pause();
            setIsPlaying(false);
        }
    };

    const formatTime = (time: number) => {
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60).toString().padStart(2, "0");
        return `${minutes}:${seconds}`;
    };

    const onLoadedMetadata = () => {
        if (audioRef) setDuration(audioRef.duration);
    };

    const onSeek = (e: number) => {
        if (audioRef) {
            audioRef.currentTime = e;
            setCurrentTime(audioRef.currentTime);
        }
    };

    onMount(() => {
        const interval = setInterval(() => {
            if (audioRef && isPlaying()) {
                setCurrentTime(audioRef.currentTime);
            }
        }, 500);
        onCleanup(() => clearInterval(interval));
    });

    createEffect(() => {
        if (isPlaying()) {
            audioRef?.play();
        } else {
            audioRef?.pause();
        }
    })

    return (
        <div style={props.style}>
            <audio
                ref={audioRef}
                {...props}
                onTimeUpdate={() => { if (audioRef) setCurrentTime(audioRef.currentTime) }}
                onLoadedMetadata={onLoadedMetadata}
                style={{
                    display: "none",
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
            >
                {props.children}
            </audio>
            <div
                style={{
                    width: "100%",
                    height: "100%",
                    display: "flex",
                    "background-color": "var(--sidebar-bg-color)",
                    "border-radius": "999px",
                    "align-items": "center",
                    "justify-content": "space-between",
                    "padding": "8px 12px 8px 12px",
                    "gap": "8px",
                    "box-sizing": "border-box"
                }}
            >
                <div
                    class="player-btn"
                    onClick={togglePlay}
                >
                    {isPlaying() ? <img src={pauseIcon}></img> : <img src={playIcon}></img>}
                </div>
                <div
                    style={{
                        width: "calc(100% - 220px)", "box-sizing": "border-box"
                    }}
                >
                    <Slider.Root
                        defaultValue={[currentTime()]}
                        value={[currentTime()]}
                        onValueChange={(details) => {
                            onSeek(details.value[0] || 0)
                        }}
                        min={0}
                        max={duration()}
                        step={0.1}
                    >
                        <Slider.Control>
                            <Slider.Track>
                                <Slider.Range />
                            </Slider.Track>
                            <Slider.Thumb index={0}>
                                <Slider.HiddenInput />
                            </Slider.Thumb>
                        </Slider.Control>
                    </Slider.Root>
                </div>
                <div style={{ "text-wrap": "nowrap", "white-space": "nowrap" }}>
                    {formatTime(currentTime())} / {formatTime(duration())}
                </div>
                <button
                    class='player-btn'
                    onClick={() => {
                        if (!audioRef) {
                            return;
                        }
                        const resultUrl = audioRef!.src;
                        if (!resultUrl) {
                            alert("No result audio to download!");
                            return;
                        }
                        const link = document.createElement('a');
                        // audioRef.get
                        link.href = resultUrl;
                        const filename = props.saveAs?.() || "";
                        if (filename) {
                            link.download = filename
                        }
                        link.click();
                        link.remove();
                    }}
                >
                    <img src={downloadIcon}></img>
                </button>
            </div>

        </div>
    );
};