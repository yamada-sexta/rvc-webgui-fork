import { Accessor, createEffect, createMemo, Setter, Show } from "solid-js";
import { AudioPlayer } from "./audio-player";

export function AudioPlayerWithStorage(props: {
  blob: Accessor<Blob | null>;
  setBlob: Setter<Blob | null>;
  key: string;
  autoplay?: boolean;
  saveAs?: Accessor<string>;
  playControls?: [Accessor<boolean>, Setter<boolean>];
}) {
  async function loadAudio() {
    try {
      const opfsRoot = await navigator.storage.getDirectory();
      const fileHandle = await opfsRoot.getFileHandle(props.key);
      const file = await fileHandle.getFile();
      const blob = new Blob([await file.arrayBuffer()], { type: file.type });
      props.setBlob(blob);
    } catch (err) {
      console.log("No existing file found in OPFS:", err);
    }
    return;
  }
  loadAudio();

  createEffect(async () => {
    try {
      const b = props.blob();
      // If blob is undefined, try to load it from storage
      const opfsRoot = await navigator.storage.getDirectory();
      if (!b) {
        return;
      }
      const fileHandle = await opfsRoot.getFileHandle(props.key, {
        create: true,
      });
      const writable = await fileHandle.createWritable();
      await writable.write(b);
      await writable.close();
    } catch (e) {
      console.log(e);
    }
  });

  const src = createMemo(() => {
    const blob = props.blob();
    if (!blob) {
      return null;
    }
    return URL.createObjectURL(blob);
  });
  return (
    <Show when={props.blob()}>
      <div class="row center-container g1" style={{ width: "100%" }}>
        <AudioPlayer
          style={{
            width: "100%",
          }}
          controls
          autoplay={props.autoplay}
          src={src() || undefined}
          saveAs={props.saveAs}
          playControls={props.playControls}
        />
      </div>
    </Show>
  );
}
