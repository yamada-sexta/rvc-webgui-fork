import { createEffect, createSignal, For, Show, type Component } from 'solid-js';
import { MediaRecorder, register } from 'extendable-media-recorder';
import { connect } from 'extendable-media-recorder-wav-encoder';

import micIcon from "./assets/mic-w.svg"
import stopIcon from "./assets/stop-w.svg"
import openIcon from "./assets/fopen-w.svg"
import { createOptions, Select } from '@thisbeyond/solid-select';
import "@thisbeyond/solid-select/style.css";
import { createDropzone } from '@solid-primitives/upload';
import { makePersisted } from '@solid-primitives/storage';
import { Client } from '@gradio/client';
import { UrlUI } from './components/url-ui';
import { AudioPlayerWithStorage } from './components/audio-player-with-storage';
import { SliderPanel } from './components/silider-pannel';
import Spinner from './components/spinner';
import { Switch } from '@kobalte/core/switch';

const App: Component = () => {
  const [client, setClient] = createSignal<Client>();
  const [isRecording, setIsRecording] = createSignal(false);
  const [isProcessing, setIsProcessing] = createSignal(false);
  const [pitch, setPitch] = makePersisted(createSignal(0), { name: "pitch" });
  const [models, setModels] = createSignal<string[]>([]);
  const [model, setModel] = makePersisted(createSignal<string>(), { name: "model" });
  const [f0Method, setF0Method] = makePersisted(createSignal("rmvpe"), { name: "f0Method" });
  const [indexRate, setIndexRate] = makePersisted(createSignal(0.75), { name: "indexRate" });
  const [protect0, setProtect0] = makePersisted(createSignal(0.33), { name: "protect0" });
  const [rmsMixRate0, setRmsMixRate0] = makePersisted(createSignal(0.25), { name: "rmsMixRate0" });
  const [resultAudioBlob, setResultAudioBlob] = (createSignal<Blob | null>(null));

  const [statusMessage, setStatusMessage] = createSignal("");
  const [inputAudioBlob, setInputAudioBlob] = createSignal<Blob | null>(null);

  // Update models
  createEffect(async () => {
    try {
      if (models().length !== 0) {
        console.log("Already fetched: ", models());
        return;
      }
      const c = client();
      if (!c) {
        console.log("No client");
        return;
      }
      const ms = await c.predict("/get_model_list", {});
      const data = ms.data as [string[]];
      console.log("ms", ms);
      setModels(data[0])
    } catch (e) {
      console.error(e);
      setStatusMessage(`Error getting models: ${e}`);
    }
  })

  // On model change
  createEffect(async () => {
    try {
      const m = model();
      if (!m) {
        console.log("No model");
        return;
      }
      const c = client();
      if (!c) {
        console.log("No client");
        return;
      }
      // Load model
      const loadModelRes = await c.predict("/infer_change_voice", {
        sid: model(),
        param_1: 0,
      })

      console.log("loadModelRes", loadModelRes);

    } catch (e) {
      console.error(e);
      setStatusMessage(`Error getting models: ${e}`);
    }
  })

  // On models change
  createEffect(() => {
    const ms = models();
    const m = model();
    if (ms.length === 0) {
      return;
    }
    if (!m) {
      setModel(ms[0]);
      return;
    }
    if (ms.includes(m)) {
      return;
    }
    setModel(ms[0]);
  })
  const [mediaRecorder, setMediaRecorder] = createSignal<MediaRecorder>();
  let chunks: BlobPart[] = [];
  console.log("svg", micIcon);

  const recordBtn = (<button onClick={handleRecordClick} class='btn-primary btn-icon'>
    {isRecording() ? [
      <img src={stopIcon} class='icon' />,
      "Stop"
    ] : [
      <img src={micIcon} class='icon' />,
      "Record"
    ]
    }
  </button>) as HTMLButtonElement;

  const uploadBtn = (<button
    class='btn-primary'
    onClick={processInput}
    disabled={!inputAudioBlob() || isProcessing()}
  >
    {isProcessing() ?
      <div class='row center-container g1'>
        <Spinner size={24} strokeWidth={6} 
        pathColor='var(--light-2)'
        trackColor='var(--blue-1)'
        ></Spinner>
        Processing...
      </div> : "Process"}
  </button>) as HTMLButtonElement;

  let fileInput: HTMLInputElement | undefined;
  const handleButtonClick = () => {
    fileInput?.click();
  };
  const filePickerBtn = (
    <div>
      <button onClick={handleButtonClick} class='btn-primary btn-icon'>
        <img src={openIcon} class='icon' />
        Open</button>
      <input
        type="file"
        ref={fileInput}
        onChange={handleFileEvent}
        style={{ display: "none" }} // Hide the input
      />
    </div>
  )

  // Check for MediaRecorder support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Your browser does not support audio recording. Please use a modern browser.");
    recordBtn.disabled = true;
  }

  async function handleRecordClick() {
    if (!isRecording()) {
      try {
        let mr = mediaRecorder() as MediaRecorder;
        if (!mr) {
          await register(await connect());
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const settings = stream.getAudioTracks()[0].getSettings();
          console.log({ settings });
          const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/wav' }) as MediaRecorder;
          setMediaRecorder(mediaRecorder);
        }

        mr = mediaRecorder()!;
        if (!mr) {
          console.log("Still no media recorder for some reason...");
          return;
        }
        // Clear existing data
        chunks = [];
        setInputAudioBlob(null);
        setResultAudioBlob(null);
        setStatusMessage("");

        mr.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            chunks.push(e.data);
          }
        };

        mr.onstop = () => {
          console.log("Stopped");
          const recordedBlob = new Blob(chunks, { type: 'audio/wav' });
          setInputAudioBlob(recordedBlob)
        };

        mr.start();
        setIsRecording(true);
      } catch (err) {
        console.error("Error accessing microphone:", err);
        mediaRecorder()?.stop();
        alert("Could not access the microphone.");
      }
    } else {
      // STOP RECORDING
      console.log(mediaRecorder());
      mediaRecorder()?.stop();
      setIsRecording(false);
    }
  };

  async function processInput() {
    const inputAudio = inputAudioBlob();
    if (!inputAudio) {
      alert("No recorded audio to upload!");
      return;
    }
    setIsProcessing(true);
    setResultAudioBlob(null);
    try {
      const c = client();
      if (!c) {
        throw new Error("No client");
      }
      const response = await c.predict("/infer_convert", {
        sr_and_audio: await inputAudioBlob(),
        f0_up_key: pitch(),
        f0_method: f0Method(),
        // file_index2: "",
        index_rate: indexRate(),
        resample_sr: 0,
        rms_mix_rate: rmsMixRate0(),
        protect: protect0(),
      });
      console.log("response", response);
      const [info, audio] = response.data as [string, {
        path: string,
        url: string,
        orig_name: string
      }];
      try {
        const serverWavBlob = await fetch(audio.url).then(res => res.blob());
        setResultAudioBlob(serverWavBlob);
        setStatusMessage("");
      } catch (e) {
        console.warn(e);
        setStatusMessage(info)
      }
    } catch (err: any) {
      console.error("Upload error:", err);
      setStatusMessage(`Error uploading/processing audio: ${err.message}`);
    }
    setIsProcessing(false);
  };

  createEffect(() => {
    if (isProcessing()) {
      setResultAudioBlob(null)
    }
  })

  function handleFile(file: File) {
    console.log("file", file);
    setInputAudioBlob(file);
    setResultAudioBlob(null);
    setStatusMessage("");
  }

  /**
 * Handle the user picking an audio file with the file input.
 */
  function handleFileEvent(e: Event & { currentTarget: HTMLInputElement }) {
    const file = e.currentTarget.files?.[0];
    if (!file) return;
    handleFile(file);
  }

  const [isFileHover, setIsFileHover] = createSignal(false);

  const { setRef: dropzoneRef, files: droppedFiles } = createDropzone({
    onDrop: async files => {
      setIsFileHover(false);
      files.forEach(f => console.log(f));
      if (!files.length) {
        console.log("No file");
        setStatusMessage("No File");
        return;
      }
      setStatusMessage("");
      const file = files[0];
      handleFile(file.file);
    },
    onDragStart: files => {
      files.forEach(f => console.log(f))
      setIsFileHover(false);
    },
    onDragOver: files => {
      setIsFileHover(true);
    },
    onDragLeave: e => {
      setIsFileHover(false);
    }
  });

  let interval: number | undefined = undefined;
  const [autoProcess, setAutoProcess] = makePersisted(createSignal(false), {
    name: "autoProcess",
  });

  createEffect(async () => {
    if (!autoProcess()) {
      return;
    }
    if (interval) {
      clearInterval(interval);
    }
    if (!inputAudioBlob()) {
      return;
    }

    const p = pitch();
    const rate = indexRate();
    const protect = protect0();
    const rmsMixRate = rmsMixRate0();

    interval = setTimeout(async () => {
      await processInput();
      interval = undefined;
    }, 500);
  });

  const [playRes, setPlayRes] = createSignal(false);
  const [autoplayResult, setAutoplayResult] = makePersisted(
    createSignal(false), { name: "autoplayResult" }
  );

  createEffect(() => {
    if (!autoplayResult()) {
      return;
    }
    // setAutoplayResult(false)
    const res = resultAudioBlob();
    if (res) {
      setPlayRes(true);
    }
  })



  return (
    <div
      class={`${isFileHover() ? "drag-over g2" : "g2"} fullscreen`}
      ref={dropzoneRef}
      style={{
        "display": "flex",
        "flex-direction": "column",
        "align-items": "center",
        "justify-content": "center",
        // gap: "16px"
      }}
    >
      <div
        style={{
          "align-self": "center",
          "align-items": "center",
          "justify-content": "center",
          display: "flex",
          "flex-direction": "row",
          gap: "20px",
        }}
      >
        {recordBtn}
        {filePickerBtn}
        <div
          class='row g1'
        >
          Auto Process
          <Switch class="switch"
            checked={autoProcess()}
            onChange={setAutoProcess}
          >
            <Switch.Input class="switch__input" />
            <Switch.Control class="switch__control">
              <Switch.Thumb class="switch__thumb" />
            </Switch.Control>
          </Switch>
        </div>

        <div
          class='row g1'
        >
          Autoplay
          <Switch class="switch"
            checked={autoplayResult()}
            onChange={setAutoplayResult}
          >
            <Switch.Input class="switch__input" />
            <Switch.Control class="switch__control">
              <Switch.Thumb class="switch__thumb" />
            </Switch.Control>
          </Switch>
        </div>

      </div>
      <UrlUI setClient={setClient} client={client}></UrlUI>
      {
        client() ? (
          <>
            <div
              class='center-container row'
              style={{
                gap: "20px",
                "width": "100%",
              }}
            >
              <div class='center-container row' style={
                {
                  "max-width": "300px",
                  "width": "100%",
                  gap: "10px",
                }
              }>
                <label >Model</label>
                <Select
                  class="select"
                  {...createOptions(models())}
                  initialValue={model()}
                  onChange={setModel}
                />
              </div>

              <div class='dropdown-container' style={{ "max-width": "200px" }}>
                <label >F0 Method</label>
                <Select
                  class="select"
                  {...createOptions(["rmvpe", "pm", "crepe"])}
                  initialValue={f0Method()}
                  onChange={setF0Method}
                />
              </div>
            </div>

            <SliderPanel
              title='Config'
              sliders={[
                { value: pitch, setValue: setPitch, min: -24, max: 24, step: 1, label: "Pitch" },
                { value: indexRate, setValue: setIndexRate, min: 0, max: 1, step: 0.01, label: "Index Rate" },
                { value: rmsMixRate0, setValue: setRmsMixRate0, min: 0, max: 1, step: 0.01, label: "RMS Mix Rate" },
                { value: protect0, setValue: setProtect0, min: 0, max: 1, step: 0.01, label: "Protect" },
              ]}
            ></SliderPanel>
            <div class='col w100 g1' style={
              {
                "max-width": "700px",
                "height": "100px",
              }
            }>
              <div class='row w100 g1'>
                <AudioPlayerWithStorage
                  blob={inputAudioBlob}
                  setBlob={setInputAudioBlob}
                  key='input-audio.wav'
                  saveAs={() => {
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    return `input_${timestamp}.wav`;
                  }}
                ></AudioPlayerWithStorage>
                <Show when={inputAudioBlob()}>
                  {uploadBtn}
                </Show>

              </div>
              <AudioPlayerWithStorage
                blob={resultAudioBlob}
                setBlob={setResultAudioBlob}
                key='result-audio.wav'
                saveAs={() => {
                  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                  const m = model();
                  if (!m) {
                    alert("No model selected!");
                    return "";
                  }
                  const fileName = `${m.replace(".pth", "")}_${timestamp}.wav`;
                  return fileName;
                }}
                playControls={[playRes, setPlayRes]}
              ></AudioPlayerWithStorage>
            </div>
            <Show
              when={statusMessage()}
            >
              <div style="font-weight: bold; width: 100%; user-select: text;">
                {statusMessage()}
              </div>
            </Show>

          </>
        ) : (
          <div>Client Not Valid</div>
        )
      }
    </div >
  );
};

export default App;
