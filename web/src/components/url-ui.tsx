import { Client } from "@gradio/client";
import { makePersisted } from "@solid-primitives/storage";
import { createOptions, Select } from "@thisbeyond/solid-select";
import { Accessor, createEffect, createSignal, Setter } from "solid-js";

export function UrlUI(props: {
  setClient: Setter<Client | undefined>;
  client: Accessor<Client | undefined>;
}) {
  const [apiUrl, setApiUrl] = makePersisted(createSignal(""), {
    name: "apiUrl",
  });
  const [apiUrls, setApiUrls] = makePersisted(createSignal<string[]>([], {}), {
    name: "apiUrls",
  });
  createEffect(async () => {
    const url = apiUrl();
    try {
      props.setClient(await Client.connect(url));
      const c = props.client();
      if (!c) {
        return;
      }
      console.log("client", c);
      const app_info = await c.view_api();
      console.log("app_info", app_info);
      if (!apiUrls().includes(url)) {
        setApiUrls([...apiUrls(), url]);
      }
    } catch (e) {
      console.error(e);
      props.setClient(undefined);
    }
  });
  return (
    <div
      class="dropdown-container"
      style={{
        "max-width": "530px",
      }}
    >
      <label>API URL{props.client() ? "" : " (not connected)"}</label>
      <Select
        class="select"
        {...createOptions(apiUrls(), {
          createable: true,
        })}
        initialValue={apiUrl()}
        onChange={setApiUrl}
      />
    </div>
  );
}
