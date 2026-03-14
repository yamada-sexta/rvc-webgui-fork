import { Component } from "solid-js"

export const Divider: Component = () => {
  return (
    <div
      style={{
        "background-color": "#E6E6E7", // Correct property for a solid divider
        width: "100%",
        height: "1px",
      }}

    ></div>
  )
}
