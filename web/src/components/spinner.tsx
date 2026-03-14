import { Component } from 'solid-js';

type SpinnerProps = {
  size?: number; // Diameter in pixels
  strokeWidth?: number;
  trackColor?: string;
  pathColor?: string;
  class?: string;
};

const Spinner: Component<SpinnerProps> = (props) => {
  const size = props.size ?? 40;
  const strokeWidth = props.strokeWidth ?? 5;
  const trackColor = props.trackColor ?? '#e0e0e0';
  const pathColor = props.pathColor ?? '#3f51b5';

  return (
    <div
      class="spinner__container"
      style={{
        width: `${size}px`,
        height: `${size}px`,
      }}
    >
      <svg
        class={`spinner ${props.class ?? ''}`}
        viewBox="25 25 50 50"
        style={{
          width: '100%',
          height: '100%',
        }}
      >
        <circle
          class="spinner__track"
          cx="50"
          cy="50"
          r="20"
          fill="none"
          stroke-width={strokeWidth}
          stroke={trackColor}
        />
        <circle
          class="spinner__path"
          cx="50"
          cy="50"
          r="20"
          fill="none"
          stroke-width={strokeWidth}
          stroke={pathColor}
          stroke-miterlimit="10"
        />
      </svg>
    </div>
  );
};

export default Spinner;
