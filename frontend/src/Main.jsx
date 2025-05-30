import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./Index.css";            // keep global styles

// Mount the root <App /> component
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);