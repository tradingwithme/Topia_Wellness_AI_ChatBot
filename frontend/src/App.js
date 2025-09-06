import React, { useState, useEffect } from "react";
import SplashPage from "./SplashPage";
import Chatbot from "./Chatbot";
import "./styles/style.css"; // global styles

function App() {
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    // Show splash for 3 seconds before switching
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="app-container">
      {showSplash ? <SplashPage /> : <Chatbot />}
    </div>
  );
}

export default App;
