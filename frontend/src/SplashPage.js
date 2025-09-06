//'./src/audio/silver-chime-290187.mp3'
import React, { useEffect, useState } from 'react';
import './styles/SplashPage.css';

function SplashPage({ onEnter }) {
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    const audio = new Audio('./src/audio/silver-chime-290187.mp3');
    audio.loop = true;
    audio.volume = 0.3;
    audio.play().catch(() => {
      console.log("Autoplay blocked, user interaction required.");
    });

    return () => {
      audio.pause();
    };
  }, []);

  const goToChatbot = () => {
    setIsLoading(true);
    setTimeout(() => {
      if (onEnter) onEnter(); // Switches to Chatbot in App.js
    }, 1200);
  };

  return (
    <div className="splash-container">
      {/* Animated bubbles background */}
      <div className="bubble-bg">
        <span></span><span></span><span></span><span></span><span></span>
      </div>

      <div className="splash-card">
        <h1 className="title">Topia Global Wellness</h1>
        <p className="tagline">Your journey to wellness begins here. Let’s chat!</p>

        <svg xmlns="http://www.w3.org/2000/svg" 
          width="120" height="120" viewBox="0 0 100 100" 
          className="bubble-svg">
          <circle cx="50" cy="50" r="40" fill="url(#grad1)" />
          <text x="50" y="55" textAnchor="middle" fill="white" fontSize="20" dy=".3em">Chat</text>
          <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style={{ stopColor: "#007bff", stopOpacity: 1 }} />
              <stop offset="100%" style={{ stopColor: "#00c6ff", stopOpacity: 1 }} />
            </linearGradient>
          </defs>
        </svg>

        {isLoading && (
          <div className="loader-container">
            <div className="loader"></div>
          </div>
        )}

        <button className="enter-btn" onClick={goToChatbot}>
          Enter Chat
        </button>
      </div>
    </div>
  );
}

export default SplashPage;
