import React from "react";
import "./styles/Message.css";

function Message({ text, sender, timestamp, showFeedback, onApprove, onCorrection }) {
  return (
    <div className={`message-row ${sender === "user" ? "user" : "bot"}`}>
      <div className="message-bubble">
        <p>{text}</p>
        <span className="message-time">
          {new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>

      {sender === "bot" && showFeedback && (
        <div className="feedback-buttons">
          <button className="approve-btn" onClick={onApprove}>👍</button>
          <button className="correction-btn" onClick={onCorrection}>✏️</button>
        </div>
      )}
    </div>
  );
}

export default Message;