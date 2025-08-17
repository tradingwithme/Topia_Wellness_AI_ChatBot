import React from 'react';
import './Message.css';

function Message({ text, sender }) {
  const messageClass = sender === 'user' ? 'user-message' : 'bot-message';
  return (
    <div className={`message ${messageClass}`}>
      <div className="message-bubble">{text}</div>
    </div>
  );
}

export default Message;
