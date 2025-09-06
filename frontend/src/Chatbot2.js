import React, { useState } from 'react';
import Message from './Message';
import './styles/Chatbot.css';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [hideTyping, setHideTyping] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const now = new Date().toISOString();

    // Add user message with timestamp
    const newMessages = [
      ...messages,
      { text: input, sender: 'user', timestamp: now }
    ];
    setMessages(newMessages);
    setInput("");

    // Show typing indicator
    setIsTyping(true);
    setHideTyping(false);

    // Simulate bot response
    setTimeout(() => {
      const botReply = "Hello! I am your sophisticated chatbot. How can I assist you today?";
      setMessages(prev => [
        ...prev,
        {
          text: botReply,
          sender: 'bot',
          timestamp: new Date().toISOString(),
          onFirstSentence: () => {
            setHideTyping(true);
            setTimeout(() => setIsTyping(false), 500);
          }
        }
      ]);
    }, 1500);
  };

  return (
    <div className="chatbot-container">
      <div className="chat-header">💬 Topia's AI Wellness Assistant</div>
      <div className="chat-window">
        {messages.map((msg, index) => (
          <Message
            key={index}
            text={msg.text}
            sender={msg.sender}
            timestamp={msg.timestamp}
            onFirstSentence={msg.onFirstSentence}
          />
        ))}

        {isTyping && (
          <div className={`typing-indicator ${hideTyping ? "hidden" : ""}`}>
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        )}
      </div>

      <div className="chat-input-area">
        <input
          type="text"
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button className="send-button" onClick={handleSend}>➤</button>
      </div>
    </div>
  );
}

export default Chatbot;