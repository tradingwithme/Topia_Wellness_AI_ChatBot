import React, { useState, useEffect } from "react";
import Message from "./Message";
import "./styles/Chatbot.css";

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    setMessages([
      { text: "Welcome to Topia's AI Wellness Chatbot. How may I assist you today?", sender: "bot", timestamp: new Date(), showFeedback: false }
    ]);
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;

    
    const newMessages = [...messages, { text: input, sender: "user", timestamp: new Date() }];
    setMessages(newMessages);
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: input }),
      });

      const data = await res.json();
      const botReply = data.response || "Sorry, I couldn’t process that.";

      setMessages((prev) => [
        ...prev,
        { text: botReply, sender: "bot", timestamp: new Date(), showFeedback: true } 
      ]);
    } catch (err) {
      console.error("Error fetching bot reply:", err);
      setMessages((prev) => [
        ...prev,
        { text: "⚠️ Error: Could not connect to backend.", sender: "bot", timestamp: new Date(), showFeedback: true }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleApprove = (message) => {
    console.log("Approved:", message);
  };

  const handleCorrection = (message) => {
    const correction = prompt("Enter a better response:");
    if (correction) {
      console.log("Correction:", correction);
    }
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
            showFeedback={msg.showFeedback}
            onApprove={() => handleApprove(msg.text)}
            onCorrection={() => handleCorrection(msg.text)}
          />
        ))}

        {isTyping && (
          <div className="typing-indicator">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        )}
      </div>

      <div className="chatbot-input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          className="chatbot-input"
        />
        <button onClick={handleSend} className="chatbot-send-btn">➤</button>
      </div>
    </div>
  );
}

export default Chatbot;