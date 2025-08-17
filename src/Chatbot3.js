import React, { useState } from 'react';
import Message from './Message';
import './Chatbot.css';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (input.trim()) {
      const newUserMessage = { text: input, sender: 'user' };
      setMessages([...messages, newUserMessage]);
      setLoading(true);

      // Send input to backend and get bot response
      try {
        const response = await fetch('http://localhost:3001/chatbot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: input })
        });
        const data = await response.json();
        const botResponse = { text: data.reply, sender: 'bot' };
        setMessages(current => [...current, botResponse]);
      } catch (error) {
        setMessages(current => [...current, { text: 'Error connecting to backend.', sender: 'bot' }]);
      }

      setInput('');
      setLoading(false);
    }
  };

  const handleInputChange = (event) => setInput(event.target.value);

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') handleSend();
  };

  return (
    <div className="chatbot-container">
      <div className="chat-history">
        {messages.map((msg, index) => (
          <Message key={index} text={msg.text} sender={msg.sender} />
        ))}
        {loading && <Message text="Bot is typing..." sender="bot" />}
      </div>
      <div className="chat-input">
        <input
          type="text"
          placeholder="Type your message..."
          value={input}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading}>Send</button>
      </div>
    </div>
  );
}

export default Chatbot;
