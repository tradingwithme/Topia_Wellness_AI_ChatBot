import React, { useState } from 'react';
import Message from './Message';
import './Chatbot.css';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastBotMessage, setLastBotMessage] = useState('');  const [lastUserMessage, setLastUserMessage] = useState('');

  const handleSend = async () => {
    if (input.trim()) {
      const newUserMessage = { text: input, sender: 'user' };
      setMessages(current => [...current, newUserMessage]);
      setLastUserMessage(input);
      setLoading(true);

      try {
        const response = await fetch('http://localhost:3001/chatbot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: input })
        });
        const data = await response.json();
        const botResponse = { text: data.reply, sender: 'bot' };
        setMessages(current => [...current, botResponse]);
        setLastBotMessage(data.reply); // Save last bot message for feedback
      } catch (error) {
        setMessages(current => [...current, { text: 'Error connecting to backend.', sender: 'bot' }]);
        setLastBotMessage('');
      }

      setInput('');
      setLoading(false);
    }
  };

  const handleInputChange = (event) => setInput(event.target.value);

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') handleSend();
  };

  // Feedback handler
  const sendFeedback = async (feedback) => {
    if (!lastUserMessage || !lastBotMessage) return;
    try {
      await fetch('http://localhost:3001/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          feedback,
          user_input: lastUserMessage,
          response: lastBotMessage
        })
      });
      // Optionally show a message or update UI
    } catch (error) {
      // Optionally handle error
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chat-history">
        {messages.map((msg, index) => (
          <Message key={index} text={msg.text} sender={msg.sender} />
        ))}
        {loading && <Message text="Bot is typing..." sender="bot" />}
        {/* Show feedback buttons only after bot response */}
        {!loading && lastBotMessage && (
          <div style={{ marginTop: '10px', textAlign: 'right' }}>
            <button onClick={() => sendFeedback('y')}>👍 Approve</button>
            <button onClick={() => sendFeedback('n')}>👎 Disapprove</button>
          </div>
        )}
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
