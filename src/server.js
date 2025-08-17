const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

app.post('/chatbot', (req, res) => {
  const userMessage = req.body.message;

  // Spawn Python process and pass userMessage as argument
  const pythonProcess = spawn('python', ['python-chatbot/chatbot_entry.py', userMessage]);

  let botReply = '';
  pythonProcess.stdout.on('data', (data) => {
    botReply += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    res.json({ reply: botReply.trim() });
  });
});

app.listen(port, () => {
  console.log(`Chatbot backend server listening at http://localhost:${port}`);
});
