const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(bodyParser.json());

// Chatbot endpoint
app.post('/chatbot', (req, res) => {
  const userMessage = req.body.message;
  // Call Python script to get chatbot response
  const pythonProcess = spawn('python', [
    '../chatbot_entry.py',
    userMessage
  ]);

  let reply = '';
  pythonProcess.stdout.on('data', (data) => {
    reply += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    res.json({ reply: reply.trim() });
  });
});

// Feedback endpoint
app.post('/feedback', (req, res) => {
  const { feedback, user_input, response } = req.body;
  // Call Python script to handle feedback and fine-tuning
  const pythonProcess = spawn('python', [
    '../chatbot_entry.py',
    '--feedback', feedback,
    '--user_input', user_input,
    '--response', response
  ]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python feedback: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    res.json({ status: 'ok' });
  });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
