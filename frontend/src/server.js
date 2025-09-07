//const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(bodyParser.json());

app.post('/chatbot', (req, res) => {
  const userMessage = req.body.message;
  const pythonProcess = spawn('python3', [
    '../scripts/chatbot.py',
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

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python process:', err);
    res.status(500).json({ reply: 'Error: Could not start the chatbot process.' });
  });
});

app.post('/feedback', (req, res) => {
  const { feedback, user_input, response } = req.body;
  const pythonProcess = spawn('python3', [
    '../scripts/chatbot_entry.py',
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

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python process:', err);
    res.status(500).json({ status: 'error' });
  });
});

app.get('/', (req, res) => {
  res.send('Topia Global Wellness Chatbot backend is running.');
});

app.listen(port, () => {
  console.log(`Chatbot backend server listening at http://localhost:${port}`);
});
