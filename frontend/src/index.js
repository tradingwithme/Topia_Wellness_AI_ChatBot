import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/style.css';  // ✅ Global styles

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { getChatbotResponse, save_approved_response, save_correction } = require('./generative_response');
const { getModel } = require('./model_handling');
const { save_csv } = require('./csv_handling');
const app = express();
const port = 3001;
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.post('/chatbot', async (req, res) => {
    const { user_input } = req.body;
    try {
        const response = await getChatbotResponse(user_input);
        res.json({ response });
    } catch (error) {
        console.error('Error getting chatbot response:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
app.post('/feedback', async (req, res) => {
    const { user_input, response, feedback } = req.body;
    if (!user_input || !response || !feedback) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    try {
        if (feedback === 'y') {
            await save_approved_response(user_input, response);
            // Optionally trigger periodic fine-tuning here
        } else if (feedback === 'n') {
            await save_correction(user_input, response);
            // Trigger fine-tuning with corrections here
        }
        res.json({ message: 'Feedback processed successfully' });
    } catch (error) {
        console.error('Error processing feedback:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
app.get('/model', async (req, res) => {
    try {
        const model = await getModel();
        res.json({ model });
    } catch (error) {
        console.error('Error getting model:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
app.post('/save_csv', async (req, res) => {
    const { data } = req.body;
    if (!data) {
        return res.status(400).json({ error: 'No data provided' });
    }
    try {
        await save_csv(data);
        res.json({ message: 'CSV saved successfully' });
    } catch (error) {
        console.error('Error saving CSV:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});
app.get('/status', (req, res) => {
    res.json({ status: 'Chatbot backend is running' });
});
app.get('/version', (req, res) => {
    res.json({ version: '1.0.0' });
});
app.get('/info', (req, res) => {
    res.json({
        name: 'Topia Global Wellness Chatbot',
        description: 'A chatbot for wellness and health advice',
        version: '1.0.0'
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});