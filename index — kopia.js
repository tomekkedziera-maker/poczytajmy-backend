// backend/index.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import multer from 'multer';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

// OpenAI SDK (teksty, fallback ASR)
import OpenAI from 'openai';

// Groq SDK (agent programista, szybki ASR)
import Groq from 'groq-sdk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(morgan('dev'));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

// --- Init klientów ---
const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const groq = process.env.GROQ_API_KEY ? new Groq({ apiKey: process.env.GROQ_API_KEY }) : null;

const MOCK_ASR = process.env.MOCK_ASR === '1';
const MOCK_TEXT = process.env.MOCK_TEXT === '1';

// --- Health ---
app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'poczytajmy-backend', version: '0.3' });
});

// --- POST /asr (rozpoznawanie mowy) ---
app.post('/asr', upload.single('audio'), async (req, res) => {
  try {
    if (MOCK_ASR) return res.json({ text: 'Ala ma kota', source: 'mock' });
    if (!req.file) return res.status(400).json({ error: 'Brak pliku w polu "audio".' });

    const tmpPath = path.join(os.tmpdir(), `rec-${Date.now()}.m4a`);
    fs.writeFileSync(tmpPath, req.file.buffer);
    const stream = fs.createReadStream(tmpPath);

    try {
      if (groq) {
        // szybki ASR na Groq
        const transcript = await groq.audio.transcriptions.create({
          file: stream,
          model: 'whisper-large-v3',
          language: 'pl',
        });
        return res.json({ text: transcript?.text ?? '', source: 'groq' });
      } else if (openai) {
        // fallback ASR na OpenAI
        const transcript = await openai.audio.transcriptions.create({
          file: stream,
          model: 'whisper-1',
          language: 'pl',
        });
        return res.json({ text: transcript?.text ?? '', source: 'openai' });
      } else {
        return res.json({ text: 'Ala ma kota', source: 'mock_fallback' });
      }
    } finally {
      fs.unlink(tmpPath, () => {});
    }
  } catch (err) {
    console.error('ASR error:', err);
    res.status(500).json({ error: 'ASR_FAILED', details: String(err?.message || err) });
  }
});

// --- POST /generate-text (krótki tekst do czytania) ---
app.post('/generate-text', async (req, res) => {
  try {
    const { language = 'pl', level = 'A1' } = req.body || {};

    if (MOCK_TEXT) {
      return res.json({ ok: true, text: 'Kasia ma kota i czyta bajki.' });
    }

    const prompt = `
Napisz jedno proste zdanie do nauki czytania po ${language} na poziomie ${level}.
Bez trudnych słów, długość ok. 6–10 wyrazów.
Zwróć tylko czysty tekst bez dodatkowych komentarzy.
`;

    if (openai) {
      const response = await openai.responses.create({
        model: 'gpt-4o-mini',
        input: prompt,
      });
      const text = response?.output_text?.trim?.() || 'Ala ma kota i czyta bajkę.';
      return res.json({ ok: true, text });
    } else if (groq) {
      const response = await groq.chat.completions.create({
        model: 'llama-3.1-8b-instant',
        messages: [{ role: 'user', content: prompt }],
      });
      const text =
        response.choices?.[0]?.message?.content?.trim?.() || 'Ala ma kota i czyta bajkę.';
      return res.json({ ok: true, text });
    } else {
      return res.json({ ok: true, text: 'Ala ma kota i czyta bajkę.' });
    }
  } catch (err) {
    console.error('generate-text error:', err);
    res.status(500).json({ ok: false, error: 'GENERATION_FAILED', details: String(err?.message || err) });
  }
});

// --- POST /agent/dev (agent programista na Groq) ---
/**
 * Body JSON: { filename, goal, source }
 * Response: { ok: true, result: "json patch/pełny plik" }
 */
app.post('/agent/dev', async (req, res) => {
  try {
    const { filename, goal, source } = req.body || {};
    if (!filename || !goal || !source) {
      return res.status(400).json({ ok: false, error: 'MISSING_FIELDS' });
    }

    if (!groq) {
      return res.status(500).json({ ok: false, error: 'NO_GROQ_KEY' });
    }

    const systemPrompt = `
Jesteś agentem programistą. Użytkownik poda plik źródłowy i cel.
Zwróć wynik w JSON: {"patch": "...", "notes": "..."}.
Patch może być unified diff (git apply) albo pełny plik w formacie:
FILE: <nazwa>
<treść>
`;

    const response = await groq.chat.completions.create({
      model: 'llama-3.1-8b-instant',
      messages: [
        { role: 'system', content: systemPrompt },
        {
          role: 'user',
          content: `Plik: ${filename}\nCel: ${goal}\nŹródło:\n${source}`,
        },
      ],
      temperature: 0,
    });

    const content = response.choices?.[0]?.message?.content || '';
    return res.json({ ok: true, result: content });
  } catch (err) {
    console.error('agent/dev error:', err);
    res.status(500).json({ ok: false, error: 'AGENT_DEV_FAILED', details: String(err?.message || err) });
  }
});

// --- Start ---
app.listen(PORT, () => {
  console.log(`🚀 Backend działa na http://localhost:${PORT}`);
  if (groq) console.log('🎧 Groq podłączony (ASR + agent).');
  if (openai) console.log('🤖 OpenAI podłączony (teksty + fallback ASR).');
  if (!groq && !openai) console.log('🧪 Tryb MOCK (brak kluczy).');
});
