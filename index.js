import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import multer from 'multer';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import http from 'node:http';

import OpenAI from 'openai';
import Groq from 'groq-sdk';

import sharp from 'sharp';
import Tesseract from 'tesseract.js';

/* ===== Paths / app ===== */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(morgan('dev'));

/* ===== Fast config / anti-sleep ===== */
const DEADLINE_MS = Number(process.env.FAST_TIMEOUT_MS || 1200);
const MAX_TOKENS_FAST = Number(process.env.MAX_TOKENS_FAST || 64);
const PREWARM_EVERY_MIN = Number(process.env.PREWARM_EVERY_MIN || 5); // 0 = tylko na starcie
const BASE_URL = process.env.BASE_URL || '';
const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';

const keepAliveAgent = new http.Agent({ keepAlive: true, timeout: 10_000 });
const now = () => (global.performance?.now?.() ?? Date.now());
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const trimUserContent = (s = '', limit = 1200) => {
  const compact = String(s || '').replace(/\s+/g, ' ').trim();
  return compact.length > limit ? compact.slice(-limit) : compact;
};
function withDeadline(promise, ms = DEADLINE_MS) {
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error('DEADLINE_EXCEEDED')), ms);
    promise.then(v => { clearTimeout(to); resolve(v); }, e => { clearTimeout(to); reject(e); });
  });
}

/* ===== Uploads ===== */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

/* ===== Clients ===== */
const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const groq   = process.env.GROQ_API_KEY   ? new Groq({ apiKey: process.env.GROQ_API_KEY })     : null;

/* ===== Mock flags ===== */
const MOCK_ASR  = process.env.MOCK_ASR  === '1';
const MOCK_TEXT = process.env.MOCK_TEXT === '1';

/* ====== ElevenLabs defaults ====== */
const DEFAULT_ELEVEN_VOICE_ID = process.env.ELEVEN_VOICE_ID || 'jJYvw04W4nFnH9II4y4C';

/* ===================== OCR helpers ===================== */
const LANG_PATH =
  process.env.OCR_LANG_PATH ||
  'https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main';

let inflight = 0;
const MAX_CONCURRENCY = Number(process.env.OCR_MAX_CONCURRENCY || 2);
async function acquire() { while (inflight >= MAX_CONCURRENCY) await sleep(40); inflight++; }
function release() { inflight = Math.max(0, inflight - 1); }

const WHITELIST =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZÄ„Ä†ÄĹĹĂ“ĹšĹąĹ»' +
  'abcdefghijklmnopqrstuvwxyzÄ…Ä‡Ä™Ĺ‚Ĺ„ĂłĹ›ĹşĹĽ' +
  '0123456789' +
  ' .,:;!?â€žâ€ť"\'()-â€“â€”/\\[]{}â€¦';

async function preprocess(buffer) {
  let img = sharp(buffer)
    .rotate()
    .resize({ width: Number(process.env.OCR_WIDTH || 2000), withoutEnlargement: true })
    .grayscale()
    .normalize();

  if (process.env.OCR_THRESHOLD === '1') {
    const thr = Number(process.env.OCR_THRESHOLD_VALUE || 185);
    img = img.threshold(thr);
  } else {
    const a = Number(process.env.OCR_LINEAR_A || 1.25);
    const b = Number(process.env.OCR_LINEAR_B || -12);
    img = img.linear(a, b).sharpen();
  }
  return img.png().toBuffer();
}

/* ===================== AUDIO helpers ===================== */
const EXT_BY_MIME = {
  'audio/webm': 'webm',
  'audio/m4a': 'm4a',
  'audio/mp4': 'mp4',
  'audio/mpeg': 'mp3',
  'audio/mp3': 'mp3',
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
  'audio/ogg': 'ogg',
};
function pickAudioExt(file) {
  const fromName = path.extname(file?.originalname || '').replace('.', '').toLowerCase();
  if (fromName) return fromName;
  const fromMime = EXT_BY_MIME[(file?.mimetype || '').toLowerCase()];
  if (fromMime) return fromMime;
  return 'dat';
}

/* ===================== ROUTES ===================== */

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'poczytajmy-backend', version: '1.15-redeploy' });
});

// Prosty root
app.get('/', (_req, res) => {
  res.type('html').send(`
    <html><head><meta charset="utf-8"><title>poczytajmy-backend</title></head>
    <body style="font-family: system-ui, sans-serif; padding:24px">
      <h1>poczytajmy-backend</h1>
      <p>Status: <a href="/health">/health</a></p>
      <ul>
        <li>POST <code>/agent/generate-greeting</code></li>
        <li>POST <code>/agent/generate-text</code></li>
        <li>POST <code>/agent/comprehend</code> âś… pytanie+klucz</li>
        <li>POST <code>/agent/check-answer-voice</code> âś… ocena+feedback</li>
        <li>POST <code>/agent/check-answer-text</code> âś… ocena+feedback (tekst)</li>
        <li>POST <code>/asr</code>, <code>/ocr</code></li>
      </ul>
    </body></html>
  `);
});

/* ===================== ASR (z timestamps + accuracy) ===================== */
app.post('/asr', upload.single('audio'), async (req, res) => {
  try {
    if (MOCK_ASR) {
      return res.json({
        ok: true,
        recognizedText: 'Ala ma kota i psa',
        wordsRead: 5,
        accuracy: 87,
        wordTimestamps: [
          { word: 'Ala', tStart: 0.0, tEnd: 0.4 },
          { word: 'ma',  tStart: 0.6, tEnd: 0.8 },
          { word: 'kota',tStart: 1.2, tEnd: 1.7 },
          { word: 'i',   tStart: 3.7, tEnd: 3.8 },
          { word: 'psa', tStart: 8.8, tEnd: 9.3 },
        ],
        source: 'mock'
      });
    }

    if (!req.file) return res.status(400).json({ ok: false, error: 'Brak pliku w polu "audio".' });

    const { expectedText = '' } = req.body || {};
    const ext = pickAudioExt(req.file);
    const tmpPath = path.join(os.tmpdir(), `rec-${Date.now()}.${ext}`);
    fs.writeFileSync(tmpPath, req.file.buffer);
    const stream = fs.createReadStream(tmpPath);

    let provider = 'none';
    let recognizedText = '';
    let wordTimestamps = [];

    try {
      if (groq) {
        const transcript = await groq.audio.transcriptions.create({
          file: stream,
          model: 'whisper-large-v3',
          language: 'pl',
          response_format: 'verbose_json',
          temperature: 0,
        });
        provider = 'groq';
        recognizedText = (transcript?.text || '').trim();

        if (Array.isArray(transcript?.words) && transcript.words.length) {
          wordTimestamps = transcript.words.map(w => ({
            word: String(w.word || w.text || '').trim(),
            tStart: Number(w.start ?? 0),
            tEnd: Number(w.end ?? 0),
          })).filter(w => w.word);
        } else if (Array.isArray(transcript?.segments)) {
          const out = [];
          for (const seg of transcript.segments) {
            if (Array.isArray(seg.words) && seg.words.length) {
              for (const w of seg.words) {
                out.push({
                  word: String(w.word || w.text || '').trim(),
                  tStart: Number(w.start ?? 0),
                  tEnd: Number(w.end ?? 0),
                });
              }
            }
          }
          wordTimestamps = out;
        }
      } else if (openai) {
        const transcript = await openai.audio.transcriptions.create({
          file: stream,
          model: 'whisper-1',
          language: 'pl',
          response_format: 'verbose_json',
          temperature: 0,
        });
        provider = 'openai';
        recognizedText = (transcript?.text || '').trim();

        const out = [];
        if (Array.isArray(transcript?.segments)) {
          for (const seg of transcript.segments) {
            if (Array.isArray(seg.words) && seg.words.length) {
              for (const w of seg.words) {
                out.push({
                  word: String(w.word || w.text || '').trim(),
                  tStart: Number(w.start ?? 0),
                  tEnd: Number(w.end ?? 0),
                });
              }
            }
          }
        }
        wordTimestamps = out;
      } else {
        return res.status(502).json({ ok: false, error: 'NO_PROVIDER' });
      }
    } finally {
      fs.unlink(tmpPath, () => {});
    }

    if (!Array.isArray(wordTimestamps) || wordTimestamps.length === 0) {
      const words = (recognizedText || '').split(/\s+/).filter(Boolean);
      let t = 0;
      wordTimestamps = words.map(w => {
        const start = t; const end = t + 0.4; t += 0.8;
        return { word: w, tStart: start, tEnd: end };
      });
    }

    const wordsRead = Number(wordTimestamps.length || 0);

    function norm(s=''){ return String(s).toLowerCase().replace(/[^\p{L}\p{M}0-9\s]+/gu,' ').replace(/\s+/g,' ').trim(); }
    function jacc(a,b){
      const A=new Set(norm(a).split(' ').filter(Boolean));
      const B=new Set(norm(b).split(' ').filter(Boolean));
      if(!A.size && !B.size) return 100;
      let inter=0; for (const x of A) if(B.has(x)) inter++;
      return Math.round((inter/(A.size+B.size-inter))*100);
    }
    const accuracy = expectedText ? jacc(recognizedText, expectedText) : 0;

    return res.json({
      ok: true,
      recognizedText,
      wordsRead,
      accuracy,
      wordTimestamps,
      source: provider,
    });
  } catch (err) {
    console.error('ASR error:', err);
    res.status(500).json({ ok: false, error: 'ASR_FAILED', details: String(err?.message || err) });
  }
});

/* ===================== AGENT POWITAĹ ===================== */

const HERO_THEMES = {
  'MiĹ›': 'przytulny i cierpliwy, kocha bajki na dobranoc',
  'LabuĹ›': 'energiczny i wesoĹ‚y, lubi ksiÄ…ĹĽki przygodowe',
  'KrĂłliczek': 'ciekawski i szybki, uwielbia zagadki w opowieĹ›ciach',
  'JeĹĽyk': 'ostroĹĽny i mÄ…dry, kocha opowieĹ›ci z moraĹ‚em'
};

const READING_TOPICS = [
  'ksiÄ…ĹĽki peĹ‚ne magii i zaklÄ™Ä‡',
  'czytanie bajek na gĹ‚os',
  'szukanie nowych sĹ‚Ăłw w opowiadaniu',
  'przeĹĽywanie przygĂłd z bohaterami ksiÄ…ĹĽek',
  'poznawanie liter i sylab',
  'czytanie komiksĂłw z obrazkami',
  'odkrywanie tajemnic w bibliotece',
  'pisanie wĹ‚asnej bajki po przeczytaniu ksiÄ…ĹĽki',
  'czytanie rozdziaĹ‚Ăłw z przygodami',
  'opowiadanie przeczytanej historii przyjacioĹ‚om'
];

function pick(arr){ return arr[Math.floor(Math.random()*arr.length)]; }

function normalize(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[â€žâ€ť"!?.,;:()\-\â€“â€”[\]{}â€¦]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}
function jaccard(a, b) {
  const A = new Set(normalize(a).split(' ').filter(Boolean));
  const B = new Set(normalize(b).split(' ').filter(Boolean));
  if (!A.size && !B.size) return 1;
  let inter = 0;
  for (const w of A) if (B.has(w)) inter++;
  return inter / (A.size + B.size - inter);
}
function chooseMostNovel(cands, history) {
  if (!history || history.length === 0) return cands[0] || '';
  let best = '';
  let bestScore = 1;
  for (const c of cands) {
    const maxSim = Math.max(0, ...history.map(h => jaccard(c, h)));
    if (maxSim < bestScore) { bestScore = maxSim; best = c; }
  }
  return best || cands[0] || '';
}

function buildGreetingPrompt({ age, character = 'TwĂłj przyjaciel', theme = '', n = 12 }) {
  const wiek = Number.isFinite(age) ? age : 'X';
  const tone =
    Number.isFinite(age) && age <= 5
      ? 'proste, ciepĹ‚e, zabawowe; rytm mowy dziecka; onomatopeje OK'
      : Number.isFinite(age) && age <= 8
      ? 'ĹĽywe, motywujÄ…ce; mini-misja; 1â€“2 emoji'
      : 'pewne, partnerskie; cel, sprawczoĹ›Ä‡; max 1â€“2 emoji';

  const heroHint = theme ? `Delikatny klimat bohatera: ${theme}.` : '';
  const chosenTopic = pick(READING_TOPICS);

  return `WymyĹ›l ${n} ZUPEĹNIE rĂłĹĽnych, krĂłtkich powitaĹ„ po polsku dla dziecka (wiek: ${wiek}).
MĂłwi ${character}. Styl: ${tone}. ${heroHint}
Temat przewodni: ${chosenTopic}.

âšˇ KaĹĽde powitanie MUSI odnosiÄ‡ siÄ™ do czytania i ksiÄ…ĹĽek, np. sĹ‚owa: ksiÄ…ĹĽka, czytanie, rozdziaĹ‚, bajka, historia, sylaba, sĹ‚owo, zdanie, ilustracje, narrator, zakĹ‚adka, biblioteka, ksiÄ™garnia, opowieĹ›Ä‡, litery.
âšˇ NIE uĹĽywaj motywĂłw typu: las, bieganie, sport, piknik, podrĂłĹĽe â€” tylko Ĺ›wiat ksiÄ…ĹĽek.
âšˇ Zakaz: nie uĹĽywaj sĹ‚Ăłw powitalnych (czeĹ›Ä‡, hej, witaj, siema, halo) oraz NIE uĹĽywaj imienia dziecka w ĹĽadnej formie.

đź“š PrzykĹ‚ady:
- DziĹ› razem odkryjemy nowy rozdziaĹ‚ bajki. đź“–
- Zajrzymy do ksiÄ…ĹĽki peĹ‚nej czarodziejskich sĹ‚Ăłw. âś¨
- Sprawdzimy, ile sylab ma najdĹ‚uĹĽsze sĹ‚owo w opowieĹ›ci. đźš€

Zasady: jedno zdanie, 6â€“14 wyrazĂłw, bez cudzysĹ‚owĂłw i bez wstÄ™pĂłw.
KaĹĽde powitanie w osobnej linii poprzedzone myĹ›lnikiem "- ".`;
}

function parseList(text) {
  const lines = (text || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const items = [];
  for (let l of lines) {
    l = l.replace(/^[-*\d.)]+\s*/, '');
    if (l) items.push(l);
  }
  const uniq = Array.from(new Set(items)).filter(s => {
    const wc = normalize(s).split(' ').filter(Boolean).length;
    return wc >= 5 && wc <= 16;
  });
  return uniq.slice(0, 20);
}

const FORBIDDEN_HELLOS = ['czeĹ›Ä‡', 'hej', 'witaj', 'siema', 'halo'];
function sanitizeNoName(name, raw) {
  let s = (raw || '').trim();
  const helloRe = new RegExp(`^\\s*(?:${FORBIDDEN_HELLOS.join('|')})\\b[\\p{L}\\p{M}\\s,!.?â€“â€”-]*`, 'iu');
  s = s.replace(helloRe, '').trim();
  if (name) {
    const forms = [name, `${name}u`, `${name}o`, `${name}e`, `${name}a`, `${name}ku`];
    const escaped = forms.map(v => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const nameRe = new RegExp(`\\b(?:${escaped.join('|')})\\b[\\s,!.?]*`, 'giu');
    s = s.replace(nameRe, '').trim();
  }
  s = s.replace(/^[,â€“â€”\-|:;!.\s]+/u, '').trim();
  return s;
}

const recentGreetings = new Map();

/* ===== Groq/OpenAI race helper ===== */
async function groqChat({ messages, max_tokens = MAX_TOKENS_FAST, temperature = 0.3, top_p = 0.95 }) {
  const t0 = now();
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.GROQ_API_KEY || ''}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      Connection: 'keep-alive'
    },
    body: JSON.stringify({ model: GROQ_MODEL, temperature, top_p, max_tokens, messages })
  });
  if (!res.ok) throw new Error(`GROQ_HTTP_${res.status}`);
  const data = await res.json();
  return { provider: 'groq', text: data?.choices?.[0]?.message?.content?.trim?.() || '', latency_ms: Math.round(now() - t0) };
}

async function generateGreetingV2({ name, age, character, theme }) {
  const prompt = buildGreetingPrompt({ age: Number(age), character, theme, n: 12 });

  const racers = [];
  if (process.env.GROQ_API_KEY) {
    racers.push(groqChat({
      messages: [{ role: 'user', content: trimUserContent(prompt) }],
      temperature: 0.9, top_p: 0.95, max_tokens: 180,
    }));
  }
  if (openai) {
    racers.push((async () => {
      const t0 = now();
      const r = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.9, top_p: 0.95, max_tokens: 180,
      });
      const txt = r?.choices?.[0]?.message?.content?.trim?.() || '';
      if (!txt) throw new Error('OPENAI_EMPTY');
      return { provider: 'openai', text: txt, latency_ms: Math.round(now() - t0) };
    })());
  }

  const winner = await withDeadline(Promise.any(racers), DEADLINE_MS);
  let raw = winner.text;

  let cands = parseList(raw);
  if (!cands.length && raw) cands = raw.split(/[.\n]/).map(s => s.trim()).filter(Boolean);
  if (!cands.length) throw new Error('EMPTY_GENERATION');

  const profileKey = `${(name || '').toLowerCase()}|${Number(age)||'X'}`;
  const history = recentGreetings.get(profileKey) || [];

  const picked = chooseMostNovel(cands, history);
  const cleaned = sanitizeNoName(name, picked);
  const finalText = cleaned || picked;

  recentGreetings.set(profileKey, [finalText, ...history].slice(0, 20));
  return { text: finalText, source: winner.provider || 'unknown' };
}

app.post('/agent/generate-greeting', async (req, res) => {
  try {
    const { name = '', age, character = 'TwĂłj przyjaciel' } = req.body || {};
    const theme = HERO_THEMES[character] || '';
    const { text, source } = await generateGreetingV2({ name, age, character, theme });
    res.json({ ok: true, text, source });
  } catch (err) {
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    if (timedOut) return res.status(504).json({ ok: false, error: 'DEADLINE_EXCEEDED', timed_out: true });
    console.error('agent/generate-greeting error:', err);
    return res.status(502).json({ ok: false, error: String(err?.message || err) });
  }
});

/* ===================== AGENT MOTYWACJI ===================== */

function bucketToneByAge(age) {
  const a = Number(age);
  if (Number.isFinite(a) && a <= 5) return 'bardzo prosto, ciepĹ‚o, Ĺ‚agodnie; krĂłtkie sĹ‚owa; 1 emoji max';
  if (Number.isFinite(a) && a <= 8) return 'prosto, energicznie, wspierajÄ…co; mini-sugestia co poprawiÄ‡; 1 emoji max';
  return 'partnersko, konkretnie, z uznaniem; 1 emoji max';
}

function rubricByAccuracy(acc) {
  const s = Math.max(0, Math.min(100, Math.round(acc || 0)));
  if (s >= 95) return 'wynik Ĺ›wietny; podkreĹ›l perfekcjÄ™ i zaproponuj trudniejsze sĹ‚owo przy nastÄ™pnej stronie';
  if (s >= 80) return 'wynik bardzo dobry; pochwal pĹ‚ynnoĹ›Ä‡ i zaproponuj jednÄ… mikro-radÄ™ (np. dokĹ‚adniej koĹ„cĂłwki)';
  if (s >= 60) return 'wynik dobry; pochwal staranie i podaj jednÄ… prostÄ… wskazĂłwkÄ™ (np. wolniej, sylabizuj trudniejsze sĹ‚owa)';
  return 'wynik na rozgrzewkÄ™; skup siÄ™ na zachÄ™cie i jednej mini-radzie (np. przeczytaj zdanie jeszcze raz spokojnie)';
}

function buildMotivationPrompt({ age, accuracy, text, characterName = 'Bohater', lang = 'pl' }) {
  const tone = bucketToneByAge(age);
  const rubric = rubricByAccuracy(accuracy);
  const excerpt = trimUserContent(text || '', 220);

  return `
JesteĹ› ${characterName} z aplikacji do nauki czytania dla dzieci. Twoje zadanie:
napisz 1 krĂłtki komentarz motywacyjny po polsku (${lang}), dopasowany do wieku dziecka i jakoĹ›ci czytania.

Zasady stylu:
- Styl: ${tone}.
- ${rubric}.
- Maks. 160 znakĂłw. 1 zdanie (wyjÄ…tkowo 2 bardzo krĂłtkie).
- Brak cudzysĹ‚owĂłw i nawiasĂłw. Bez liczb procentowych ani ocen wprost.
- MĂłw do dziecka w 2. osobie (â€žczytaszâ€ť, â€ždasz radÄ™â€ť), NIE uĹĽywaj imienia dziecka.
- UĹĽyj co najwyĹĽej 1 emoji (opcjonalnie).

Kontekst (fragment przeczytanego tekstu â€“ opcjonalnie moĹĽesz nawiÄ…zaÄ‡ ogĂłlnie, bez cytowania):
"${excerpt}"

Podaj tylko gotowÄ… wypowiedĹş.`.trim();
}

function tightenMotivation(s, maxChars = 160) {
  if (!s) return s;
  s = String(s)
    .replace(/[\"â€śâ€ťâ€žâ€ť'()]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  s = s.replace(/[Â«Â»â€žâ€ť"'].*?[Â«Â»â€žâ€ť"']/g, '').replace(/\s+/g, ' ').trim();
  const parts = s.split(/(?<=[.!?â€¦])\s+/).filter(Boolean);
  s = parts.slice(0, 2).join(' ').trim();
  const emojiRe = /[\p{Extended_Pictographic}\uFE0F]/gu;
  let seen = 0;
  s = s.replace(emojiRe, m => (++seen > 1 ? '' : m));
  if (s.length > maxChars) {
    s = s.slice(0, maxChars).replace(/\s+\S*$/, '').trim();
  }
  if (!/[.!?â€¦]$/.test(s)) s += '.';
  return s;
}

async function generateMotivation({ age, accuracy, text, characterName, lang = 'pl' }) {
  const prompt = buildMotivationPrompt({ age, accuracy, text, characterName, lang });

  const racers = [];
  if (process.env.GROQ_API_KEY) {
    racers.push(groqChat({
      messages: [{ role: 'user', content: trimUserContent(prompt) }],
      temperature: 0.9, top_p: 0.95, max_tokens: 120,
    }));
  }
  if (openai) {
    racers.push((async () => {
      const t0 = now();
      const r = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.9, top_p: 0.95, max_tokens: 120,
      });
      const txt = r?.choices?.[0]?.message?.content?.trim?.() || '';
      if (!txt) throw new Error('OPENAI_EMPTY');
      return { provider: 'openai', text: txt, latency_ms: Math.round(now() - t0) };
    })());
  }

  const winner = await withDeadline(Promise.any(racers), DEADLINE_MS);
  let out = String(winner.text || '').trim();
  out = out.replace(/^["'â€žâ€ť]+|["'â€žâ€ť]+$/g, '').trim();
  out = tightenMotivation(out, 160);
  if (!out) throw new Error('EMPTY_MOTIVATION');
  return { text: out, source: winner.provider || 'unknown' };
}

app.post('/agent/motivate', async (req, res) => {
  try {
    const {
      age,
      accuracy = 0,
      text = '',
      name,
      characterName = 'Bohater',
      lang = 'pl',
    } = req.body || {};

    const { text: rawMsg, source } = await generateMotivation({
      age, accuracy, text, characterName, lang
    });

    const msg = tightenMotivation(rawMsg, 160);

    res.json({ ok: true, text: msg, source });
  } catch (err) {
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    if (timedOut) return res.status(504).json({ ok: false, error: 'DEADLINE_EXCEEDED', timed_out: true });
    console.error('agent/motivate error:', err);
    return res.status(502).json({
      ok: false,
      error: String(err?.message || err),
      fallback: 'Ĺšwietna prĂłba! Z kaĹĽdÄ… stronÄ… bÄ™dzie coraz lepiej â€” sprĂłbujmy jeszcze raz! đź’Ş'
    });
  }
});

/* ===================== GENERATOR ZDAĹ DO CZYTANIA ===================== */

const BANK_A1 = [
  "Ala ma kota i lubi czytaÄ‡ bajki wieczorem.",
  "MiĹ› je miodek, a potem sĹ‚ucha krĂłtkiej opowieĹ›ci.",
  "PiĹ‚ka leĹĽy na trawie, a Julek czyta na Ĺ‚awce.",
  "Pies biegnie do domu, gdzie czeka nowa ksiÄ…ĹĽka.",
  "SĹ‚oĹ„ce Ĺ›wieci jasno, a my czytamy w ogrodzie."
];
const BANK_A2 = [
  "W ogrodzie rosnÄ… kwiaty, a my czytamy o motylach.",
  "Kasia czyta ksiÄ…ĹĽkÄ™ o zwierzÄ™tach i szuka trudnych sĹ‚Ăłw.",
  "Na spacerze opowiadamy historiÄ™ o maĹ‚ej latarni morskiej.",
  "Po poĹ‚udniu wybieramy rozdziaĹ‚ o odwaĹĽnym krĂłliku."
];
const BANK_B1 = [
  "ChoÄ‡ padaĹ‚ deszcz, przeczytaliĹ›my rozdziaĹ‚ o podrĂłĹĽy po mapie.",
  "LubiÄ™ zagadki, bo rozwijajÄ… wyobraĹşniÄ™ i pomagajÄ… w czytaniu.",
  "Z zachwytem Ĺ›ledziĹ‚em, jak narrator opisuje lot kolorowego motyla.",
  "Po kolacji wspĂłlnie czytamy i planujemy jutrzejszÄ… przygodÄ™."
];
function bankByLevel(level = "A1") {
  const L = String(level).toUpperCase();
  if (L === "B1") return BANK_B1;
  if (L === "A2") return BANK_A2;
  return BANK_A1;
}

function onlyOneSentence(s) {
  const parts = String(s).split(/(?<=[.!?â€¦])\s+/).filter(Boolean);
  return (parts[0] || s).trim();
}
function cleanSentence(s) {
  let out = String(s)
    .replace(/[â€žâ€ť"â€śâ€ť'()Â«Â»]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  out = onlyOneSentence(out);
  if (!/[.!?â€¦]$/.test(out)) out += ".";
  return out;
}
function countWords(s) {
  return (String(s).trim().match(/\b[\p{L}\p{M}0-9'-]+\b/gu) || []).length;
}
const PROFANITY = [
  "kurwa","cholera","debil","idiota","gĹ‚upi","szmata",
  "pedaĹ‚","lesba","spier","nienawidzÄ™","zabij","Ĺ›mierÄ‡"
];
function hasForbidden(s) {
  const low = String(s).toLowerCase();
  return PROFANITY.some(p => low.includes(p));
}
function hasPolishDiacritics(s) {
  return /[Ä…Ä‡Ä™Ĺ‚Ĺ„ĂłĹ›ĹşĹĽ]/i.test(String(s));
}
function validateKidsSentencePL(s, { minWords=8, maxWords=16 } = {}) {
  const issues = [];
  const txt = cleanSentence(onlyOneSentence(s));
  const words = countWords(txt);
  if (words < minWords || words > maxWords) {
    issues.push(`Liczba sĹ‚Ăłw ${words} poza zakresem ${minWords}â€“${maxWords}.`);
  }
  if (hasForbidden(txt)) issues.push("SĹ‚owa niedozwolone.");
  if (!hasPolishDiacritics(txt)) issues.push("Brak polskich znakĂłw.");
  const tokens = (txt.match(/\b[\p{L}\p{M}0-9'-]+\b/gu) || []);
  const long = tokens.filter(w => w.replace(/[^a-zÄ…Ä‡Ä™Ĺ‚Ĺ„ĂłĹ›ĹşĹĽ-]/gi,"").length > 12).length;
  const ratio = tokens.length ? long / tokens.length : 0;
  if (tokens.length > 24 || ratio > 0.4) issues.push("Zbyt trudne lub nienaturalne sĹ‚ownictwo.");
  return { ok: issues.length === 0, issues, text: txt };
}

async function correctPolishSentence(raw) {
  const prompt = `
Popraw zdanie dla dziecka w wieku wczesnoszkolnym.
Zasady:
- Jedno zdanie po polsku, 8â€“16 sĹ‚Ăłw.
- Proste, naturalne, bez ĹĽargonu i cudzysĹ‚owĂłw.
- Popraw ortografiÄ™ i interpunkcjÄ™.
ZwrĂłÄ‡ tylko gotowe zdanie.
Tekst:
${raw}`.trim();

  const racers = [];
  if (process.env.GROQ_API_KEY) {
    racers.push(groqChat({
      messages: [{ role: "user", content: prompt }],
      temperature: 0.2, top_p: 0.9, max_tokens: 60,
    }));
  }
  if (openai) {
    racers.push((async () => {
      const t0 = now();
      const r = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.2, top_p: 0.9, max_tokens: 60,
      });
      const txt = r?.choices?.[0]?.message?.content?.trim?.() || "";
      if (!txt) throw new Error("OPENAI_EMPTY");
      return { provider: "openai", text: txt, latency_ms: Math.round(now() - t0) };
    })());
  }
  const winner = await withDeadline(Promise.any(racers), DEADLINE_MS);
  return cleanSentence(winner.text || "");
}

app.post("/agent/generate-text", async (req, res) => {
  try {
    const { language = "pl", level = "A1" } = req.body || {};

    const prompt =
`Napisz jedno proste zdanie po polsku na poziomie ${String(level).toUpperCase()} do gĹ‚oĹ›nego czytania przez dziecko.
Wymagania:
- Jedno zdanie (8â€“16 sĹ‚Ăłw), naturalne i poprawne.
- SĹ‚ownictwo codzienne, bez ĹĽargonu i neologizmĂłw.
- Zero przemocy, straszenia, polityki, chorĂłb.
- Brak cudzysĹ‚owĂłw i nawiasĂłw.
- UĹĽywaj peĹ‚nych polskich znakĂłw.
Podaj tylko gotowe zdanie.`;

    const racers = [];
    if (process.env.GROQ_API_KEY) {
      racers.push(groqChat({
        messages: [{ role: "user", content: trimUserContent(prompt) }],
        temperature: 0.3, top_p: 0.9, max_tokens: 60,
      }));
    }
    if (openai) {
      racers.push((async () => {
        const t0 = now();
        const r = await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: prompt }],
          temperature: 0.3, top_p: 0.9, max_tokens: 60,
        });
        const txt = r?.choices?.[0]?.message?.content?.trim?.() || "";
        if (!txt) throw new Error("OPENAI_EMPTY");
        return { provider: "openai", text: txt, latency_ms: Math.round(now() - t0) };
      })());
    }

    const winner = await withDeadline(Promise.any(racers), DEADLINE_MS);
    let sentence = cleanSentence(winner.text || "");
    if (!sentence) throw new Error("EMPTY_GENERATION");

    let check = validateKidsSentencePL(sentence);
    if (!check.ok) {
      const fixed = cleanSentence(await correctPolishSentence(sentence));
      const check2 = validateKidsSentencePL(fixed);
      if (check2.ok) {
        return res.json({ ok: true, text: check2.text, level, language, source: `${winner.provider}+corrector` });
      }
      const backup = pick(bankByLevel(level));
      return res.json({ ok: true, text: backup, level, language, source: "fallback-bank" });
    }
    return res.json({ ok: true, text: check.text, level, language, source: winner.provider });
  } catch (err) {
    const timedOut = String(err?.message || err) === "DEADLINE_EXCEEDED";
    if (timedOut) return res.status(504).json({ ok: false, error: "DEADLINE_EXCEEDED", timed_out: true });
    console.error("agent/generate-text error:", err);
    const { level = "A1", language = "pl" } = req.body || {};
    const backup = pick(bankByLevel(level));
    return res.status(200).json({ ok: true, text: backup, level, language, source: "fallback-bank" });
  }
});
app.post("/generate-text", (req, res) => {
  res.redirect(307, "/agent/generate-text");
});

/* ===================== OCR ===================== */
app.post('/ocr', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ ok: false, error: 'NO_FILE' });
    if (process.env.MOCK_OCR === '1') return res.json({ ok: true, text: 'PrzykĹ‚adowy tekst z OCR.' });

    if (process.env.USE_OPENAI_OCR === '1' && openai) {
      const b64 = `data:image/jpeg;base64,${req.file.buffer.toString('base64')}`;
      const prompt = 'WyodrÄ™bnij czysty tekst z obrazu (po polsku). ZwrĂłÄ‡ tylko tekst.';
      const resp = await openai.responses.create({
        model: 'gpt-4o-mini',
        input: [{ role: 'user', content: [
          { type: 'input_text', text: prompt },
          { type: 'input_image', image_url: b64 }
        ]}],
      });
      const text = resp?.output_text?.trim?.() || '';
      return res.json({ ok: true, text });
    }

    await acquire();
    try {
      const pre = await preprocess(req.file.buffer);
      const psm = Number(process.env.OCR_PSM || 6);
      const result = await Tesseract.recognize(pre, 'pol+eng', {
        langPath: LANG_PATH,
        tessedit_pageseg_mode: psm,
        tessedit_char_whitelist: WHITELIST,
        preserve_interword_spaces: '1',
        user_defined_dpi: '300',
        logger: () => {},
      });
      const text = (result?.data?.text || '').trim();
      const confidence = Number(result?.data?.confidence ?? 0);
      return res.json({ ok: true, text, confidence });
    } finally {
      release();
    }
  } catch (err) {
    console.error('OCR error:', err);
    res.status(500).json({ ok: false, error: 'OCR_FAILED', details: String(err?.message || err) });
  }
});

/* ===================== ElevenLabs TTS proxy ===================== */
app.post('/tts', async (req, res) => {
  try {
    const apiKey = process.env.ELEVEN_API_KEY || process.env.ELEVENLABS_API_KEY;
    if (!apiKey) return res.status(500).json({ ok: false, error: 'NO_ELEVEN_API_KEY' });

    const {
      text = '',
      voiceId = DEFAULT_ELEVEN_VOICE_ID,
      output_format = 'mp3_44100_128',
      stability = 0.5,
      similarity_boost = 0.75,
    } = req.body || {};

    const clean = String(text).trim().slice(0, 600);
    if (!clean) return res.status(400).json({ ok: false, error: 'EMPTY_TEXT' });

    const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'xi-api-key': apiKey,
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg'
      },
      body: JSON.stringify({
        text: clean,
        model_id: 'eleven_multilingual_v2',
        output_format,
        voice_settings: { stability, similarity_boost }
      })
    });

    if (!r.ok) {
      let details = '';
      try { details = await r.text(); } catch {}
      return res.status(r.status).json({ ok: false, error: `ELEVEN_HTTP_${r.status}`, details: details?.slice(0, 800) });
    }

    const buf = Buffer.from(await r.arrayBuffer());
    res.json({ ok: true, audioB64: buf.toString('base64'), mime: 'audio/mpeg', voiceId });
  } catch (err) {
    console.error('TTS proxy error:', err);
    res.status(500).json({ ok: false, error: 'TTS_PROXY_FAILED', details: String(err?.message || err) });
  }
});

app.get('/tts-voices', async (_req, res) => {
  try {
    const apiKey = process.env.ELEVEN_API_KEY || process.env.ELEVENLABS_API_KEY;
    if (!apiKey) return res.status(500).json({ ok: false, error: 'NO_ELEVEN_API_KEY' });

    const r = await fetch('https://api.elevenlabs.io/v1/voices', {
      headers: { 'xi-api-key': apiKey, 'Accept': 'application/json' }
    });

    if (!r.ok) {
      let details = '';
      try { details = await r.text(); } catch {}
      return res.status(r.status).json({ ok: false, error: `ELEVEN_HTTP_${r.status}`, details: details?.slice(0, 800) });
    }

    const data = await r.json();
    const voices = Array.isArray(data?.voices) ? data.voices.map(v => ({ id: v.voice_id, name: v.name })) : [];
    return res.json({ ok: true, voices });
  } catch (err) {
    console.error('TTS voices error:', err);
    return res.status(500).json({ ok: false, error: 'VOICES_FAILED', details: String(err?.message || err) });
  }
});

/* ===================== OpenAI TTS proxy ===================== */
app.post('/tts-openai', async (req, res) => {
  try {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) return res.status(500).json({ ok: false, error: 'NO_OPENAI_API_KEY' });

    const { text = '', voice = 'alloy', format = 'mp3' } = req.body || {};
    const clean = String(text).trim().slice(0, 600);
    if (!clean) return res.status(400).json({ ok: false, error: 'EMPTY_TEXT' });

    const r = await fetch('https://api.openai.com/v1/audio/speech', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        Accept: format === 'wav' ? 'audio/wav' : (format === 'ogg' ? 'audio/ogg' : 'audio/mpeg')
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini-tts',
        voice,
        input: clean
      })
    });

    if (!r.ok) {
      const errText = await r.text().catch(() => '');
      return res.status(502).json({ ok: false, error: `OPENAI_HTTP_${r.status}`, details: errText.slice(0, 300) });
    }

    const buf = Buffer.from(await r.arrayBuffer());
    res.json({ ok: true, provider: 'openai', format, audioB64: buf.toString('base64') });
  } catch (err) {
    console.error('TTS-OPENAI REST error:', err);
    res.status(500).json({ ok: false, error: 'TTS_OPENAI_FAILED' });
  }
});

/* ===================================================================== */
/* =====================  QUIZ / COMPREHEND â€“ NOWE  ==================== */
/* ===================================================================== */

// uniwersalny wyĹ›cig LLM (zwraca tekst)
async function raceLLM({ prompt, max_tokens = 150, temperature = 0.3 }) {
  const racers = [];
  if (process.env.GROQ_API_KEY) {
    racers.push(groqChat({
      messages: [{ role: 'user', content: trimUserContent(prompt) }],
      temperature, top_p: 0.95, max_tokens,
    }));
  }
  if (openai) {
    racers.push((async () => {
      const t0 = now();
      const r = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        temperature, top_p: 0.95, max_tokens,
      });
      const txt = r?.choices?.[0]?.message?.content?.trim?.() || '';
      if (!txt) throw new Error('OPENAI_EMPTY');
      return { provider: 'openai', text: txt, latency_ms: Math.round(now() - t0) };
    })());
  }
  const winner = await withDeadline(Promise.any(racers), DEADLINE_MS);
  return (winner?.text || '').trim();
}
function extractJSON(s) {
  const m = String(s || '').match(/\{[\s\S]*\}/);
  if (!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
}

// pomocnicze walidacje pytaĹ„/odpowiedzi
function isGenericQuestion(q = '') {
  const s = String(q).toLowerCase();
  return /opowiedz.*jednym zdaniem|co.*zapamiÄ™ta|o czym byĹ‚|co siÄ™ wydarzyĹ‚o|streĹ›Ä‡|podsumuj/.test(s);
}
function isDefinitionQuestion(q='') {
  const s = String(q).toLowerCase();
  return /\bkim jest\b|\bkim byĹ‚\b|\bco to jest\b|\bczym jest\b/.test(s);
}
function answerWordCount(a = '') {
  return (String(a).trim().match(/\b[\p{L}\p{M}0-9'-]+\b/gu) || []).length;
}
function endsWithQuestionMark(q = '') {
  return /\?\s*$/.test(String(q));
}
function isFirstPersonText(t = '') {
  const s = String(t).toLowerCase();
  return /\b(ja|mnie|mi|mnÄ…|mĂłj|moja|moje|jestem|mam|idÄ™|robiÄ™|czytam|siedzÄ™|bÄ™dÄ™|chcÄ™|przeczytam|jem|pijÄ™|oglÄ…dam|sĹ‚ucham|gram)\b/.test(s);
}
function answerIsExtractive(text = '', answer = '') {
  const t = String(text).toLowerCase();
  const a = String(answer).toLowerCase().trim();
  if (!a) return false;
  return t.includes(a);
}

/* === NOWE: 1os twarde dopasowanie i ekstrakcja dopeĹ‚nienia === */
function findTransitive1stFormHard(text="") {
  const s = String(text).trim().toLowerCase();
  let m = s.match(/^(ja\s+)?(czytam|jem|pijÄ™|oglÄ…dam|sĹ‚ucham|gram)\b/i);
  return m ? (m[2] || '').toLowerCase() : "";
}
function extractObjectAfter1st(text="", verbForm="") {
  if (!verbForm) return "";
  const rx = new RegExp(`\\b${verbForm}\\b\\s+([^.!?]+)`, "i");
  const m = String(text).match(rx);
  if (!m) return "";
  let seg = m[1].trim();

  if (verbForm === "gram") {
    return seg.replace(/\s*[,;:].*$/u, "").trim(); // np. "w piĹ‚kÄ™"
  }

  const cut = seg.split(/\s+(?:(?!o\b)(w|na|przy|pod|obok|koĹ‚o|u|do|z|ze|po))\s+/i)[0];
  return (cut || seg).replace(/\s*[,;:].*$/u, "").trim();
}

/* === NOWE: 3os prosta heurystyka (ImiÄ™ + czasownik + dopeĹ‚nienie) === */
function thirdPersonDirectQA(text="") {
  const s = String(text).trim();
  const m = s.match(/\b([A-ZĹĹšĹ»ĹąÄ†Ĺ][a-zÄ…Ä‡Ä™Ĺ‚Ĺ„ĂłĹ›ĹşĹĽ]+)\s+(czyta|oglÄ…da|sĹ‚ucha|je|pije|gra)\b\s+([^.!?]+)/u);
  if (!m) return null;
  const name = m[1], verb = m[2].toLowerCase();
  let rest = m[3].trim();

  if (verb === "gra") {
    const obj = rest.replace(/\s*[,;:].*$/u,"").trim();
    if (obj) return { question: `W co gra ${name}?`, answer: obj };
    return null;
  }
  if (verb === "sĹ‚ucha") {
    const obj = rest.split(/\s+(?:(?!o\b)(w|na|przy|pod|obok|koĹ‚o|u|do|z|ze|po))\s+/i)[0].replace(/\s*[,;:].*$/u,"").trim();
    if (obj) return { question: `Czego sĹ‚ucha ${name}?`, answer: obj };
    return null;
  }
  const obj = rest.split(/\s+(?:(?!o\b)(w|na|przy|pod|obok|koĹ‚o|u|do|z|ze|po))\s+/i)[0].replace(/\s*[,;:].*$/u,"").trim();
  if (!obj) return null;
  const qVerb = (verb === "czyta") ? "czyta" : (verb === "oglÄ…da") ? "oglÄ…da" : (verb === "je") ? "je" : (verb === "pije") ? "pije" : "robi";
  return { question: `Co ${qVerb} ${name}?`, answer: obj };
}

/* â€” Pytanie + krĂłtka poprawna odpowiedĹş (klucz) â€” */
function buildQuestionPrompt({ text, age }) {
  const wiek = Number(age);
  const target =
    Number.isFinite(wiek) && wiek <= 8
      ? 'bardzo proste, jednoznaczne pytanie. OdpowiedĹş 1â€“5 sĹ‚Ăłw.'
      : 'proste, faktograficzne pytanie. OdpowiedĹş krĂłtka (max 6 sĹ‚Ăłw).';

  return `
JesteĹ› nauczycielem w klasach 1â€“3. Na podstawie fragmentu napisz JEDNO pytanie sprawdzajÄ…ce zrozumienie i krĂłtki KLUCZ odpowiedzi.

FRAGMENT:
"""${trimUserContent(text, 1000)}"""

WYMAGANIA DLA PYTANIA:
- Po polsku, ${target}
- Gramatycznie poprawne i naturalne dla dziecka.
- OdnoĹ› siÄ™ do KONKRETNEGO elementu z fragmentu: czynnoĹ›Ä‡, miejsce, cel, obiekt, czas.
- Dopuszczalne sĹ‚owa pytajÄ…ce: Kto, Co, Gdzie, Kiedy, Po co, Czym. (Preferuj: Gdzie/Co/Po co/Kiedy.)
- JeĹ›li fragment jest w 1. osobie (np. â€žSiedzÄ™â€¦â€ť, â€žIdÄ™â€¦â€ť, â€žPrzeczytamâ€¦â€ť), NIE uĹĽywaj â€žKtoâ€¦?â€ť. Zadaj â€žGdzieâ€¦?â€ť, â€žDokÄ…dâ€¦?â€ť, â€žCoâ€¦?â€ť, â€žPo coâ€¦?â€ť lub â€žKiedyâ€¦?â€ť.
- Pytanie zakoĹ„cz znakiem zapytania.

ZAKAZY (BEZWZGLÄDNIE):
- OgĂłlne: â€žO czym byĹ‚ tekst?â€ť, â€žCo siÄ™ wydarzyĹ‚o?â€ť, â€žOpowiedz jednym zdaniemâ€¦â€ť, â€žCo zapamiÄ™taĹ‚eĹ›â€¦â€ť
- Definicyjne: â€žKim jestâ€¦?â€ť, â€žCo to jestâ€¦?â€ť, â€žCzym jestâ€¦?â€ť
- Nienaturalne formy (â€žKim poszedĹ‚â€¦â€ť, â€žCzym poszedĹ‚â€¦â€ť).

WYMAGANIA DLA ODPOWIEDZI (KLUCZA):
- Bardzo krĂłtka (1â€“5 sĹ‚Ăłw, max 6), jednoznaczna.
- EKSTRAKTYWNA: odpowiedĹş MUSI byÄ‡ dosĹ‚ownym fragmentem powyĹĽszego tekstu (bez parafrazy).
- Bez kropek/cudzysĹ‚owĂłw; maĹ‚e/duĹĽe litery dowolnie.

FORMAT ZWRACANY (Tylko JSON, bez komentarzy):
{
  "question": "â€¦jedno krĂłtkie pytanieâ€¦?",
  "answer": "â€¦krĂłtka odpowiedĹş â€“ dokĹ‚adny fragment z tekstuâ€¦"
}`.trim();
}

/* ===================== /agent/comprehend ===================== */
app.post('/agent/comprehend', async (req, res) => {
  try {
    const { text = '', age } = req.body || {};
    if (!text.trim()) return res.status(400).json({ ok: false, error: 'NO_TEXT' });

    const firstPerson = isFirstPersonText(text);

    // === 1. OSOBA â€” PRIORYTET: dopeĹ‚nienie (Co/Czego/W co) ===
    function buildFirstPersonQA(txt) {
      const hard = findTransitive1stFormHard(txt);
      if (hard) {
        const obj = extractObjectAfter1st(txt, hard);
        if (obj) {
          if (hard === "sĹ‚ucham") return { question: `Czego ${hard}?`, answer: obj };
          if (hard === "gram")    return { question: `W co ${hard}?`,  answer: obj };
          return { question: `Co ${hard}?`, answer: obj }; // czytam/jem/pijÄ™/oglÄ…dam
        }
      }

      // Cel â†’ Czas â†’ Miejsce â†’ fallback
      const purpose = (String(txt).match(/\b(ĹĽeby|aby)\s+[^.?!,]+/i)?.[0] || '').trim();
      const when =
        (String(txt).match(/\b(jutro|dzisiaj|dziĹ›|wczoraj|rano|wieczorem|po poĹ‚udniu|popoĹ‚udniu)\b/i)?.[0] ||
         String(txt).match(/\bw\s+(poniedziaĹ‚ek|wtorek|Ĺ›rodÄ™|czwartek|piÄ…tek|sobotÄ™|niedzielÄ™)\b/i)?.[0] || '').trim();
      const place = (String(txt).match(/\b(przy|w|na|pod|obok|do)\s+[^,.!?]+/i)?.[0] || '').trim();

      if (purpose) return { question: 'Po co to robiÄ™?', answer: purpose };
      if (when)    return { question: 'Kiedy to robiÄ™?', answer: when };
      if (place)   return { question: 'Gdzie jestem?',   answer: place };

      return { question: 'Co robiÄ™?', answer: 'robiÄ™' };
    }

    if (firstPerson) {
      const qa = buildFirstPersonQA(text);
      return res.json({ ok: true, question: qa.question, answer: qa.answer });
    }

    // === 3. OSOBA â€” sprĂłbuj heurystyki przed LLM
    const h3 = thirdPersonDirectQA(text);
    if (h3 && h3.question && h3.answer) {
      return res.json({ ok: true, question: h3.question, answer: h3.answer });
    }

    // === 3. osoba â†’ generuje LLM + walidacja ===
    const prompt = buildQuestionPrompt({ text, age });
    const out = await raceLLM({ prompt, max_tokens: 180, temperature: 0.35 });

    const json = extractJSON(out) || {};
    let question = (json.question || '').trim();
    let answer   = (json.answer   || '').trim();

    const BAD =
      !question || !answer ||
      isGenericQuestion(question) ||
      isDefinitionQuestion(question) ||
      answerWordCount(answer) > 6 ||
      !endsWithQuestionMark(question) ||
      !answerIsExtractive(text, answer);

    if (BAD) {
      const retryPrompt = buildQuestionPrompt({ text, age }) +
        `\nUWAGA: Poprzednia prĂłba nie speĹ‚niĹ‚a zasad (zbyt ogĂłlna/definicyjna lub odpowiedĹş nie byĹ‚a fragmentem tekstu). ` +
        `ZwrĂłÄ‡ NOWY JSON. OdpowiedĹş musi byÄ‡ DOSĹOWNIE zaczerpniÄ™ta z fragmentu, maks. 6 sĹ‚Ăłw.`;
      const out2 = await raceLLM({ prompt: retryPrompt, max_tokens: 160, temperature: 0.2 });
      const j2 = extractJSON(out2) || {};
      question = (j2.question || question || '').trim();
      answer   = (j2.answer   || answer   || '').trim();
    }

    const qClean = question.replace(/[â€žâ€ť"']/g, '').trim();
    const aClean = answer.replace(/[â€žâ€ť"']/g, '').trim();

    if (!qClean || !aClean ||
        isGenericQuestion(qClean) ||
        isDefinitionQuestion(qClean) ||
        answerWordCount(aClean) > 6 ||
        !endsWithQuestionMark(qClean) ||
        !answerIsExtractive(text, aClean)) {
      return res.json({
        ok: true,
        question: 'Gdzie byĹ‚ gĹ‚Ăłwny bohater?',
        answer: 'w podanym miejscu',
        fallback: true
      });
    }

    return res.json({ ok: true, question: qClean, answer: aClean });
  } catch (err) {
    console.error('comprehend error:', err);
    return res.status(200).json({
      ok: true,
      question: 'Gdzie byĹ‚ gĹ‚Ăłwny bohater?',
      answer: 'w podanym miejscu',
      fallback: true
    });
  }
});

/* â€” Ocena odpowiedzi gĹ‚osowej dziecka â€” */
function buildCheckPrompt({ text, age, question, childAnswer, expectedAnswer }) {
  const wiek = Number(age);
  const styl =
    Number.isFinite(wiek) && wiek <= 8
      ? 'feedback jedno krĂłtkie zdanie, bardzo proste i motywujÄ…ce'
      : 'feedback 1â€“2 krĂłtkie zdania, proste i motywujÄ…ce';

  return `
Wciel siÄ™ w nauczyciela jÄ™zyka polskiego w klasach 1â€“3 i oceĹ„ odpowiedĹş dziecka.

Fragment:
"""${trimUserContent(text, 1000)}"""

Pytanie:
"${question}"

OdpowiedĹş dziecka:
"${childAnswer || ''}"

Oczekiwana poprawna odpowiedĹş (klucz):
"${expectedAnswer || ''}"

Zasady oceny:
- OceĹ„ TYLKO sens merytoryczny; bĹ‚Ä™dy jÄ™zykowe ignoruj.
- JeĹ›li odpowiedĹş jest bliska znaczeniowo â€“ zaakceptuj jako poprawnÄ….
- ZwrĂłÄ‡ TYLKO JSON:
{
  "ok": true/false,
  "feedback": "krĂłtki komentarz dla dziecka",
  "expectedAnswer": "powtĂłrz poprawnÄ… odpowiedĹş jednym krĂłtkim zdaniem lub 1-5 sĹ‚owami"
}

Styl feedbacku: ${styl}. ZAWSZE po polsku.`.trim();
}

app.post('/agent/check-answer-voice', upload.single('audio'), async (req, res) => {
  try {
    const { question = '', text = '', age, expectedAnswer = '' } = req.body || {};

    if (!req.file) return res.status(400).json({ ok: false, error: 'NO_AUDIO' });
    if (!question || !text) return res.status(400).json({ ok: false, error: 'NO_Q_OR_TEXT' });

    // 1) ASR
    const ext = pickAudioExt(req.file);
    const tmpPath = path.join(os.tmpdir(), `ans-${Date.now()}.${ext}`);
    fs.writeFileSync(tmpPath, req.file.buffer);
    const stream = fs.createReadStream(tmpPath);

    let childAnswer = '';
    try {
      if (groq) {
        const tr = await groq.audio.transcriptions.create({
          file: stream,
          model: 'whisper-large-v3',
          language: 'pl',
          response_format: 'json',
          temperature: 0
        });
        childAnswer = (tr?.text || '').trim();
      } else if (openai) {
        const tr = await openai.audio.transcriptions.create({
          file: stream,
          model: 'whisper-1',
          language: 'pl',
          response_format: 'json',
          temperature: 0
        });
        childAnswer = (tr?.text || '').trim();
      } else {
        return res.status(502).json({ ok: false, error: 'NO_ASR_PROVIDER' });
      }
    } finally {
      fs.unlink(tmpPath, () => {});
    }

    // 2) Ocena
    const checkPrompt = buildCheckPrompt({
      text,
      age,
      question,
      childAnswer,
      expectedAnswer
    });

    const out = await raceLLM({ prompt: checkPrompt, max_tokens: 160, temperature: 0.2 });
    const json = extractJSON(out) || {};
    const ok = !!json.ok;
    const feedback = (json.feedback || '').trim();
    const expected = (json.expectedAnswer || expectedAnswer || '').trim();

    return res.json({
      ok: true,
      recognizedText: childAnswer,
      result: ok ? 'ok' : 'bad',
      feedback,
      expectedAnswer: expected
    });
  } catch (e) {
    console.error('check-answer-voice error:', e);
    return res.status(200).json({
      ok: true,
      recognizedText: '',
      result: 'bad',
      feedback: 'Nie udaĹ‚o siÄ™ oceniÄ‡ odpowiedzi, sprĂłbuj powiedzieÄ‡ jÄ… jeszcze raz.',
      expectedAnswer: expectedAnswer || ''
    });
  }
});

/* â€” Ocena odpowiedzi TEKSTOWEJ dziecka (bez audio) â€” */
app.post('/agent/check-answer-text', async (req, res) => {
  try {
    const {
      question = '',
      text = '',
      age,
      expectedAnswer = '',
      childAnswer = ''
    } = req.body || {};

    if (!question || !text) return res.status(400).json({ ok: false, error: 'NO_Q_OR_TEXT' });

    const checkPrompt = buildCheckPrompt({
      text,
      age,
      question,
      childAnswer,
      expectedAnswer
    });

    const out = await raceLLM({ prompt: checkPrompt, max_tokens: 160, temperature: 0.2 });
    const json = extractJSON(out) || {};
    const ok = !!json.ok;
    const feedback = (json.feedback || '').trim();
    const expected = (json.expectedAnswer || expectedAnswer || '').trim();

    return res.json({
      ok: true,
      recognizedText: childAnswer,
      result: ok ? 'ok' : 'bad',
      feedback,
      expectedAnswer: expected
    });
  } catch (e) {
    console.error('check-answer-text error:', e);
    return res.status(200).json({
      ok: true,
      recognizedText: '',
      result: 'bad',
      feedback: 'Nie udaĹ‚o siÄ™ oceniÄ‡ odpowiedzi, sprĂłbuj wpisaÄ‡ jÄ… ponownie.',
      expectedAnswer: ''
    });
  }
});

/* ===================== START ===================== */
async function prewarmOnce() {
  try {
    if (process.env.GROQ_API_KEY) {
      await groqChat({ messages: [{ role: 'user', content: 'ping' }], max_tokens: 8, temperature: 0.0 });
    }
    if (BASE_URL) {
      await fetch(`${BASE_URL}/health`, { headers: { Connection: 'keep-alive' } }).catch(()=>{});
    }
  } catch { /* noop */ }
}

app.listen(PORT, () => {
  console.log(`đźš€ Backend dziaĹ‚a na http://localhost:${PORT}`);
  console.log(`đźŽ§ Groq ${groq ? 'podĹ‚Ä…czony' : 'OFF'} (model=${GROQ_MODEL})`);
  console.log(`đź¤– OpenAI ${openai ? 'podĹ‚Ä…czony' : 'OFF'}`);
  prewarmOnce();
  if (PREWARM_EVERY_MIN > 0) {
    setInterval(prewarmOnce, PREWARM_EVERY_MIN * 60_000);
    console.log(`đź›Ś Anti-sleep: ping co ${PREWARM_EVERY_MIN} min${BASE_URL ? ` â†’ ${BASE_URL}/health` : ''}`);
  }
});

