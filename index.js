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

const keepAliveAgent = new http.Agent({ keepAlive: true, timeout: 10_000 }); // (pozostawiony – już nie wpinamy go do fetch)
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

/* ===================== OCR helpers ===================== */
const LANG_PATH =
  process.env.OCR_LANG_PATH ||
  'https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main';

let inflight = 0;
const MAX_CONCURRENCY = Number(process.env.OCR_MAX_CONCURRENCY || 2);
async function acquire() { while (inflight >= MAX_CONCURRENCY) await sleep(40); inflight++; }
function release() { inflight = Math.max(0, inflight - 1); }

const WHITELIST =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZĄĆĘŁŃÓŚŹŻ' +
  'abcdefghijklmnopqrstuvwxyząćęłńóśźż' +
  '0123456789' +
  ' .,:;!?„”"\'()-–—/\\[]{}…';

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
  res.json({ ok: true, service: 'poczytajmy-backend', version: '1.5-text-endpoint' });
});

// Prosty root, żeby nie było 404 przy wejściu na URL
app.get('/', (_req, res) => {
  res.type('html').send(`
    <html><head><meta charset="utf-8"><title>poczytajmy-backend</title></head>
    <body style="font-family: system-ui, sans-serif; padding:24px">
      <h1>poczytajmy-backend</h1>
      <p>Status: <a href="/health">/health</a></p>
      <ul>
        <li>POST <code>/agent/generate-greeting</code></li>
        <li>POST <code>/agent/generate-text</code></li>
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
        // Groq Whisper – verbose_json pozwala wyciągnąć words/segments
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
        // OpenAI Whisper – verbose_json z segments/words
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

    // Fallback gdy brak word-level timestamps – rozsmaruj po czasie
    if (!Array.isArray(wordTimestamps) || wordTimestamps.length === 0) {
      const words = (recognizedText || '').split(/\s+/).filter(Boolean);
      let t = 0;
      wordTimestamps = words.map(w => {
        const start = t; const end = t + 0.4; t += 0.8; // 0.4s artykulacji + 0.4s krótka pauza
        return { word: w, tStart: start, tEnd: end };
      });
    }

    const wordsRead = Number(wordTimestamps.length || 0);

    // Accuracy: prosty Jaccard po tokenach (szybki i stabilny)
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

/* ===================== AGENT POWITAŃ: tematy czytelnicze + SANITYZACJA ===================== */

const HERO_THEMES = {
  'Miś': 'przytulny i cierpliwy, kocha bajki na dobranoc',
  'Labuś': 'energiczny i wesoły, lubi książki przygodowe',
  'Króliczek': 'ciekawski i szybki, uwielbia zagadki w opowieściach',
  'Jeżyk': 'ostrożny i mądry, kocha opowieści z morałem'
};

const READING_TOPICS = [
  'książki pełne magii i zaklęć',
  'czytanie bajek na głos',
  'szukanie nowych słów w opowiadaniu',
  'przeżywanie przygód z bohaterami książek',
  'poznawanie liter i sylab',
  'czytanie komiksów z obrazkami',
  'odkrywanie tajemnic w bibliotece',
  'pisanie własnej bajki po przeczytaniu książki',
  'czytanie rozdziałów z przygodami',
  'opowiadanie przeczytanej historii przyjaciołom'
];

function pick(arr){ return arr[Math.floor(Math.random()*arr.length)]; }

function normalize(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[„”"!?.,;:()\-\–—[\]{}…]/g, '')
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

function buildGreetingPrompt({ age, character = 'Twój przyjaciel', theme = '', n = 12 }) {
  const wiek = Number.isFinite(age) ? age : 'X';
  const tone =
    Number.isFinite(age) && age <= 5
      ? 'proste, ciepłe, zabawowe; rytm mowy dziecka; onomatopeje OK'
      : Number.isFinite(age) && age <= 8
      ? 'żywe, motywujące; mini-misja; 1–2 emoji'
      : 'pewne, partnerskie; cel, sprawczość; max 1–2 emoji';

  const heroHint = theme ? `Delikatny klimat bohatera: ${theme}.` : '';
  const chosenTopic = pick(READING_TOPICS);

  return `Wymyśl ${n} ZUPEŁNIE różnych, krótkich powitań po polsku dla dziecka (wiek: ${wiek}).
Mówi ${character}. Styl: ${tone}. ${heroHint}
Temat przewodni: ${chosenTopic}.

⚡ Każde powitanie MUSI odnosić się do czytania i książek, np. słowa: książka, czytanie, rozdział, bajka, historia, sylaba, słowo, zdanie, ilustracje, narrator, zakładka, biblioteka, księgarnia, opowieść, litery.
⚡ NIE używaj motywów typu: las, bieganie, sport, piknik, podróże — tylko świat książek.
⚡ Zakaz: nie używaj słów powitalnych (cześć, hej, witaj, siema, halo) oraz NIE używaj imienia dziecka w żadnej formie.

📚 Przykłady:
- Dziś razem odkryjemy nowy rozdział bajki. 📖
- Zajrzymy do książki pełnej czarodziejskich słów. ✨
- Sprawdzimy, ile sylab ma najdłuższe słowo w opowieści. 🚀

Zasady: jedno zdanie, 6–14 wyrazów, bez cudzysłowów i bez wstępów.
Każde powitanie w osobnej linii poprzedzone myślnikiem "- ".`;
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

const FORBIDDEN_HELLOS = ['cześć', 'hej', 'witaj', 'siema', 'halo'];
function sanitizeNoName(name, raw) {
  let s = (raw || '').trim();
  const helloRe = new RegExp(`^\\s*(?:${FORBIDDEN_HELLOS.join('|')})\\b[\\p{L}\\p{M}\\s,!.?–—-]*`, 'iu');
  s = s.replace(helloRe, '').trim();
  if (name) {
    const forms = [name, `${name}u`, `${name}o`, `${name}e`, `${name}a`, `${name}ku`];
    const escaped = forms.map(v => v.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'));
    const nameRe = new RegExp(`\\b(?:${escaped.join('|')})\\b[\\s,!.?]*`, 'giu');
    s = s.replace(nameRe, '').trim();
  }
  s = s.replace(/^[,–—\-|:;!.\s]+/u, '').trim();
  return s;
}

const recentGreetings = new Map();

/* ===== Groq/OpenAI race helpers (no local text) ===== */
async function groqChat({ messages, max_tokens = MAX_TOKENS_FAST, temperature = 0.3, top_p = 0.95 }) {
  const t0 = now();
  // Zmiana: usunięty `agent: keepAliveAgent` (fetch w Node ignoruje tę opcję)
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
    const { name = '', age, character = 'Twój przyjaciel' } = req.body || {};
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

/* ===================== AGENT MOTYWACJI (Groq/OpenAI) ===================== */

function bucketToneByAge(age) {
  const a = Number(age);
  if (Number.isFinite(a) && a <= 5) return 'bardzo prosto, ciepło, łagodnie; krótkie słowa; 1 emoji max';
  if (Number.isFinite(a) && a <= 8) return 'prosto, energicznie, wspierająco; mini-sugestia co poprawić; 1 emoji max';
  return 'partnersko, konkretnie, z uznaniem; 1 emoji max';
}

function rubricByAccuracy(acc) {
  const s = Math.max(0, Math.min(100, Math.round(acc || 0)));
  if (s >= 95) return 'wynik świetny; podkreśl perfekcję i zaproponuj trudniejsze słowo przy następnej stronie';
  if (s >= 80) return 'wynik bardzo dobry; pochwal płynność i zaproponuj jedną mikro-radę (np. dokładniej końcówki)';
  if (s >= 60) return 'wynik dobry; pochwal staranie i podaj jedną prostą wskazówkę (np. wolniej, sylabizuj trudniejsze słowa)';
  return 'wynik na rozgrzewkę; skup się na zachęcie i jednej mini-radzie (np. przeczytaj zdanie jeszcze raz spokojnie)';
}

function buildMotivationPrompt({ age, accuracy, text, characterName = 'Bohater', lang = 'pl' }) {
  const tone = bucketToneByAge(age);
  const rubric = rubricByAccuracy(accuracy);
  const excerpt = trimUserContent(text || '', 220);

  return `
Jesteś ${characterName} z aplikacji do nauki czytania dla dzieci. Twoje zadanie:
napisz 1 krótki komentarz motywacyjny po polsku (${lang}), dopasowany do wieku dziecka i jakości czytania.

Zasady stylu:
- Styl: ${tone}.
- ${rubric}.
- Maks. 160 znaków. 1 zdanie (wyjątkowo 2 bardzo krótkie).
- Brak cudzysłowów i nawiasów. Bez liczb procentowych ani ocen wprost.
- Mów do dziecka w 2. osobie („czytasz”, „dasz radę”), NIE używaj imienia dziecka.
- Użyj co najwyżej 1 emoji (opcjonalnie).

Kontekst (fragment przeczytanego tekstu – opcjonalnie możesz nawiązać ogólnie, bez cytowania):
"${excerpt}"

Podaj tylko gotową wypowiedź.`;
}

// --- Hard limiter: 1–2 zdania, <= maxChars, max 1 emoji, bez cudzysłowów/nawiasów/cytatów
function tightenMotivation(s, maxChars = 160) {
  if (!s) return s;

  // usuń cudzysłowy, nawiasy i nadmiar spacji
  s = String(s)
    .replace(/[\"“”„”'()]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  // usuń fragmenty w cytatach (np. «kota», „kota”, "kota")
  s = s.replace(/[«»„”"'].*?[«»„”"']/g, '').replace(/\s+/g, ' ').trim();

  // rozbij na zdania i weź maks 2
  const parts = s.split(/(?<=[.!?…])\s+/).filter(Boolean);
  s = parts.slice(0, 2).join(' ').trim();

  // zostaw max 1 emoji
  const emojiRe = /[\p{Extended_Pictographic}\uFE0F]/gu;
  let seen = 0;
  s = s.replace(emojiRe, m => (++seen > 1 ? '' : m));

  // twardy limit znaków (ucięcie na granicy wyrazu)
  if (s.length > maxChars) {
    s = s.slice(0, maxChars).replace(/\s+\S*$/, '').trim();
  }

  // domknij kropką, jeśli brak
  if (!/[.!?…]$/.test(s)) s += '.';
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
  // lekkie sanity: usuń otaczające cudzysłowy
  out = out.replace(/^["'„”]+|["'„”]+$/g, '').trim();

  // TWARDY LIMIT w generatorze
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
      name,                 // świadomie ignorujemy w treści (priv + prostota)
      characterName = 'Bohater',
      lang = 'pl',
    } = req.body || {};

    const { text: rawMsg, source } = await generateMotivation({
      age, accuracy, text, characterName, lang
    });

    // DRUGI BEZPIECZNIK w endpointzie
    const msg = tightenMotivation(rawMsg, 160);

    res.json({ ok: true, text: msg, source });
  } catch (err) {
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    if (timedOut) return res.status(504).json({ ok: false, error: 'DEADLINE_EXCEEDED', timed_out: true });
    console.error('agent/motivate error:', err);
    return res.status(502).json({
      ok: false,
      error: String(err?.message || err),
      // bezpieczny fallback zgodny z UI
      fallback: 'Świetna próba! Z każdą stroną będzie coraz lepiej — spróbujmy jeszcze raz! 💪'
    });
  }
});

/* ===================== GENERATOR ZDAŃ DO CZYTANIA ===================== */
const BANK_A1 = [
  'Ala ma kota.',
  'Miś je miodek.',
  'Piłka leży na trawie.',
  'Pies biegnie do domu.',
  'Słońce świeci jasno.',
];
const BANK_A2 = [
  'W ogrodzie rosną kolorowe kwiaty.',
  'Kasia czyta ciekawą książkę o zwierzętach.',
  'Na spacerze spotkaliśmy wesołego psa.',
  'Dziś po południu pojedziemy na rowerach.',
];
const BANK_B1 = [
  'Choć padał deszcz, wybraliśmy się na długi spacer.',
  'Lubię zagadki, bo rozwijają wyobraźnię i spostrzegawczość.',
  'Z zachwytem obserwowałem, jak motyl siada na liściu.',
  'Po kolacji wspólnie ułożyliśmy plan jutrzejszej wycieczki.',
];
function bankByLevel(level = 'A1') {
  const L = String(level).toUpperCase();
  if (L === 'B1') return BANK_B1;
  if (L === 'A2') return BANK_A2;
  return BANK_A1;
}

app.post('/agent/generate-text', async (req, res) => {
  try {
    const { language = 'pl', level = 'A1' } = req.body || {};

    if (MOCK_TEXT) {
      const list = bankByLevel(level);
      return res.json({ ok: true, text: pick(list), level, language, source: 'mock' });
    }

    const prompt =
`Napisz jedno proste zdanie po ${language === 'pl' ? 'polsku' : 'angielsku'} na poziomie ${String(level).toUpperCase()} do głośnego czytania przez dziecko.
Zasady: jedno zdanie, jasno i naturalnie, bez cudzysłowów, 12–16 słów.`;

    const racers = [];
    if (process.env.GROQ_API_KEY) {
      racers.push(groqChat({
        messages: [{ role: 'user', content: trimUserContent(prompt) }],
        temperature: 0.7, top_p: 0.95, max_tokens: 60,
      }));
    }
    if (openai) {
      racers.push((async () => {
        const t0 = now();
        const r = await openai.chat.completions.create({
          model: 'gpt-4o-mini',
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.8, top_p: 0.95, max_tokens: 60,
        });
        const txt = r?.choices?.[0]?.message?.content?.trim?.() || '';
        if (!txt) throw new Error('OPENAI_EMPTY');
        return { provider: 'openai', text: txt, latency_ms: Math.round(now() - t0) };
      })());
    }

    const winner = await withDeadline(Promise.any(racers), DEADLINE_MS);
    let text = String(winner.text || '').replace(/^["'„”]+|["'„”]+$/g, '').trim();
    if (!text) throw new Error('EMPTY_GENERATION');

    return res.json({ ok: true, text, level, language, source: winner.provider });
  } catch (err) {
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    if (timedOut) return res.status(504).json({ ok: false, error: 'DEADLINE_EXCEEDED', timed_out: true });
    console.error('agent/generate-text error:', err);
    return res.status(502).json({ ok: false, error: String(err?.message || err) });
  }
});

// Alias zgodności wstecznej
app.post('/generate-text', (req, res) => {
  res.redirect(307, '/agent/generate-text');
});

/* ===================== OCR ===================== */
app.post('/ocr', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ ok: false, error: 'NO_FILE' });
    if (process.env.MOCK_OCR === '1') return res.json({ ok: true, text: 'Przykładowy tekst z OCR.' });

    if (process.env.USE_OPENAI_OCR === '1' && openai) {
      const b64 = `data:image/jpeg;base64,${req.file.buffer.toString('base64')}`;
      const prompt = 'Wyodrębnij czysty tekst z obrazu (po polsku). Zwróć tylko tekst.';
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

/* ===================== ElevenLabs TTS proxy (nowe) ===================== */
/* ENV: ELEVEN_API_KEY lub ELEVENLABS_API_KEY */
app.post('/tts', async (req, res) => {
  try {
    const apiKey = process.env.ELEVEN_API_KEY || process.env.ELEVENLABS_API_KEY;
    if (!apiKey) return res.status(500).json({ ok: false, error: 'NO_ELEVEN_API_KEY' });

    const { text = '', voiceId = '21m00Tcm4TlvDq8ikWAM' } = req.body || {}; // Rachel (domyślna)
    const clean = String(text).trim().slice(0, 500);
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
        voice_settings: { stability: 0.5, similarity_boost: 0.75 }
      })
    });
    if (!r.ok) return res.status(502).json({ ok: false, error: `ELEVEN_HTTP_${r.status}` });

    const buf = Buffer.from(await r.arrayBuffer());
    res.json({ ok: true, audioB64: buf.toString('base64') });
  } catch (err) {
    console.error('TTS proxy error:', err);
    res.status(500).json({ ok: false, error: 'TTS_PROXY_FAILED' });
  }
});

/* ===================== START ===================== */
async function prewarmOnce() {
  try {
    if (process.env.GROQ_API_KEY) {
      await groqChat({ messages: [{ role: 'user', content: 'ping' }], max_tokens: 8, temperature: 0.0 });
    }
    if (BASE_URL) {
      // Zmiana: usunięty `agent: keepAliveAgent`
      await fetch(`${BASE_URL}/health`, { headers: { Connection: 'keep-alive' } }).catch(()=>{});
    }
  } catch { /* noop */ }
}

app.listen(PORT, () => {
  console.log(`🚀 Backend działa na http://localhost:${PORT}`);
  console.log(`🎧 Groq ${groq ? 'podłączony' : 'OFF'} (model=${GROQ_MODEL})`);
  console.log(`🤖 OpenAI ${openai ? 'podłączony' : 'OFF'}`);
  prewarmOnce();
  if (PREWARM_EVERY_MIN > 0) {
    setInterval(prewarmOnce, PREWARM_EVERY_MIN * 60_000);
    console.log(`🛌 Anti-sleep: ping co ${PREWARM_EVERY_MIN} min${BASE_URL ? ` → ${BASE_URL}/health` : ''}`);
  }
});
