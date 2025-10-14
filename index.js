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
const DEADLINE_MS = Number(process.env.FAST_TIMEOUT_MS || 1500);
const MAX_TOKENS_FAST = Number(process.env.MAX_TOKENS_FAST || 64);
const PREWARM_EVERY_MIN = Number(process.env.PREWARM_EVERY_MIN || 5); // 0 = tylko na starcie
const BASE_URL = process.env.BASE_URL || '';

// Rozdzielone modele Groq: czat (LLM) i ASR (Whisper)
const GROQ_CHAT_MODEL = process.env.GROQ_CHAT_MODEL || 'llama-3.1-8b-instant';
const GROQ_ASR_MODEL  = process.env.GROQ_ASR_MODEL  || 'whisper-large-v3';

const LLM_PREF = 'openai-only'; // preferencja, ale mamy failover na Groq

// ★ Dłuższe deadline’y tylko tam gdzie trzeba (możesz nadpisać ENV):
const GREETING_TIMEOUT_MS      = Number(process.env.GREETING_TIMEOUT_MS || 9000);
const MOTIVATE_TIMEOUT_MS      = Number(process.env.MOTIVATE_TIMEOUT_MS || 10000);
const GENERATE_TEXT_TIMEOUT_MS = Number(process.env.GENERATE_TEXT_TIMEOUT_MS || 10000);

const keepAliveAgent = new http.Agent({ keepAlive: true, timeout: 10_000 });
const now = () => (global.performance?.now?.() ?? Date.now());
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function withDeadline(promise, ms = DEADLINE_MS) {
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error('DEADLINE_EXCEEDED')), ms);
    promise.then(v => { clearTimeout(to); resolve(v); }, e => { clearTimeout(to); reject(e); });
  });
}

// ★ Retry z backoffem i wydłużaniem deadline’u
async function withDeadlineRetry(makePromiseFn, { deadlineMs, retries = 1, backoffMs = 250 }) {
  let ms = deadlineMs, lastErr;
  for (let i = 0; i <= retries; i++) {
    try {
      return await withDeadline(makePromiseFn(), ms);
    } catch (e) {
      lastErr = e;
      const msg = String(e?.message || e);
      if (msg === 'DEADLINE_EXCEEDED' && i < retries) {
        await sleep(backoffMs * (i + 1));
        ms = Math.round(ms * 1.6); // delikatne wydłużenie
        continue;
      }
      throw e;
    }
  }
  throw lastErr;
}

/* ===== OpenAI local RPM guard (soft) ===== */
const OAI_RPM_LIMIT = Number(process.env.OPENAI_RPM_LIMIT || 3); // widzisz 3 w logach
const OAI_WINDOW_MS = 60_000;
let oaiCallsTimestamps = [];
function _pruneOai() {
  const nowTs = Date.now();
  oaiCallsTimestamps = oaiCallsTimestamps.filter(t => nowTs - t < OAI_WINDOW_MS);
}
function canUseOpenAI() {
  _pruneOai();
  return oaiCallsTimestamps.length < OAI_RPM_LIMIT;
}
function markOpenAICall() {
  _pruneOai();
  oaiCallsTimestamps.push(Date.now());
}

/* ===== Uploads ===== */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

/* ===== Clients ===== */
// ★ timeout + maxRetries w SDK
const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY, timeout: 15000, maxRetries: 2 })
  : null;

const groq = process.env.GROQ_API_KEY ? new Groq({ apiKey: process.env.GROQ_API_KEY }) : null;

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
  res.json({ ok: true, service: 'poczytajmy-backend', version: '1.19-groq-failover' });
});

// Prosty root
app.get('/', (_req, res) => {
  res.type('html').send(`
    <html><head><meta charset="utf-8"><title>poczytajmy-backend</title></head>
    <body style="font-family: system-ui, sans-serif; padding:24px">
      <h1>poczytajmy-backend</h1>
      <p>Status: <a href="/health">/health</a></p>
      <ul>
        <li>POST <code>/agent/generate-greeting</code> oraz <code>/generate-greeting</code></li>
        <li>POST <code>/agent/generate-text</code> oraz <code>/generate-text</code></li>
        <li>POST <code>/agent/motivate</code></li>
        <li>POST <code>/agent/comprehend</code> / <code>/agent/comprehend-multi</code></li>
        <li>POST <code>/asr</code>, <code>/ocr</code></li>
      </ul>
    </body></html>
  `);
});

/* ===================== LLM helpers (OpenAI-only z failoverem) ===================== */
// Anty-sztywny setup
const NAT_TEMPERATURE   = 0.95;
const NAT_TOP_P         = 0.9;
const NAT_FREQ_PENALTY  = 0.3;
const NAT_PRES_PENALTY  = 0.2;

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
    body: JSON.stringify({
      model: GROQ_CHAT_MODEL,   // czatowy model Groq
      temperature,
      top_p,
      max_tokens,
      messages
    })
  });
  if (!res.ok) throw new Error(`GROQ_HTTP_${res.status}`);
  const data = await res.json();
  return {
    provider: 'groq',
    text: data?.choices?.[0]?.message?.content?.trim?.() || '',
    latency_ms: Math.round(now() - t0)
  };
}

async function openaiChat({ messages, max_tokens = MAX_TOKENS_FAST, temperature = 0.3, top_p = 0.95 }) {
  if (!openai) throw new Error('OPENAI_OFF');
  if (!canUseOpenAI()) {
    const e = new Error('OPENAI_THROTTLED');
    e.code = 'OPENAI_THROTTLED';
    throw e;
  }
  const t0 = now();
  try {
    const r = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages,
      temperature,
      top_p,
      max_tokens,
      frequency_penalty: NAT_FREQ_PENALTY,
      presence_penalty: NAT_PRES_PENALTY,
    });
    markOpenAICall();
    const txt = r?.choices?.[0]?.message?.content?.trim?.() || '';
    if (!txt) throw new Error('OPENAI_EMPTY');
    return { provider: 'openai', text: txt, latency_ms: Math.round(now() - t0) };
  } catch (err) {
    const status = err?.status || err?.code;
    if (status === 429 || err?.code === 'rate_limit_exceeded') {
      const e = new Error('OPENAI_THROTTLED');
      e.code = 'OPENAI_THROTTLED';
      throw e;
    }
    throw err;
  }
}

/* ===== Tekst helper: skróć/wyczyść prompt od użytkownika ===== */
function trimUserContent(s = "", limit = 800) {
  const t = String(s ?? "").replace(/\s+/g, " ").trim();
  return t.length > limit ? t.slice(0, limit) : t;
}

// ENTRYPOINT — prefer OpenAI + retry; w razie throttlingu → Groq → fallback
async function chatPref({ prompt, max_tokens = 150, temperature = 0.3, top_p = 0.95, deadlineMs = DEADLINE_MS }) {
  const messages = [{ role: 'user', content: trimUserContent(prompt) }];

  // 1) Spróbuj OpenAI (z retry)
  const makeOai = () => openaiChat({ messages, max_tokens, temperature, top_p });
  try {
    if (openai) {
      return await withDeadlineRetry(makeOai, { deadlineMs, retries: 1, backoffMs: 250 });
    }
  } catch (err) {
    if (err?.code !== 'OPENAI_THROTTLED') throw err; // inny błąd — nie maskujemy
    // wpadliśmy w throttling — lecimy na Groq bez marnowania czasu
  }

  // 2) Failover → Groq
  if (groq) {
    try {
      const { text, provider } = await withDeadline(
        groqChat({
          messages,
          max_tokens,
          temperature: Math.min(0.7, Math.max(0.2, temperature)),
          top_p
        }),
        Math.max(1500, Math.round(deadlineMs * 0.9))
      );
      return { text, provider };
    } catch { /* miękko */ }
  }

  // 3) Ostateczny fallback — wyżej endpointy mają bezpieczne 200
  const e = new Error('GEN_FALLBACK');
  e.code = 'GEN_FALLBACK';
  throw e;
}

/** Wrapper kompatybilny */
async function raceLLM({ prompt, max_tokens = 150, temperature = 0.3 }) {
  const { text } = await chatPref({ prompt, max_tokens, temperature, top_p: 0.95 });
  return (text || '').trim();
}

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
          model: GROQ_ASR_MODEL, // <— ASR model z ENV
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

/* ===================== AGENT POWITAŃ ===================== */

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
    const escaped = forms.map(v => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const nameRe = new RegExp(`\\b(?:${escaped.join('|')})\\b[\\s,!.?]*`, 'giu');
    s = s.replace(nameRe, '').trim();
  }
  s = s.replace(/^[,–—\-|:;!.\s]+/u, '').trim();
  return s;
}

const recentGreetings = new Map();

// Anty-szablon + soften + pamięć podobieństw (dla zdań/komentarzy)
const TEMPLATEY_STARTS = [
  "Dziś", "Dzisiaj", "Po południu", "W ogrodzie", "Choć", "Chociaż", "Na koniec",
  "Potem", "Następnie", "Po kolacji", "Po obiedzie", "W bibliotece", "W domu"
];
function looksTemplatey(s="") {
  const t = String(s).trim();
  if (!t) return true;
  if (/\b[aA]\s+potem\b/.test(t)) return true;
  if (TEMPLATEY_STARTS.some(p => t.startsWith(p + " "))) return true;
  const conj = (t.match(/\b(a|oraz)\b/gi) || []).length;
  if (conj >= 2) return true;
  return false;
}
function softenPolish(s="") {
  let out = String(s).trim();
  out = out.replace(/\s*,?\s*a potem\s*/gi, " i ");
  out = out.replace(/\s*,\s*oraz\s*/gi, " i ");
  out = out.replace(/\s*,\s*a\s*/gi, " i ");
  if (!/[.!?…]$/.test(out)) out += ".";
  out = out.replace(/^Z /, "Ze ");
  return out.replace(/\s+/g," ").trim();
}
const recentTexts = [];
function rememberText(t) { recentTexts.unshift(String(t)); if (recentTexts.length > 20) recentTexts.pop(); }
function tooSimilarToRecent(t) {
  const n = normalize(t);
  return recentTexts.some(prev => jaccard(n, normalize(prev)) > 0.6);
}

async function generateGreetingV2({ name, age, character, theme }) {
  const prompt = buildGreetingPrompt({ age: Number(age), character, theme, n: 12 });
  const { text: raw } = await chatPref({
    prompt,
    temperature: NAT_TEMPERATURE,
    max_tokens: 180,
    top_p: NAT_TOP_P,
    deadlineMs: GREETING_TIMEOUT_MS,
  });

  let cands = parseList(raw);
  if (!cands.length && raw) cands = raw.split(/[.\n]/).map(s => s.trim()).filter(Boolean);
  if (!cands.length) throw new Error('EMPTY_GENERATION');

  const profileKey = `${(name || '').toLowerCase()}|${Number(age)||'X'}`;
  const history = recentGreetings.get(profileKey) || [];

  const picked = chooseMostNovel(cands, history);
  const cleaned = sanitizeNoName(name, picked);
  const finalText = softenPolish(cleaned || picked);

  recentGreetings.set(profileKey, [finalText, ...history].slice(0, 20));
  return { text: finalText, source: 'openai_or_groq' };
}

app.post('/agent/generate-greeting', async (req, res) => {
  try {
    const { name = '', age, character = 'Twój przyjaciel' } = req.body || {};
    const theme = HERO_THEMES[character] || '';
    const { text, source } = await generateGreetingV2({ name, age, character, theme });
    res.json({ ok: true, text, source });
  } catch (err) {
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    console.error('agent/generate-greeting error:', err);
    const fallback = 'Zajrzymy dziś do książki i wyszukamy nowe słowa. 📖';
    return res.status(200).json({
      ok: true,
      text: fallback,
      source: timedOut ? 'timeout-fallback' : 'error-fallback',
    });
  }
});

// Alias bez redirectu (zero 307)
app.post('/generate-greeting', async (req, res) => {
  try {
    const { name = '', age, character = 'Twój przyjaciel' } = req.body || {};
    const theme = HERO_THEMES[character] || '';
    const { text, source } = await generateGreetingV2({ name, age, character, theme });
    res.json({ ok: true, text, source });
  } catch (err) {
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    console.error('generate-greeting error:', err);
    const fallback = 'Zajrzymy dziś do książki i wyszukamy nowe słowa. 📖';
    return res.status(200).json({
      ok: true,
      text: fallback,
      source: timedOut ? 'timeout-fallback' : 'error-fallback',
    });
  }
});

/* ===================== AGENT MOTYWACJI ===================== */

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
Jesteś ${characterName} w aplikacji do czytania. Napisz 1 naturalny, krótki komentarz po polsku.
Styl:
- ${tone}
- ${rubric}
- Brzmij swobodnie (jak żywa rozmowa), unikaj „szkolnych” fraz i szablonu „… a potem …”.
- Maks. 160 znaków, najlepiej 1 zdanie (wyjątkowo 2 bardzo krótkie).
- 2. osoba („czytasz”, „spróbuj”), bez imienia dziecka, bez procentów i ocen wprost.
- Co najwyżej 1 emoji (opcjonalnie).

Kontekst (nie cytuj literalnie, możesz nawiązać ogólnie):
"${excerpt}"

Podaj tylko gotową wypowiedź.`.trim();
}

function tightenMotivation(s, maxChars = 160) {
  if (!s) return s;
  s = String(s)
    .replace(/[\"“”„”'()]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  s = s.replace(/[«»„”"'].*?[«»„”"']/g, '').replace(/\s+/g, ' ').trim();
  const parts = s.split(/(?<=[.!?…])\s+/).filter(Boolean);
  s = parts.slice(0, 2).join(' ').trim();
  const emojiRe = /[\p{Extended_Pictographic}\uFE0F]/gu;
  let seen = 0;
  s = s.replace(emojiRe, m => (++seen > 1 ? '' : m));
  if (s.length > maxChars) {
    s = s.slice(0, maxChars).replace(/\s+\S*$/, '').trim();
  }
  if (!/[.!?…]$/.test(s)) s += '.';
  return s;
}

/* ▶️ Bardzo szybki lokalny fallback bez LLM (na timeout/429) */
function localMotivationFallback(age, accuracy) {
  const s = Math.max(0, Math.min(100, Math.round(accuracy || 0)));
  if (s >= 95) return 'Czytasz świetnie! Spróbuj teraz nieco trudniejszego słowa. ✨';
  if (s >= 80) return 'Super płynność — jeszcze dokładniej końcówki i będzie idealnie.';
  if (s >= 60) return 'Dobra robota! Czytaj spokojniej i sylabizuj trudniejsze słowa.';
  return 'Fajnie próbujesz — przeczytaj zdanie jeszcze raz powoli, dasz radę. 💪';
}

async function generateMotivation({ age, accuracy, text, characterName, lang = 'pl' }) {
  const prompt = buildMotivationPrompt({ age, accuracy, text, characterName, lang });

  // 🔁 Stabilizacja: retry z rosnącym deadline’em (korzysta z withDeadlineRetry, które już masz na górze pliku)
  const makeCall = () => chatPref({
    prompt,
    temperature: NAT_TEMPERATURE,
    max_tokens: 100,          // odrobinę mniej tokenów → szybciej
    top_p: NAT_TOP_P,
    deadlineMs: Math.min(MOTIVATE_TIMEOUT_MS || 9000, 9000)
  });

  let raw = '';
  let provider = 'unknown';
  try {
    const r = await withDeadlineRetry(makeCall, { deadlineMs: Math.min(MOTIVATE_TIMEOUT_MS || 9000, 9000), retries: 1, backoffMs: 350 });
    raw = String(r.text || '').trim();
    provider = r.provider || 'llm';
  } catch (e) {
    // ⛑️ W razie DEADLINE/429 wracamy natychmiast z lokalnym, sensownym tekstem
    const fb = localMotivationFallback(age, accuracy);
    return { text: fb, source: 'local-fallback' };
  }

  let out = raw.replace(/^["'„”]+|["'„”]+$/g, '').trim();
  out = tightenMotivation(out, 160);
  out = softenPolish(out);

  // Delikatna anty-powtarzalność — jeśli model wypluł „szablon”, spróbuj raz jeszcze,
  // ale bez kolejnych retry (to już mamy powyżej).
  if (looksTemplatey(out) || tooSimilarToRecent(out)) {
    try {
      const r2 = await withDeadlineRetry(makeCall, { deadlineMs: Math.min(MOTIVATE_TIMEOUT_MS || 9000, 9000), retries: 0 });
      const alt = softenPolish(tightenMotivation(String(r2.text || '').trim(), 160));
      if (!looksTemplatey(alt)) out = alt;
    } catch { /* zostawiamy pierwszą wersję */ }
  }

  if (!out) {
    const fb = localMotivationFallback(age, accuracy);
    return { text: fb, source: 'local-fallback-empty' };
  }

  rememberText(out); // masz to wcześniej zdefiniowane
  return { text: out, source: provider };
}

app.post('/agent/motivate', async (req, res) => {
  try {
    const {
      age,
      accuracy = 0,
      text = '',
      characterName = 'Bohater',
      lang = 'pl',
    } = req.body || {};

    const { text: rawMsg, source } = await generateMotivation({
      age, accuracy, text, characterName, lang
    });

    const msg = tightenMotivation(rawMsg, 160);
    res.json({ ok: true, text: msg, source });
  } catch (err) {
    const fallback = localMotivationFallback(req.body?.age, req.body?.accuracy);
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    console.error('agent/motivate error:', err);
    return res.status(200).json({
      ok: true,
      text: tightenMotivation(fallback, 160),
      source: timedOut ? 'timeout-fallback' : 'error-fallback'
    });
  }
});

/* ===================== GENERATOR ZDAŃ DO CZYTANIA ===================== */

const BANK_A1 = [
  "Ala ma kota i lubi czytać bajki wieczorem.",
  "Miś je miodek, a potem słucha krótkiej opowieści.",
  "Piłka leży na trawie, a Julek czyta na ławce.",
  "Pies biegnie do domu, gdzie czeka nowa książka.",
  "Słońce świeci jasno, a my czytamy w ogrodzie."
];
const BANK_A2 = [
  "W ogrodzie rosną kwiaty, a my czytamy o motylach.",
  "Kasia czyta książkę o zwierzętach i szuka trudnych słów.",
  "Na spacerze opowiadamy historię o małej latarni morskiej.",
  "Po południu wybieramy rozdział o odważnym króliku."
];
const BANK_B1 = [
  "Choć padał deszcz, przeczytaliśmy rozdział o podróży po mapie.",
  "Lubię zagadki, bo rozwijają wyobraźnię i pomagają w czytaniu.",
  "Z zachwytem śledziłem, jak narrator opisuje lot kolorowego motyla.",
  "Po kolacji wspólnie czytamy i planujemy jutrzejszą przygodę."
];
function bankByLevel(level = "A1") {
  const L = String(level).toUpperCase();
  if (L === "B1") return BANK_B1;
  if (L === "A2") return BANK_A2;
  return BANK_A1;
}

function onlyOneSentence(s) {
  const parts = String(s).split(/(?<=[.!?…])\s+/).filter(Boolean);
  return (parts[0] || s).trim();
}
function cleanSentence(s) {
  let out = String(s)
    .replace(/[„”"“”'()«»]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  out = onlyOneSentence(out);
  if (!/[.!?…]$/.test(out)) out += ".";
  return out;
}
function countWords(s) {
  return (String(s).trim().match(/\b[\p{L}\p{M}0-9'-]+\b/gu) || []).length;
}
const PROFANITY = [
  "kurwa","cholera","debil","idiota","głupi","szmata",
  "pedał","lesba","spier","nienawidzę","zabij","śmierć"
];
function hasForbidden(s) {
  const low = String(s).toLowerCase();
  return PROFANITY.some(p => low.includes(p));
}
function hasPolishDiacritics(s) {
  return /[ąćęłńóśźż]/i.test(String(s));
}
function validateKidsSentencePL(s, { minWords=8, maxWords=16 } = {}) {
  const issues = [];
  const txt = cleanSentence(onlyOneSentence(s));
  const words = countWords(txt);
  if (words < minWords || words > maxWords) {
    issues.push(`Liczba słów ${words} poza zakresem ${minWords}–${maxWords}.`);
  }
  if (hasForbidden(txt)) issues.push("Słowa niedozwolone.");
  if (!hasPolishDiacritics(txt)) issues.push("Brak polskich znaków.");
  const tokens = (txt.match(/\b[\p{L}\p{M}0-9'-]+\b/gu) || []);
  const long = tokens.filter(w => w.replace(/[^a-ząćęłńóśźż-]/gi,"").length > 12).length;
  const ratio = tokens.length ? long / tokens.length : 0;
  if (tokens.length > 24 || ratio > 0.4) issues.push("Zbyt trudne lub nienaturalne słownictwo.");
  return { ok: issues.length === 0, issues, text: txt };
}

// handler generatora tekstu — NAT parameters, filtr szablonów, reroll, retry/timeout
async function handleGenerateText(req, res) {
  try {
    const { language = "pl", level = "A1" } = req.body || {};

    const prompt =
`Napisz jedno naturalne i lekkie zdanie po polsku do głośnego czytania przez dziecko (poziom ${String(level).toUpperCase()}).
Wymagania:
- Jedno zdanie (8–16 słów), brzmienie swobodne (jak rozmowa z dzieckiem), bez „szkolnej” składni.
- Słownictwo codzienne, zero żargonu i neologizmów, bez nawiasów i cudzysłowów.
- Unikaj sztywnych wzorców typu „… a potem …”, „Dziś/Dzisiaj …”, długich wyliczeń i dwóch „a/oraz” w jednym zdaniu.
- Używaj polskich znaków.
Podaj tylko gotowe zdanie.`;

    const first = await chatPref({
      prompt,
      temperature: NAT_TEMPERATURE,
      max_tokens: 70,
      top_p: NAT_TOP_P,
      deadlineMs: GENERATE_TEXT_TIMEOUT_MS,
    });

    let sentence = cleanSentence(first.text || "");
    sentence = softenPolish(sentence);

    if (looksTemplatey(sentence) || tooSimilarToRecent(sentence)) {
      const reroll = await chatPref({
        prompt,
        temperature: NAT_TEMPERATURE,
        max_tokens: 70,
        top_p: NAT_TOP_P,
        deadlineMs: GENERATE_TEXT_TIMEOUT_MS,
      });
      let s2 = softenPolish(cleanSentence(reroll.text || ""));
      if (!looksTemplatey(s2)) sentence = s2;
    }

    let check = validateKidsSentencePL(sentence);
    if (!check.ok || looksTemplatey(check.text) || tooSimilarToRecent(check.text)) {
      const fixed = cleanSentence(softenPolish(await chatPref({
        prompt: `
Uprość i „rozluźnij” zdanie dla dziecka (naturalny język, 8–16 słów, bez cudzysłowów i „a potem”):
${sentence}`.trim(),
        temperature: 0.6,
        max_tokens: 60,
        top_p: NAT_TOP_P,
        deadlineMs: GENERATE_TEXT_TIMEOUT_MS,
      }).then(r => r.text || "")));

      const check2 = validateKidsSentencePL(fixed);
      if (check2.ok && !looksTemplatey(check2.text) && !tooSimilarToRecent(check2.text)) {
        rememberText(check2.text);
        return res.json({ ok: true, text: check2.text, level, language, source: `${first.provider}+soft-corrector` });
      }
      const backup = pick(bankByLevel(level));
      rememberText(backup);
      return res.json({ ok: true, text: backup, level, language, source: "fallback-bank" });
    }

    rememberText(check.text);
    return res.json({ ok: true, text: check.text, level, language, source: first.provider });
  } catch (err) {
    const timedOut = String(err?.message || err) === "DEADLINE_EXCEEDED";
    console.error("agent/generate-text error:", err);
    const { level = "A1", language = "pl" } = req.body || {};
    const backup = pick(bankByLevel(level));
    return res.status(200).json({
      ok: true,
      text: backup,
      level,
      language,
      source: timedOut ? "timeout-fallback" : "error-fallback",
    });
  }
}

app.post("/agent/generate-text", handleGenerateText);
app.post("/generate-text",      handleGenerateText); // alias bez 307

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
/* ===================================================================== */
/* ==============  QUIZ / COMPREHEND – NAUCZYCIEL PL 1–3  =============== */
/* ===================================================================== */

const COMPREHEND_DEBUG = process.env.COMPREHEND_DEBUG === '1';
const COMPREHEND_TIMEOUT_MS = Number(process.env.COMPREHEND_TIMEOUT_MS || 3200);

/* ===== Regexy sygnałów w tekście ===== */
// MIEJSCE: prepozycje miejsca, z wyjątkami i krótkim NP (max 3 wyrazy), stop na spójnikach
// - "do" tylko gdy NIE występuje po nim liczebnik (cyfra lub słowo: jeden… dziesięciu)
// - "w" nie łapie "w celu"
const RE_PLACE = new RegExp(
  String.raw`\b(?:` +
  String.raw`w(?!\s+celu)\b|we\b|na\b|` +
  String.raw`do(?!\s+(?:\d+|jedn\w*|dwu\w*|dwie\w*|trzech|czterech|pięciu|piec\w*|sześciu|siedmiu|ośmiu|dziewięciu|dziesięciu))\b|` +
  String.raw`przy\b|pod\b|u\b|obok\b` +
  String.raw`)` +
  // po przyimku: max 3 tokeny; zatrzymaj przed spójnikami i typowymi czasownikami (np. „wieje”)
  String.raw`\s+(?!\s)(?:` +
    String.raw`(?!i\b|oraz\b|ale\b|a\b|potem\b|jest\b|stoi\b|leży\b|lezy\b|idzie\b|biegnie\b|wieje\b|patrzy\b|pisze\b|czyta\b)[^\s0-9.,;!?]+` +
    String.raw`(?:\s+(?!i\b|oraz\b|ale\b|a\b|potem\b|jest\b|stoi\b|leży\b|lezy\b|idzie\b|biegnie\b|wieje\b|patrzy\b|pisze\b|czyta\b)[^\s0-9.,;!?]+){0,2}` +
  String.raw`)`,
  'iu'
);
const RE_TIME     = /\b(rano|wieczorem|w południe|po południu|wczoraj|dzisiaj|dziś|jutro|po obiedzie|w niedzielę|później)\b/iu;
const RE_PURPOSE  = /\b(żeby|aby|by|bo|ponieważ|dlatego że|w celu|po to)\b/iu;
const RE_PLACE_ANS_START = /^\s*(w|we|na|do|przy|pod|u|obok)\b/iu;

/* ===== utilsy ===== */
function flag(v){ return v === true || v === 1 || v === '1' || String(v).toLowerCase() === 'true'; }
function qz_trim(s="", limit=900){ const t=String(s||"").replace(/\s+/g," ").trim(); return t.length>limit?t.slice(0,limit):t; }
function qz_splitSentences(s=""){ return String(s||"").replace(/\s*[\r\n]+\s*/g," ").split(/(?<=[.!?…])\s+/u).map(t=>t.trim()).filter(Boolean); }
function safeHeaderASCII(v){ return String(v ?? '').replace(/[\r\n]+/g,' ').replace(/[^\x20-\x7E]/g,'').trim().slice(0,160); }
function setComprehendHeaders(res, payload){
  if (!payload) return;
  res.setHeader('X-Comprehend-Path', safeHeaderASCII(payload.source_path || payload.path || 'unknown') || '-');
  const q = safeHeaderASCII(payload.question || ''), a = safeHeaderASCII(payload.answer || ''), s = safeHeaderASCII(payload.sentence || '');
  if (q) res.setHeader('X-Comprehend-Question', q);
  if (a) res.setHeader('X-Comprehend-Answer', a);
  if (s) res.setHeader('X-Comprehend-Sentence', s);
}

/* twardy timeout: zawsze coś zwrócimy */
function hardTimeout(promise, ms) {
  return new Promise((resolve) => {
    let done = false;
    const t = setTimeout(() => { if(!done){ done=true; resolve({ __timeout: true }); } }, ms);
    promise.then(v => { if(!done){ done=true; clearTimeout(t); resolve(v); } })
           .catch(e => { if(!done){ done=true; clearTimeout(t); resolve({ __error: e }); } });
  });
}

/* pomocnicze: przytnij frazę miejsca do 1–3 słów + usuń końcowe czasowniki */
const PLACE_VERBS = /(jest|stoi|leży|lezy|śpi|spi|idzie|biegnie|wieje|patrzy|pisze|czyta|maluje|rysuje|słucha|slucha|gra|je|pije)$/i;
function trimPlace(p='') {
  let s = String(p).trim();

  // utnij wszystko po pierwszym dużym słowie (np. "Na spacerze Kuba" -> "Na spacerze")
  s = s.replace(/\s+[A-ZŁŚŻŹĆŃÓ][\p{L}\p{M}\-']+.*$/u, '');

  let toks = s.split(/\s+/)
    .filter(w => !/^(i|oraz|ale|a|potem)$/i.test(w))
    .slice(0, 3);
  // usuń końcowy czasownik jeśli wylezie
  while (toks.length && PLACE_VERBS.test(toks[toks.length-1])) toks.pop();
  return toks.join(' ');
}

/* mini-heurystyka jako ostatnia deska ratunku (ładniejsze pytania/odpowiedzi) */
function heuristicQA(sentence){
  const text = String(sentence||'').trim();
  const subj = (text.match(/^([A-ZŁŚŻŹĆŃÓ][\p{L}\p{M}\-']+)/iu)||[])[1] || ''; // pierwsze słowo – imię/subiekt
  const placeRaw = (text.match(RE_PLACE)||[])[0];
  const place = placeRaw ? trimPlace(placeRaw) : '';
  const time  = (text.match(RE_TIME)||[])[0];
  const mVerb = (text.match(/\b(śpi|spi|czyta|pisze|rysuje|maluje|je|pije|ogląda|oglad|słucha|slucha|idzie|biegnie|gra|stoi|siedzi|leży|lezy|patrzy)\b/iu)||[])[0];

  if (place && subj && mVerb)  return { question: `Gdzie ${subj} ${mVerb.toLowerCase()}?`, answer: place };
  if (place && subj)           return { question: `Gdzie jest ${subj}?`,            answer: place };
  if (time && subj && mVerb)   return { question: `Kiedy ${subj} ${mVerb.toLowerCase()}?`, answer: time };
  if (time && subj)            return { question: `Kiedy to było?`,                 answer: time };
  if (mVerb && subj)           return { question: `Co robi ${subj}?`,               answer: mVerb.toLowerCase() };
  if (mVerb)                   return { question: 'Co się dzieje w zdaniu?',        answer: mVerb.toLowerCase() };
  return { question: 'O co chodzi w zdaniu?', answer: '' };
}

/* porządkowanie bzdurek typu „czyta czyta”, „pije pije”, itp. */
function dedupeVerbInQuestion(q){
  if (!q) return q;
  let s = String(q).replace(/\s+/g,' ').trim();
  s = s.replace(/\b(czyta|pije|je|gra|idzie|biegnie|stoi|siedzi|leży|lezy|ogląda|oglad|pisze|rysuje|maluje)\s+\1\b/gi, '$1');
  s = s.replace(/\b(ksiazke|książkę|kakao|zup[eęa]|komiks|piłk[ae]|muzyke|muzykę)\s+(czyta|pije|je|gra)\?/gi, '$2?');
  return s;
}

/* czyszczenie pytania i sanity-check formy */
function cleanQuestion(q) {
  if (!q) return q;
  let s = q.trim().replace(/\s+/g,' ').replace(/[„”"']/g, '');

  // usuń nienaturalne "się" po słowie pytającym (Kto/Gdzie/Co)
  s = s.replace(/^(Kto|Gdzie|Co)\s+się\s+/iu, '$1 ');

  // „Kiedy Po …” → „Kiedy to było?”
  s = s.replace(/^Kiedy\s+Po\b.*$/iu, 'Kiedy to było?');

  // „Gdzie … (jest|stoi|leży)?” → „Gdzie to było?”
  s = s.replace(/^Gdzie\s+.+\b(jest|stoi|leży|lezy)\?$/iu, 'Gdzie to było?');

  // „Co robi Po …” (z „Po południu…”) → „Co robi?”
  s = s.replace(/^Co robi\s+Po\b.*$/iu, 'Co robi?');

  // „Co robi Czyta/… mape” → „Co robi?”
  s = s.replace(/^Co robi\s+(Czyta|Pisze|Rysuje|Maluje|Słucha|Slucha|Ogląda|Oglad)\b.*$/iu, 'Co robi?');

  // „Gdzie jest Potem …” (z „Potem…”) → „Gdzie to było?”
  s = s.replace(/^Gdzie\s+jest\s+Potem\b.*$/iu, 'Gdzie to było?');

  // „Kto się [czasownik]?” / „Gdzie się [czasownik]?” → bez „się”
  s = s.replace(/^Kto\s+się\s+([a-ząćęłńóśźż]+)\?/iu, 'Kto $1?');
  s = s.replace(/^Gdzie\s+się\s+([a-ząćęłńóśźż]+)\?/iu, 'Gdzie $1?');

  // pytanie zawiera przecinki/kropki → skróć
  if (/[.,;:].*\?$/u.test(s)) {
    s = /^(?=.*\bGdzie\b)/iu.test(s) ? 'Gdzie to było?' :
        /^(?=.*\bKiedy\b)/iu.test(s) ? 'Kiedy to było?' : 'Co się dzieje?';
  }

  const words = s.split(' ');
  if (words.length > 8) s = words.slice(0, 8).join(' ') + '?';
  if (!/\?\s*$/.test(s)) s = s.replace(/[?]*\s*$/,'') + '?';
  s = s[0].toUpperCase() + s.slice(1);
  return dedupeVerbInQuestion(s);
}

/* parser JSON z modelu (odporny na prawie-JSON) */
function parseQuestionsFromJSON(raw){
  if (!raw) return [];
  let m = String(raw).match(/\{[\s\S]*\}/);
  if (!m) return [];
  let j; try { j = JSON.parse(m[0]); } catch { return []; }
  const arr = Array.isArray(j?.questions) ? j.questions : [];
  return arr.map(x => {
      const q = cleanQuestion(String(x?.question||''));
      const a = String(x?.answer||'').trim().replace(/[„”"']/g,'').split(/\s+/).slice(0,6).join(' ');
      return { question: q, answer: a };
    })
    .filter(x => x.question && x.answer && /\?\s*$/.test(x.question));
}

/* PROMPT — krótki i twardy: zero wymysłów, zero miejsca/czasu jeśli nie ma w tekście */
function buildTeacherPrompt(text, maxQ=3){
  const clamped = Math.max(1, Math.min(5, Number(maxQ)||3));
  return `
Jesteś nauczycielem języka polskiego (klasy 1–3).
Na podstawie TEKSTU przygotuj od 1 do ${clamped} PROSTYCH pytań i krótkich poprawnych odpowiedzi (max 6 słów) — WYŁĄCZNIE z treści.

Zasady:
- Nie wymyślaj nowych imion/miejsc/czasów/celów.
- Jeśli w tekście NIE ma miejsca → nie pytaj „Gdzie…?”.
- Jeśli w tekście NIE ma czasu → nie pytaj „Kiedy…?”.
- „Po co…?” lub „Dlaczego…?” tylko jeśli w TEKŚCIE występują markery celu (żeby, aby, bo, ponieważ, dlatego że, w celu, po to).
- Formy pytań ogranicz do: „Kto…?”, „Co robi…?”, „Gdzie…?”, „Kiedy…?”, ewentualnie „Po co…?” (tylko jeśli są markery celu).
- Pytania krótkie (max 8 słów), bez cytowania całych zdań.
- Odpowiedzi krótkie (1–6 słów), z polskimi znakami, nazwy własne z wielkiej litery.

Zwróć dokładnie JSON:
{"questions":[{"question":"…?","answer":"…"}]}

TEKST:
"""${qz_trim(text, 900)}"""`.trim();
}

/* walidacja i post-processing par QA */
function isValidQA(q, a, text) {
  if (/^\s*Gdzie\b/i.test(q) && !RE_PLACE.test(text)) return false;
  if (/^\s*Kiedy\b/i.test(q) && !RE_TIME.test(text))  return false;
  if (/^\s*(Po co|Dlaczego)\b/i.test(q) && !RE_PURPOSE.test(text)) return false;
  if (/^\s*(Po co|Dlaczego)\b/i.test(q) && RE_PLACE_ANS_START.test(a)) return false;
  if (a.trim().split(/\s+/).length > 6) return false;
  if (/[.,;:]{1}.*\?$/i.test(q)) return false; // echo całych zdań
  return true;
}

function postProcessLLMItems(items, text, want=3) {
  const out = [];
  const placeMatch = (text.match(RE_PLACE)||[])[0];

  for (const it of items || []) {
    let q = cleanQuestion(it.question || '');
    let a = (it.answer || '').trim();

    // Jeżeli „Gdzie …?” a odpowiedź wygląda na czas → przerób pytanie na „Kiedy…?”
    if (/^\s*Gdzie\b/i.test(q) && RE_TIME.test(a)) {
      q = 'Kiedy to było?';
    }

    // Po co/Dlaczego bez celu → zmiana pytania
    if (/^\s*(Po co|Dlaczego)\b/i.test(q) && !RE_PURPOSE.test(text)) {
      if (RE_PLACE.test(text)) q = 'Gdzie to było?';
      else if (RE_TIME.test(text)) q = 'Kiedy to było?';
      else q = 'Co się dzieje?';
    }

    // „Gdzie …?” – dopilnuj odpowiedzi zaczynającej się przyimkiem
    if (/^\s*Gdzie\b/i.test(q)) {
      if (RE_PURPOSE.test(a)) { if (placeMatch) a = trimPlace(placeMatch); else continue; }
      if (!RE_PLACE_ANS_START.test(a)) { if (placeMatch) a = trimPlace(placeMatch); else continue; }
    }

    // Odpowiedź z markerem celu, a pytanie nie „Po co/Dlaczego” → zmień pytanie
    if (RE_PURPOSE.test(a) && !/^\s*(Po co|Dlaczego)\b/i.test(q)) {
      q = 'Po co?';
    }

    if (!isValidQA(q, a, text)) continue;

    // Skracanie odpowiedzi miejsca do 1–3 słów + ucięcie końcowych czasowników
    if (RE_PLACE_ANS_START.test(a)) a = trimPlace(a);

    out.push({ question: q, answer: a, fallback: false, source_path: 'llm+post' });
    if (out.length >= want) break;
  }
  return out;
}

/* prosty fallback regułowy, by zmniejszyć brzydkie echa */
function ruleFallback(text, want=1) {
  const out = [];
  if (/Ala .*ćwiczy .*na boisku/i.test(text) && out.length < want) {
    out.push({ question: 'Gdzie ćwiczy Ala?', answer: 'na boisku', fallback: true, source_path: 'rule' });
  }
  if (/biegnie .*do klasy/i.test(text) && out.length < want) {
    out.push({ question: 'Dokąd biegnie?', answer: 'do klasy', fallback: true, source_path: 'rule' });
  }
  if (/U babci\b.*piecze/i.test(text) && out.length < want) {
    out.push({ question: 'Gdzie piecze?', answer: 'u babci', fallback: true, source_path: 'rule' });
  }
  if (/w ogrodzie\b/i.test(text) && out.length < want) {
    out.push({ question: 'Gdzie to było?', answer: 'w ogrodzie', fallback: true, source_path: 'rule' });
  }
  return out;
}

/* równoległy wyścig: OpenAI i Groq jednocześnie — bierzemy pierwszą sensowną odpowiedź */
async function llmQuestionsRace(text, countHint){
  const sentences = qz_splitSentences(text);
  const maxQ = Math.min(5, Math.max(1, countHint || sentences.length || 1));
  const prompt = buildTeacherPrompt(text, maxQ);

  const oaiPromise = (async () => {
    if (!openai) throw new Error('OPENAI_OFF');
    const r = await withDeadlineRetry(
      () => openai.chat.completions.create({
        model: 'gpt-4o-mini',
        temperature: 0.2,
        max_tokens: 180,
        top_p: 0.95,
        messages: [{ role: 'user', content: prompt }]
      }),
      { deadlineMs: Math.min(COMPREHEND_TIMEOUT_MS-300, 2500), retries: 0 }
    );
    const out = r?.choices?.[0]?.message?.content || '';
    const items = parseQuestionsFromJSON(out);
    if (!items.length) throw new Error('OAI_EMPTY');
    return { items, provider: 'openai' };
  })();

  const groqPromise = (async () => {
    if (!groq) throw new Error('GROQ_OFF');
    const r = await hardTimeout(
      groqChat({
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 180,
        temperature: 0.2,
        top_p: 0.95
      }),
      Math.min(COMPREHEND_TIMEOUT_MS-300, 2500)
    );
    if (r?.__timeout || r?.__error) throw new Error('GROQ_FAIL');
    const out = r?.text || '';
    const items = parseQuestionsFromJSON(out);
    if (!items.length) throw new Error('GROQ_EMPTY');
    return { items, provider: 'groq' };
  })();

  try {
    const res = await hardTimeout(
      Promise.any([oaiPromise, groqPromise]),
      COMPREHEND_TIMEOUT_MS
    );
    if (res?.__timeout) throw new Error('RACE_TIMEOUT');
    return res;
  } catch {
    return { items: [], provider: null };
  }
}

/* multi-heurystyka (ostatni fallback) */
function heuristicMulti(text, want=1){
  const sents = qz_splitSentences(text).slice(0, Math.max(1, want));
  return sents.map(s => {
    const h = heuristicQA(s);
    return { question: h.question, answer: h.answer, fallback: true, sentence: s, source_path: 'heuristic-fallback' };
  });
}

/* ===================== /agent/comprehend-multi ===================== */
app.post('/agent/comprehend-multi', async (req, res) => {
  const dbg = flag(req.query?.debug) || flag(req.body?.debug) || COMPREHEND_DEBUG;
  try {
    const { text = '', count = 3 } = req.body || {};
    const src = String(text || '').trim();
    if (!src) return res.status(400).json({ ok: false, error: 'NO_TEXT' });

    const want = Math.min(5, Math.max(1, Number(count)||3));

    let out = [];
    let provider = null;

    // 1) LLM race
    const race = await llmQuestionsRace(src, want);
    provider = race.provider;

    // 2) Post-process + walidacja
    out = postProcessLLMItems(race.items, src, want);

    // 3) Jeśli brak kompletnego wyniku – dopełnij fallbackami
    if (out.length < want) {
      const rf = ruleFallback(src, want - out.length);
      out = out.concat(rf);
    }
    if (out.length < want) {
      const hf = heuristicMulti(src, want - out.length);
      out = out.concat(hf);
    }

    // 4) Dodaj pola pomocnicze (sentence, provider) i przytnij do want
    const sents = qz_splitSentences(src);
    out = out.slice(0, want).map((x,i)=>({
      question: dedupeVerbInQuestion(x.question),
      answer: String(x.answer||'').trim().split(/\s+/).slice(0,6).join(' '),
      fallback: !!x.fallback,
      sentence: sents[i] || sents[0] || src,
      source_path: x.source_path || 'llm+post',
      llm_provider: provider || (x.fallback ? 'fallback' : null)
    }));

    setComprehendHeaders(res, out[0]);
    if (dbg) console.log('[COMPREHEND-MULTI]', out.map(i=>({path:i.source_path,prov:i.llm_provider,q:i.question,a:i.answer,s:i.sentence})));
    return res.json({ ok: true, count: out.length, items: out });
  } catch (err) {
    console.error('comprehend-multi error:', err);
    return res.status(200).json({ ok: true, count: 0, items: [] });
  }
});

/* ===================== /agent/comprehend ===================== */
app.post('/agent/comprehend', async (req, res) => {
  const dbg = flag(req.query?.debug) || flag(req.body?.debug) || COMPREHEND_DEBUG;
  try {
    const { text = '' } = req.body || {};
    const src = String(text || '').trim();
    if (!src) return res.status(400).json({ ok: false, error: 'NO_TEXT' });

    const bestSent = qz_splitSentences(src)[0] || src;

    // 1) LLM race (1 szt.) + post-process
    const race = await llmQuestionsRace(bestSent, 1);
    const pp = postProcessLLMItems(race.items, bestSent, 1);

    let qa;
    if (pp.length) {
      qa = {
        question: dedupeVerbInQuestion(pp[0].question),
        answer: String(pp[0].answer||'').trim().split(/\s+/).slice(0,6).join(' '),
        fallback: false,
        sentence: bestSent,
        source_path: 'llm+post',
        llm_provider: race.provider
      };
    } else {
      const h = heuristicQA(bestSent);
      qa = { question: h.question, answer: h.answer, fallback: true, sentence: bestSent, source_path: 'heuristic-fallback' };
    }

    setComprehendHeaders(res, qa);
    if (dbg) console.log('[COMPREHEND-ONE]', { path: qa.source_path, provider: qa.llm_provider, q: qa.question, a: qa.answer, sent: qa.sentence });
    return res.json({ ok: true, question: qa.question, answer: qa.answer, fallback: !!qa.fallback, source_path: qa.source_path });
  } catch (err) {
    console.error('comprehend error:', err);
    return res.status(200).json({ ok: true, question: 'Co się dzieje w zdaniu?', answer: '', fallback: true, source_path: 'error-fallback' });
  }
});
/* ===================================================================== */


/* ===================== START ===================== */
async function prewarmOnce() {
  try {
    if (process.env.GROQ_API_KEY) {
      await groqChat({ messages: [{ role: 'user', content: 'ping' }], max_tokens: 8, temperature: 0.0 }).catch(()=>{});
    }
    // ★ prewarm OpenAI (ultrakrótki)
    if (openai) {
      await openaiChat({ messages: [{ role: 'user', content: 'ok' }], max_tokens: 1, temperature: 0.0, top_p: 1.0 }).catch(()=>{});
    }
    if (BASE_URL) {
      await fetch(`${BASE_URL}/health`, { headers: { Connection: 'keep-alive' } }).catch(()=>{});
    }
  } catch { /* noop */ }
}

app.listen(PORT, () => {
  console.log(`🚀 Backend działa na http://localhost:${PORT}`);
  console.log(`🎧 Groq ${groq ? 'podłączony' : 'OFF'} (chat=${GROQ_CHAT_MODEL}, asr=${GROQ_ASR_MODEL})`);
  console.log(`🤖 OpenAI ${openai ? 'podłączony' : 'OFF'}`);
  console.log(`🧠 LLM_PREF=${LLM_PREF}`);
  prewarmOnce();
  if (PREWARM_EVERY_MIN > 0) {
    setInterval(prewarmOnce, PREWARM_EVERY_MIN * 60_000);
    console.log(`🛌 Anti-sleep: ping co ${PREWARM_EVERY_MIN} min${BASE_URL ? ` → ${BASE_URL}/health` : ''}`);
  }
});
