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

/** Możesz włączyć stały tryb debug: COMPREHEND_DEBUG=1 w ENV */
const COMPREHEND_DEBUG = process.env.COMPREHEND_DEBUG === '1';

/* — drobne utils — */
function flag(v){ return v === true || v === 1 || v === '1' || String(v).toLowerCase() === 'true'; }
function qz_trim(s="", limit=1000){ const t=String(s||"").replace(/\s+/g," ").trim(); return t.length>limit?t.slice(0,limit):t; }
function qz_splitSentences(s=""){
  return String(s||"")
    .replace(/\s*[\r\n]+\s*/g," ")
    .split(/(?<=[.!?…])\s+/u)
    .map(t=>t.trim())
    .filter(Boolean);
}
function safeHeaderASCII(v){
  return String(v ?? '')
    .replace(/[\r\n]+/g, ' ')
    .replace(/[^\x20-\x7E]/g, '')   // tylko ASCII, by uniknąć ERR_INVALID_CHAR
    .trim()
    .slice(0, 160);
}
function setComprehendHeaders(res, payload){
  if (!payload) return;
  const q = safeHeaderASCII(payload.question || '');
  const a = safeHeaderASCII(payload.answer   || '');
  const s = safeHeaderASCII(payload.sentence || '');
  const p = safeHeaderASCII(payload.source_path || payload.path || 'unknown');
  res.setHeader('X-Comprehend-Path', p || '-');
  if (q) res.setHeader('X-Comprehend-Question', q);
  if (a) res.setHeader('X-Comprehend-Answer', a);
  if (s) res.setHeader('X-Comprehend-Sentence', s);
}

/* — fallback heurystyczny na 100% bezpieczeństwa — */
function heuristicQA(sentence){
  const text = String(sentence||'').trim();
  // prosty: wyłap kto/verb/miejsce/czas
  const mSubject = text.match(/^([A-ZŁŚŻŹĆŃÓ][\p{L}\p{M}\-']+(?:\s+[A-ZŁŚŻŹĆŃÓ][\p{L}\p{M}\-']+)*)\b/iu);
  const subj = mSubject ? mSubject[1] : '';
  const hasPlace = /\b(w|we|na|do|przy|pod|u|obok)\s+[^.,;!?]+/iu.test(text);
  const hasTime  = /\b(rano|wieczorem|w południe|po południu|wczoraj|dzisiaj|dziś|jutro)\b/iu.test(text);
  const mVerb = text.match(/\b(śpi|spi|czyta|pisze|rysuje|maluje|je|pije|ogląda|slucha|słucha|idzie|gra)\b/iu);
  if (hasPlace && subj)  return { question: `Gdzie ${/je|pije|czyta|ogląda|pisze|rysuje|maluje|śpi|spi|idzie|słucha|slucha|gra/i.test(text) ? subj : 'to się dzieje'}?`, answer: (text.match(/\b(w|we|na|do|przy|pod|u|obok)\s+[^.,;!?]+/iu)||[])[0] || '' };
  if (hasTime && subj)   return { question: `Kiedy ${subj.replace(/\s+$/,'')} ${mVerb ? mVerb[0] : 'to robi'}?`, answer: (text.match(/\b(rano|wieczorem|w południe|po południu|wczoraj|dzisiaj|dziś|jutro)\b/iu)||[])[0] || '' };
  if (mVerb && subj)     return { question: `Co robi ${subj}?`, answer: mVerb[0].toLowerCase() };
  if (mVerb)             return { question: 'Co się dzieje w zdaniu?', answer: mVerb[0].toLowerCase() };
  return { question: 'O co chodzi w zdaniu?', answer: '' };
}

/* — parser JSON z modelu — */
function parseQuestionsFromJSON(raw){
  if (!raw) return [];
  const m = String(raw).match(/\{[\s\S]*\}/);
  if (!m) return [];
  let j;
  try { j = JSON.parse(m[0]); } catch { return []; }
  const arr = Array.isArray(j?.questions) ? j.questions : [];
  // sanity: trim i krótkie odpowiedzi
  return arr
    .map(x => ({
      question: String(x?.question || '').replace(/[„”"']/g,'').trim(),
      answer: String(x?.answer || '').replace(/[„”"']/g,'').trim().split(/\s+/).slice(0,6).join(' ')
    }))
    .filter(x => x.question && x.answer && /\?\s*$/.test(x.question));
}

/* — delikatna dywersyfikacja (gdy model da kilka „Kto…?”) — */
function diversifyQAList(items, text){
  const arr = Array.isArray(items) ? items.slice() : [];
  if (arr.length <= 1) return arr;

  const t = String(text||'');
  const hasPlace = /\b(w|we|na|do|przy|pod|u|obok)\s+[^.,;!?]+/iu.test(t);
  const hasTime  = /\b(rano|wieczorem|w południe|po południu|wczoraj|dzisiaj|dziś|jutro)\b/iu.test(t);

  const idxKto = arr
    .map((q,i)=> (/^\s*Kto\b/i.test(String(q?.question||'')) ? i : -1))
    .filter(i => i >= 0);

  if (idxKto.length <= 1) return arr; // już ok

  // Weź drugie „Kto…?” i zamień, ale tylko jeśli treść na to pozwala
  const i = idxKto.slice(1)[0];
  const q = String(arr[i].question||'').trim();
  const a = String(arr[i].answer||'').trim();

  const subj = a || (q.replace(/^Kto\s+/i,'').replace(/\?$/,'').trim());
  if (!subj) return arr;

  // Preferencje: jeśli w tekście jest miejsce/czas, dodaj Gdzie/Kiedy — inaczej „Co robi …?”
  const hasGdzie = arr.some(x => /^\s*Gdzie\b/i.test(String(x?.question||'')));
  const hasKiedy = arr.some(x => /^\s*Kiedy\b/i.test(String(x?.question||'')));

  if (hasPlace && !hasGdzie) {
    arr[i] = { question: `Gdzie ${subj}?`, answer: '', ...arr[i] };
  } else if (hasTime && !hasKiedy) {
    arr[i] = { question: `Kiedy ${subj}?`, answer: '', ...arr[i] };
  } else {
    arr[i] = { question: `Co robi ${subj}?`, answer: '', ...arr[i] };
  }
  return arr;
}

/* — główny PROMPT NAUCZYCIELA PL 1–3 — */
function buildTeacherPrompt(text, maxQ=3){
  const clamped = Math.max(1, Math.min(5, Number(maxQ)||3));
  return `
Jesteś nauczycielem języka polskiego w klasach 1–3 szkoły podstawowej.
Twoim zadaniem jest sprawdzenie, czy dziecko zrozumiało przeczytany tekst.

Na podstawie TEKSTU poniżej przygotuj od 1 do ${clamped} prostych pytań (dla dzieci 1–3), oraz krótkie poprawne odpowiedzi (max 6 słów), oparte WYŁĄCZNIE na treści tekstu.
Nie wymyślaj nowych imion ani faktów.
Pytaj tylko o to, co naprawdę jest w tekście. W szczególności:
- „Kto…?” — gdy w tekście jest wyraźny bohater (imię/postać).
- „Co robi…?” — gdy jest jasno opisana czynność.
- „Gdzie…?” — tylko jeśli miejsce jest podane (np. „w pokoju”, „na boisku”).
- „Kiedy…?” — tylko jeśli czas jest podany (np. „rano”, „po obiedzie”).
- „Po co…?” (cel) — tylko jeśli cel jest wyraźnie podany (np. „żeby…”).
Jeśli tekst to 1 zdanie, daj 1 pytanie. Jeśli dłuższy — 2–${clamped} pytań.
Urozmaicaj typy pytań: nie zadawaj kilku „Kto…?” pod rząd, jeśli da się zapytać o czynność/miejsce/czas wynikające z treści.
Używaj tylko polskiego.

Zwróć DOKŁADNIE czysty JSON:
{
  "questions": [
    { "question": "…?", "answer": "…" }
  ]
}

TEKST:
"""${qz_trim(text, 1000)}"""
`.trim();
}

/* — LLM-first: OpenAI via chatPref (z Groq w failoverze przez chatPref/groqChat) — */
async function llmQuestions(text, countHint){
  const sentences = qz_splitSentences(text);
  const maxQ = Math.min(5, Math.max(1, countHint || sentences.length || 1));
  const prompt = buildTeacherPrompt(text, maxQ);

  // używamy istniejącego chatPref (OpenAI-first + retry, fallback na Groq)
  const { text: out, provider } = await chatPref({
    prompt,
    temperature: 0.3,
    top_p: 0.95,
    max_tokens: 300,
    deadlineMs: 4000
  });

  let items = parseQuestionsFromJSON(out);
  items = diversifyQAList(items, text);   // ★ delikatna dywersyfikacja

  return { items, provider };
}

/* — fallback multi: generuj Q/A z pierwszych Z zdań heurystycznie — */
function heuristicMulti(text, want = 1){
  const sents = qz_splitSentences(text).slice(0, Math.max(1, want));
  const arr = [];
  for (const s of sents){
    const qa = heuristicQA(s);
    arr.push({ question: qa.question, answer: qa.answer, fallback: true, sentence: s, source_path: 'heuristic-fallback' });
  }
  return arr;
}

/* ===================== /agent/comprehend-multi ===================== */
app.post('/agent/comprehend-multi', async (req, res) => {
  try {
    const dbg = flag(req.query?.debug) || flag(req.body?.debug) || COMPREHEND_DEBUG;
    const { text = '', age, count = 3 } = req.body || {};
    const src = String(text || '').trim();
    if (!src) return res.status(400).json({ ok: false, error: 'NO_TEXT' });

    let out = [];
    let path = 'llm';
    let provider = null;

    try {
      const r = await llmQuestions(src, count);
      provider = r.provider || 'llm';
      out = (r.items || []).slice(0, Math.min(5, Math.max(1, count)));
    } catch (e) {
      path = 'llm-error';
      if (dbg) console.log('[COMPREHEND-MULTI][LLM_ERROR]', String(e?.message || e));
      out = []; // przejdziemy do heurystyki poniżej
    }

    if (!out.length) {
      path = 'heuristic-fallback';
      out = heuristicMulti(src, Math.min(3, Math.max(1, count)));
    } else {
      // dorzuć pola dla spójności z heurystyką
      out = out.map((x, i) => ({
        question: x.question,
        answer: x.answer,
        fallback: false,
        sentence: qz_splitSentences(src)[i] || src,
        source_path: 'llm',
        llm_provider: provider
      }));
    }

    // nagłówki (ASCII safe)
    setComprehendHeaders(res, out[0]);

    if (dbg) {
      console.log('[COMPREHEND-MULTI]', out.map(i => ({
        path: i.source_path, provider: i.llm_provider, q: i.question, a: i.answer, sent: i.sentence
      })));
    }

    return res.json({ ok: true, count: out.length, items: out });
  } catch (err) {
    console.error('comprehend-multi error:', err);
    return res.status(200).json({ ok: true, count: 0, items: [] });
  }
});

/* ===================== /agent/comprehend ===================== */
app.post('/agent/comprehend', async (req, res) => {
  try {
    const dbg = flag(req.query?.debug) || flag(req.body?.debug) || COMPREHEND_DEBUG;
    const { text = '', age } = req.body || {};
    const src = String(text || '').trim();
    if (!src) return res.status(400).json({ ok: false, error: 'NO_TEXT' });

    let qa = null;
    let path = 'llm';
    let provider = null;

    try {
      const r = await llmQuestions(src, 1);
      provider = r.provider || 'llm';
      const first = (r.items || [])[0];
      if (first) {
        qa = {
          question: first.question,
          answer: first.answer,
          fallback: false,
          sentence: qz_splitSentences(src)[0] || src,
          source_path: 'llm',
          llm_provider: provider
        };
      }
    } catch (e) {
      path = 'llm-error';
      if (dbg) console.log('[COMPREHEND-ONE][LLM_ERROR]', String(e?.message || e));
    }

    if (!qa) {
      const h = heuristicQA(qz_splitSentences(src)[0] || src);
      qa = { question: h.question, answer: h.answer, fallback: true, sentence: qz_splitSentences(src)[0] || src, source_path: 'heuristic-fallback' };
    }

    setComprehendHeaders(res, qa);

    if (dbg) {
      console.log('[COMPREHEND-ONE]', { path: qa.source_path, provider: qa.llm_provider, q: qa.question, a: qa.answer, sent: qa.sentence });
    }

    return res.json({
      ok: true,
      question: qa.question,
      answer: qa.answer,
      fallback: !!qa.fallback,
      source_path: qa.source_path
    });
  } catch (err) {
    console.error('comprehend error:', err);
    return res.status(200).json({
      ok: true,
      question: 'Co się dzieje w zdaniu?',
      answer: '',
      fallback: true,
      source_path: 'error-fallback'
    });
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
