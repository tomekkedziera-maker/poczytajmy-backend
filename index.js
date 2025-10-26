// index.js
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
import { COMMON_VERBS } from "./verbs.js";

import OpenAI from 'openai';
import Groq from 'groq-sdk';

import sharp from 'sharp';
import Tesseract from 'tesseract.js';

/* ===== Opcjonalny keep-alive dla fetch (undici, jeśli dostępne) ===== */
try {
  import('undici').then(({ Agent: UndiciAgent, setGlobalDispatcher }) => {
    try {
      setGlobalDispatcher(
        new UndiciAgent({
          keepAliveTimeout: 10_000,
          keepAliveMaxTimeout: 10_000,
          connections: 128,
        })
      );
      console.log('🌐 undici keep-alive enabled');
    } catch (e) {
      console.warn('🌐 undici available but failed to configure:', String(e));
    }
  }).catch(() => {
    console.log('🌐 undici not installed; using native fetch');
  });
} catch {
  console.log('🌐 undici dynamic import failed; using native fetch');
}

// --- END NEW ---

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

const GROQ_CHAT_MODEL = process.env.GROQ_CHAT_MODEL || 'llama-3.1-8b-instant';
const GROQ_ASR_MODEL  = process.env.GROQ_ASR_MODEL  || 'whisper-large-v3';

const LLM_PREF = process.env.LLM_PREF || 'openai-first';


const GREETING_TIMEOUT_MS      = Number(process.env.GREETING_TIMEOUT_MS || 2800);
const MOTIVATE_TIMEOUT_MS      = Number(process.env.MOTIVATE_TIMEOUT_MS || 10000);
const GENERATE_TEXT_TIMEOUT_MS = Number(process.env.GENERATE_TEXT_TIMEOUT_MS || 10000);

/* ===== Text Pool (prefetch) ===== */
const POOL_LEVELS = (process.env.POOL_LEVELS || "A1,A2,B1").split(",").map(s=>s.trim().toUpperCase());
const POOL_TARGET_SIZE = Number(process.env.POOL_TARGET_SIZE || 24);
const POOL_REFILL_BATCH = Number(process.env.POOL_REFILL_BATCH || 8);
const POOL_REFILL_INTERVAL_MS = Number(process.env.POOL_REFILL_INTERVAL_MS || 60_000);

const keepAliveAgent = new http.Agent({ keepAlive: true, timeout: 10_000 });
const now = () => (global.performance?.now?.() ?? Date.now());
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function withDeadline(promise, ms = DEADLINE_MS) {
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error('DEADLINE_EXCEEDED')), ms);
    promise.then(v => { clearTimeout(to); resolve(v); }, e => { clearTimeout(to); reject(e); });
  });
}

// Retry z backoffem i wydłużaniem deadline’u
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
        ms = Math.round(ms * 1.6);
        continue;
      }
      throw e;
    }
  }
  throw lastErr;
}

/* ===== OpenAI local RPM guard (soft) ===== */
const OAI_RPM_LIMIT = Number(process.env.OPENAI_RPM_LIMIT || 3);
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

/* ===================== LLM helpers (OpenAI + Groq failover) ===================== */
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
      model: GROQ_CHAT_MODEL,
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

function trimUserContent(s = "", limit = 800) {
  const t = String(s ?? "").replace(/\s+/g, " ").trim();
  return t.length > limit ? t.slice(0, limit) : t;
}

// --- RESILIENT chatPref (phased, anti-429) ---
function isSoftFail(err) {
  const m = String(err?.message || err);
  return m.includes('DEADLINE_EXCEEDED') || m.includes('429') ||
         err?.code === 'OPENAI_THROTTLED';
}

async function callWithDeadline(makePromiseFn, ms, retries = 0, backoffMs = 300) {
  let lastErr = null, cur = ms;
  for (let i = 0; i <= retries; i++) {
    try {
      return await withDeadline(makePromiseFn(), cur);
    } catch (e) {
      lastErr = e;
      if (isSoftFail(e) && i < retries) {
        await sleep(backoffMs * (i + 1));
        cur = Math.round(cur * 1.6);
        continue;
      }
      throw e;
    }
  }
  throw lastErr;
}

async function chatPref({
  prompt,
  max_tokens = 150,
  temperature = 0.3,
  top_p = 0.95,
  deadlineMs = DEADLINE_MS,
  prefer = (process.env.LLM_PREF || 'openai-first')  // 'openai-first' | 'groq-first' | 'race'
}) {
  const messages = [{ role: 'user', content: trimUserContent(prompt) }];

  const tryOpenAI = async (ms, retries=0) =>
    openai ? await callWithDeadline(
      () => openaiChat({ messages, max_tokens, temperature, top_p }),
      ms, retries
    ) : Promise.reject(new Error('OPENAI_OFF'));

  const tryGroq = async (ms, retries=0) =>
    groq ? await callWithDeadline(
      () => groqChat({ messages, max_tokens, temperature, top_p }),
      ms, retries
    ) : Promise.reject(new Error('GROQ_OFF'));

  // 1) try preferred first (short-ish)
  const short = Math.max(1200, Math.min(deadlineMs, 2500));
  const long  = Math.max(2000, Math.min(deadlineMs * 1.8, 6000));

  const order = (() => {
    if (prefer === 'groq-first') return ['groq', 'openai'];
    if (prefer === 'race')       return ['race'];
    return ['openai', 'groq']; // default openai-first
  })();

  try {
    if (order[0] === 'race' && openai && groq) {
      // conservative race: krótszy timeout, kto pierwszy — ten lepszy
      const p1 = tryOpenAI(short, 0).catch(e => { throw e; });
      const p2 = tryGroq(short, 0).catch(e => { throw e; });
      return await Promise.any([p1, p2]);
    }

    for (const who of order) {
      try {
        if (who === 'openai') return await tryOpenAI(short, 0);
        if (who === 'groq')   return await tryGroq(short, 0);
      } catch (e) {
        if (!isSoftFail(e)) throw e; // błąd twardy — nie ma co odraczać
        // miękki błąd — spróbujemy drugiego w pętli
      }
    }
  } catch (e) {
    if (!isSoftFail(e)) throw e;
  }

  // 2) last-chance: dwie próby łącznie (po jednej na provider) z dłuższym deadline’em
  const attempts = [];
  if (prefer !== 'groq-first') {
    if (openai) attempts.push(() => tryOpenAI(long, 1));
    if (groq)   attempts.push(() => tryGroq(long, 1));
  } else {
    if (groq)   attempts.push(() => tryGroq(long, 1));
    if (openai) attempts.push(() => tryOpenAI(long, 1));
  }

  let lastErr = null;
  for (const make of attempts) {
    try { return await make(); } catch (e) { lastErr = e; }
  }

  const err = new Error('GEN_FALLBACK');
  err.cause = lastErr;
  err.code = 'GEN_FALLBACK';
  throw err;
}
// --- END RESILIENT chatPref ---


async function raceLLM({ prompt, max_tokens = 150, temperature = 0.3 }) {
  const { text } = await chatPref({ prompt, max_tokens, temperature, top_p: 0.95 });
  return (text || '').trim();
}

/* ===================== ASR ===================== */
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
          model: GROQ_ASR_MODEL,
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

const TEMPLATEY_STARTS = [
  'Dziś','Dzisiaj','Po południu','W ogrodzie','Choć','Chociaż','Na koniec',
  'Potem','Następnie','Po kolacji','Po obiedzie','W bibliotece','W domu'
];
function looksTemplatey(s='') {
  const t = String(s).trim();
  if (!t) return true;
  if (/\b[aA]\s+potem\b/.test(t)) return true;
  if (TEMPLATEY_STARTS.some(p => t.startsWith(p + ' '))) return true;
  const conj = (t.match(/\b(a|oraz)\b/gi) || []).length;
  if (conj >= 2) return true;
  return false;
}
function softenPolish(s='') {
  let out = String(s).trim();
  out = out.replace(/\s*,?\s*a potem\s*/gi, ' i ');
  out = out.replace(/\s*,\s*oraz\s*/gi, ' i ');
  out = out.replace(/\s*,\s*a\s*/gi, ' i ');
  if (!/[.!?…]$/.test(out)) out += '.';
  out = out.replace(/^Z /, 'Ze ');
  return out.replace(/\s+/g,' ').trim();
}
const recentTexts = [];
function rememberText(t) { recentTexts.unshift(String(t)); if (recentTexts.length > 20) recentTexts.pop(); }
function tooSimilarToRecent(t) {
  const n = normalize(t);
  return recentTexts.some(prev => jaccard(n, normalize(prev)) > 0.6);
}

async function generateGreetingV2({ name, age, character, theme }) {
  const prompt = buildGreetingPrompt({ age: Number(age), character, theme, n: 12 });

  // więcej tokenów, by uniknąć ucięć typu „Oto 12…”
  const { text: raw, provider } = await chatPref({
    prompt,
    temperature: NAT_TEMPERATURE,
    max_tokens: 120,
    top_p: NAT_TOP_P,
    deadlineMs: GREETING_TIMEOUT_MS,
  });

  // oczyść nagłówki typu „Oto 12…”, „Przykłady:”, puste linie
  const cleanedRaw = String(raw || '')
    .replace(/^[^\n]{1,120}:\s*$/gmi, '')                 // linie zakończone dwukropkiem
    .replace(/^(?:\s*[-*]\s*)?Oto\b[^\n]*$/gmi, '')       // „Oto …”
    .replace(/^(?:\s*[-*]\s*)?Przykłady\b[^\n]*$/gmi, '') // „Przykłady …”
    .replace(/^Oto\s+\d+\s+[^\n]*$/gmi, '')               // „Oto 12 …”
    .replace(/^\s*$/gm, '')                               // puste linie
    .trim();

  // zbuduj listę kandydatów
  let cands = parseList(cleanedRaw);
  if (!cands.length && cleanedRaw) {
    cands = cleanedRaw.split(/[.\n]/).map(s => s.trim()).filter(Boolean);
  }
  if (!cands.length) throw new Error('EMPTY_GENERATION');

  // wybór najbardziej „nowego” względem historii
  const profileKey = `${(name || '').toLowerCase()}|${Number(age) || 'X'}`;
  const history = recentGreetings.get(profileKey) || [];

  const picked  = chooseMostNovel(cands, history);
  const cleaned = sanitizeNoName(name, picked);
  const finalText = softenPolish(cleaned || picked);

  recentGreetings.set(profileKey, [finalText, ...history].slice(0, 20));
  return { text: finalText, provider: provider || 'llm' };
}

// --- endpoints ---

app.post('/agent/generate-greeting', async (req, res) => {
  try {
    const { name = '', age, character = 'Twój przyjaciel' } = req.body || {};
    const theme = HERO_THEMES[character] || '';
    const { text, provider } = await generateGreetingV2({ name, age, character, theme });
    res.json({ ok: true, text, source: provider });
  } catch (err) {
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    console.error('agent/generate-greeting error:', err);
    const fallback = 'Zajrzymy dziś do książki i wyszukamy nowe słowa. 📖';
    res.status(200).json({
      ok: true,
      text: fallback,
      source: timedOut ? 'timeout-fallback' : 'error-fallback',
    });
  }
});

app.post('/generate-greeting', async (req, res) => {
  try {
    const { name = '', age, character = 'Twój przyjaciel' } = req.body || {};
    const theme = HERO_THEMES[character] || '';
    const { text, provider } = await generateGreetingV2({ name, age, character, theme });
    res.json({ ok: true, text, source: provider });
  } catch (err) {
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    console.error('generate-greeting error:', err);
    const fallback = 'Zajrzymy dziś do książki i wyszukamy nowe słowa. 📖';
    res.status(200).json({
      ok: true,
      text: fallback,
      source: timedOut ? 'timeout-fallback' : 'error-fallback',
    });
  }
});

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

function localMotivationFallback(age, accuracy) {
  const s = Math.max(0, Math.min(100, Math.round(accuracy || 0)));
  if (s >= 95) return 'Czytasz świetnie! Spróbuj teraz nieco trudniejszego słowa. ✨';
  if (s >= 80) return 'Super płynność — jeszcze dokładniej końcówki i będzie idealnie.';
  if (s >= 60) return 'Dobra robota! Czytaj spokojniej i sylabizuj trudniejsze słowa.';
  return 'Fajnie próbujesz — przeczytaj zdanie jeszcze raz powoli, dasz radę. 💪';
}

async function generateMotivation({ age, accuracy, text, characterName, lang = 'pl' }) {
  const prompt = buildMotivationPrompt({ age, accuracy, text, characterName, lang });
  const makeCall = () => chatPref({
    prompt,
    temperature: NAT_TEMPERATURE,
    max_tokens: 100,
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
    const fb = localMotivationFallback(age, accuracy);
    return { text: fb, source: 'local-fallback' };
  }

  let out = raw.replace(/^["'„”]+|["'„”]+$/g, '').trim();
  out = tightenMotivation(out, 160);
  out = softenPolish(out);

  if (looksTemplatey(out) || tooSimilarToRecent(out)) {
    try {
      const r2 = await withDeadlineRetry(makeCall, { deadlineMs: Math.min(MOTIVATE_TIMEOUT_MS || 9000, 9000), retries: 0 });
      const alt = softenPolish(tightenMotivation(String(r2.text || '').trim(), 160));
      if (!looksTemplatey(alt)) out = alt;
    } catch { /* ignore */ }
  }

  if (!out) {
    const fb = localMotivationFallback(age, accuracy);
    return { text: fb, source: 'local-fallback-empty' };
  }

  rememberText(out);
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

/* ===================== GENERATOR ZDAŃ ===================== */
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

    let sentence = softenPolish(cleanSentence(first.text || ""));

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
app.post("/generate-text",      handleGenerateText);

/* ===================== TEXT POOL (prefetch cache) ===================== */
const PREFETCH_PROVIDER = (process.env.PREFETCH_PROVIDER || 'groq').toLowerCase();

const textPool = new Map();
for (const lv of POOL_LEVELS) textPool.set(lv, []);

async function chatPrefBg({
  prompt,
  max_tokens = 150,
  temperature = 0.3,
  top_p = 0.95,
  deadlineMs = 9000
}) {
  if (PREFETCH_PROVIDER === 'groq' && groq) {
    try {
      return await withDeadlineRetry(
        () => groqChat({
          messages: [{ role: 'user', content: trimUserContent(prompt) }],
          max_tokens,
          temperature,
          top_p
        }),
        { deadlineMs: Math.max(4000, deadlineMs), retries: 1, backoffMs: 400 }
      );
    } catch { /* fallback */ }
  }
  return await chatPref({ prompt, max_tokens, temperature, top_p, deadlineMs });
}

function _tp_onlyOneSentence(s) {
  const parts = String(s).split(/(?<=[.!?…])\s+/).filter(Boolean);
  return (parts[0] || s).trim();
}
function _tp_cleanSentence(s) {
  let out = String(s).replace(/[„”"“”'()«»]/g, "").replace(/\s+/g, " ").trim();
  out = _tp_onlyOneSentence(out);
  if (!/[.!?…]$/.test(out)) out += ".";
  return out;
}
function _tp_countWords(s) {
  return (String(s).trim().match(/\b[\p{L}\p{M}0-9'-]+\b/gu) || []).length;
}
function _tp_hasPolishDiacritics(s) { return /[ąćęłńóśźż]/i.test(String(s)); }
function _tp_softenPolish(s="") {
  let out = String(s).trim();
  out = out.replace(/\s*,?\s*a potem\s*/gi, " i ");
  out = out.replace(/\s*,\s*oraz\s*/gi, " i ");
  out = out.replace(/\s*,\s*a\s*/gi, " i ");
  if (!/[.!?…]$/.test(out)) out += ".";
  out = out.replace(/^Z /, "Ze ");
  return out.replace(/\s+/g," ").trim();
}
const _TP_TEMPLATEY_STARTS = ["Dziś","Dzisiaj","Po południu","W ogrodzie","Choć","Chociaż","Na koniec","Potem","Następnie","Po kolacji","Po obiedzie","W bibliotece","W domu"];
function _tp_looksTemplatey(s="") {
  const t = String(s).trim();
  if (!t) return true;
  if (/\b[aA]\s+potem\b/.test(t)) return true;
  if (_TP_TEMPLATEY_STARTS.some(p => t.startsWith(p + " "))) return true;
  const conj = (t.match(/\b(a|oraz)\b/gi) || []).length;
  return conj >= 2;
}
function _tp_validateKidsSentencePL(s, { minWords=8, maxWords=16 } = {}) {
  const txt = _tp_cleanSentence(_tp_onlyOneSentence(s));
  const words = _tp_countWords(txt);
  if (words < minWords || words > maxWords) return { ok:false, text:txt };
  if (!_tp_hasPolishDiacritics(txt)) return { ok:false, text:txt };
  const tokens = (txt.match(/\b[\p{L}\p{M}0-9'-]+\b/gu) || []);
  const long = tokens.filter(w => w.replace(/[^a-ząćęłńóśźż-]/gi,"").length > 12).length;
  const ratio = tokens.length ? long / tokens.length : 0;
  if (tokens.length > 24 || ratio > 0.4) return { ok:false, text:txt };
  return { ok:true, text:txt };
}
const _TP_BANK_A1 = BANK_A1;
const _TP_BANK_A2 = BANK_A2;
const _TP_BANK_B1 = BANK_B1;
function _tp_bankByLevel(level="A1") {
  const L = String(level).toUpperCase();
  if (L === "B1") return _TP_BANK_B1;
  if (L === "A2") return _TP_BANK_A2;
  return _TP_BANK_A1;
}
const _tp_recentTexts = [];
function _tp_rememberText(t) { _tp_recentTexts.unshift(String(t)); if (_tp_recentTexts.length > 20) _tp_recentTexts.pop(); }
function _tp_norm(x){ return String(x||'').toLowerCase().replace(/[^\p{L}\p{M}0-9\s]+/gu,' ').replace(/\s+/g,' ').trim(); }
function _tp_jacc(a,b){
  const A=new Set(_tp_norm(a).split(' ').filter(Boolean));
  const B=new Set(_tp_norm(b).split(' ').filter(Boolean));
  if (!A.size && !B.size) return 1;
  let inter=0; for (const w of A) if(B.has(w)) inter++;
  return inter/(A.size+B.size-inter);
}
function _tp_tooSimilarToRecent(t){ const n=_tp_norm(t); return _tp_recentTexts.some(prev => _tp_jacc(n,_tp_norm(prev)) > 0.6); }

async function genOneSentence(level = "A1") {
  const prompt =
`Napisz jedno naturalne i lekkie zdanie po polsku do głośnego czytania przez dziecko (poziom ${String(level).toUpperCase()}).
Wymagania:
- Jedno zdanie (8–16 słów), brzmienie swobodne (jak rozmowa z dzieckiem), bez „szkolnej” składni.
- Słownictwo codzienne, zero żargonu i neologizmów, bez nawiasów i cudzysłowów.
- Unikaj sztywnych wzorców typu: „… a potem …”, „Dziś/Dzisiaj …”, długich wyliczeń i dwóch „a/oraz” w jednym zdaniu.
- Używaj polskich znaków.
Podaj tylko gotowe zdanie.`;

  const r = await chatPrefBg({
    prompt,
    temperature: 0.9,
    max_tokens: 70,
    top_p: 0.9,
    deadlineMs: Math.min(GENERATE_TEXT_TIMEOUT_MS, 7000)
  }).catch(() => null);

  let s = r?.text
    ? _tp_softenPolish(_tp_cleanSentence(r.text))
    : _tp_bankByLevel(level)[Math.floor(Math.random()*_tp_bankByLevel(level).length)];

  let v = _tp_validateKidsSentencePL(s);
  if (!v.ok || _tp_looksTemplatey(v.text) || _tp_tooSimilarToRecent(v.text)) {
    const r2 = await chatPrefBg({
      prompt,
      temperature: 0.9,
      max_tokens: 70,
      top_p: 0.9,
      deadlineMs: Math.min(GENERATE_TEXT_TIMEOUT_MS, 7000)
    }).catch(() => null);
    s = r2?.text
      ? _tp_softenPolish(_tp_cleanSentence(r2.text))
      : _tp_bankByLevel(level)[Math.floor(Math.random()*_tp_bankByLevel(level).length)];

    v = _tp_validateKidsSentencePL(s);
    if (!v.ok || _tp_looksTemplatey(v.text) || _tp_tooSimilarToRecent(v.text)) {
      const bank = _tp_bankByLevel(level);
      s = bank[Math.floor(Math.random()*bank.length)];
    } else {
      s = v.text;
    }
  } else {
    s = v.text;
  }

  _tp_rememberText(s);
  return s;
}

let _refillLock = false;
async function refillPoolOnce(levels = POOL_LEVELS) {
  if (_refillLock) return;
  _refillLock = true;
  try {
    for (const lv of levels) {
      const buf = textPool.get(lv) || [];
      const need = Math.max(0, POOL_TARGET_SIZE - buf.length);
      if (need <= 0) continue;

      const want = Math.max(POOL_REFILL_BATCH, Math.ceil(need / 2));
      const batch = [];
      for (let i = 0; i < want; i++) {
        try { batch.push(await genOneSentence(lv)); } catch {}
        await sleep(150);
      }
      textPool.set(lv, buf.concat(batch).slice(-POOL_TARGET_SIZE));
    }
  } finally {
    _refillLock = false;
  }
}

function popFromPool(level = "A1", n = 1) {
  const lv = String(level).toUpperCase();
  if (!textPool.has(lv)) textPool.set(lv, []);
  const buf = textPool.get(lv);
  const out = [];
  for (let i = 0; i < n; i++) {
    const s = buf.shift();
    if (s) out.push(s);
  }
  textPool.set(lv, buf);
  setTimeout(() => refillPoolOnce([lv]).catch(()=>{}), 0);
  return out;
}

app.get('/pool/next', async (req, res) => {
  try {
    const level = (req.query?.level || 'A1').toString().toUpperCase();
    const n = Math.min(20, Math.max(1, Number(req.query?.n) || 1));
    if (!textPool.has(level)) textPool.set(level, []);
    let out = popFromPool(level, n);
    if (!out.length) {
      const batch = [];
      for (let i = 0; i < n; i++) batch.push(await genOneSentence(level));
      out = batch;
    }
    return res.json({ ok: true, level, count: out.length, items: out });
  } catch (e) {
    console.error('pool/next error:', e);
    return res.status(200).json({ ok: true, level: String(req.query?.level || 'A1').toUpperCase(), count: 0, items: [] });
  }
});

app.post('/pool/refill', async (req, res) => {
  try {
    const levels = Array.isArray(req.body?.levels) ? req.body.levels : POOL_LEVELS;
    await refillPoolOnce(levels.map(l=>String(l).toUpperCase()));
    return res.json({ ok: true, levels });
  } catch (e) {
    console.error('pool/refill error:', e);
    return res.status(500).json({ ok: false, error: 'REFILL_FAILED' });
  }
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


// ===================== QUIZ: rule-teacher-strict-v7.7-verbs (tune-2) =====================

export function generateQuestionAndAnswerTeacher(textRaw) {
  if (!textRaw || typeof textRaw !== "string") {
    return { ok:false, qtype:"Co się dzieje?", question:"Co się dzieje?", answer:"", source_path:"error-empty" };
  }

  const text = String(textRaw).normalize("NFC").trim();
  const t = text.replace(/[!?]/g, " ").replace(/\s+/g, " ").trim();
  const tLower = t.toLowerCase();

  // ===== utils =====
  const normalizeSpaces = (s="") => String(s).replace(/\s+/g," ").trim();
  const stripPunct = (s="") => String(s).trim().replace(/[.,!?;:]+$/,"");
  const deaccent = (s="") => String(s)
    .normalize("NFD").replace(/[\u0300-\u036f]/g,"")
    .replace(/[łŁ]/g,"l").replace(/[ąĄ]/g,"a").replace(/[ćĆ]/g,"c")
    .replace(/[ęĘ]/g,"e").replace(/[ńŃ]/g,"n").replace(/[óÓ]/g,"o")
    .replace(/[śŚ]/g,"s").replace(/[źŹżŻ]/g,"z");

  const stripOnConjOrComma = (s="") => normalizeSpaces(String(s).split(/(?:,|\s+(?:i|oraz|a)\s)/i)[0] || "");
  const removeDiscourseMarkers = (s="") => normalizeSpaces(String(s).replace(/^\s*(a|i|oraz|potem|więc|następnie|a\s+potem)\s+/i, ""));
  const stripDanglingPrzez = (s="") => String(s).replace(/\s+przez\b.*$/i,"").trim();

  // ===== cięcie przy kolejnym czasowniku =====
  const VERB_CUT_RE = new RegExp(`\\s+(${COMMON_VERBS.join("|")})\\b.*$`, "i");
  // Fallback bez słownika – typowe formy fleksyjne PL (past/present)
  const VERB_CUT_FALLBACK = /\s+\b(usiad\w*|usied\w*|siedz\w*|czyta\w*|zjad\w*|zjed\w*|poszed\w*|poszl\w*|pojech\w*|wr[óo]c\w*|przesz\w*|czeka\w*|wesz\w*|wsiad\w*|wsied\w*)\b.*$/i;
  function cutAtVerb(s="") {
    const before = s;
    s = String(s).replace(VERB_CUT_RE, "");
    if (s === before) s = String(s).replace(VERB_CUT_FALLBACK, "");
    return normalizeSpaces(s);
  }

  // ===== cues ruchu, dopasowanie pytania Dokąd? =====
  const MOTION_CUE_RE = /\b(poszed\w*|poszl\w*|pojech\w*|wr[óo]c\w*|wszed\w*|weszl\w*|wyszed\w*|przesz\w*|pobiegl\w*|wsiad\w*|wsied\w*)\b/i;
  const dokadQuestionByCue = (low="") => {
    if (/\b(wsied\w*|wsiad\w*)\b/.test(low)) return "Dokąd oni wsiedli?";
    if (/\bwr[óo]c\w*/.test(low)) return "Dokąd oni wrócili?";
    return "Dokąd oni poszli?";
  };

  // ===== czas =====
  const TIME_RE = /\b(o\s*\d{1,2}[:.]\d{2}|przed\s*\d{1,2}[:.]\d{2}|rano|wieczorem|w\s+południe|po\s+południu|w\s+sobotni\s+poranek|w\s+niedzielny\s+poranek|w\s+poniedziałek|w\s+wtorek|w\s+środę|w\s+czwartek|w\s+piątek|dzisiaj|dziś|wczoraj|jutro)\b/i;
  const TIME_TAIL = /\s+(rano|wieczorem|w\s+południe|po\s+południu|dzisiaj|dziś|jutro|wczoraj|o\s+\d{1,2}[:.]\d{2}|przed\s+\d{1,2}[:.]\d{2}|w\s+sobotni\s+poranek|w\s+niedzielny\s+poranek)\b/i;

  // ===== „na …” lokatyw – nie cel ruchu =====
  const NA_LOCATIVE_STARTS = [
    "na trawie","na lawce","na ławce","na dywanie","na rynku","na peronie",
    "na przystanku","na stadionie","na boisku","na plaży","na plazy"
  ].map(s => deaccent(s));

  // ===== miejsce (gdzie?) =====
  function pickPlaceFromSentence(s="") {
    const base = removeDiscourseMarkers(s);
    const PREP = "(?:we?|na|pod|nad|przy|mi[eę]dzy|miedzy|za|przed|obok|kolo|koło|u|w)";
    const re = new RegExp(String.raw`\b${PREP}\s+\S+(?:\s+(?!${PREP}\b|przez\b)\S+){0,6}`,"gi");

    let pps = (base.match(re) || []).map(x => normalizeSpaces(x));

    pps = pps
      .map(x => x.replace(/\s+(przez\s+chwil[ęe]|na\s+chwil[ęe]|przez\s+moment)\b$/i, ""))
      .map(x => x.replace(TIME_TAIL, ""))
      .map(x => stripDanglingPrzez(x))
      .map(x => cutAtVerb(x))
      .map(x => stripOnConjOrComma(x))
      .map(x => normalizeSpaces(x));

    const isTemporalPP = (pp="") => TIME_RE.test(pp);
    pps = pps.filter(x => x && !isTemporalPP(x));
    if (!pps.length) return null;

    // Heurystyka wyboru:
    // - preferuj PP z „przy …” (często lokalizacja siedzenia)
    // - preferuj PP znajdujące się PO czasowniku „usiąść/siedzieć/czekać/czytać”
    // - w razie remisu — weź prawostronną (ostatnią) PP
    const verbIdx = (() => {
      const m = s.match(/\b(usiad\w*|usied\w*|siedz\w*|czeka\w*|czyta\w*)\b/i);
      return m ? s.indexOf(m[0]) : -1;
    })();

    let best = pps[pps.length - 1], bestScore = -1;
    for (const cand of pps) {
      let sc = 0;
      const low = deaccent(cand.toLowerCase());
      if (/^\s*przy\b/i.test(cand)) sc += 2.5;
      if (/^\s*(w|we|na)\b/i.test(cand)) sc += 1.5;
      if (/^\s*(pod|za|przed|obok|u)\b/i.test(cand)) sc += 1.0;
      if (/\b(klasie|salonie|kuchni|pokoju|ławce|lawce|oknie|rynku|kinie|przystanku|fontannie|stole|zoo)\b/i.test(low)) sc += 1.5;

      const pos = s.indexOf(cand);
      if (verbIdx >= 0 && pos > verbIdx) sc += 1.8; // po czasowniku
      sc += pos / 1000; // lekki bonus za „prawiej”

      if (sc > bestScore) { best = cand; bestScore = sc; }
    }
    return stripPunct(normalizeSpaces(best));
  }

  // ===== pytania =====
  const qKiedy = () => "Kiedy to się działo?";
  const qGdzie = (low="") => {
    if (/\b(usiad\w*|usied\w*|siedzieli)\b/.test(low)) return "Gdzie usiedli?";
    if (/\bczekal\w*|czekali\b/.test(low)) return "Gdzie czekali?";
    if (/\bczytal\w*|czyta(ła|li)\b/.test(low)) return "Gdzie czytali?";
    return "Gdzie byli?";
  };

  let qtype = "", question = "", answer = "";

  // 1) DOKĄD? (priorytet – tylko jeśli jest cue ruchu)
  if (MOTION_CUE_RE.test(tLower)) {
    // wsiedli/wsiadł → „do …” po czasowniku
    const mBoard = t.match(/\b(wsied\w*|wsiad\w*)\s+do\s+([^\s,.;!?]+(?:\s+[^\s,.;!?]+){0,6})/i);
    if (mBoard) {
      qtype = "Dokąd?";
      question = dokadQuestionByCue(tLower);
      let body = mBoard[2];
      body = body.replace(TIME_TAIL,"");
      body = cutAtVerb(body);
      body = stripOnConjOrComma(body);
      answer = stripPunct(`do ${body}`);
    }

    // ogólna „do|na …”
    if (!answer) {
      const mDest = t.match(/\b(do|na)\s+([^\s,.;!?]+(?:\s+[^\s,.;!?]+){0,6})/i);
      if (mDest) {
        let prep = mDest[1].toLowerCase();
        let body = mDest[2];
        body = body.replace(TIME_TAIL,"");
        body = cutAtVerb(body);
        body = stripOnConjOrComma(body);
        let cand = stripPunct(`${prep} ${body}`);

        const low = deaccent(cand.toLowerCase());
        const isNaLoc = prep === "na" && NA_LOCATIVE_STARTS.some(st => low.startsWith(st));
        if (!isNaLoc) {
          qtype = "Dokąd?";
          question = dokadQuestionByCue(tLower);
          answer = cand;
        }
      }
    }
  }

  // 2) KIEDY?
  if (!answer) {
    const mTime = t.match(TIME_RE);
    if (mTime) {
      qtype = "Kiedy?";
      question = (/^\s*o\s*\d{1,2}[:.]\d{2}\b/i.test(mTime[0])) ? "O której godzinie to się wydarzyło?" : qKiedy();
      answer = stripPunct(mTime[0]);
    }
  }

  // 3) GDZIE?
  if (!answer) {
    const place = pickPlaceFromSentence(t);
    if (place) {
      qtype = "Gdzie?";
      question = qGdzie(tLower);
      answer = place;
    }
  }

  // 4) KTO? – np. „Do parku poszedł Tomek.”
  if (!answer && MOTION_CUE_RE.test(tLower)) {
    const mName = t.match(/\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+(?:\s+i\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)*\b/);
    if (mName) {
      qtype = "Kto?";
      question = "Kto poszedł do ...?";
      answer = stripPunct(mName[0]);
    }
  }

  return {
    ok: Boolean(question && answer),
    qtype,
    question,
    answer: stripPunct(answer),
    source_path: "rule-teacher-strict-v7.7-verbs(tune-2)"
  };
}

// --- QUIZ endpoints ---
app.post("/agent/comprehend", (req, res) => {
  try {
    const { text } = req.body || {};
    const r = generateQuestionAndAnswerTeacher(text);
    res.json(r);
  } catch (e) {
    res.status(500).json({ ok:false, error:String(e?.message || e), source_path:"error-exception" });
  }
});

app.post("/agent/comprehend-aggregate", (req, res) => {
  try {
    const { text, max_questions = 10 } = req.body || {};
    if (!text?.trim()) return res.json({ ok:true, count:0, items:[] });

    const cleaned = String(text)
      .split(/\r?\n/).map(line => line.replace(/^\s*[>›»]{1,2}\s*/,""))
      .join(" ").replace(/\s+/g," ").trim();

    const sentences = cleaned
      .split(/(?<=[.!?;])\s+/).map(s => s.trim()).filter(Boolean).slice(0, 30);

    const deaccent2 = s => String(s).normalize("NFD")
      .replace(/[\u0300-\u036f]/g,"")
      .replace(/[łŁ]/g,"l").replace(/[ąĄ]/g,"a").replace(/[ćĆ]/g,"c")
      .replace(/[ęĘ]/g,"e").replace(/[ńŃ]/g,"n").replace(/[óÓ]/g,"o")
      .replace(/[śŚ]/g,"s").replace(/[źŹżŻ]/g,"z");

    const scored = sentences
      .map(s => {
        const r = generateQuestionAndAnswerTeacher(s);
        let sc = 0;
        if (r.qtype === "Dokąd?") sc = 100;
        else if (r.qtype === "Kto?") sc = 85;
        else if (r.qtype === "Kiedy?") sc = 90;
        else if (r.qtype === "Gdzie?") sc = 80;
        else sc = 40;
        return { ...r, sentence: s, score: sc, src: "rule-teacher-strict-v7.7-verbs(tune-2)" };
      })
      .sort((a,b) => b.score - a.score);

    const top = [];
    const seen = new Set();
    for (const r of scored) {
      if (top.length >= max_questions) break;
      if (!r.answer) continue;
      const key = `${r.qtype}|${deaccent2((r.answer||"").toLowerCase())}`;
      if (seen.has(key)) continue;
      seen.add(key);
      top.push(r);
    }
    if (!top.length) {
      for (const r of scored) {
        if (top.length >= max_questions) break;
        if (r.answer) top.push(r);
      }
    }
    res.json({ ok:true, count: top.length, items: top });
  } catch (e) {
    res.status(500).json({ ok:false, error:String(e?.message || e), source_path:"error-exception" });
  }
});

app.get("/version", (_req, res) =>
  res.json({ build: "2025-10-26 rule-teacher-strict-v7.7-verbs(tune-2)" })
);
// ===================== /QUIZ =====================






















/* ===================== START ===================== */
let _prewarmRunning = false;
async function prewarmOnce() {
  if (_prewarmRunning) return;
  _prewarmRunning = true;
  try {
    if (process.env.GROQ_API_KEY) {
      await groqChat({
        messages: [{ role: 'user', content: 'ping' }],
        max_tokens: 8,
        temperature: 0.0
      }).catch(() => {});
    }
    if (openai) {
      await openaiChat({
        messages: [{ role: 'user', content: 'ok' }],
        max_tokens: 1,
        temperature: 0.0,
        top_p: 1.0
      }).catch(() => {});
    }
    if (BASE_URL) {
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 2000);
      try {
        await fetch(`${BASE_URL}/health`, {
          method: 'HEAD',
          headers: { Connection: 'keep-alive' },
          signal: ctrl.signal
        });
      } catch {}
      clearTimeout(to);
    }

    await refillPoolOnce().catch(() => {});
  } finally {
    _prewarmRunning = false;
  }
}

app.listen(PORT, () => {
  console.log(`🚀 Backend działa na http://localhost:${PORT}`);
  console.log(`🎧 Groq ${groq ? 'podłączony' : 'OFF'} (chat=${GROQ_CHAT_MODEL}, asr=${GROQ_ASR_MODEL})`);
  console.log(`🤖 OpenAI ${openai ? 'podłączony' : 'OFF'}`);
  console.log(`🧠 LLM_PREF=${LLM_PREF}`);
  console.log("Build tag: 2025-10-21 rehydrate+placefix+multi");


  // mały delay, żeby połączenia mogły się ustawić
  setTimeout(prewarmOnce, 500);

  setInterval(() => { refillPoolOnce().catch(() => {}); }, POOL_REFILL_INTERVAL_MS);

  if (PREWARM_EVERY_MIN > 0) {
    setInterval(() => { prewarmOnce().catch(()=>{}); }, PREWARM_EVERY_MIN * 60_000);
    console.log(`🛌 Anti-sleep: ping co ${PREWARM_EVERY_MIN} min${BASE_URL ? ` → ${BASE_URL}/health` : ''}`);
  }

  console.log(`🧺 Text pool: levels=${POOL_LEVELS.join(', ')} target=${POOL_TARGET_SIZE}, refill every ${POOL_REFILL_INTERVAL_MS}ms`);
});
