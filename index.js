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
const GROQ_MODEL = process.env.GROQ_MODEL || 'whisper-large-v3';
const LLM_PREF = 'openai-only'; // <<<<<<<<<<<<<< TYLKO OPENAI dla pytań/odpowiedzi

const keepAliveAgent = new http.Agent({ keepAlive: true, timeout: 10_000 });
const now = () => (global.performance?.now?.() ?? Date.now());
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

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
  res.json({ ok: true, service: 'poczytajmy-backend', version: '1.16-openai-only' });
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
        <li>POST <code>/agent/comprehend</code> ✅ pytanie+klucz</li>
        <li>POST <code>/agent/check-answer-voice</code> ✅ ocena+feedback</li>
        <li>POST <code>/agent/check-answer-text</code> ✅ ocena+feedback (tekst)</li>
        <li>POST <code>/asr</code>, <code>/ocr</code></li>
      </ul>
    </body></html>
  `);
});

/* ===================== LLM helpers (OpenAI-only dla czatu) ===================== */
async function groqChat({ messages, max_tokens = MAX_TOKENS_FAST, temperature = 0.3, top_p = 0.95 }) {
  // Zostawione do ewentualnego pingu/prewarm — NIE używamy do pytań/odpowiedzi.
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
async function openaiChat({ messages, max_tokens = MAX_TOKENS_FAST, temperature = 0.3, top_p = 0.95 }) {
  if (!openai) throw new Error('OPENAI_OFF');
  const t0 = now();
  const r = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages,
    temperature,
    top_p,
    max_tokens,
  });
  const txt = r?.choices?.[0]?.message?.content?.trim?.() || '';
  if (!txt) throw new Error('OPENAI_EMPTY');
  return { provider: 'openai', text: txt, latency_ms: Math.round(now() - t0) };
}
/** ENTRYPOINT — tylko OpenAI */
async function chatPref({ prompt, max_tokens = 150, temperature = 0.3, top_p = 0.95 }) {
  if (!openai) throw new Error('NO_OPENAI');
  const messages = [{ role: 'user', content: trimUserContent(prompt) }];
  return await withDeadline(openaiChat({ messages, max_tokens, temperature, top_p }), DEADLINE_MS);
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

async function generateGreetingV2({ name, age, character, theme }) {
  const prompt = buildGreetingPrompt({ age: Number(age), character, theme, n: 12 });
  const { text: raw, provider } = await chatPref({ prompt, temperature: 0.9, max_tokens: 180 });

  let cands = parseList(raw);
  if (!cands.length && raw) cands = raw.split(/[.\n]/).map(s => s.trim()).filter(Boolean);
  if (!cands.length) throw new Error('EMPTY_GENERATION');

  const profileKey = `${(name || '').toLowerCase()}|${Number(age)||'X'}`;
  const history = recentGreetings.get(profileKey) || [];

  const picked = chooseMostNovel(cands, history);
  const cleaned = sanitizeNoName(name, picked);
  const finalText = cleaned || picked;

  recentGreetings.set(profileKey, [finalText, ...history].slice(0, 20));
  return { text: finalText, source: provider || 'unknown' };
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

async function generateMotivation({ age, accuracy, text, characterName, lang = 'pl' }) {
  const prompt = buildMotivationPrompt({ age, accuracy, text, characterName, lang });
  const { text: raw, provider } = await chatPref({ prompt, temperature: 0.9, max_tokens: 120 });
  let out = String(raw || '').trim();
  out = out.replace(/^["'„”]+|["'„”]+$/g, '').trim();
  out = tightenMotivation(out, 160);
  if (!out) throw new Error('EMPTY_MOTIVATION');
  return { text: out, source: provider || 'unknown' };
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
    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
    if (timedOut) return res.status(504).json({ ok: false, error: 'DEADLINE_EXCEEDED', timed_out: true });
    console.error('agent/motivate error:', err);
    return res.status(502).json({
      ok: false,
      error: String(err?.message || err),
      fallback: 'Świetna próba! Z każdą stroną będzie coraz lepiej — spróbujmy jeszcze raz! 💪'
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

async function correctPolishSentence(raw) {
  const prompt = `
Popraw zdanie dla dziecka w wieku wczesnoszkolnym.
Zasady:
- Jedno zdanie po polsku, 8–16 słów.
- Proste, naturalne, bez żargonu i cudzysłowów.
- Popraw ortografię i interpunkcję.
Zwróć tylko gotowe zdanie.
Tekst:
${raw}`.trim();

  const { text } = await chatPref({ prompt, temperature: 0.2, max_tokens: 60, top_p: 0.9 });
  return cleanSentence(text || "");
}

app.post("/agent/generate-text", async (req, res) => {
  try {
    const { language = "pl", level = "A1" } = req.body || {};

    const prompt =
`Napisz jedno proste zdanie po polsku na poziomie ${String(level).toUpperCase()} do głośnego czytania przez dziecko.
Wymagania:
- Jedno zdanie (8–16 słów), naturalne i poprawne.
- Słownictwo codzienne, bez żargonu i neologizmów.
- Zero przemocy, straszenia, polityki, chorób.
- Brak cudzysłowów i nawiasów.
- Używaj pełnych polskich znaków.
Podaj tylko gotowe zdanie.`;

    const first = await chatPref({ prompt, temperature: 0.3, max_tokens: 60, top_p: 0.9 });
    let sentence = cleanSentence(first.text || "");
    if (!sentence) throw new Error("EMPTY_GENERATION");

    let check = validateKidsSentencePL(sentence);
    if (!check.ok) {
      const fixed = cleanSentence(await correctPolishSentence(sentence));
      const check2 = validateKidsSentencePL(fixed);
      if (check2.ok) {
        return res.json({ ok: true, text: check2.text, level, language, source: `${first.provider}+corrector` });
      }
      const backup = pick(bankByLevel(level));
      return res.json({ ok: true, text: backup, level, language, source: "fallback-bank" });
    }
    return res.json({ ok: true, text: check.text, level, language, source: first.provider });
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
/* =====================  QUIZ / COMPREHEND – V2  ====================== */
/* ===================================================================== */

/* — drobne utils (lokalne, z prefiksem qz_) — */
function qz_trim(s = "", limit = 600) {
  const t = String(s || "").replace(/\s+/g, " ").trim();
  return t.length > limit ? t.slice(0, limit) : t;
}
function qz_qmark(q = "") { return /\?\s*$/.test(String(q)); }
function qz_wc(a = "") {
  return (String(a).trim().match(/\b[\p{L}\p{M}0-9'-]+\b/gu) || []).length;
}

/* — wykryj imię na początku zdania + 3. os. — */
function qz_detectThirdName(text) {
  const m = String(text).trim()
    .match(/^([A-ZŁŚŻŹĆŃÓ][\p{L}\-']+)\s+(czyta|ogląda|słucha|je|pije|gra|idzie|siedzi|rysuje|maluje|pisze|gotuje|otwiera|uczy)\b/u);
  if (!m) return null;
  const name = m[1];
  if (/^Ja$/i.test(name)) return null;
  return name;
}

/* — wytnij minimalny fragment po czasowniku do separatora — */
function qz_sliceAfter(text, reVerb, { keepPreposition = false } = {}) {
  const src = String(text);
  const re = new RegExp(
    reVerb.source + "\\s+(.+?)\\s*(?=(,|\\.|;|\\s+i\\s+|\\s+oraz\\s+|\\s+a\\s+|$))",
    reVerb.flags
  );
  const m = src.match(re);
  if (!m) return null;
  // WAŻNE: reVerb ma własną ( ) wokół czasownika → nasz fragment to GRUPA 2
  let out = (m[2] || "").trim();
  // odcięcie celu/uzasadnienia
  out = out.replace(/\s*,?\s*(żeby|aby)\s+.*$/i, "").trim();
  // kosmetyka
  out = out.replace(/\s{2,}/g, " ").trim();
  if (!keepPreposition) {
    out = out.replace(/^(w|we|na|do|przy|pod|u|obok|o)\s+/i, "").trim();
  }
  return out || null;
}
/* — ekstrakcje: czas / miejsce / cel — */
const QZ_RE_TIME = /\b(rano|wieczorem|w południe|po południu|wczoraj|dzisiaj|dziś|jutro|po (obiedzie|szkole|kolacji))\b/iu;
function qz_time(text) { const m = String(text).match(QZ_RE_TIME); return m ? m[0] : null; }

function qz_place(text) {
  let m = String(text).match(/\b(w|we|na|do|przy|pod|u|obok)\s+([^.,;!?]+)/i);
  if (!m) return null;
  let out = m[0].trim();
  // odetnij trailing time (np. „po południu”)
  out = out.replace(new RegExp('\\s*' + QZ_RE_TIME.source + '.*$', 'iu'), '').trim();
  return out;
}
function qz_destination(text) {
  const m = String(text).match(/\b(do|na)\s+([^.,;!?]+)/iu);
  if (!m) return null;
  let out = m[0].trim();
  // usuń czas, jeśli się podczepił
  out = out.replace(new RegExp(QZ_RE_TIME.source + '.*$', 'iu'), '').trim();
  // usuń ogon celu/uzasadnienia
  out = out.replace(/\s*,?\s*(żeby|aby)\s+.*$/i, '').trim();
  return out;
}

/* — mapa czasowników i strategii — */
const QZ_VERBS = {
  OBJ_CO_1OS: [
    { re: /\b(czytam)\b/iu, q: "Co czytam?" },
    { re: /\b(piszę)\b/iu,  q: "Co piszę?" },
    { re: /\b(rysuję)\b/iu, q: "Co rysuję?" },
    { re: /\b(gotuję)\b/iu, q: "Co gotuję?" },
    { re: /\b(otwieram)\b/iu, q: "Co otwieram?" },
    { re: /\b(oglądam)\b/iu,  q: "Co oglądam?" },
    { re: /\b(jem)\b/iu,      q: "Co jem?" },
    { re: /\b(piję)\b/iu,     q: "Co piję?" },
    { re: /\b(słucham)\b/iu,  q: "Czego słucham?", genitive: true },
  ],
  OBJ_CO_3OS: [
    { re: /\b(czyta)\b/iu,   q: n => `Co czyta ${n}?` },
    { re: /\b(pisze)\b/iu,   q: n => `Co pisze ${n}?` },
    { re: /\b(rysuje)\b/iu,  q: n => `Co rysuje ${n}?` },
    { re: /\b(maluje)\b/iu,  q: n => `Co maluje ${n}?` },
    { re: /\b(gotuje)\b/iu,  q: n => `Co gotuje ${n}?` },
    { re: /\b(otwiera)\b/iu, q: n => `Co otwiera ${n}?` },
    { re: /\b(ogląda)\b/iu,  q: n => `Co ogląda ${n}?` },
    { re: /\b(je)\b/iu,      q: n => `Co je ${n}?` },
    { re: /\b(pije)\b/iu,    q: n => `Co pije ${n}?` },
    { re: /\b(słucha)\b/iu,  q: n => `Czego słucha ${n}?`, genitive: true },
  ],
  PLAY_1OS_W:  { re: /\b(gram)\b/iu, prep: "w",  q: "W co gram?" },
  PLAY_3OS_W:  { re: /\b(gra)\b/iu,  prep: "w",  q: n => `W co gra ${n}?` },
  PLAY_1OS_NA: { re: /\b(gram)\b/iu, prep: "na", q: "Na czym gram?" },
  PLAY_3OS_NA: { re: /\b(gra)\b/iu,  prep: "na", q: n => `Na czym gra ${n}?` },
  PLACE_1OS: [
    { re: /\b(siedzę)\b/iu, q: "Gdzie siedzę?" },
    { re: /\b(stoję)\b/iu,  q: "Gdzie stoję?" },
    { re: /\b(leżę)\b/iu,   q: "Gdzie leżę?" },
  ],
  PLACE_3OS: [
    { re: /\b(siedzi)\b/iu, q: n => `Gdzie siedzi ${n}?` },
    { re: /\b(stoi)\b/iu,   q: n => `Gdzie stoi ${n}?` },
    { re: /\b(leży)\b/iu,   q: n => `Gdzie leży ${n}?` },
  ],
  MOVE_1OS: { re: /\b(idę|idziemy)\b/iu, qDest: "Dokąd idę?", qTime: "Kiedy idę?" },
  MOVE_3OS: { re: /\b(idzie)\b/iu, qDest: n => `Dokąd idzie ${n}?`, qTime: n => `Kiedy idzie ${n}?` },
  LEARN_1OS: { re: /\b(uczę się)\b/iu, q: "Czego się uczę?" },
  LEARN_3OS: { re: /\b(uczy się)\b/iu, q: n => `Czego uczy się ${n}?` },
};

/* — główna heurystyka — */
function qz_heuristic(textRaw, age) {
  const text = String(textRaw || "").trim();
  const name3 = qz_detectThirdName(text);
  const isThird = !!name3;

  // 1) gram NA …
  {
    const v = isThird ? QZ_VERBS.PLAY_3OS_NA : QZ_VERBS.PLAY_1OS_NA;
    if (text.match(v.re)) {
      const m = text.match(/\bna\s+([^.,;!?]+)/i);
      if (m) {
        const ans = (m[1] || "").trim();
        const q = isThird ? v.q(name3) : v.q;
        if (ans) return { question: q, answer: ans, fallback: false };
      }
    }
  }
  // 2) gram W …
  {
    const v = isThird ? QZ_VERBS.PLAY_3OS_W : QZ_VERBS.PLAY_1OS_W;
    if (text.match(v.re)) {
      const m = text.match(/\bw\s+([^.,;!?]+)/i);
      if (m) {
        const ans = (m[1] || "").trim();
        const q = isThird ? v.q(name3) : v.q;
        if (ans) return { question: q, answer: ans, fallback: false };
      }
    }
  }
  // 3) pozycja (miejsce > czas)
  {
    const list = isThird ? QZ_VERBS.PLACE_3OS : QZ_VERBS.PLACE_1OS;
    for (const v of list) {
      if (!text.match(v.re)) continue;
      const p = qz_place(text);
      if (p) {
        const ans = p.replace(/^(w|we|na|do|przy|pod|u|obok)\s+/i, "").trim();
        const q = isThird ? v.q(name3) : v.q;
        return { question: q, answer: ans, fallback: false };
      }
      const t = qz_time(text);
      if (t) {
        const q = (isThird ? v.q(name3) : v.q).replace(/^Gdzie/, "Kiedy");
        return { question: q, answer: t, fallback: false };
      }
      const q = isThird ? v.q(name3) : v.q;
      return { question: q, answer: "", fallback: true };
    }
  }
  // 4) ruch (cel > czas)
  {
    const v = isThird ? QZ_VERBS.MOVE_3OS : QZ_VERBS.MOVE_1OS;
    if (text.match(v.re)) {
      const dest = qz_destination(text);
      if (dest) {
        const q = isThird ? v.qDest(name3) : v.qDest;
        const ans = dest.replace(/\s*,.*$/, "");
        return { question: q, answer: ans, fallback: false };
      }
      const t = qz_time(text);
      if (t) {
        const q = isThird ? v.qTime(name3) : v.qTime;
        return { question: q, answer: t, fallback: false };
      }
      const q = isThird ? v.qDest(name3) : v.qDest;
      return { question: q, answer: "", fallback: true };
    }
  }
  // 5) uczę się …
  {
    const v = isThird ? QZ_VERBS.LEARN_3OS : QZ_VERBS.LEARN_1OS;
    if (text.match(v.re)) {
      const obj = qz_sliceAfter(text, v.re, { keepPreposition: false }) || "";
      const ans = obj.replace(/^się\s+/i, "").replace(/^(czego|z|o)\s+/i, "").trim();
      const q = isThird ? v.q(name3) : v.q;
      if (ans) return { question: q, answer: ans, fallback: false };
      return { question: q, answer: "", fallback: true };
    }
  }
  // 6) Co/Czego…
  {
    const bucket = isThird ? QZ_VERBS.OBJ_CO_3OS : QZ_VERBS.OBJ_CO_1OS;
    for (const v of bucket) {
      if (!text.match(v.re)) continue;
      let obj = qz_sliceAfter(text, v.re, { keepPreposition: false });
      if (obj) {
  // skróć do samego obiektu (odetnij fragmenty typu „w/na/do ...”)
  obj = obj.replace(/\s+(w|we|na|do|przy|pod|u|obok)\s+.*$/i, '').trim();
  const q = isThird ? v.q(name3) : v.q;
  return { question: q, answer: obj, fallback: false };
}
      const p = qz_place(text);
      if (p) {
        const q = (isThird ? (typeof v.q === "function" ? v.q(name3) : v.q) : v.q)
          .replace(/^Co|^Czego/, "Gdzie");
        const ans = p.replace(/^(w|we|na|do|przy|pod|u|obok)\s+/i, "").trim();
        return { question: q, answer: ans, fallback: true };
      }
      const t = qz_time(text);
      if (t) {
        const q = (isThird ? (typeof v.q === "function" ? v.q(name3) : v.q) : v.q)
          .replace(/^Co|^Czego/, "Kiedy");
        return { question: q, answer: t, fallback: true };
      }
      const q = isThird ? (typeof v.q === "function" ? v.q(name3) : v.q) : v.q;
      return { question: q, answer: "", fallback: true };
    }
  }
  // 7) fallback ogólny
  const p = qz_place(text);
  if (p) return { question: "Gdzie to się dzieje?", answer: p.replace(/^(w|we|na|do|przy|pod|u|obok)\s+/i, "").trim(), fallback: true };
  const t = qz_time(text);
  if (t) return { question: "Kiedy to się dzieje?", answer: t, fallback: true };
  return { question: "O co chodzi w zdaniu?", answer: "", fallback: true };
}

/* ===================== /agent/comprehend ===================== */
app.post("/agent/comprehend", async (req, res) => {
  try {
    const { text = "", age } = req.body || {};
    const src = String(text || "").trim();
    if (!src) return res.status(400).json({ ok: false, error: "NO_TEXT" });

    // 1) Heurystyka
    let h = qz_heuristic(src, age);
    if (qz_qmark(h.question) && h.answer && qz_wc(h.answer) <= 8) {
      return res.json({ ok: true, question: h.question, answer: h.answer, fallback: !!h.fallback });
    }

    // 2) Delikatny fallback LLM (OpenAI, jeśli dostępny) — bez zależności od zewn. trimUserContent
    if (typeof openai !== "undefined" && openai) {
      const prompt =
`Na podstawie fragmentu napisz JEDNO bardzo proste pytanie kontrolne i krótki klucz odpowiedzi (max 6 słów).
Preferuj: „Co…?”, „Czego…?”, dla ruchu: „Dokąd…?”, dla pozycji: „Gdzie…?”.
Fragment:
"""${qz_trim(src, 600)}"""
Zwróć wyłącznie JSON: {"question":"…?","answer":"…"} — bez komentarza.`;

      try {
        const r = await withDeadline(
          openai.chat.completions.create({
            model: "gpt-4o-mini",
            temperature: 0.2,
            max_tokens: 120,
            messages: [{ role: "user", content: prompt }]
          }),
          DEADLINE_MS
        );
        const out = (r?.choices?.[0]?.message?.content || "").trim();
        const m = out.match(/\{[\s\S]*\}/);
        if (m) {
          const j = JSON.parse(m[0]);
          const q = String(j?.question || "").replace(/[„”"']/g, "").trim();
          const a = String(j?.answer || "").replace(/[„”"']/g, "").trim();
          if (qz_qmark(q) && a && qz_wc(a) <= 8) {
            return res.json({ ok: true, question: q, answer: a, fallback: false });
          }
        }
      } catch (_) {
        // miękki fallback na heurystykę niżej
      }
    }

    // 3) Ostateczny fallback na heurystykę
    h = qz_heuristic(src, age);
    const question = qz_qmark(h.question) ? h.question : "Gdzie to się dzieje?";
    const answer = h.answer || qz_place(src) || qz_time(src) || "";
    return res.json({ ok: true, question, answer, fallback: true });
  } catch (err) {
    console.error("comprehend error:", err);
    const src = String(req.body?.text || "");
    return res.status(200).json({
      ok: true,
      question: "Gdzie to się dzieje?",
      answer: qz_place(src) || qz_time(src) || "",
      fallback: true
    });
  }
});


/* ===================== START ===================== */
async function prewarmOnce() {
  try {
    if (process.env.GROQ_API_KEY) {
      // lekki ping by utrzymać połączenie przy życiu
      await groqChat({ messages: [{ role: 'user', content: 'ping' }], max_tokens: 8, temperature: 0.0 }).catch(()=>{});
    }
    if (BASE_URL) {
      await fetch(`${BASE_URL}/health`, { headers: { Connection: 'keep-alive' } }).catch(()=>{});
    }
  } catch { /* noop */ }
}

app.listen(PORT, () => {
  console.log(`🚀 Backend działa na http://localhost:${PORT}`);
  console.log(`🎧 Groq ${groq ? 'podłączony' : 'OFF'} (model=${GROQ_MODEL})`);
  console.log(`🤖 OpenAI ${openai ? 'podłączony' : 'OFF'}`);
  console.log(`🧠 LLM_PREF=${LLM_PREF}`);
  prewarmOnce();
  if (PREWARM_EVERY_MIN > 0) {
    setInterval(prewarmOnce, PREWARM_EVERY_MIN * 60_000);
    console.log(`🛌 Anti-sleep: ping co ${PREWARM_EVERY_MIN} min${BASE_URL ? ` → ${BASE_URL}/health` : ''}`);
  }
});
