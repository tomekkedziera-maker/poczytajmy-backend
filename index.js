import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import multer from 'multer';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import OpenAI from 'openai';
import Groq from 'groq-sdk';

import sharp from 'sharp';
import Tesseract from 'tesseract.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(morgan('dev'));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const groq   = process.env.GROQ_API_KEY   ? new Groq({ apiKey: process.env.GROQ_API_KEY })     : null;

const MOCK_ASR  = process.env.MOCK_ASR  === '1';
const MOCK_TEXT = process.env.MOCK_TEXT === '1';

/* ===================== OCR helpers ===================== */
const LANG_PATH =
  process.env.OCR_LANG_PATH ||
  'https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main';

let inflight = 0;
const MAX_CONCURRENCY = Number(process.env.OCR_MAX_CONCURRENCY || 2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
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

// === ASR ===
app.post('/asr', upload.single('audio'), async (req, res) => {
  try {
    if (MOCK_ASR) return res.json({ text: 'Ala ma kota', source: 'mock' });
    if (!req.file) return res.status(400).json({ error: 'Brak pliku w polu "audio".' });

    const ext = pickAudioExt(req.file);
    const tmpPath = path.join(os.tmpdir(), `rec-${Date.now()}.${ext}`);
    fs.writeFileSync(tmpPath, req.file.buffer);
    const stream = fs.createReadStream(tmpPath);

    try {
      if (groq) {
        const transcript = await groq.audio.transcriptions.create({
          file: stream, model: 'whisper-large-v3', language: 'pl',
        });
        return res.json({ text: transcript?.text ?? '', source: 'groq' });
      }
      if (openai) {
        const transcript = await openai.audio.transcriptions.create({
          file: stream, model: 'whisper-1', language: 'pl',
        });
        return res.json({ text: transcript?.text ?? '', source: 'openai' });
      }
      return res.json({ text: 'Ala ma kota', source: 'mock_fallback' });
    } finally {
      fs.unlink(tmpPath, () => {});
    }
  } catch (err) {
    console.error('ASR error:', err);
    res.status(500).json({ error: 'ASR_FAILED', details: String(err?.message || err) });
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
    .replace(/[„”"!?.,;:()\-–—[\]{}…]/g, '')
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

function buildGreetingPrompt({
  name = 'Przyjacielu',
  age,
  character = 'Twój przyjaciel',
  theme = '',
  n = 12
}) {
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

function localGreeting() {
  const pool = [
    'Otwórzmy dziś nowy rozdział niezwykłej książki! 📖',
    'Czeka nas opowieść pełna ciekawych słów i ilustracji. ✨',
    'Zbierajmy gwiazdki za płynne czytanie kolejnych zdań! 🌟',
    'Zajrzyjmy do biblioteczki i wybierzmy świetną bajkę!',
    'Rozgrzewka z sylabami, a potem czytelnicza przygoda! 🚀'
  ];
  return pick(pool);
}

const FORBIDDEN_HELLOS = ['cześć', 'hej', 'witaj', 'siema', 'halo'];
function sanitizeNoName(name, raw) {
  let s = (raw || '').trim();

  // usuń powitania na początku
  const helloRe = new RegExp(`^\\s*(?:${FORBIDDEN_HELLOS.join('|')})\\b[\\p{L}\\p{M}\\s,!.?–—-]*`, 'iu');
  s = s.replace(helloRe, '').trim();

  // usuń odmiany imienia (prosty zestaw)
  if (name) {
    const forms = [name, `${name}u`, `${name}o`, `${name}e`, `${name}a`, `${name}ku`];
    const escaped = forms.map(v => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const nameRe = new RegExp(`\\b(?:${escaped.join('|')})\\b[\\s,!.?]*`, 'giu');
    s = s.replace(nameRe, '').trim();
  }

  // przód z przecinków/znaków
  s = s.replace(/^[,–—\-|:;!.\s]+/u, '').trim();
  return s;
}

const recentGreetings = new Map();

async function generateGreetingV2({ name, age, character, theme }) {
  const prompt = buildGreetingPrompt({ age: Number(age), character, theme, n: 12 });

  const tasks = [];
  if (openai) {
    tasks.push((async () => {
      const resp = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.95, top_p: 0.95, max_tokens: 220,
      });
      const txt = resp?.choices?.[0]?.message?.content?.trim?.();
      if (!txt) throw new Error('openai_empty');
      return { text: txt, source: 'openai' };
    })());
  }
  if (groq) {
    tasks.push((async () => {
      const resp = await groq.chat.completions.create({
        model: 'llama-3.1-8b-instant',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.95, top_p: 0.95, max_tokens: 220,
      });
      const txt = resp?.choices?.[0]?.message?.content?.trim?.();
      if (!txt) throw new Error('groq_empty');
      return { text: txt, source: 'groq' };
    })());
  }
  // teksty ma generować nasz agent Groq – OpenAI jako fallback

  let raw = '';
  let provider = 'local';
  if (tasks.length) {
    try {
      const win = await Promise.any(tasks);
      raw = win.text;
      provider = win.source;
    } catch {}
  }

  let cands = parseList(raw);
  if (!cands.length && raw) cands = raw.split(/[.\n]/).map(s => s.trim()).filter(Boolean);
  if (!cands.length) return { text: localGreeting(), source: 'local' };

  const profileKey = `${(name || '').toLowerCase()}|${Number(age)||'X'}`;
  const history = recentGreetings.get(profileKey) || [];

  const picked = chooseMostNovel(cands, history);
  const cleaned = sanitizeNoName(name, picked);
  const finalText = cleaned || picked;

  recentGreetings.set(profileKey, [finalText, ...history].slice(0, 20));
  return { text: finalText, source: provider };
}

app.post('/agent/generate-greeting', async (req, res) => {
  try {
    const { name = '', age, character = 'Twój przyjaciel' } = req.body || {};
    const theme = HERO_THEMES[character] || '';
    const { text, source } = await generateGreetingV2({ name, age, character, theme });
    res.json({ ok: true, text, source });
  } catch (err) {
    console.error('agent/generate-greeting error:', err);
    res.json({ ok: true, text: localGreeting(), source: 'local-fallback' });
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

    let text = '';
    let source = 'local';

    if (groq) {
      const r = await groq.chat.completions.create({
        model: 'llama-3.1-8b-instant',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.8, top_p: 0.95, max_tokens: 60,
      });
      text = r?.choices?.[0]?.message?.content?.trim?.() || '';
      source = 'groq';
    } else if (openai) {
      const r = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.8, top_p: 0.95, max_tokens: 60,
      });
      text = r?.choices?.[0]?.message?.content?.trim?.() || '';
      source = 'openai';
    }

    if (!text) {
      const list = bankByLevel(level);
      text = pick(list);
      source = 'local';
    }

    // zdejmij ewentualne cudzysłowy/ozdobniki
    text = String(text).replace(/^["'„”]+|["'„”]+$/g, '').trim();

    return res.json({ ok: true, text, level, language, source });
  } catch (err) {
    console.error('agent/generate-text error:', err);
    const lvl = req?.body?.level || 'A1';
    const list = bankByLevel(lvl);
    return res
      .status(200)
      .json({ ok: true, text: pick(list), level: lvl, language: req?.body?.language || 'pl', source: 'local-fallback' });
  }
});

// Alias zgodności wstecznej
app.post('/generate-text', (req, res) => {
  // 307 zachowuje metodę POST i body
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

/* ===================== START ===================== */
app.listen(PORT, () => {
  console.log(`🚀 Backend działa na http://localhost:${PORT}`);
  console.log(`🎧 Groq ${groq ? 'podłączony' : 'OFF'}`);
  console.log(`🤖 OpenAI ${openai ? 'podłączony' : 'OFF'}`);
});

