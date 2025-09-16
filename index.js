--- a/index.js
+++ b/index.js
@@ -1,12 +1,17 @@
 import 'dotenv/config';
 import express from 'express';
 import cors from 'cors';
 import morgan from 'morgan';
 import multer from 'multer';
 import fs from 'fs';
 import os from 'os';
 import path from 'path';
 import { fileURLToPath } from 'url';
+import http from 'node:http';
 
 import OpenAI from 'openai';
 import Groq from 'groq-sdk';
 
 import sharp from 'sharp';
 import Tesseract from 'tesseract.js';
@@ -20,11 +25,39 @@
 const app = express();
 const PORT = process.env.PORT || 3001;
 
 app.use(cors());
 app.use(express.json({ limit: '10mb' }));
 app.use(morgan('dev'));
 
+/* ======== Low-latency config (race + deadline, no local text) ======== */
+const DEADLINE_MS = Number(process.env.FAST_TIMEOUT_MS || 1200);            // globalny deadline
+const MAX_TOKENS_FAST = Number(process.env.MAX_TOKENS_FAST || 64);
+const PREWARM_EVERY_MIN = Number(process.env.PREWARM_EVERY_MIN || 5);
+const BASE_URL = process.env.BASE_URL || '';
+const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';
+const keepAliveAgent = new http.Agent({ keepAlive: true, timeout: 10_000 });
+const now = () => (global.performance?.now?.() ?? Date.now());
+const sleep = (ms) => new Promise(r => setTimeout(r, ms));
+const trimUserContent = (s = '', limit = 1200) => {
+  const compact = String(s || '').replace(/\s+/g, ' ').trim();
+  return compact.length > limit ? compact.slice(-limit) : compact;
+};
+
+function withDeadline(promise, ms = DEADLINE_MS) {
+  return new Promise((resolve, reject) => {
+    const to = setTimeout(() => reject(new Error('DEADLINE_EXCEEDED')), ms);
+    promise.then((v) => { clearTimeout(to); resolve(v); }, (e) => { clearTimeout(to); reject(e); });
+  });
+}
+
+async function groqChat({ messages, max_tokens = MAX_TOKENS_FAST, temperature = 0.3, top_p = 0.95 }) {
+  const t0 = now();
+  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
+    method: 'POST', agent: keepAliveAgent,
+    headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY || ''}`, 'Content-Type': 'application/json', Accept: 'application/json', Connection: 'keep-alive' },
+    body: JSON.stringify({ model: GROQ_MODEL, temperature, top_p, max_tokens, messages })
+  });
+  if (!res.ok) throw new Error(`GROQ_HTTP_${res.status}`);
+  const data = await res.json();
+  return { provider: 'groq', text: data?.choices?.[0]?.message?.content?.trim?.() || '', latency_ms: Math.round(now() - t0) };
+}
+
 const upload = multer({
   storage: multer.memoryStorage(),
   limits: { fileSize: 50 * 1024 * 1024 },
 });
 
@@ -124,6 +157,7 @@
 }
 
 function buildGreetingPrompt({
   name = 'Przyjacielu',
   age,
   character = 'Twój przyjaciel',
   theme = '',
   n = 12
@@ -219,41 +253,50 @@
 const recentGreetings = new Map();
 
 async function generateGreetingV2({ name, age, character, theme }) {
   const prompt = buildGreetingPrompt({ age: Number(age), character, theme, n: 12 });
 
-  // 1) Najpierw superszybka ścieżka Groq (1.2s). 2) Gdy nie zdąży, lecimy fallbackiem (OpenAI lub lokalnie).
-  const tasks = [
-    (async () => {
-      if (!process.env.GROQ_API_KEY) throw new Error('groq_off');
-      const fast = await callGroqFast({
-        messages: [{ role: 'user', content: trimUserContent(prompt) }],
-        temperature: 0.9,
-        top_p: 0.95,
-        max_tokens: 180,
-      });
-      if (!fast.ok || !fast.text) throw new Error(fast.error || 'groq_empty');
-      return { text: fast.text, source: 'groq' };
-    })()
-  ];
-  if (openai) {
-    tasks.push((async () => {
-      const resp = await openai.chat.completions.create({
-        model: 'gpt-4o-mini',
-        messages: [{ role: 'user', content: prompt }],
-        temperature: 0.9, top_p: 0.95, max_tokens: 180,
-      });
-      const txt = resp?.choices?.[0]?.message?.content?.trim?.();
-      if (!txt) throw new Error('openai_empty');
-      return { text: txt, source: 'openai' };
-    })());
-  }
-
-  let raw = '';
-  let provider = 'local';
-  if (tasks.length) {
-    try {
-      const win = await Promise.any(tasks);
-      raw = win.text;
-      provider = win.source;
-    } catch {}
-  }
+  // RACE: Groq vs (opcjonalnie) OpenAI — z globalnym DEADLINE_MS. Brak lokalnych treści.
+  const racers = [];
+  if (process.env.GROQ_API_KEY) {
+    racers.push(groqChat({
+      messages: [{ role: 'user', content: trimUserContent(prompt) }],
+      temperature: 0.9, top_p: 0.95, max_tokens: 180,
+    }));
+  }
+  if (openai) {
+    racers.push((async () => {
+      const t0 = now();
+      const r = await openai.chat.completions.create({
+        model: 'gpt-4o-mini',
+        messages: [{ role: 'user', content: prompt }],
+        temperature: 0.9, top_p: 0.95, max_tokens: 180,
+      });
+      const txt = r?.choices?.[0]?.message?.content?.trim?.() || '';
+      if (!txt) throw new Error('OPENAI_EMPTY');
+      return { provider: 'openai', text: txt, latency_ms: Math.round(now() - t0) };
+    })());
+  }
+
+  let raw = '', provider = '';
+  try {
+    const winner = await withDeadline(Promise.any(racers), DEADLINE_MS);
+    raw = winner.text; provider = winner.provider;
+  } catch (e) {
+    // brak zwycięzcy w deadline → sygnalizujemy timeout / 504
+    const err = String(e && e.message || e);
+    throw new Error(err === 'DEADLINE_EXCEEDED' ? 'DEADLINE_EXCEEDED' : err);
+  }
 
   let cands = parseList(raw);
   if (!cands.length && raw) cands = raw.split(/[.\n]/).map(s => s.trim()).filter(Boolean);
-  if (!cands.length) return { text: localGreeting(), source: 'local' };
+  if (!cands.length) {
+    // źródło nie zwróciło poprawnej listy — raportujemy błąd do frontu
+    throw new Error('EMPTY_GENERATION');
+  }
 
   const profileKey = `${(name || '').toLowerCase()}|${Number(age)||'X'}`;
   const history = recentGreetings.get(profileKey) || [];
 
   const picked = chooseMostNovel(cands, history);
   const cleaned = sanitizeNoName(name, picked);
   const finalText = cleaned || picked;
 
   recentGreetings.set(profileKey, [finalText, ...history].slice(0, 20));
-  return { text: finalText, source: provider };
+  return { text: finalText, source: provider || 'unknown' };
 }
 
 app.post('/agent/generate-greeting', async (req, res) => {
   try {
     const { name = '', age, character = 'Twój przyjaciel' } = req.body || {};
     const theme = HERO_THEMES[character] || '';
     const { text, source } = await generateGreetingV2({ name, age, character, theme });
     res.json({ ok: true, text, source });
   } catch (err) {
-    console.error('agent/generate-greeting error:', err);
-    res.json({ ok: true, text: localGreeting(), source: 'local-fallback' });
+    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
+    if (timedOut) return res.status(504).json({ ok: false, error: 'DEADLINE_EXCEEDED', timed_out: true });
+    console.error('agent/generate-greeting error:', err);
+    return res.status(502).json({ ok: false, error: String(err?.message || err) });
   }
 });
 
 /* ===================== GENERATOR ZDAŃ DO CZYTANIA ===================== */
@@ -281,40 +324,53 @@
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
 
-    // Najpierw szybki Groq; jeśli nie zdąży → OpenAI; jeśli brak → lokalny bank
-    let text = '';
-    let source = 'local';
-    if (process.env.GROQ_API_KEY) {
-      const fast = await callGroqFast({
-        messages: [{ role: 'user', content: trimUserContent(prompt) }],
-        temperature: 0.7, top_p: 0.95, max_tokens: 60,
-      });
-      if (fast.ok && fast.text) { text = fast.text; source = 'groq'; }
-    }
-    if (!text && openai) {
-      const r = await openai.chat.completions.create({
-        model: 'gpt-4o-mini',
-        messages: [{ role: 'user', content: prompt }],
-        temperature: 0.8, top_p: 0.95, max_tokens: 60,
-      });
-      text = r?.choices?.[0]?.message?.content?.trim?.() || '';
-      source = text ? 'openai' : 'local';
-    }
+    // RACE (deadline) — brak lokalnego banku jako fallbacku.
+    const racers = [];
+    if (process.env.GROQ_API_KEY) {
+      racers.push(groqChat({
+        messages: [{ role: 'user', content: trimUserContent(prompt) }],
+        temperature: 0.7, top_p: 0.95, max_tokens: 60,
+      }));
+    }
+    if (openai) {
+      racers.push((async () => {
+        const t0 = now();
+        const r = await openai.chat.completions.create({
+          model: 'gpt-4o-mini',
+          messages: [{ role: 'user', content: prompt }],
+          temperature: 0.8, top_p: 0.95, max_tokens: 60,
+        });
+        const txt = r?.choices?.[0]?.message?.content?.trim?.() || '';
+        if (!txt) throw new Error('OPENAI_EMPTY');
+        return { provider: 'openai', text: txt, latency_ms: Math.round(now() - t0) };
+      })());
+    }
+
+    let text = '', source = '';
+    try {
+      const winner = await withDeadline(Promise.any(racers), DEADLINE_MS);
+      text = winner.text; source = winner.provider;
+    } catch (e) {
+      const timedOut = String(e?.message || e) === 'DEADLINE_EXCEEDED';
+      if (timedOut) return res.status(504).json({ ok: false, error: 'DEADLINE_EXCEEDED', timed_out: true, level, language });
+      throw e;
+    }
 
-    if (!text) {
-      const list = bankByLevel(level);
-      text = pick(list);
-      source = 'local';
-    }
+    if (!text) throw new Error('EMPTY_GENERATION');
 
     // zdejmij ewentualne cudzysłowy/ozdobniki
     text = String(text).replace(/^["'„”]+|["'„”]+$/g, '').trim();
 
-    return res.json({ ok: true, text, level, language, source });
+    return res.json({ ok: true, text, level, language, source });
   } catch (err) {
     console.error('agent/generate-text error:', err);
-    const lvl = req?.body?.level || 'A1';
-    const list = bankByLevel(lvl);
-    return res
-      .status(200)
-      .json({ ok: true, text: pick(list), level: lvl, language: req?.body?.language || 'pl', source: 'local-fallback' });
+    const timedOut = String(err?.message || err) === 'DEADLINE_EXCEEDED';
+    if (timedOut) return res.status(504).json({ ok: false, error: 'DEADLINE_EXCEEDED', timed_out: true });
+    return res.status(502).json({ ok: false, error: String(err?.message || err) });
   }
 });
 
@@ -351,6 +407,28 @@
   }
 });
 
 /* ===================== START ===================== */
+async function prewarmOnce() {
+  try {
+    if (process.env.GROQ_API_KEY) {
+      // króciutki „ping” do Groq – bez znaczenia treść, ważne rozgrzanie połączenia
+      await groqChat({ messages: [{ role: 'user', content: 'ping' }], max_tokens: 8, temperature: 0.0 });
+    }
+    if (BASE_URL) {
+      await fetch(`${BASE_URL}/health`, { agent: keepAliveAgent, headers: { Connection: 'keep-alive' } }).catch(()=>{});
+    }
+  } catch { /* noop */ }
+}
+
 app.listen(PORT, () => {
   console.log(`🚀 Backend działa na http://localhost:${PORT}`);
-  console.log(`🎧 Groq ${groq ? 'podłączony' : 'OFF'}`);
+  console.log(`🎧 Groq ${groq ? 'podłączony' : 'OFF'} (model=${GROQ_MODEL})`);
   console.log(`🤖 OpenAI ${openai ? 'podłączony' : 'OFF'}`);
+  // Anti-sleep + pre-warm
+  prewarmOnce();
+  if (PREWARM_EVERY_MIN > 0) {
+    setInterval(prewarmOnce, PREWARM_EVERY_MIN * 60_000);
+    console.log(`🛌 Anti-sleep: ping co ${PREWARM_EVERY_MIN} min${BASE_URL ? ` → ${BASE_URL}/health` : ''}`);
+  }
 });

