#!/usr/bin/env python3
"""박스 라벨링 + 자동 검출 올인원 도구 (로컬 전용, 파일 하나).

실행:
    python labeler.py --data dataset --model ./weights/yolo.pt

    dataset/images/*.jpg   프레임 (images/ 가 없으면 --data 폴더 자체를 프레임 폴더로 본다)
    dataset/labels/*.txt   YOLO 형식 라벨(자동 생성), classes.txt, _status.json(확인 완료 표시)

동작:
    이 컴퓨터(127.0.0.1)에서만 접속되는 작은 서버를 띄우고 브라우저로 화면을 연다.
    프레임과 라벨은 이 컴퓨터의 폴더를 직접 읽고 쓰며, 인터넷으로 나가지 않는다.
    --model 을 주면 화면의 [자동 검출](P)과 "열 때 자동"이 켜진다. 없으면 라벨링 도구로만 동작한다.

옵션:
    --map coco   COCO 사전학습 모델(기본): car->차, motorcycle->오토바이, person->보행자, bus->버스, truck->트럭
    --map ours   미세조정으로 0 car, 1 motorcycle, 2 pedestrian, 3 bus, 4 truck 으로 학습한 모델: 번호를 그대로 쓴다
    --imgsz 1280 멀리 있는 작은 객체를 잡으려면 크게. --device 는 cpu, 0 등.
    --model 은 이미 내려받은 가중치 '파일 경로'만 받는다(이름만 주면 인터넷에서 받으려 하므로 막는다).

주의: 자동 박스는 정답이 아니라 출발점이다. 사람이 모두 확인해 '확인 완료'(Enter)로 바꿔야 평가에 쓴다.
"""
import argparse
import hmac
import json
import mimetypes
import os
import re
import secrets
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

CLASSES = ["car", "motorcycle", "pedestrian", "bus", "truck"]
# COCO id -> 우리 클래스 id
COCO_TO_OURS = {0: 2, 2: 0, 3: 1, 5: 3, 7: 4}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_BODY = 1_000_000
CSP = ("default-src 'none'; connect-src 'self'; img-src blob:; "
       "style-src 'unsafe-inline'; script-src 'unsafe-inline'")

HTML = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<!-- 이 페이지는 이 컴퓨터의 서버(127.0.0.1)와만 통신한다: 아래 정책이 그 밖의 모든 요청을 브라우저 수준에서 막는다. -->
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'self'; img-src blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>박스 라벨링 + 자동 검출 (로컬 전용)</title>
<style>
  :root { --bg:#16181d; --panel:#1e2128; --line:#2f343e; --txt:#e6e8ec; --dim:#8b93a1; --acc:#4f8cff; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--txt); font:13px/1.45 system-ui,-apple-system,"Malgun Gothic",sans-serif; }
  #app { display:grid; grid-template-rows:auto 1fr; height:100%; }
  header { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:6px 10px; background:var(--panel); border-bottom:1px solid var(--line); }
  button { background:#2a2f39; color:var(--txt); border:1px solid var(--line); border-radius:6px; padding:5px 10px; cursor:pointer; font:inherit; }
  button:hover { border-color:var(--acc); }
  button.on { background:var(--acc); border-color:var(--acc); color:#fff; }
  .sep { width:1px; height:22px; background:var(--line); }
  .dim { color:var(--dim); }
  #status { margin-left:auto; color:var(--dim); }
  #main { display:grid; grid-template-columns:200px 1fr 250px; min-height:0; }
  aside { background:var(--panel); overflow:auto; min-height:0; }
  #left { border-right:1px solid var(--line); }
  #right { border-left:1px solid var(--line); padding:8px; }
  #stage { position:relative; min-width:0; min-height:0; overflow:hidden; }
  canvas { position:absolute; left:0; top:0; touch-action:none; }
  .row { padding:3px 8px; cursor:pointer; display:flex; gap:6px; align-items:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .row:hover { background:#262b34; }
  .row.cur { background:#2b3a57; }
  .dot { width:8px; height:8px; border-radius:50%; flex:none; background:#434a57; }
  .dot.lab { background:#e0a53c; } .dot.chk { background:#3ec27a; }
  h4 { margin:10px 0 4px; font-size:12px; color:var(--dim); font-weight:600; }
  .cls { display:flex; gap:4px; flex-wrap:wrap; }
  .cls button { display:flex; gap:6px; align-items:center; }
  .sw { width:10px; height:10px; border-radius:2px; display:inline-block; }
  .brow { display:flex; gap:6px; align-items:center; padding:2px 4px; border-radius:4px; cursor:pointer; }
  .brow.sel { background:#2b3a57; }
  .brow .x { margin-left:auto; color:var(--dim); }
  .brow .x:hover { color:#ff6b6b; }
  kbd { background:#2a2f39; border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:11px; }
  .help { color:var(--dim); font-size:12px; line-height:1.8; }
  input[type=range] { width:90px; vertical-align:middle; }
  #empty { position:absolute; inset:0; display:none; align-items:center; justify-content:center; text-align:center; color:var(--dim); padding:30px; pointer-events:none; }
</style>
</head>
<body>
<div id="app">
  <header>
    <button id="btnDet" title="P">자동 검출</button>
    <span class="dim">신뢰도</span><input id="conf" type="range" min="5" max="80" value="25"><span id="confv" class="dim">0.25</span>
    <label class="dim" style="display:flex;gap:4px;align-items:center"><input id="autoDet" type="checkbox" checked>열 때 자동</label>
    <span class="sep"></span>
    <span class="dim">밝기</span><input id="bri" type="range" min="50" max="250" value="100">
    <span class="dim">대비</span><input id="con" type="range" min="50" max="250" value="100">
    <span class="sep"></span>
    <button id="btnFit" title="F">맞춤</button>
    <button id="btn100" title="1">100%</button>
    <button id="btnTags" class="on" title="H">이름표</button>
    <button id="btnUndo" title="Ctrl+Z">되돌리기</button>
    <span id="status">불러오는 중…</span>
  </header>
  <div id="main">
    <aside id="left"><div class="dim" style="padding:6px 8px;font-size:11px;line-height:1.6"><span class="dot" style="display:inline-block"></span> 미작업 <span class="dot lab" style="display:inline-block"></span> 자동/작업 중 <span class="dot chk" style="display:inline-block"></span> 확인 완료</div><div id="list"></div></aside>
    <div id="stage">
      <canvas id="cv"></canvas>
      <div id="empty"><div><div id="emptyMsg" style="font-size:16px;color:var(--txt)">불러오는 중…</div></div></div>
    </div>
    <aside id="right">
      <h4>클래스 (숫자 키)</h4>
      <div class="cls" id="clsBtns"></div>
      <h4 id="boxHead">이 프레임의 박스</h4>
      <div id="boxList"></div>
      <h4>단축키</h4>
      <div class="help">
        드래그 <span class="dim">새 박스</span><br>
        클릭 <span class="dim">박스 선택</span> · 선택 후 드래그 <span class="dim">이동</span> · 손잡이 <span class="dim">크기</span><br>
        <kbd>1</kbd>~<kbd>9</kbd> <span class="dim">클래스(선택 박스가 있으면 바꿈)</span><br>
        <kbd>Del</kbd> <span class="dim">삭제</span> · <kbd>Esc</kbd> <span class="dim">선택 해제</span> · <kbd>Tab</kbd> <span class="dim">다음 박스</span><br>
        휠 <span class="dim">확대/축소</span> · <kbd>Space</kbd>+드래그/우클릭 드래그 <span class="dim">이동</span><br>
        <kbd>A</kbd>/<kbd>←</kbd> <kbd>D</kbd>/<kbd>→</kbd> <span class="dim">이전/다음</span><br>
        <kbd>Enter</kbd> <span class="dim">확인 완료 표시 후 다음</span> · <kbd>Shift</kbd>+<kbd>Enter</kbd> <span class="dim">확인 표시 토글</span><br>
        <kbd>N</kbd> <span class="dim">다음 미확인</span> · <kbd>H</kbd> <span class="dim">이름표</span><br>
        <kbd>P</kbd> <span class="dim">자동 검출(없는 박스만 추가)</span><br>
        <kbd>Ctrl</kbd>+<kbd>Z</kbd> <span class="dim">되돌리기</span> · 변경은 자동 저장
      </div>
    </aside>
  </div>
</div>

<script>
'use strict';
const $ = id => document.getElementById(id);
const DEFAULT_CLASSES = ['car', 'motorcycle', 'pedestrian', 'bus', 'truck'];
const KO = { car: '차', motorcycle: '오토바이', pedestrian: '보행자', bus: '버스', truck: '트럭' };
const COLORS = ['#3ec27a', '#f0a63a', '#4fb4ff', '#e879f9', '#ff5d73', '#a3e635', '#fb923c', '#2dd4bf'];
const IMG_RE = /\.(jpe?g|png|bmp|webp)$/i;

const TOKEN = '__TOKEN__';
let detectOn = false, autoDetect = true, dataName = '';
let classes = DEFAULT_CLASSES.slice();
let files = [];                 // 이미지 파일 이름(정렬됨)
const hasLabel = new Set();     // 라벨 txt가 있는 이미지 이름
let checked = {};               // 이름 -> true (확인 완료)
let idx = -1, bmp = null, iw = 0, ih = 0, loadTok = 0;
let boxes = [], hist = [], sel = -1, curClass = 0;
let view = { s: 1, ox: 0, oy: 0 };
let showTags = true, dirty = false, saveTimer = null, saving = Promise.resolve(), pendingSaves = 0;
let drag = null, spaceDown = false, mouse = null;

const cv = $('cv'), ctx = cv.getContext('2d');
const stem = n => n.replace(/\.[^.]+$/, '');
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const clsName = c => classes[c] !== undefined ? classes[c] : 'class' + c;
const clsColor = c => COLORS[c % COLORS.length];

/* ---------- 서버(127.0.0.1)와 통신 ---------- */
const api = (path, opt = {}) => fetch(path, { ...opt, headers: { 'X-Token': TOKEN, ...(opt.headers || {}) } });
async function readLabel(stemName) {
  const r = await api('/api/label/' + encodeURIComponent(stemName));
  if (r.status === 404 || r.status === 204) return null;
  if (!r.ok) throw new Error(await r.text());
  return await r.text();
}
async function writeLabel(stemName, text) {
  const r = await api('/api/label/' + encodeURIComponent(stemName), { method: 'PUT', body: text });
  if (!r.ok) throw new Error(await r.text());
}
async function init() {
  try {
    const r = await api('/api/info');
    if (!r.ok) throw new Error(await r.text());
    const info = await r.json();
    files = info.files; classes = info.classes; checked = info.checked || {};
    hasLabel.clear(); info.labeled.forEach(n => hasLabel.add(n));
    detectOn = !!info.detect; dataName = info.data;
  } catch (e) { $('emptyMsg').textContent = '서버에 연결하지 못했습니다: ' + e.message; return; }
  const bd = $('btnDet');
  if (!detectOn) { bd.disabled = true; bd.title = '서버를 --model 없이 실행했거나 ultralytics를 찾지 못했습니다'; bd.style.opacity = .5; $('autoDet').disabled = true; }
  curClass = 0; buildClassButtons(); renderList();
  if (!files.length) { $('emptyMsg').textContent = '이미지가 없습니다'; return; }
  $('empty').style.display = 'none';
  const firstUnchecked = files.findIndex(n => !checked[n]);
  await go(firstUnchecked >= 0 ? firstUnchecked : 0);
}
const iou = (a, b) => {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1), u = a.w * a.h + b.w * b.h - i;
  return u > 0 ? i / u : 0;
};
/* mode 'auto': 박스가 하나도 없을 때만 채움 / 'manual': 기존 박스와 겹치지 않는 박스만 추가 */
async function runDetect(mode) {
  if (!detectOn || idx < 0) return;
  const tok = loadTok, name = files[idx];
  setStatus('검출 중… ' + name);
  let res;
  try {
    const r = await api('/api/predict', { method: 'POST', body: JSON.stringify({ name, conf: Number($('conf').value) / 100 }) });
    if (!r.ok) throw new Error(await r.text());
    res = await r.json();
  } catch (e) { updateStatus(); setStatus('검출 실패: ' + e.message); return; }
  if (tok !== loadTok) return;                          // 검출 중 다른 프레임으로 이동함
  let add = res.boxes.map(b => ({ c: b.c, x: (b.cx - b.w / 2) * iw, y: (b.cy - b.h / 2) * ih, w: b.w * iw, h: b.h * ih }));
  if (mode === 'auto') { if (boxes.length) { updateStatus(); return; } }
  else add = add.filter(f => !boxes.some(b => iou(b, f) > 0.4));
  if (add.length || mode === 'auto') { pushHist(); boxes = boxes.concat(add); changed(); }
  updateStatus();
  setStatus($('status').textContent + ` · 검출 ${add.length}개 추가`);
}

/* ---------- 라벨 변환 (YOLO txt <-> 픽셀 박스) ---------- */
function parseLabels(text) {
  const out = [];
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (t.length !== 5) continue;                       // 다각형 등 박스가 아닌 줄은 무시
    const [c, cx, cy, w, h] = t.map(Number);
    if ([c, cx, cy, w, h].some(v => !isFinite(v))) continue;
    out.push({ c: Math.round(c), x: (cx - w / 2) * iw, y: (cy - h / 2) * ih, w: w * iw, h: h * ih });
  }
  return out;
}
function toText(bs, W, H) {
  return bs.map(b => {
    const cx = clamp((b.x + b.w / 2) / W, 0, 1), cy = clamp((b.y + b.h / 2) / H, 0, 1);
    const w = clamp(b.w / W, 0, 1), h = clamp(b.h / H, 0, 1);
    return `${b.c} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
  }).join('\n') + (bs.length ? '\n' : '');
}

/* ---------- 저장 (변경 후 자동 저장) ---------- */
function scheduleSave() { clearTimeout(saveTimer); saveTimer = setTimeout(() => { saveCur(); }, 400); }
function saveCur() {
  clearTimeout(saveTimer);
  if (idx < 0 || !dirty) return saving;
  const name = files[idx], text = toText(boxes, iw, ih);
  dirty = false;
  pendingSaves++;
  saving = saving.then(async () => {
    try { await writeLabel(stem(name), text); hasLabel.add(stem(name)); renderList(); }
    catch (e) { alert('저장 실패: ' + e.message); dirty = true; }
    finally { pendingSaves--; }
  });
  return saving;
}
async function flushSave() { await saveCur(); await saving; }
async function saveStatus() { try { const r = await api('/api/status', { method: 'PUT', body: JSON.stringify(checked) }); if (!r.ok) throw new Error(await r.text()); } catch (e) { alert('상태 저장 실패: ' + e.message); } }

/* ---------- 이미지 이동 ---------- */
async function go(i) {
  if (i < 0 || i >= files.length) return;
  await flushSave();
  const tok = ++loadTok;
  const name = files[i];
  const ir = await api('/api/image/' + encodeURIComponent(name));
  if (!ir.ok) { alert('이미지를 불러오지 못했습니다: ' + name); return; }
  const b = await createImageBitmap(await ir.blob());
  const t = await readLabel(stem(name));
  if (tok !== loadTok) { b.close(); return; }
  if (bmp) bmp.close();
  bmp = b; iw = b.width; ih = b.height; idx = i;
  boxes = t ? parseLabels(t) : [];
  hist = []; sel = -1; dirty = false;
  fit();
  renderList(); refreshSide(); updateStatus(); draw();
  if (t === null && autoDetect && detectOn && !checked[name]) runDetect('auto');
}

/* ---------- 좌표 변환 / 보기 ---------- */
const toScreen = (x, y) => [x * view.s + view.ox, y * view.s + view.oy];
const toImg = (sx, sy) => [(sx - view.ox) / view.s, (sy - view.oy) / view.s];
function stageSize() { const r = $('stage').getBoundingClientRect(); return [r.width, r.height]; }
function fit() {
  if (!bmp) return;
  const [W, H] = stageSize();
  view.s = Math.min(W / iw, H / ih);
  view.ox = (W - iw * view.s) / 2; view.oy = (H - ih * view.s) / 2;
}
function zoomAt(f, sx, sy) {
  const [ix, iy] = toImg(sx, sy);
  view.s = clamp(view.s * f, 0.05, 60);
  view.ox = sx - ix * view.s; view.oy = sy - iy * view.s;
  draw();
}
function resizeCanvas() {
  const [W, H] = stageSize(), d = devicePixelRatio || 1;
  cv.width = Math.round(W * d); cv.height = Math.round(H * d);
  cv.style.width = W + 'px'; cv.style.height = H + 'px';
  draw();
}

/* ---------- 그리기 ---------- */
function handlePoints(b) {
  const [x1, y1] = toScreen(b.x, b.y), [x2, y2] = toScreen(b.x + b.w, b.y + b.h), mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  return [['nw', x1, y1], ['n', mx, y1], ['ne', x2, y1], ['e', x2, my], ['se', x2, y2], ['s', mx, y2], ['sw', x1, y2], ['w', x1, my]]
    .map(([k, x, y]) => ({ k, x, y }));
}
function drawBox(b, i, tmp) {
  const [x, y] = toScreen(b.x, b.y), w = b.w * view.s, h = b.h * view.s, col = clsColor(b.c), isSel = i === sel;
  ctx.lineWidth = isSel ? 2.5 : 1.5; ctx.strokeStyle = col;
  ctx.fillStyle = col + (isSel ? '33' : '14');
  ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
  if (showTags && !tmp) {
    const label = KO[clsName(b.c)] || clsName(b.c);
    ctx.font = '11px system-ui,"Malgun Gothic",sans-serif';
    const tw = ctx.measureText(label).width + 6;
    ctx.fillStyle = col; ctx.fillRect(x, y - 14, tw, 14);
    ctx.fillStyle = '#10131a'; ctx.fillText(label, x + 3, y - 3);
  }
  if (isSel) {
    ctx.fillStyle = '#fff'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
    for (const p of handlePoints(b)) { ctx.fillRect(p.x - 4, p.y - 4, 8, 8); ctx.strokeRect(p.x - 4, p.y - 4, 8, 8); }
  }
}
function draw() {
  const d = devicePixelRatio || 1, W = cv.width / d, H = cv.height / d;
  ctx.setTransform(d, 0, 0, d, 0, 0);
  ctx.fillStyle = '#0d0f13'; ctx.fillRect(0, 0, W, H);
  if (!bmp) return;
  ctx.imageSmoothingEnabled = view.s < 2.5;
  ctx.filter = `brightness(${$('bri').value}%) contrast(${$('con').value}%)`;
  ctx.drawImage(bmp, view.ox, view.oy, iw * view.s, ih * view.s);
  ctx.filter = 'none';
  boxes.forEach((b, i) => drawBox(b, i));
  if (drag && drag.mode === 'draw') drawBox(drag.box, -1, true);
  if (mouse && !(drag && (drag.mode === 'pan'))) {
    ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(mouse[0] + .5, 0); ctx.lineTo(mouse[0] + .5, H); ctx.moveTo(0, mouse[1] + .5); ctx.lineTo(W, mouse[1] + .5); ctx.stroke();
    ctx.setLineDash([]);
  }
}

/* ---------- 마우스 ---------- */
const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
function boxAt(ix, iy) {
  const pad = 3 / view.s; let best = -1, ba = Infinity;
  boxes.forEach((b, i) => {
    if (ix >= b.x - pad && ix <= b.x + b.w + pad && iy >= b.y - pad && iy <= b.y + b.h + pad) { const a = b.w * b.h; if (a < ba) { ba = a; best = i; } }
  });
  return best;
}
function handleAt(sx, sy) {
  if (sel < 0) return null;
  for (const p of handlePoints(boxes[sel])) if (Math.abs(sx - p.x) <= 7 && Math.abs(sy - p.y) <= 7) return p.k;
  return null;
}
function pushHist() { hist.push(JSON.stringify(boxes)); if (hist.length > 100) hist.shift(); }
function changed() { dirty = true; refreshSide(); draw(); scheduleSave(); }

cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('pointerdown', e => {
  if (!bmp) return;
  cv.setPointerCapture(e.pointerId);
  const [sx, sy] = pos(e), [ix, iy] = toImg(sx, sy);
  if (e.button === 1 || e.button === 2 || spaceDown) { drag = { mode: 'pan', sx0: sx, sy0: sy, ox0: view.ox, oy0: view.oy }; return; }
  if (e.button !== 0) return;
  const hk = handleAt(sx, sy);
  if (hk) { pushHist(); drag = { mode: 'resize', key: hk, b0: { ...boxes[sel] }, moved: false }; return; }
  drag = { mode: 'pending', sx0: sx, sy0: sy, ix0: clamp(ix, 0, iw), iy0: clamp(iy, 0, ih), hit: boxAt(ix, iy) };
});
cv.addEventListener('pointermove', e => {
  const [sx, sy] = pos(e); mouse = [sx, sy];
  if (!drag) { draw(); return; }
  const [ix, iy] = toImg(sx, sy);
  if (drag.mode === 'pan') { view.ox = drag.ox0 + (sx - drag.sx0); view.oy = drag.oy0 + (sy - drag.sy0); }
  else if (drag.mode === 'pending') {
    if (Math.hypot(sx - drag.sx0, sy - drag.sy0) > 4) {
      if (drag.hit >= 0 && drag.hit === sel) { pushHist(); drag = { mode: 'move', i: sel, b0: { ...boxes[sel] }, ix0: drag.ix0, iy0: drag.iy0, moved: false }; }
      else drag = { mode: 'draw', ix0: drag.ix0, iy0: drag.iy0, box: { c: curClass, x: drag.ix0, y: drag.iy0, w: 0, h: 0 } };
    }
  }
  if (drag.mode === 'draw') {
    const x2 = clamp(ix, 0, iw), y2 = clamp(iy, 0, ih);
    drag.box = { c: curClass, x: Math.min(drag.ix0, x2), y: Math.min(drag.iy0, y2), w: Math.abs(x2 - drag.ix0), h: Math.abs(y2 - drag.iy0) };
  } else if (drag.mode === 'move') {
    const b = boxes[drag.i], b0 = drag.b0;
    b.x = clamp(b0.x + (ix - drag.ix0), 0, iw - b0.w); b.y = clamp(b0.y + (iy - drag.iy0), 0, ih - b0.h); drag.moved = true;
  } else if (drag.mode === 'resize') {
    const b0 = drag.b0, k = drag.key;
    let x1 = b0.x, y1 = b0.y, x2 = b0.x + b0.w, y2 = b0.y + b0.h;
    const cx = clamp(ix, 0, iw), cy = clamp(iy, 0, ih);
    if (k.includes('w')) x1 = cx; if (k.includes('e')) x2 = cx;
    if (k.includes('n')) y1 = cy; if (k.includes('s')) y2 = cy;
    Object.assign(boxes[sel], { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) }); drag.moved = true;
  }
  draw();
});
cv.addEventListener('pointerup', e => {
  if (!drag) return;
  const d = drag; drag = null;
  if (d.mode === 'pending') { sel = d.hit; if (sel >= 0) curClass = boxes[sel].c; refreshSide(); buildClassButtons(); }
  else if (d.mode === 'draw') {
    if (d.box.w * view.s >= 3 && d.box.h * view.s >= 3) { pushHist(); boxes.push(d.box); sel = boxes.length - 1; changed(); }
  } else if (d.mode === 'move' || d.mode === 'resize') {
    if (d.moved) changed(); else hist.pop();
  }
  draw();
});
cv.addEventListener('pointerleave', () => { mouse = null; draw(); });
cv.addEventListener('wheel', e => { e.preventDefault(); const [sx, sy] = pos(e); zoomAt(Math.pow(1.0018, -e.deltaY), sx, sy); }, { passive: false });

/* ---------- 오른쪽/왼쪽 패널 ---------- */
function buildClassButtons() {
  const box = $('clsBtns'); box.innerHTML = '';
  classes.forEach((n, i) => {
    const b = document.createElement('button');
    if (i === curClass) b.className = 'on';
    b.innerHTML = `<span class="sw" style="background:${clsColor(i)}"></span>${i + 1} ${KO[n] || n}`;
    b.addEventListener('click', () => setClass(i));
    box.appendChild(b);
  });
}
function setClass(c) {
  if (c >= classes.length) return;
  curClass = c;
  if (sel >= 0 && boxes[sel].c !== c) { pushHist(); boxes[sel].c = c; changed(); }
  buildClassButtons(); draw();
}
function refreshSide() {
  const bl = $('boxList'); bl.innerHTML = '';
  $('boxHead').textContent = `이 프레임의 박스 (${boxes.length})`;
  boxes.forEach((b, i) => {
    const r = document.createElement('div'); r.className = 'brow' + (i === sel ? ' sel' : '');
    r.innerHTML = `<span class="sw" style="background:${clsColor(b.c)}"></span>${KO[clsName(b.c)] || clsName(b.c)} <span class="dim">${Math.round(b.w)}×${Math.round(b.h)}px</span><span class="x" title="삭제">✕</span>`;
    r.addEventListener('click', ev => {
      if (ev.target.classList.contains('x')) { pushHist(); boxes.splice(i, 1); sel = -1; changed(); return; }
      sel = i; curClass = b.c; buildClassButtons(); refreshSide(); draw();
    });
    bl.appendChild(r);
  });
}
function renderList() {
  const el = $('list'); el.innerHTML = '';
  files.forEach((n, i) => {
    const r = document.createElement('div'); r.className = 'row' + (i === idx ? ' cur' : '');
    const st = checked[n] ? 'chk' : (hasLabel.has(stem(n)) ? 'lab' : '');
    r.innerHTML = `<span class="dot ${st}"></span>${n}`;
    r.title = n;
    r.addEventListener('click', () => go(i));
    el.appendChild(r);
  });
  const cur = el.children[idx]; if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: 'nearest' });
  updateStatus();
}
function setStatus(t) { $('status').textContent = t; }
function updateStatus() {
  if (!files.length) return;
  const done = files.filter(n => checked[n]).length;
  setStatus(`${idx + 1}/${files.length} · ${files[idx] || ''} · ${iw}×${ih} · 확인 완료 ${done}/${files.length}`);
}

/* ---------- 키보드 ---------- */
async function toggleChecked(andNext) {
  if (idx < 0) return;
  const n = files[idx];
  if (checked[n] && !andNext) delete checked[n]; else checked[n] = true;
  dirty = true;                       // 박스가 0개여도 빈 txt를 만들어 "배경 프레임"으로 남긴다
  await flushSave(); await saveStatus(); renderList();
  if (andNext) await go(Math.min(idx + 1, files.length - 1));
}
function nextUnchecked() {
  for (let k = 1; k <= files.length; k++) { const j = (idx + k) % files.length; if (!checked[files[j]]) return go(j); }
}
window.addEventListener('keydown', e => {
  if (e.target && e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  const k = e.key;
  if (k === ' ') { spaceDown = true; cv.style.cursor = 'grab'; e.preventDefault(); return; }
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 's') { e.preventDefault(); flushSave(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (/^[1-9]$/.test(k)) { setClass(Number(k) - 1); return; }
  if (k === 'Delete' || k === 'Backspace') { if (sel >= 0) { pushHist(); boxes.splice(sel, 1); sel = -1; changed(); } e.preventDefault(); return; }
  if (k === 'Escape') { sel = -1; refreshSide(); draw(); return; }
  if (k === 'Tab') { e.preventDefault(); if (boxes.length) { sel = (sel + 1) % boxes.length; curClass = boxes[sel].c; buildClassButtons(); refreshSide(); draw(); } return; }
  if (k === 'a' || k === 'A' || k === 'ArrowLeft') { e.preventDefault(); go(idx - 1); return; }
  if (k === 'd' || k === 'D' || k === 'ArrowRight') { e.preventDefault(); go(idx + 1); return; }
  if (k === 'Enter') { e.preventDefault(); toggleChecked(!e.shiftKey); return; }
  if (k === 'n' || k === 'N') { nextUnchecked(); return; }
  if (k === 'f' || k === 'F') { fit(); draw(); return; }
  if (k === 'p' || k === 'P') { runDetect('manual'); return; }
  if (k === 'h' || k === 'H') { toggleTags(); return; }
});
window.addEventListener('keyup', e => { if (e.key === ' ') { spaceDown = false; cv.style.cursor = 'crosshair'; } });
function undo() { if (!hist.length) return; boxes = JSON.parse(hist.pop()); sel = -1; changed(); }
function toggleTags() { showTags = !showTags; $('btnTags').classList.toggle('on', showTags); draw(); }

$('btnDet').addEventListener('click', () => runDetect('manual'));
$('conf').addEventListener('input', () => { $('confv').textContent = (Number($('conf').value) / 100).toFixed(2); });
$('conf').addEventListener('change', e => e.target.blur());
$('autoDet').addEventListener('change', e => { autoDetect = e.target.checked; e.target.blur(); });
$('btnFit').addEventListener('click', () => { fit(); draw(); });
$('btn100').addEventListener('click', () => { const [W, H] = stageSize(); zoomAt(1 / view.s, W / 2, H / 2); });
$('btnTags').addEventListener('click', toggleTags);
$('btnUndo').addEventListener('click', undo);
for (const id of ['bri', 'con']) { $(id).addEventListener('input', draw); $(id).addEventListener('change', e => e.target.blur()); }
window.addEventListener('resize', () => { resizeCanvas(); });
window.addEventListener('beforeunload', e => { if (dirty || pendingSaves > 0) { e.preventDefault(); e.returnValue = ''; } });
cv.style.cursor = 'crosshair';
$('empty').style.display = 'flex';
buildClassButtons();
resizeCanvas();
init();
</script>
</body>
</html>
'''


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def valid_label_text(text: str) -> bool:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        t = line.split()
        if len(t) != 5:
            return False
        try:
            c, *vals = [float(v) for v in t]
        except ValueError:
            return False
        if c < 0 or c != int(c) or any(v < 0 or v > 1 for v in vals):
            return False
    return True


class Detector:
    """YOLO를 처음 쓸 때 한 번만 불러오고, 동시에 한 장씩만 추론한다."""

    def __init__(self, model_path: Path, imgsz: int, device, mapping: str):
        self.model_path, self.imgsz, self.device, self.mapping = model_path, imgsz, device, mapping
        self.model = None
        self.lock = threading.Lock()

    def _load(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))

    def predict(self, image_path: Path, conf: float):
        with self.lock:
            self._load()
            kw = dict(conf=conf, imgsz=self.imgsz, verbose=False)
            if self.mapping == "coco":
                kw["classes"] = sorted(COCO_TO_OURS)
            if self.device is not None:
                kw["device"] = self.device
            res = self.model.predict(str(image_path), **kw)[0]
            out = []
            for b in res.boxes:
                cid = int(b.cls)
                if self.mapping == "coco":
                    cid = COCO_TO_OURS.get(cid)
                    if cid is None:
                        continue
                cx, cy, w, h = (min(max(float(v), 0.0), 1.0) for v in b.xywhn[0])
                out.append({"c": cid, "cx": cx, "cy": cy, "w": w, "h": h})
            return out


class App:
    def __init__(self, data: Path, detector, token: str):
        self.images = data / "images" if (data / "images").is_dir() else data
        self.labels = data / "labels"
        self.labels.mkdir(exist_ok=True)
        self.detector = detector
        self.token = token
        self.status_lock = threading.Lock()
        self.files = sorted((p.name for p in self.images.iterdir()
                             if p.is_file() and p.suffix.lower() in IMG_EXT), key=natural_key)
        self.fileset = set(self.files)
        self.stems = {Path(n).stem for n in self.files}
        cf = self.labels / "classes.txt"
        if not cf.exists():
            atomic_write(cf, ("\n".join(CLASSES) + "\n").encode("utf-8"))
        self.classes = [s.strip() for s in cf.read_text(encoding="utf-8").splitlines() if s.strip()] or CLASSES[:]

    def info(self):
        labeled = sorted(s for s in self.stems if (self.labels / (s + ".txt")).exists())
        sf = self.labels / "_status.json"
        try:
            checked = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
        except (OSError, ValueError):
            checked = {}
        return {"files": self.files, "classes": self.classes, "labeled": labeled, "checked": checked,
                "detect": self.detector is not None, "data": self.images.parent.name or str(self.images)}


class Handler(BaseHTTPRequestHandler):
    server_version = "BoxLabeler"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # 조용히
        pass

    @property
    def app(self) -> App:
        return self.server.app

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        if code != 204:
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        if code != 204:
            self.wfile.write(body)

    def _guard(self, need_token: bool) -> bool:
        if self.headers.get("Host", "") not in self.server.allowed_hosts:      # DNS 재바인딩 방지
            self._send(403, "forbidden host")
            return False
        if need_token and not hmac.compare_digest(self.headers.get("X-Token", ""), self.app.token):
            self._send(403, "forbidden")
            return False
        return True

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self._send(413, "too large")
            return None
        return self.rfile.read(n)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if self._guard(False):
                self._send(200, HTML.replace("__TOKEN__", self.app.token), "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            self._send(204)
            return
        if not path.startswith("/api/") or not self._guard(True):
            if not path.startswith("/api/"):
                self._send(404, "not found")
            return
        parts = path.split("/", 3)          # ['', 'api', kind, key]
        kind = parts[2]
        key = unquote(parts[3]) if len(parts) > 3 else ""
        if kind == "info":
            self._send(200, json.dumps(self.app.info(), ensure_ascii=False), "application/json; charset=utf-8")
        elif kind == "image" and key in self.app.fileset:
            ctype = mimetypes.guess_type(key)[0] or "application/octet-stream"
            self._send(200, (self.app.images / key).read_bytes(), ctype)
        elif kind == "label" and key in self.app.stems:
            f = self.app.labels / (key + ".txt")
            if f.exists():
                self._send(200, f.read_bytes())
            else:
                self._send(204)               # 라벨이 아직 없음(오류 아님)
        else:
            self._send(404, "not found")

    def do_PUT(self):
        path = urlparse(self.path).path
        if not self._guard(True):
            return
        body = self._body()
        if body is None:
            return
        if path == "/api/status":
            try:
                data = json.loads(body.decode("utf-8"))
                assert isinstance(data, dict) and all(k in self.app.fileset and v is True for k, v in data.items())
            except Exception:
                self._send(400, "bad status")
                return
            with self.app.status_lock:
                atomic_write(self.app.labels / "_status.json", json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"))
            self._send(200, "ok")
        elif path.startswith("/api/label/"):
            key = unquote(path[len("/api/label/"):])
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                self._send(400, "bad text")
                return
            if key not in self.app.stems or not valid_label_text(text):
                self._send(400, "bad label")
                return
            atomic_write(self.app.labels / (key + ".txt"), text.encode("utf-8"))
            self._send(200, "ok")
        else:
            self._send(404, "not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self._send(404, "not found")
            return
        if not self._guard(True):
            return
        body = self._body()
        if body is None:
            return
        if self.app.detector is None:
            self._send(503, "자동 검출이 꺼져 있습니다(--model 필요)")
            return
        try:
            req = json.loads(body.decode("utf-8"))
            name, conf = req["name"], float(req.get("conf", 0.25))
            assert name in self.app.fileset
        except Exception:
            self._send(400, "bad request")
            return
        try:
            boxes = self.app.detector.predict(self.app.images / name, min(max(conf, 0.01), 0.95))
        except Exception as e:                       # 모델 오류는 화면에 그대로 알린다
            self._send(500, f"{type(e).__name__}: {e}")
            return
        self._send(200, json.dumps({"boxes": boxes}), "application/json; charset=utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="데이터 폴더 (images/ 와 labels/ 를 가짐)")
    ap.add_argument("--model", default=None, help="YOLO 가중치 파일 경로 (없으면 자동 검출 끔)")
    ap.add_argument("--map", choices=["coco", "ours"], default="coco")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_dir():
        print(f"폴더가 없습니다: {data}", file=sys.stderr)
        return 1

    detector = None
    if args.model:
        mp = Path(args.model)
        if not mp.is_file():
            print(f"모델 파일을 찾을 수 없습니다: {mp}\n(이름만 주면 인터넷에서 받으려 하므로, 받아 둔 파일의 경로를 주세요)", file=sys.stderr)
            return 1
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            print("경고: ultralytics 가 없어 자동 검출을 끄고 라벨링 도구로만 실행합니다.", file=sys.stderr)
        else:
            detector = Detector(mp, args.imgsz, args.device, args.map)

    token = secrets.token_urlsafe(24)
    app = App(data, detector, token)
    if not app.files:
        print(f"이미지가 없습니다: {app.images}", file=sys.stderr)
        return 1

    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        print("사용할 수 있는 포트를 찾지 못했습니다. --port 를 바꿔 보세요.", file=sys.stderr)
        return 1
    server.app = app
    server.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    url = f"http://127.0.0.1:{port}/"
    print(f"프레임 {len(app.files)}장 · 이미지 {app.images} · 라벨 {app.labels}")
    print(f"자동 검출: {'켜짐 (' + str(args.model) + ')' if detector else '꺼짐'}")
    print(f"열기: {url}   (종료: Ctrl+C)  — 이 컴퓨터에서만 접속됩니다")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
