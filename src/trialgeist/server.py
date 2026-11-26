"""`trialgeist serve`: zero-cloud local studio.

Standard-library HTTP server bound to localhost.  Drop a PDF, every extracted
cell / KM vertex is overlaid on the rendered page (colour = verdict tier),
and any PROVEN_IMPOSSIBLE finding can be exported as its standalone
verification script.  No external network requests are made (no CDNs, no
registry lookups unless an NCT id is typed in).
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pymupdf

from .scan import Report, scan_pdf, self_verify

STATE: dict[str, dict] = {}
LOCK = threading.Lock()
WORK = Path(tempfile.gettempdir()) / "trialgeist-studio"
SCALE = 1.5

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>TRIALGEIST studio</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;display:flex;height:100vh;color:#111}
#side{width:430px;overflow:auto;border-right:1px solid #ddd;padding:12px;box-sizing:border-box}
#main{flex:1;overflow:auto;background:#f4f4f4;position:relative}
#drop{border:2px dashed #999;padding:18px;text-align:center;border-radius:8px;cursor:pointer}
.f{padding:6px 8px;margin:4px 0;border-radius:6px;cursor:pointer;border:1px solid #ddd;background:#fff}
.f:hover{border-color:#333}.t{font-size:11px;font-weight:600;letter-spacing:.03em}
.PROVEN_IMPOSSIBLE{color:#b00020}.STATISTICALLY_EXTREME{color:#b35c00}.REGISTRY_DISCORDANT{color:#6a1b9a}
.CONSISTENT{color:#1b5e20}.INCONCLUSIVE{color:#555}
.pg{position:relative;margin:16px auto;box-shadow:0 1px 4px #0003;background:#fff}
.box{position:absolute;border:1.5px solid;pointer-events:auto;cursor:pointer}
.box.PROVEN_IMPOSSIBLE{border-color:#b00020;background:#b0002022}.box.CONSISTENT{border-color:#2e7d3288}
.box.STATISTICALLY_EXTREME{border-color:#b35c00;background:#b35c0018}.box.INCONCLUSIVE{border-color:#7778}
.box.sel{outline:3px solid #1565c0}
.v{position:absolute;width:4px;height:4px;margin:-2px 0 0 -2px;border-radius:2px;background:#1565c0}
pre{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;padding:8px;font-size:12px}
button{margin:6px 6px 0 0}
</style></head><body>
<div id="side"><h3 style="margin:4px 0 10px">TRIALGEIST studio</h3>
<div id="drop">Drop a trial PDF here or click to choose<input id="file" type="file" accept="application/pdf" hidden></div>
<p><label>NCT id (optional, queries ClinicalTrials.gov): <input id="nct" size="14"></label></p>
<div id="summary"></div><div id="detail"></div><div id="list"></div></div>
<div id="main"><div id="pages"></div></div>
<script>
const $=s=>document.querySelector(s); let R=null, SCALE=__SCALE__;
const drop=$('#drop'), file=$('#file');
drop.onclick=()=>file.click();
drop.ondragover=e=>{e.preventDefault();drop.style.borderColor='#1565c0'};
drop.ondrop=e=>{e.preventDefault();up(e.dataTransfer.files[0])};
file.onchange=()=>up(file.files[0]);
async function up(f){ if(!f) return; $('#summary').textContent='Scanning '+f.name+' (exact solvers + certificate self-check)...';
  const nct=$('#nct').value.trim(); const r=await fetch('/api/scan'+(nct?'?nct='+encodeURIComponent(nct):''),{method:'POST',body:await f.arrayBuffer()});
  R=await r.json(); if(R.error){$('#summary').textContent=R.error;return} render(); }
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(){
  const c=R.counts; $('#summary').innerHTML=`<p><b>${esc(R.name)}</b>: ${R.n_claims} printed claims checked</p>`+
    Object.entries(c).map(([k,v])=>`<span class="t ${k}">${k}: ${v}</span>`).join('<br>');
  const order=['PROVEN_IMPOSSIBLE','STATISTICALLY_EXTREME','REGISTRY_DISCORDANT','INCONCLUSIVE','CONSISTENT'];
  const fs=[...R.findings].sort((a,b)=>order.indexOf(a.tier)-order.indexOf(b.tier));
  $('#list').innerHTML=fs.map((f,i)=>`<div class="f" data-i="${R.findings.indexOf(f)}"><div class="t ${f.tier}">${f.tier}</div>${esc(f.title)}</div>`).join('');
  document.querySelectorAll('.f').forEach(d=>d.onclick=()=>show(+d.dataset.i));
  $('#pages').innerHTML=Array.from({length:R.pages},(_,p)=>`<div class="pg" id="pg${p}"><img src="/api/page/${R.id}/${p}.png" onload="this.parentNode.style.width=this.naturalWidth+'px'"></div>`).join('');
  R.findings.forEach((f,i)=>(f.provenance||[]).forEach(p=>{ if(p.page==null||!p.bbox) return;
    const [x0,y0,x1,y1]=p.bbox, d=document.createElement('div'); d.className='box '+f.tier; d.dataset.i=i;
    Object.assign(d.style,{left:x0*SCALE-2+'px',top:y0*SCALE-2+'px',width:(x1-x0)*SCALE+4+'px',height:(y1-y0)*SCALE+4+'px'});
    d.title=f.title; d.onclick=()=>show(i); $('#pg'+p.page).appendChild(d);
    const m=(p.note||'').match(/^vertices: (.*)$/); if(m) m[1].split(';').forEach(v=>{const [x,y]=v.split(',').map(Number);
      const e=document.createElement('div'); e.className='v'; Object.assign(e.style,{left:x*SCALE+'px',top:y*SCALE+'px'}); $('#pg'+p.page).appendChild(e)});
  }));
}
function show(i){ const f=R.findings[i];
  document.querySelectorAll('.box.sel').forEach(b=>b.classList.remove('sel'));
  document.querySelectorAll(`.box[data-i="${i}"]`).forEach((b,k)=>{b.classList.add('sel'); if(!k) b.scrollIntoView({block:'center'})});
  $('#detail').innerHTML=`<h4 class="${f.tier}">${esc(f.title)}</h4><p>${esc(f.detail)}</p>`+
   (f.assumptions?.length?`<p><b>Assumptions</b><br>${f.assumptions.map(esc).join('<br>')}</p>`:'')+
   (f.provenance||[]).map(p=>`<pre>${p.page!=null?'page '+(p.page+1)+': ':''}${esc(p.text)}${p.note&&!p.note.startsWith('vertices')?'\n'+esc(p.note):''}</pre>`).join('')+
   (f.certificate?`<button onclick="location.href='/api/cert/${R.id}/${f.id}'">Export Standalone Verification Script</button>`:'')+
   (f.stats?.certificate_seconds!=null?`<p>Self-check: certificate re-derived the contradiction in ${f.stats.certificate_seconds}s (${f.stats.certificate_lines} lines).</p>`:'');
}
</script></body></html>"""


def _report_json(rid: str, rep: Report, pages: int, name: str) -> dict:
    from .scan import TIER_ORDER

    return {
        "id": rid, "name": name, "pages": pages, "n_claims": rep.n_claims, "notes": rep.notes,
        "counts": {t.value: len(rep.by_tier(t)) for t in TIER_ORDER},
        "findings": [f.model_dump(mode="json") for f in rep.findings],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "trialgeist"

    def log_message(self, format, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'self'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.replace("__SCALE__", str(SCALE)).encode(), "text/html; charset=utf-8")
        m = re.fullmatch(r"/api/page/([0-9a-f]{16})/(\d+)\.png", u.path)
        if m and m.group(1) in STATE:
            pno = int(m.group(2))
            with pymupdf.open(STATE[m.group(1)]["pdf"]) as doc:
                if pno >= len(doc):
                    return self._send(404, b"no page", "text/plain")
                png = doc[pno].get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE)).tobytes("png")
            return self._send(200, png, "image/png")
        m = re.fullmatch(r"/api/cert/([0-9a-f]{16})/([\w\-]+)", u.path)
        if m and m.group(1) in STATE:
            f = next((f for f in STATE[m.group(1)]["report"].findings if f.id == m.group(2)), None)
            if f and f.certificate:
                return self._send(200, f.certificate.encode(), "text/x-python",
                                  {"Content-Disposition": f'attachment; filename="certificate_{f.id}.py"'})
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/scan":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > 200 * 2**20:
            return self._send(400, b'{"error":"empty or too large"}', "application/json")
        data = self.rfile.read(n)
        if not data.startswith(b"%PDF"):
            return self._send(400, b'{"error":"not a PDF"}', "application/json")
        rid = hashlib.sha256(data).hexdigest()[:16]
        d = WORK / rid
        d.mkdir(parents=True, exist_ok=True)
        pdf = d / "paper.pdf"
        pdf.write_bytes(data)
        nct = (parse_qs(u.query).get("nct") or [None])[0]
        if nct and not re.fullmatch(r"NCT\d{8}", nct):
            return self._send(400, b'{"error":"NCT id must look like NCT01234567"}', "application/json")
        try:
            rep = scan_pdf(pdf, nct=nct)
            self_verify(rep, d / "certificates")
        except Exception as e:  # surface parse failures to the UI, never as findings
            return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")
        with LOCK:
            STATE[rid] = {"pdf": str(pdf), "report": rep}
        with pymupdf.open(pdf) as doc:
            npages = len(doc)
        body = json.dumps(_report_json(rid, rep, npages, self.headers.get("X-Filename", "paper.pdf")))
        return self._send(200, body.encode(), "application/json")


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    srv = make_server(host, port)
    print(f"TRIALGEIST studio on http://{host}:{port}  (local only; Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
