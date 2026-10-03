#!/usr/bin/env python3
"""
live_dashboard.py -- watch training happen in a browser.

Serves a single page that polls ``runs/<run_id>/live/state.json`` (written by
the trainer every window) and redraws the actual-vs-predicted chart and the
accuracy progress bars.  Everything is inlined -- no CDN, no build step -- so
it works on an air-gapped box.

    # terminal 1
    python train.py --speed 120 --live
    # terminal 2
    python live_dashboard.py --run-dir runs --port 8080

If you only want a shareable artefact of a *finished* run, use
``btcpred.viz.build_standalone_html`` instead: it embeds the data so the file
needs no server at all.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bitcoin Price Predictor -- live</title><style>
:root{--bg:#0e1117;--fg:#e6edf3;--grid:#262c36;--price:#4cc9f0;--pred:#ffb703;
      --ok:#3fb950;--warn:#f0883e;--dim:#8b949e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.45 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:18px}
h1{font-size:19px;margin:0 0 2px}.sub{color:var(--dim);font-size:12px;margin-bottom:14px}
.card{background:#11161d;border:1px solid var(--grid);border-radius:10px;padding:12px;margin-bottom:12px}
canvas{width:100%;display:block}.row{display:flex;gap:12px;flex-wrap:wrap}.col{flex:1 1 320px}
.bar{margin:9px 0}.bar .lab{display:flex;justify-content:space-between;font-size:12px;color:var(--dim)}
.bar .trk{height:9px;background:var(--grid);border-radius:5px;overflow:hidden;margin-top:3px}
.bar .fil{height:100%;width:0;border-radius:5px;transition:width .25s}
.kv{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--dim)}.kv b{color:var(--fg)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ok);margin-right:6px}
.leg{display:flex;gap:16px;font-size:12px;color:var(--dim);margin-bottom:6px}
.sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:5px}
</style></head><body><div class="wrap">
<h1><span class="dot" id="dot"></span>Bitcoin Price Predictor &mdash; live training</h1>
<div class="sub">actual vs 30-minute-ahead forecast, updated every window</div>
<div class="card">
 <div class="leg">
  <span><i class="sw" style="background:var(--price)"></i>actual price</span>
  <span><i class="sw" style="border-top:3px dotted var(--pred)"></i>predicted (t+30m)</span>
 </div>
 <canvas id="c" height="340"></canvas></div>
<div class="row">
 <div class="card col"><div style="font-size:12px;color:var(--dim)">TRAINING ACCURACY</div>
  <div id="bars"></div></div>
 <div class="card col"><div style="font-size:12px;color:var(--dim)">RUN</div>
  <div id="stats" class="kv" style="margin-top:8px"></div></div>
</div></div><script>
const B=[["band_accuracy","band 10bps",0,1,v=>(100*v).toFixed(1)+"%"],
         ["direction_acc","direction",.3,.8,v=>(100*v).toFixed(1)+"%"],
         ["skill_vs_naive","skill vs naive",-.5,.5,v=>v.toFixed(3)],
         ["calibration","calibration",0,1,v=>(100*v).toFixed(1)+"%"]];
const c=document.getElementById("c"),x=c.getContext("2d");
let S=null;
function bars(m){const h=document.getElementById("bars");
 if(!h.dataset.i){h.innerHTML=B.map(b=>`<div class="bar"><div class="lab">
  <span>${b[1]}</span><span id="v_${b[0]}">--</span></div>
  <div class="trk"><div class="fil" id="f_${b[0]}"></div></div></div>`).join("");
  h.dataset.i=1;}
 B.forEach(([k,l,lo,hi,f])=>{const v=(m&&m[k]!=null)?m[k]:lo;
  const p=Math.max(0,Math.min(1,(v-lo)/(hi-lo)));
  const e=document.getElementById("f_"+k);e.style.width=(100*p)+"%";
  e.style.background=p>.5?"var(--ok)":"var(--warn)";
  document.getElementById("v_"+k).textContent=f(v);});}
function draw(){if(!S)return;const r=c.getBoundingClientRect(),d=devicePixelRatio||1;
 c.width=r.width*d;c.height=340*d;x.setTransform(d,0,0,d,0,0);
 const W=r.width,H=340;x.clearRect(0,0,W,H);
 const P=S.price,Q=S.pred,T=S.ts;if(!P||P.length<2)return;
 let lo=Math.min(...P,...Q),hi=Math.max(...P,...Q);
 const pad=(hi-lo)*.12+1e-6;lo-=pad;hi+=pad;
 const t0=T[0],t1=T[T.length-1]+1800,PX=54,PY=12,PB=22;
 const X=t=>PX+(t-t0)/(t1-t0)*(W-PX-10),Y=v=>PY+(hi-v)/(hi-lo)*(H-PY-PB);
 x.strokeStyle="#262c36";x.fillStyle="#8b949e";x.font="10px ui-sans-serif";
 for(let g=0;g<=4;g++){const v=lo+(hi-lo)*g/4,y=Y(v);x.beginPath();
  x.moveTo(PX,y);x.lineTo(W-10,y);x.stroke();x.fillText(v.toFixed(0),4,y+3);}
 x.beginPath();x.strokeStyle="#4cc9f0";x.lineWidth=1.6;
 P.forEach((p,i)=>{const xx=X(T[i]),yy=Y(p);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke();
 x.beginPath();x.strokeStyle="#ffb703";x.lineWidth=1.7;x.setLineDash([3,3]);
 Q.forEach((q,i)=>{const xx=X(T[i]+1800),yy=Y(q);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});
 x.stroke();x.setLineDash([]);
 bars(S.metrics||{});
 document.getElementById("stats").innerHTML=
  `<span>window <b>${S.window||0}</b></span><span>sim time <b>${S.utc||"-"}</b></span>
   <span>throughput <b>${(S.throughput||0).toLocaleString()} sim-s/s</b></span>
   <span>loss <b>${S.loss!=null?(+S.loss).toFixed(4):"-"}</b></span>
   <span>elapsed <b>${(S.elapsed||0).toFixed(0)}s</b></span>`;}
async function poll(){try{const r=await fetch("/state.json?"+Date.now());
 if(r.ok){S=await r.json();document.getElementById("dot").style.background="var(--ok)";draw();}
 else document.getElementById("dot").style.background="var(--warn)";}
 catch(e){document.getElementById("dot").style.background="var(--warn)";}}
addEventListener("resize",draw);poll();setInterval(poll,1000);
</script></body></html>"""


def newest_run(run_dir: str) -> str:
    runs = [d for d in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, d))
            and os.path.exists(os.path.join(run_dir, d, "live", "state.json"))]
    if not runs:
        raise FileNotFoundError(
            f"no run with live/state.json under {run_dir} -- start training first")
    runs.sort(key=lambda d: os.path.getmtime(
        os.path.join(run_dir, d, "live", "state.json")))
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console clean
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/state.json"):
                try:
                    rid = args.run_id or newest_run(args.run_dir)
                    p = os.path.join(args.run_dir, rid, "live", "state.json")
                    with open(p, "rb") as fh:
                        self._send(fh.read(), "application/json")
                except Exception as e:                   # noqa: BLE001
                    self._send(json.dumps({"error": str(e)}).encode(),
                               "application/json", 503)
                return
            self._send(PAGE.encode(), "text/html; charset=utf-8")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((args.host, args.port), Handler) as srv:
        print(f"[dashboard] http://{args.host}:{args.port}  "
              f"(watching {os.path.abspath(args.run_dir)})")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[dashboard] bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
