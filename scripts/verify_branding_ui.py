import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import sqlite3, tempfile, threading, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from werkzeug.serving import make_server, WSGIRequestHandler
class QuietHandler(WSGIRequestHandler):
 def log_request(self,*args): pass
from playwright.sync_api import sync_playwright
tmp=tempfile.TemporaryDirectory(prefix="railpulse-ui-")
real=sqlite3.connect
def connect(database,*args,**kwargs):
    return real(str(Path(tmp.name)/"test.db") if str(database)=="railpulse.db" else database,*args,**kwargs)
sqlite3.connect=connect
import app as module
client=module.app.test_client()
client.get("/")
with client.session_transaction() as s: token=s["csrf_token"]
client.post("/signup",data=dict(csrf_token=token,username="ui_test",full_name="UI Test User",password="test-pass-123",confirm_password="test-pass-123"))
server=make_server("127.0.0.1",5058,module.app,threaded=True,request_handler=QuietHandler)
threading.Thread(target=server.serve_forever,daemon=True).start()
out=Path("ui-verification");out.mkdir(exist_ok=True)
errors=[]
with sync_playwright() as pw:
 b=pw.chromium.launch(channel="msedge",headless=True)
 page=b.new_page(viewport=dict(width=1440,height=900))
 page.on("pageerror",lambda e:errors.append(str(e)))
 base="http://127.0.0.1:5058"
 def login(password="test-pass-123"):
  page.goto(base+"/")
  page.locator('[name="username"]').fill("ui_test")
  page.locator('[name="password"]').fill(password)
  page.locator('button[type="submit"]').click()
 login("wrong")
 assert page.locator("#rp-startup").count()==0
 login()
 page.wait_for_url("**/dashboard")
 page.wait_for_timeout(1400)
 assert page.locator("#rp-startup").count()==1
 assert page.locator(".rp-startup__train").is_visible()
 page.screenshot(path=str(out/"signin-transition.png"))
 page.locator("#rp-startup").wait_for(state="detached",timeout=6000)
 assert not page.locator(".app-layout").evaluate("(e)=>e.inert")
 assert not page.evaluate("document.documentElement.classList.contains('rp-startup-running')")
 page.locator(".rp-account summary").click()
 assert page.locator(".rp-account-dropdown").is_visible()
 page.keyboard.press("Escape")
 assert not page.locator(".rp-account-dropdown").is_visible()
 page.locator(".rp-brand-link").click()
 assert page.locator("#rp-startup").count()==0
 page.screenshot(path=str(out/"dashboard-desktop.png"),full_page=True)
 print("PASS browser authentication, train transition, cleanup, logo link, account menu",flush=True)
 for width in [320,390,768,1024,1440]:
  page.set_viewport_size(dict(width=width,height=900))
  for path in ["/dashboard","/predict","/analytics","/history","/route-map","/about","/account"]:
   page.goto(base+path)
   assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"),(width,path,page.evaluate("document.documentElement.scrollWidth"))
   assert page.locator("#rp-startup").count()==0
  if width==390:
   page.goto(base+"/about")
   page.locator(".rp-account summary").click()
   box=page.locator(".rp-account-dropdown").bounding_box()
   assert box["x"]>=0 and box["x"]+box["width"]<=390
   page.keyboard.press("Escape")
   page.screenshot(path=str(out/"about-mobile.png"),full_page=True)
 print("PASS all authenticated layouts at 320, 390, 768, 1024 and 1440 pixels",flush=True)
 page.set_viewport_size(dict(width=390,height=844))
 page.goto(base+"/predict")
 page.locator("#generateButton").click()
 assert not page.locator("#rp-processing").is_visible()
 record=module.df.iloc[0]
 page.select_option("#from_station",str(record["from_station"]))
 page.wait_for_function("!document.getElementById('to_station').disabled")
 page.select_option("#to_station",str(record["to_station"]))
 page.wait_for_function("!document.getElementById('departure_time').disabled")
 page.select_option("#departure_time",str(record["departure_time"]))
 page.locator('[name="prediction_date"]').fill("2026-10-05")
 # Pause only this test submission after the production submit listener.
 page.evaluate("document.getElementById('predictionForm').addEventListener('submit', e => e.preventDefault(), {once:true})")
 page.locator("#generateButton").click()
 assert page.locator("#rp-processing").is_visible()
 assert page.locator(".app-layout").evaluate("(e)=>e.inert")
 page.screenshot(path=str(out/"forecast-loading.png"))
 page.evaluate("document.getElementById('predictionForm').submit()")
 page.wait_for_url("**/result/**",timeout=120000)
 assert page.locator("#dailyCrowdChart").count()==1
 assert page.get_by_text("Operational Action Plan",exact=False).count()>0
 assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1")
 page.screenshot(path=str(out/"result-mobile.png"),full_page=True)
 for width in [320,768,1024,1440]:
  page.set_viewport_size(dict(width=width,height=900))
  if page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"):
   print(page.evaluate("Array.from(document.querySelectorAll('body *')).map(e=>({tag:e.tagName,cls:e.className,right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})).filter(e=>e.right>innerWidth+1).slice(0,25)"),flush=True)
  assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"),("result",width)
 assert page.evaluate("!!Chart.getChart(document.getElementById('dailyCrowdChart'))")
 page.goto(base+"/history")
 assert "Alwal" in page.locator(".data-table").inner_text()
 page.goto(base+"/route-map")
 assert page.locator("#city-map .leaflet-pane").count()>0
 page.locator(".map-route-row").first.click()
 assert page.locator(".map-route-row").first.get_attribute("aria-pressed")=="true"
 page.locator("#map-zoom-in").click()
 page.locator("#map-fit").click()
 page.goto(base+"/logout")
 page.set_viewport_size(dict(width=320,height=640))
 login()
 page.wait_for_url("**/dashboard")
 page.wait_for_timeout(1000)
 box=page.locator(".rp-startup__logo").bounding_box()
 assert box["x"]>=0 and box["x"]+box["width"]<=320
 page.locator("#rp-startup").wait_for(state="detached",timeout=6000)
 print("PASS actual forecast, chart, action plan, history, result widths, map interaction and mobile train transition",flush=True)
 page.goto(base+"/logout")
 for width in [320,390,768,1440]:
  page.set_viewport_size(dict(width=width,height=900))
  for path in ["/","/signup"]:
   page.goto(base+path)
   assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"),(width,path)
   assert page.locator("#rp-startup").count()==0
 page.set_viewport_size(dict(width=390,height=844))
 page.emulate_media(reduced_motion="reduce")
 login()
 page.wait_for_url("**/dashboard")
 page.locator("#rp-startup").wait_for(state="detached",timeout=2000)
 assert not page.locator(".app-layout").evaluate("(e)=>e.inert")
 assert not errors,errors
 b.close()
server.shutdown()
tmp.cleanup()
print("PASS logout, auth page layouts, reduced motion; no JavaScript errors",flush=True)