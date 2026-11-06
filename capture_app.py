import asyncio
import base64
import json
import os
import subprocess
import time
import urllib.request
import websockets


async def cdp_call(ws, method, params=None, msg_id=[1]):
    mid = msg_id[0]
    msg_id[0] += 1
    req = {"id": mid, "method": method, "params": params or {}}
    await ws.send(json.dumps(req))
    while True:
        resp = json.loads(await ws.recv())
        if resp.get("id") == mid:
            return resp.get("result", {})


async def main():
    chrome_proc = subprocess.Popen([
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--user-data-dir=/tmp/chrome_profile_cdp",
        "--remote-debugging-port=9222",
        "--window-size=1280,1200",
        "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(2)

    try:
        targets_url = "http://127.0.0.1:9222/json"
        with urllib.request.urlopen(targets_url) as resp:
            targets = json.loads(resp.read().decode())
        page_targets = [t for t in targets if t.get("type") == "page"]
        ws_url = page_targets[0]["webSocketDebuggerUrl"] if page_targets else targets[0]["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            await cdp_call(ws, "Page.enable")
            await cdp_call(ws, "Runtime.enable")

            print("Navigating to http://localhost:8501 ...")
            await cdp_call(ws, "Page.navigate", {"url": "http://localhost:8501"})

            for attempt in range(25):
                await asyncio.sleep(1)
                res = await cdp_call(ws, "Runtime.evaluate", {
                    "expression": "document.body.innerText"
                })
                text = res.get("result", {}).get("value", "")
                if "Chaotic Text Generator" in text:
                    print("Found Streamlit app content!")
                    break

            await asyncio.sleep(2)

            # Screenshot 1: Tab 1 (Generation & Metrics + Table)
            shot1 = await cdp_call(ws, "Page.captureScreenshot", {"format": "png"})
            with open("screenshot_generation.png", "wb") as f:
                f.write(base64.b64decode(shot1["data"]))
            print("Saved screenshot_generation.png")

            # Click Tab 2: Viterbi Tagger
            print("Switching to Tab 2 (Tagger)...")
            await cdp_call(ws, "Runtime.evaluate", {
                "expression": "const el = Array.from(document.querySelectorAll('button, div, p, span')).find(e => e.innerText && e.innerText.trim() === '🏷️ Interactive Viterbi Tagger'); if (el) el.click();"
            })
            await asyncio.sleep(2)

            # Scroll down to display the predicted alignment boxes and result
            await cdp_call(ws, "Runtime.evaluate", {
                "expression": "window.scrollTo(0, 180);"
            })
            await asyncio.sleep(1)

            shot2 = await cdp_call(ws, "Page.captureScreenshot", {"format": "png"})
            with open("screenshot_tagger.png", "wb") as f:
                f.write(base64.b64decode(shot2["data"]))
            print("Saved screenshot_tagger.png")

    finally:
        chrome_proc.terminate()
        chrome_proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
