const targetUrl = 'http://localhost:9222/json';

async function main() {
  console.log("Fetching Chrome DevTools targets...");
  const res = await fetch(targetUrl);
  const targets = await res.json();
  const target = targets.find(t => t.url.includes("localhost:5174"));
  if (!target) {
    console.error("Target localhost:5174 not found. Active targets:", targets);
    process.exit(1);
  }

  const wsUrl = target.webSocketDebuggerUrl;
  console.log(`Connecting to WebSocket: ${wsUrl}`);
  const ws = new WebSocket(wsUrl);

  let messageId = 1;
  const pendingRequests = new Map();

  function sendCommand(method, params = {}) {
    const id = messageId++;
    const payload = JSON.stringify({ id, method, params });
    ws.send(payload);
    return new Promise((resolve, reject) => {
      pendingRequests.set(id, { resolve, reject });
    });
  }

  ws.onopen = async () => {
    console.log("Connected to Chrome. Enabling Runtime and Console domains...");
    await sendCommand("Runtime.enable");
    await sendCommand("Console.enable");
    await sendCommand("Log.enable");
    await sendCommand("Page.enable");

    console.log("Reloading page to ensure fresh code state...");
    await sendCommand("Page.reload");

    // Wait 3 seconds for load
    await new Promise(r => setTimeout(r, 3000));

    // Click "Script & Dialogue" in sidebar
    console.log("Clicking 'Script & Dialogue' in sidebar...");
    await evaluateInPage(`{
      const btn = Array.from(document.querySelectorAll('button')).find(el => el.textContent.includes('Script & Dialogue'));
      if (btn) {
        btn.click();
        'Clicked Script & Dialogue';
      } else {
        throw new Error('Script & Dialogue button not found');
      }
    }`);

    await new Promise(r => setTimeout(r, 1500));

    // Click "Open Editor"
    console.log("Clicking 'Open Editor'...");
    await evaluateInPage(`{
      const btn = Array.from(document.querySelectorAll('button')).find(el => el.textContent.includes('Open Editor'));
      if (btn) {
        btn.click();
        'Clicked Open Editor';
      } else {
        throw new Error('Open Editor button not found');
      }
    }`);

    await new Promise(r => setTimeout(r, 1500));

    // Click "Multitrack Timeline"
    console.log("Clicking 'Multitrack Timeline' tab...");
    await evaluateInPage(`{
      const btn = Array.from(document.querySelectorAll('button')).find(el => el.textContent.includes('Multitrack Timeline'));
      if (btn) {
        btn.click();
        'Clicked Multitrack Timeline';
      } else {
        throw new Error('Multitrack Timeline button not found');
      }
    }`);

    // Wait for render/crash
    console.log("Waiting for 3 seconds to capture any crashes...");
    await new Promise(r => setTimeout(r, 3000));

    ws.close();
    process.exit(0);
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pendingRequests.has(msg.id)) {
      const { resolve, reject } = pendingRequests.get(msg.id);
      pendingRequests.delete(msg.id);
      if (msg.error) reject(msg.error);
      else resolve(msg.result);
    }

    if (msg.method === "Runtime.exceptionThrown") {
      console.error("\n❌ BROWSER EXCEPTION DETECTED:", JSON.stringify(msg.params.exceptionDetails, null, 2));
    }
    if (msg.method === "Runtime.consoleAPICalled") {
      console.log("\n🌐 BROWSER CONSOLE:", msg.params.type, ...msg.params.args.map(a => a.value || a.description || JSON.stringify(a)));
    }
  };

  async function evaluateInPage(expression) {
    try {
      const result = await sendCommand("Runtime.evaluate", { expression, returnByValue: true });
      if (result.exceptionDetails) {
        console.error("Evaluation exception:", result.exceptionDetails);
      } else {
        console.log("Evaluation result:", result.result.value);
      }
    } catch (e) {
      console.error("Evaluation failed:", e);
    }
  }
}

main().catch(console.error);
