// Push-to-talk kiosk client. Browsers allow the microphone on http://localhost without HTTPS.
const $ = (id) => document.getElementById(id);
const talk = $("talk"), statusEl = $("status"), video = $("avatar"), audioOnly = $("voice-only");
let recorder = null, chunks = [], voices = {};

function setStatus(text) { statusEl.textContent = text; }

async function loadVoices() {
  const res = await fetch("/voices");
  const data = await res.json();
  voices = data.voices;
  const select = $("voice");
  for (const [lang, list] of Object.entries(voices)) {
    const group = document.createElement("optgroup");
    group.label = data.languages[lang];
    for (const v of list) group.append(new Option(v, v));
    select.append(group);
  }
}

function show(result) {
  $("you").textContent = result.transcript ?? result.question ?? "–";
  $("kiosk").textContent = result.answer;
  $("lang-badge").textContent = `lang: ${result.lang}${result.refused ? " · refused" : ""}`;
  const t = result.timings_ms ?? (result.ms !== undefined ? { total: result.ms } : {});
  $("timings").textContent = Object.entries(t).map(([k, v]) => `${k} ${v} ms`).join(" · ");
}

function play(result) {
  if (result.video_url) {
    video.src = result.video_url;
    video.play().catch(() => setStatus("Tap the video to play the answer"));
  } else if (result.audio_url) {
    audioOnly.src = result.audio_url;
    audioOnly.play();
  }
}

async function sendAudio(blob) {
  setStatus("Thinking…");
  const form = new FormData();
  form.append("audio", blob, "question.webm");
  const voice = $("voice").value;
  if (voice) form.append("voice", voice);
  const res = await fetch("/turn", { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok) { setStatus(body.detail ?? "Something went wrong. Try again."); return; }
  show(body);
  play(body);
  setStatus("Hold the button and ask another question");
}

let holding = false;

async function startRecording() {
  if (recorder) return;
  holding = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: recorder.mimeType });
      recorder = null;
      if (blob.size > 2000) sendAudio(blob); else setStatus("That was too short. Hold the button while you speak.");
    };
    recorder.start();
    talk.classList.add("recording");
    talk.textContent = "Listening… release to send";
    setStatus("Listening…");
    if (!holding) stopRecording();  // button was released while the mic permission prompt was open
  } catch (err) {
    setStatus("Microphone blocked. Allow mic access for this page and try again.");
  }
}

function stopRecording() {
  holding = false;
  if (!recorder || recorder.state !== "recording") return;
  talk.classList.remove("recording");
  talk.textContent = "Hold to talk";
  recorder.stop();
}

talk.addEventListener("pointerdown", startRecording);
talk.addEventListener("pointerup", stopRecording);
talk.addEventListener("pointerleave", stopRecording);
talk.addEventListener("keydown", (e) => { if (e.code === "Space" && !e.repeat) { e.preventDefault(); startRecording(); } });
talk.addEventListener("keyup", (e) => { if (e.code === "Space") stopRecording(); });

$("type-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("typed").value.trim();
  if (!question) return;
  setStatus("Thinking…");
  const res = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
  const body = await res.json();
  if (!res.ok) { setStatus("Something went wrong. Check the server log."); return; }
  show(body);
  setStatus("Text answers are not spoken. Use the button for voice.");
});

loadVoices().catch(() => setStatus("Server not reachable. Is uvicorn running?"));
