"use strict";

function createSpeechInput({ canListen, onQuestion, onError }) {
  const wave = document.getElementById("mic-wave");
  const levelBars = [...wave.children];
  const state = document.getElementById("mic-state");
  const AudioEngine = window.AudioContext ?? window.webkitAudioContext;
  const barWeights = [0.35, 0.62, 0.83, 1, 0.83, 0.62, 0.35];
  let current;
  let frame;
  let lastSample;
  let quietUntil;
  let noise = 0.004;
  let lastLoud;
  let loudDuration = 0;
  let level = 0;

  function renderLevel(next) {
    for (const [i, bar] of levelBars.entries()) {
      bar.style.transform = `scaleY(${0.16 + next * 0.84 * barWeights[i]})`;
    }
  }

  function reset() {
    cancelAnimationFrame(frame);
    frame = undefined;
    lastSample = undefined;
    quietUntil = performance.now() + 260;
    loudDuration = 0;
    lastLoud = undefined;
    level = 0;
    renderLevel(0);
  }

  function disable() {
    reset();
    const session = current;
    current = undefined;
    if (!session) return;
    session.track?.stop();
    session.source?.disconnect();
    session.context.close().catch(() => {});
  }

  function fail(message) {
    disable();
    onError(message);
  }

  function sample(now) {
    frame = undefined;
    if (!current?.track?.enabled || !canListen()) return;
    const elapsed = now - (lastSample ?? now);
    if (lastSample === undefined || elapsed >= 30) {
      lastSample = now;
      current.analyser.getFloatTimeDomainData(current.samples);
      let total = 0;
      for (const value of current.samples) total += value * value;
      const rms = Math.sqrt(total / current.samples.length);
      const threshold = Math.max(0.012, noise * 3.1);
      const audible = rms > threshold;
      const visual = audible ? Math.min(1, Math.sqrt((rms - threshold) * 16)) : 0;
      level += (visual - level) * (visual > level ? 0.7 : 0.25);
      renderLevel(level);

      if (!audible) noise += (rms - noise) * 0.04;
      if (now >= quietUntil) {
        if (audible) {
          if (lastLoud !== undefined && now - lastLoud > 190 && loudDuration < 320) loudDuration = 0;
          lastLoud = now;
          loudDuration += Math.min(elapsed, 48);
        } else if (lastLoud !== undefined && now - lastLoud >= 480) {
          const ready = loudDuration >= 320;
          loudDuration = 0;
          lastLoud = undefined;
          if (ready) {
            reset();
            onQuestion();
            return;
          }
        }
      }
    }
    frame = requestAnimationFrame(sample);
  }

  function refresh() {
    const track = current?.track;
    if (!track || !current.analyser) return;
    const listening = canListen();
    track.enabled = listening;
    if (!listening) {
      state.textContent = "المايك متوقف مؤقتًا";
      reset();
    } else if (frame === undefined) {
      state.textContent = "أسمعك، تكلّم ثم اسكت لحظة";
      frame = requestAnimationFrame(sample);
    }
  }

  async function enable() {
    if (!AudioEngine || !navigator.mediaDevices?.getUserMedia) {
      fail("متصفحك ما قدر يشغّل المايك. استخدم زر «ماهذا؟».");
      return;
    }
    disable();
    state.textContent = "جاري تشغيل المايك…";
    let session;
    try {
      const context = new AudioEngine();
      session = { context };
      current = session;
      const running = context.resume().catch(() => false);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: false,
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      if (current !== session) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      [session.track] = stream.getAudioTracks();
      await running;
      if (current !== session) return;
      if (context.state !== "running" || !session.track) throw new Error("Microphone unavailable");
      session.track.addEventListener("ended", () => {
        if (current === session) fail("توقف المايك. جرّب تشغيله مرة ثانية.");
      }, { once: true });
      session.source = context.createMediaStreamSource(stream);
      session.analyser = context.createAnalyser();
      session.analyser.fftSize = 1024;
      session.samples = new Float32Array(session.analyser.fftSize);
      session.source.connect(session.analyser);
      noise = 0.004;
      reset();
      refresh();
    } catch (error) {
      if (session && current !== session) return;
      fail(error.name === "NotAllowedError"
        ? "اسمح بالمايك من المتصفح أو استخدم زر «ماهذا؟»."
        : "تعذّر تشغيل المايك. جرّب مرة ثانية أو استخدم زر «ماهذا؟».");
    }
  }

  return { enable, disable, refresh };
}
