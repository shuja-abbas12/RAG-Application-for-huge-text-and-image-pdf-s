// ---- Plot UMAP with fallback -------------------------------------------
async function plotUMAP() {
    const container = document.getElementById("umap-plot");
    try {
      const res = await fetch("/api/umap");
      const { x, y, labels } = await res.json();
      if (!x.length) throw new Error("No data");
      Plotly.newPlot(container, [{
        x, y, text: labels, mode: "markers",
        marker: { size: 5, color: labels, colorscale: "Rainbow" }
      }], { height: 500, margin: { t: 10 } });
    } catch (e) {
      container.innerText = "UMAP visualization unavailable.";
    }
  }
  plotUMAP();
  
  // ---- Immediate Markdown rendering as you type -------------------------
  function typeWriter(text, targetEl, delay = 25) {
    let i = 0, buf = "";
    targetEl.innerHTML = "";  
    const tid = setInterval(() => {
      if (i < text.length) {
        buf += text[i++];
        // render Markdown on the fly
        targetEl.innerHTML = marked.parse(buf);
      } else {
        clearInterval(tid);
      }
    }, delay);
  }
  
  const form = document.getElementById("ask-form");
  const spinner = document.getElementById("spinner");
  const btn     = document.getElementById("ask-btn");
  
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
  
    const queryText = document.getElementById("query").value.trim();
    const hasImage  = document.getElementById("image").files.length > 0;
    if (!queryText && !hasImage) {
      alert("Please enter a question or upload an image.");
      return;
    }
  
    spinner.classList.remove("hidden");
    btn.disabled = true;
  
    const fd  = new FormData(form);
    const res = await fetch("/api/ask", { method: "POST", body: fd });
    spinner.classList.add("hidden");
    btn.disabled = false;
  
    if (res.status !== 200) {
      const err = await res.json();
      alert(err.error || "An error occurred.");
      return;
    }
  
    const { answer, sources, latency } = await res.json();
    document.getElementById("answer-card").classList.remove("hidden");
  
    const answerEl = document.getElementById("answer");
    typeWriter(answer, answerEl);
  
    const ul = document.getElementById("sources");
    ul.innerHTML = "";
    sources.forEach(s => {
      const li = document.createElement("li");
      li.textContent = `${s.doc_id} p.${s.page} – ${s.id}`;
      ul.appendChild(li);
    });
  
    document.getElementById("latency")
            .innerText = `Latency: ${latency.toFixed(2)} s`;
  });
  