let chartB = null;
let chartC = null;

async function loadAndRender(freq) {
  const status = document.getElementById("status");
  const noteB = document.getElementById("noteB");
  const noteC = document.getElementById("noteC");

  status.textContent = "Carico i dati reali...";
  noteB.textContent = "";
  noteC.textContent = "";

  try {
    const res = await fetch(`/api/combined?freq=${encodeURIComponent(freq)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    status.textContent = "";
    noteB.textContent = `Base (100) dal: ${data.base_date} | punti: ${data.points} | pesi: LS80 ${data.weights.ls80}, Gold ${data.weights.gold}, BTC ${data.weights.btc} | freq: ${data.freq}`;
    noteC.textContent = `Base (100) dal: ${data.base_date} | punti: ${data.points} | freq: ${data.freq}`;

    const labels = data.labels;

    // Distruggi grafici precedenti (se cambio frequenza)
    if (chartB) chartB.destroy();
    if (chartC) chartC.destroy();

    // ---- Grafico B (Portafoglio) ----
    const ctxB = document.getElementById("chartB").getContext("2d");
    chartB = new Chart(ctxB, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Portafoglio (B)",
            data: data.portfolio,
            tension: 0.2,
            pointRadius: 0,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
          y: { beginAtZero: false }
        }
      }
    });

    // ---- Grafico C (Componenti normalizzate) ----
    const ctxC = document.getElementById("chartC").getContext("2d");
    chartC = new Chart(ctxC, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "LS80", data: data.ls80, tension: 0.2, pointRadius: 0, borderWidth: 2 },
          { label: "Gold", data: data.gold, tension: 0.2, pointRadius: 0, borderWidth: 2 },
          { label: "BTC",  data: data.btc,  tension: 0.2, pointRadius: 0, borderWidth: 2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
          y: { beginAtZero: false }
        }
      }
    });

  } catch (err) {
    console.error(err);
    status.textContent = "Errore nel caricamento dati (vedi Console F12).";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const freqSelect = document.getElementById("freq");

  // primo render (daily)
  loadAndRender(freqSelect.value);

  // cambia frequenza al volo
  freqSelect.addEventListener("change", () => {
    loadAndRender(freqSelect.value);
  });
});
