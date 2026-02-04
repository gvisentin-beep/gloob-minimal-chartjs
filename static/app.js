document.addEventListener("DOMContentLoaded", () => {
  const status = document.getElementById("status");
  const ctx = document.getElementById("portfolioChart").getContext("2d");

  fetch("/api/data")
    .then(response => response.json())
    .then(data => {
      status.textContent = "";

      new Chart(ctx, {
        type: "line",
        data: {
          labels: data.labels,
          datasets: [{
            label: "Portafoglio (API test)",
            data: data.values,
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59,130,246,0.15)",
            tension: 0.3
          }]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: false
            }
          }
        }
      });
    })
    .catch(err => {
      status.textContent = "Errore nel caricamento dati";
      console.error(err);
    });
});
