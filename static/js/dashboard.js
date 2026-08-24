// Smart Civic Monitoring — dashboard.js
// Handles: sidebar scrollspy, evidence lightbox, Chart.js analytics.
// No new backend calls are made; all data comes from attributes already
// rendered server-side by Flask/Jinja2 from the existing log/evidence data.

(function () {
  "use strict";

  /* ---------- Scrollspy: highlight active sidebar link ---------- */

  const navLinks = Array.from(document.querySelectorAll(".nav-link"));
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  function updateActiveNav() {
    const scrollPos = window.scrollY + 120;
    let currentIndex = 0;

    sections.forEach((section, i) => {
      if (section.offsetTop <= scrollPos) currentIndex = i;
    });

    navLinks.forEach((link, i) => {
      link.classList.toggle("active", i === currentIndex);
    });
  }

  window.addEventListener("scroll", updateActiveNav, { passive: true });
  updateActiveNav();

  /* ---------- Evidence lightbox ---------- */

  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxCaption = document.getElementById("lightbox-caption");
  const lightboxClose = document.getElementById("lightbox-close");

  function openLightbox(src, caption) {
    if (!lightbox) return;
    lightboxImg.src = src;
    lightboxCaption.textContent = caption || "";
    lightbox.classList.add("open");
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.classList.remove("open");
    lightboxImg.src = "";
  }

  document.querySelectorAll(".evidence-card").forEach((card) => {
    card.addEventListener("click", () => {
      openLightbox(card.dataset.fullsrc, card.dataset.caption);
    });
    card.setAttribute("tabindex", "0");
    card.setAttribute("role", "button");
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openLightbox(card.dataset.fullsrc, card.dataset.caption);
      }
    });
  });

  if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
  if (lightbox) {
    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox) closeLightbox();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });

  /* ---------- Analytics charts ---------- */
  // Reads real counts from data-* attributes rendered by Flask. No values
  // are invented here — if the data container isn't present, charts skip.

  const chartData = document.getElementById("chart-data");

  if (chartData && window.Chart) {
    const garbage = Number(chartData.dataset.garbage || 0);
    const trespass = Number(chartData.dataset.trespass || 0);
    const noparking = Number(chartData.dataset.noparking || 0);

    const palette = {
      garbage: "#e8a33d",
      trespass: "#e5484d",
      noparking: "#4b9ef0",
    };

    Chart.defaults.color = "#8a94a6";
    Chart.defaults.font.family = "'JetBrains Mono', monospace";
    Chart.defaults.font.size = 11;

    const barCtx = document.getElementById("barChart");
    if (barCtx) {
      new Chart(barCtx, {
        type: "bar",
        data: {
          labels: ["Garbage", "Trespassing", "No Parking"],
          datasets: [
            {
              label: "Verified violations",
              data: [garbage, trespass, noparking],
              backgroundColor: [palette.garbage, palette.trespass, palette.noparking],
              borderRadius: 4,
              maxBarThickness: 56,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false }, border: { color: "#232b38" } },
            y: {
              beginAtZero: true,
              ticks: { precision: 0 },
              grid: { color: "#1c2330" },
              border: { color: "#232b38" },
            },
          },
        },
      });
    }

    const pieCtx = document.getElementById("pieChart");
    if (pieCtx) {
      const total = garbage + trespass + noparking;
      new Chart(pieCtx, {
        type: "doughnut",
        data: {
          labels: ["Garbage", "Trespassing", "No Parking"],
          datasets: [
            {
              data: total > 0 ? [garbage, trespass, noparking] : [1],
              backgroundColor:
                total > 0
                  ? [palette.garbage, palette.trespass, palette.noparking]
                  : ["#1c2330"],
              borderColor: "#10151d",
              borderWidth: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "62%",
          plugins: {
            legend: {
              position: "bottom",
              labels: { boxWidth: 10, padding: 14 },
            },
            tooltip: { enabled: total > 0 },
          },
        },
      });
    }
  }
})();
