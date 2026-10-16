// ==========================================================
// CivicGuard AI — Security Operations Center UI Controller
// ==========================================================
//
// Responsibilities:
//   1. Navigation & Scrollspy (Sidebar & breadcrumbs)
//   2. Real-time Telemetry HUD clock (Live camera overlay)
//   3. Live Global Search & Table Category Filtering
//   4. High-Resolution Evidence Lightbox Inspector
//   5. Chart.js Surveillance Analytics & Risk Modeling
//   6. Heartbeat Polling (/api/status every 2s) & Reactive Swapping
//   7. Real-Time Verified Event Dispatch Toasts
//   8. Perspective-Aware Zone Calibration Canvas Editor
//
// Zero page reloads required after initial load.
// ==========================================================

(function () {
  "use strict";

  const POLL_INTERVAL_MS = 2000;

  let lastSeenEventId = window.__INITIAL_EVENT_ID__ || 0;
  let barChartInstance = null;
  let riskChartInstance = null;
  let currentActiveFilter = "all";
  let currentSearchQuery = "";

  /* ==========================================================
     1. Navigation, Scrollspy & Header Breadcrumbs
     ========================================================== */

  const navItems = Array.from(document.querySelectorAll(".nav-item, .nav-link"));
  const headerSectionName = document.getElementById("header-section-name");

  const sectionMap = navItems.map((item) => {
    const href = item.getAttribute("href");
    if (!href || !href.startsWith("#")) return null;
    const target = document.querySelector(href);
    return target ? { item, target, title: item.querySelector(".nav-text")?.textContent.trim() || "" } : null;
  }).filter(Boolean);

  function updateActiveNav() {
    const scrollPos = window.scrollY + 140;
    let current = sectionMap[0];

    sectionMap.forEach((entry) => {
      if (entry.target.offsetTop <= scrollPos) {
        current = entry;
      }
    });

    if (current) {
      navItems.forEach((el) => el.classList.remove("active"));
      current.item.classList.add("active");
      if (headerSectionName && current.title) {
        headerSectionName.textContent = current.title;
      }
    }
  }

  window.addEventListener("scroll", updateActiveNav, { passive: true });
  updateActiveNav();

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      setTimeout(updateActiveNav, 50);
    });
  });

  /* ==========================================================
     2. Real-time Telemetry HUD Clock
     ========================================================== */

  const hudClock = document.getElementById("hud-clock");
  function tickHudClock() {
    if (!hudClock) return;
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const h = pad(now.getHours());
    const m = pad(now.getMinutes());
    const s = pad(now.getSeconds());
    hudClock.textContent = `${h}:${m}:${s} LOC`;
  }
  tickHudClock();
  setInterval(tickHudClock, 1000);

  /* ==========================================================
     3. Table Filtering: Live Search & Category Chips
     ========================================================== */

  function applyTableFilters() {
    const rows = document.querySelectorAll(".table-surveillance tbody tr[data-search]");
    let visibleCount = 0;

    rows.forEach((row) => {
      const rowSearch = (row.getAttribute("data-search") || "").toLowerCase();
      const rowCategory = (row.getAttribute("data-category") || "").toLowerCase();

      const matchesCategory = (currentActiveFilter === "all") || (rowCategory === currentActiveFilter);
      const matchesSearch = !currentSearchQuery || rowSearch.includes(currentSearchQuery);

      if (matchesCategory && matchesSearch) {
        row.style.display = "";
        visibleCount++;
      } else {
        row.style.display = "none";
      }
    });

    // Handle empty state notice if all rows are filtered
    const emptyNotice = document.getElementById("table-empty-notice");
    if (emptyNotice) {
      emptyNotice.style.display = visibleCount === 0 && rows.length > 0 ? "flex" : "none";
    }
  }

  // Global search input in header
  const globalSearch = document.getElementById("global-search");
  if (globalSearch) {
    globalSearch.addEventListener("input", (e) => {
      currentSearchQuery = e.target.value.trim().toLowerCase();
      applyTableFilters();
    });

    // Keyboard shortcut: '/' focuses search input
    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && document.activeElement !== globalSearch) {
        e.preventDefault();
        globalSearch.focus();
      }
    });
  }

  // Category filter chips delegation
  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".filter-chip");
    if (!chip) return;

    document.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");

    currentActiveFilter = chip.getAttribute("data-filter") || "all";
    applyTableFilters();
  });

  /* ==========================================================
     4. High-Resolution Evidence Lightbox Inspector
     ========================================================== */

  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxCaption = document.getElementById("lightbox-caption");
  const lightboxClose = document.getElementById("lightbox-close");

  function openLightbox(src, caption) {
    if (!lightbox || !src) return;
    lightboxImg.src = src;
    if (lightboxCaption) lightboxCaption.textContent = caption || "Forensic Evidence Frame";
    lightbox.classList.add("open");
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.classList.remove("open");
    if (lightboxImg) lightboxImg.src = "";
  }

  // Delegated click handler on document to capture dynamically refreshed elements
  document.addEventListener("click", (e) => {
    // 1. Evidence dossier cards
    const dossierCard = e.target.closest(".evidence-dossier-card");
    if (dossierCard && dossierCard.dataset.fullsrc) {
      e.preventDefault();
      openLightbox(dossierCard.dataset.fullsrc, dossierCard.dataset.caption);
      return;
    }

    // 2. Active incident thumbnail
    const activeCard = e.target.closest(".active-evidence-card");
    if (activeCard && activeCard.dataset.fullsrc) {
      e.preventDefault();
      openLightbox(activeCard.dataset.fullsrc, activeCard.dataset.caption);
      return;
    }

    // 3. Inspect buttons or table thumbnail inspects
    const inspectBtn = e.target.closest(".btn-evidence-inspect, .evidence-thumb-inspect");
    if (inspectBtn && inspectBtn.dataset.fullsrc) {
      e.preventDefault();
      openLightbox(inspectBtn.dataset.fullsrc, inspectBtn.dataset.caption);
      return;
    }

    // 4. Lightbox backdrop click to dismiss
    if (e.target.classList.contains("lightbox-backdrop") || e.target === lightbox) {
      closeLightbox();
    }
  });

  if (lightboxClose) {
    lightboxClose.addEventListener("click", closeLightbox);
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightbox && lightbox.classList.contains("open")) {
      closeLightbox();
    }
  });

  function makeEvidenceAccessible() {
    document.querySelectorAll(".evidence-dossier-card, .active-evidence-card").forEach((card) => {
      card.setAttribute("tabindex", "0");
      card.setAttribute("role", "button");
    });
  }
  makeEvidenceAccessible();

  /* ==========================================================
     5. Chart.js Surveillance Analytics & Risk Modeling
     ========================================================== */

  function renderCharts(garbage, trespass, noparking, riskLow, riskMedium, riskHigh) {
    if (!window.Chart) return;

    const colors = {
      garbage: "#f59e0b",
      trespass: "#f43f5e",
      noparking: "#3b82f6",
      low: "#10b981",
      medium: "#f59e0b",
      high: "#f43f5e",
    };

    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = "'JetBrains Mono', monospace";
    Chart.defaults.font.size = 11;

    const barCtx = document.getElementById("barChart");
    const riskCtx = document.getElementById("riskChart");

    if (barChartInstance) barChartInstance.destroy();
    if (riskChartInstance) riskChartInstance.destroy();

    if (barCtx) {
      barChartInstance = new Chart(barCtx, {
        type: "bar",
        data: {
          labels: ["Littering", "Trespassing", "No Parking"],
          datasets: [{
            label: "Verified Incidents",
            data: [garbage, trespass, noparking],
            backgroundColor: [colors.garbage, colors.trespass, colors.noparking],
            borderRadius: 6,
            maxBarThickness: 54,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: "#0d131f",
              borderColor: "#222e44",
              borderWidth: 1,
              padding: 10,
              titleFont: { family: "'Space Grotesk', sans-serif", size: 12 },
              bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
            },
          },
          scales: {
            x: {
              grid: { display: false },
              border: { color: "#192233" },
              ticks: { font: { weight: "600" } },
            },
            y: {
              beginAtZero: true,
              ticks: { precision: 0 },
              grid: { color: "#131b29" },
              border: { color: "#192233" },
            },
          },
        },
      });
    }

    if (riskCtx) {
      riskChartInstance = new Chart(riskCtx, {
        type: "bar",
        data: {
          labels: ["Low (<40)", "Medium (40-69)", "High (≥70)"],
          datasets: [{
            label: "Severity Incidents",
            data: [riskLow, riskMedium, riskHigh],
            backgroundColor: [colors.low, colors.medium, colors.high],
            borderRadius: 6,
            maxBarThickness: 46,
          }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: "#0d131f",
              borderColor: "#222e44",
              borderWidth: 1,
              padding: 10,
              titleFont: { family: "'Space Grotesk', sans-serif", size: 12 },
              bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
            },
          },
          scales: {
            x: {
              beginAtZero: true,
              ticks: { precision: 0 },
              grid: { color: "#131b29" },
              border: { color: "#192233" },
            },
            y: {
              grid: { display: false },
              border: { color: "#192233" },
              ticks: { font: { weight: "600" } },
            },
          },
        },
      });
    }
  }

  const initialChartData = document.getElementById("chart-data");
  if (initialChartData) {
    renderCharts(
      Number(initialChartData.dataset.garbage || 0),
      Number(initialChartData.dataset.trespass || 0),
      Number(initialChartData.dataset.noparking || 0),
      Number(initialChartData.dataset.riskLow || 0),
      Number(initialChartData.dataset.riskMedium || 0),
      Number(initialChartData.dataset.riskHigh || 0)
    );
  }

  /* ==========================================================
     6. KPI Cards & Telemetry Updates
     ========================================================== */

  function flashHighlight(el) {
    if (!el) return;
    el.classList.remove("just-updated");
    void el.offsetWidth; // Force CSS reflow
    el.classList.add("just-updated");
  }

  function updateCounts(counts) {
    if (!counts) return;

    const countMap = {
      "kpi-total": counts.total,
      "kpi-today": counts.today !== undefined ? counts.today : 0,
      "kpi-garbage": counts.littering,
      "kpi-trespass": counts.trespassing,
      "kpi-noparking": counts.noparking,
      "kpi-highrisk": counts.high_risk,
      "nav-violations-count": counts.total,
      "notif-badge": counts.high_risk,
    };

    Object.entries(countMap).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (el && el.textContent !== String(val)) {
        el.textContent = val;
        flashHighlight(el.closest(".stat-card") || el);
      }
    });
  }

  function updateLiveStatus(cameraRunning) {
    // 1. Header status indicator
    const headerPill = document.getElementById("header-status-pill");
    const headerText = document.getElementById("header-status-text");
    if (headerPill) {
      headerPill.classList.toggle("is-live", cameraRunning);
      headerPill.classList.toggle("is-offline", !cameraRunning);
    }
    if (headerText) {
      headerText.textContent = cameraRunning ? "SYSTEM ONLINE" : "SYSTEM OFFLINE";
    }

    // 2. Sidebar stream status
    const sidebarStatus = document.getElementById("sidebar-camera-status");
    const sidebarDot = document.querySelector(".meta-stream-dot");
    if (sidebarStatus) {
      sidebarStatus.textContent = cameraRunning ? "LIVE STREAMING" : "SYSTEM OFFLINE";
    }
    if (sidebarDot) {
      sidebarDot.classList.toggle("live", cameraRunning);
      sidebarDot.classList.toggle("offline", !cameraRunning);
    }

    // 3. Sidebar nav pill
    const navLivePill = document.getElementById("nav-live-pill");
    if (navLivePill) {
      navLivePill.textContent = cameraRunning ? "REC" : "IDLE";
      navLivePill.className = `nav-pill ${cameraRunning ? "pill-online" : "pill-offline"}`;
    }

    // 4. CCTV HUD top telemetry
    const hudCamStatus = document.getElementById("hud-cam-status");
    const hudDot = document.querySelector(".hud-dot");
    if (hudCamStatus) {
      hudCamStatus.textContent = cameraRunning ? "CAMERA 01 · STREAMING 30 FPS" : "CAMERA 01 · SIGNAL DISCONNECTED";
    }
    if (hudDot) {
      hudDot.classList.toggle("hud-green", cameraRunning);
      hudDot.classList.toggle("hud-red", !cameraRunning);
    }

    // 5. Controls telemetry tags
    const metaCamera = document.getElementById("meta-camera");
    const metaDetection = document.getElementById("meta-detection");
    if (metaCamera) {
      metaCamera.textContent = cameraRunning ? "ONLINE" : "OFFLINE";
      metaCamera.className = `tag-val ${cameraRunning ? "val-online" : "val-offline"}`;
    }
    if (metaDetection) {
      metaDetection.textContent = cameraRunning ? "PROCESSING" : "STANDBY";
      metaDetection.className = `tag-val ${cameraRunning ? "val-online" : "val-offline"}`;
    }
  }

  /* ==========================================================
     7. Toast Notification Dispatch
     ========================================================== */

  function riskTierClass(risk) {
    if (risk >= 70) return "risk-high";
    if (risk >= 40) return "risk-medium";
    return "risk-low";
  }

  function showToast(event) {
    const stack = document.getElementById("toast-stack");
    if (!stack || !event) return;

    const toast = document.createElement("div");
    toast.className = `toast ${riskTierClass(event.risk || 0)}`;

    const confidencePct = Math.round((event.confidence || 0) * 100);
    const associationLine = (event.association !== undefined && event.association !== null)
      ? `Object Distance: <b>${Math.round(event.association * 100)}%</b><br>`
      : "";

    const evidenceFilename = event.evidence
      ? event.evidence.replace(/\\/g, "/").split("/").pop()
      : null;
    const evidenceUrl = event.evidence_url || (evidenceFilename ? `/evidence/${event.category || "garbage"}/${evidenceFilename}` : "");

    toast.innerHTML = `
      <div class="toast-head">
        <div class="toast-title">🚨 ${event.violation || "VIOLATION"} VERIFIED</div>
        <button class="toast-close" aria-label="Dismiss">&times;</button>
      </div>
      <div class="toast-body">
        Confidence: <b>${confidencePct}%</b><br>
        ${associationLine}
        Persistence: <b>${event.persistence || 1} frames</b><br>
        Risk Score: <b>${event.risk || 0}/100</b><br>
        Timestamp: <b>${event.timestamp || ""}</b>
      </div>
      ${evidenceUrl ? `<a class="toast-link" href="${evidenceUrl}" target="_blank" rel="noopener">Inspect Evidence Frame &rarr;</a>` : ""}
    `;

    stack.appendChild(toast);

    toast.querySelector(".toast-close").addEventListener("click", () => toast.remove());

    setTimeout(() => {
      if (toast.parentElement) toast.remove();
    }, 10000);
  }

  /* ==========================================================
     8. Reactive Fragment Swapping (/api/events)
     ========================================================== */

  async function refreshFragments() {
    try {
      const res = await fetch("/api/events");
      if (!res.ok) return;
      const data = await res.json();

      // 1. Active Incident Partial
      const incidentBodyWrap = document.getElementById("incident-body-wrap");
      if (incidentBodyWrap && data.active_event_html) {
        incidentBodyWrap.innerHTML = data.active_event_html;
        flashHighlight(incidentBodyWrap.closest(".active-incident-panel") || incidentBodyWrap);
      }

      // 2. Violations Table Partial
      const tableWrap = document.getElementById("violations-table-wrap");
      if (tableWrap && data.violations_table_html) {
        tableWrap.innerHTML = data.violations_table_html;
        applyTableFilters();
        flashHighlight(tableWrap);
      }

      // 3. Evidence Gallery Partial
      const gallery = document.getElementById("evidence-gallery");
      if (gallery && data.evidence_gallery_html) {
        gallery.innerHTML = data.evidence_gallery_html;
        makeEvidenceAccessible();
        flashHighlight(gallery);
      }

      // 4. System Diagnostics Partial
      const statusGrid = document.getElementById("system-status-grid");
      if (statusGrid && data.system_status_html) {
        statusGrid.innerHTML = data.system_status_html;
      }

      // 5. Counts & Chart.js Re-rendering
      if (data.counts) {
        updateCounts(data.counts);
        const rd = data.risk_distribution || { low: 0, medium: 0, high: 0 };
        renderCharts(
          data.counts.littering,
          data.counts.trespassing,
          data.counts.noparking,
          rd.low,
          rd.medium,
          rd.high
        );
      }
    } catch (err) {
      console.warn("Fragment refresh skipped:", err);
    }
  }

  async function fetchAndToastLatestEvent() {
    try {
      const res = await fetch("/api/latest-event");
      if (!res.ok) return;
      const event = await res.json();
      if (event && event.id) {
        showToast(event);
      }
    } catch (err) {
      console.warn("Toast dispatch skipped:", err);
    }
  }

  /* ==========================================================
     9. Heartbeat Polling Loop (/api/status)
     ========================================================== */

  async function pollStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();

      updateCounts(data.counts);
      updateLiveStatus(data.camera_running);

      if (data.last_event_id && data.last_event_id !== lastSeenEventId) {
        lastSeenEventId = data.last_event_id;
        await refreshFragments();
        await fetchAndToastLatestEvent();
      }
    } catch (err) {
      console.warn("Status poll error:", err);
    }
  }

  /* ==========================================================
     10. Perspective-Aware Zone Calibration Canvas
     ========================================================== */

  (function initZoneEditor() {
    const snapshotImg = document.getElementById("zone-snapshot");
    const canvas = document.getElementById("zone-canvas");
    if (!snapshotImg || !canvas) return;

    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("zone-editor-status");

    const zonePoints = {
      trespass_zone: (window.__INITIAL_ZONES__ && window.__INITIAL_ZONES__.trespass_zone) || [],
      no_parking_zone: (window.__INITIAL_ZONES__ && window.__INITIAL_ZONES__.no_parking_zone) || [],
    };

    let activeZone = "trespass_zone";

    const zoneColors = {
      trespass_zone: "rgba(244, 63, 94, 0.9)",
      no_parking_zone: "rgba(59, 130, 246, 0.9)",
    };
    const zoneFill = {
      trespass_zone: "rgba(244, 63, 94, 0.2)",
      no_parking_zone: "rgba(59, 130, 246, 0.2)",
    };

    function sizeCanvasToImage() {
      canvas.width = snapshotImg.naturalWidth || snapshotImg.width || 640;
      canvas.height = snapshotImg.naturalHeight || snapshotImg.height || 480;
      canvas.style.width = snapshotImg.clientWidth + "px";
      canvas.style.height = snapshotImg.clientHeight + "px";
      redraw();
    }

    function redraw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ["trespass_zone", "no_parking_zone"].forEach((zoneKey) => {
        const pts = zonePoints[zoneKey];
        if (!pts || pts.length === 0) return;

        ctx.beginPath();
        pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
        if (pts.length >= 3) ctx.closePath();

        ctx.strokeStyle = zoneColors[zoneKey];
        ctx.lineWidth = Math.max(2, canvas.width / 240);
        ctx.stroke();

        if (pts.length >= 3) {
          ctx.fillStyle = zoneFill[zoneKey];
          ctx.fill();
        }

        pts.forEach(([x, y]) => {
          ctx.beginPath();
          ctx.arc(x, y, Math.max(3, canvas.width / 160), 0, Math.PI * 2);
          ctx.fillStyle = zoneColors[zoneKey];
          ctx.fill();
        });
      });
    }

    function setStatus(msg, isError) {
      if (!statusEl) return;
      statusEl.textContent = msg;
      statusEl.style.color = isError ? "var(--trespass)" : "var(--online)";
    }

    if (snapshotImg.complete && snapshotImg.naturalWidth) {
      sizeCanvasToImage();
    }
    snapshotImg.addEventListener("load", sizeCanvasToImage);
    window.addEventListener("resize", () => {
      canvas.style.width = snapshotImg.clientWidth + "px";
      canvas.style.height = snapshotImg.clientHeight + "px";
    });

    canvas.addEventListener("click", (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = Math.round((e.clientX - rect.left) * scaleX);
      const y = Math.round((e.clientY - rect.top) * scaleY);

      zonePoints[activeZone].push([x, y]);
      redraw();
      setStatus(`Added vertex (${x}, ${y}). Total: ${zonePoints[activeZone].length} point(s).`);
    });

    const btnTrespass = document.getElementById("zone-toggle-trespass");
    const btnNoParking = document.getElementById("zone-toggle-noparking");
    const btnUndo = document.getElementById("zone-undo");
    const btnClear = document.getElementById("zone-clear");
    const btnRefresh = document.getElementById("zone-refresh");
    const btnSave = document.getElementById("zone-save");

    if (btnTrespass) {
      btnTrespass.addEventListener("click", () => {
        activeZone = "trespass_zone";
        btnTrespass.classList.add("active");
        if (btnNoParking) btnNoParking.classList.remove("active");
        setStatus("Active zone set to: Trespass Restricted Area");
      });
    }

    if (btnNoParking) {
      btnNoParking.addEventListener("click", () => {
        activeZone = "no_parking_zone";
        btnNoParking.classList.add("active");
        if (btnTrespass) btnTrespass.classList.remove("active");
        setStatus("Active zone set to: No Parking Enforcement Zone");
      });
    }

    if (btnUndo) {
      btnUndo.addEventListener("click", () => {
        if (zonePoints[activeZone].length > 0) {
          zonePoints[activeZone].pop();
          redraw();
          setStatus(`Undid last point. ${zonePoints[activeZone].length} remaining.`);
        }
      });
    }

    if (btnClear) {
      btnClear.addEventListener("click", () => {
        zonePoints[activeZone] = [];
        redraw();
        setStatus(`Cleared points for ${activeZone === "trespass_zone" ? "Trespass" : "No Parking"} zone.`);
      });
    }

    if (btnRefresh) {
      btnRefresh.addEventListener("click", () => {
        snapshotImg.src = "/api/snapshot?ts=" + Date.now();
        setStatus("Camera frame refreshed.");
      });
    }

    if (btnSave) {
      btnSave.addEventListener("click", async () => {
        if (zonePoints[activeZone].length < 3) {
          setStatus("A valid polygon requires at least 3 points.", true);
          return;
        }

        try {
          const res = await fetch("/api/zones", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ [activeZone]: zonePoints[activeZone] }),
          });
          const data = await res.json();

          if (data.ok) {
            setStatus(`Successfully persisted ${activeZone === "trespass_zone" ? "Trespass" : "No Parking"} zone geometry.`);
          } else {
            setStatus((data.errors && data.errors.join("; ")) || "Failed to save polygon.", true);
          }
        } catch (err) {
          setStatus("Could not connect to server to persist polygon.", true);
        }
      });
    }
  })();

  // Kick off heartbeat polling loop
  setInterval(pollStatus, POLL_INTERVAL_MS);
})();
