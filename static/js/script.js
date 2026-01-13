document.addEventListener("DOMContentLoaded", () => {
    const path = window.location.pathname;

    if (path === "/" || path === "/index.html") {
        initIndexPage();
    } else if (path.includes("/details")) {
        initDetailPage();
    }
});

/* =========================================
   1. LOGIC TRANG CHỦ (INDEX)
   ========================================= */
function initIndexPage() {
    const searchBtn = document.getElementById("searchBtn");
    const textQuery = document.getElementById("textQuery");
    const fileUpload = document.getElementById("fileUpload");
    const uploadZone = document.getElementById("uploadZone");
    const removeImageBtn = document.getElementById("removeImageBtn");
    const resultsGrid = document.getElementById("resultsGrid");
    const resultsPanel = document.getElementById("resultsPanel");
    const loadingBadge = document.getElementById("loadingBadge");
    const emptyState = document.getElementById("emptyState");

    let selectedFile = null;

    // --- YÊU CẦU 1: KHÔI PHỤC KẾT QUẢ CŨ NẾU CÓ ---
    const cachedResults = sessionStorage.getItem("lastSearchResults");
    if (cachedResults) {
        try {
            const results = JSON.parse(cachedResults);
            if (results && results.length > 0) {
                renderResults(results);
            }
        } catch (e) {
            console.error("Cache error", e);
        }
    }

    // Handle File Selection
    uploadZone.addEventListener("click", () => fileUpload.click());
    fileUpload.addEventListener("change", (e) => {
        if (e.target.files[0]) {
            selectedFile = e.target.files[0];
            showPreview(selectedFile);
            checkEnableSearch();
        }
    });

    removeImageBtn.addEventListener("click", () => {
        selectedFile = null;
        fileUpload.value = "";
        document.getElementById("previewContainer").style.display = "none";
        uploadZone.style.display = "flex";
        checkEnableSearch();
    });

    textQuery.addEventListener("input", checkEnableSearch);

    function checkEnableSearch() {
        const hasText = textQuery.value.trim().length > 0;
        searchBtn.disabled = !(hasText || selectedFile);
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById("previewImage").src = e.target.result;
            document.getElementById("previewContainer").style.display = "block";
            uploadZone.style.display = "none";
        };
        reader.readAsDataURL(file);
    }

    // Handle Search Action
    searchBtn.addEventListener("click", async () => {
        const formData = new FormData();
        formData.append("query", textQuery.value.trim());
        if (selectedFile) {
            formData.append("file", selectedFile);
        }

        // Xóa cache cũ khi search mới
        sessionStorage.removeItem("lastSearchResults");

        // UI State: Loading
        loadingBadge.style.display = "flex";
        emptyState.style.display = "none";
        resultsPanel.style.display = "none";
        searchBtn.disabled = true;
        searchBtn.textContent = "Processing...";

        try {
            const response = await fetch("/api/search", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Search failed");

            const results = await response.json();

            // Lưu cache mới
            sessionStorage.setItem("lastSearchResults", JSON.stringify(results));

            renderResults(results);

        } catch (error) {
            alert("Error during search: " + error.message);
            emptyState.style.display = "flex";
        } finally {
            loadingBadge.style.display = "none";
            searchBtn.disabled = false;
            searchBtn.textContent = "Run Query";
        }
    });

    function renderResults(results) {
        resultsGrid.innerHTML = "";
        document.getElementById("resultsCount").textContent = `Found ${results.length} Matches`;

        if (results.length === 0) {
            emptyState.style.display = "flex";
            document.querySelector(".empty-title").textContent = "No Matches Found";
            return;
        }

        emptyState.style.display = "none"; // Ẩn empty state
        resultsPanel.style.display = "block";

        results.forEach((item) => {
            const card = document.createElement("div");
            card.className = "result-card";

            const imgSrc = item.thum_url || "";
            const scorePct = (item.score * 100).toFixed(1) + "%";

            card.innerHTML = `
                <div class="card-img-wrapper">
                    <img src="${imgSrc}" alt="ID ${item.global_id}" onerror="this.src='static/assets/SmartTraceRetrieval.png'">
                    <div class="card-score">${scorePct}</div>
                </div>
                <div class="card-info">
                    <h3>ID: ${item.global_id}</h3>
                    <p>Cams: ${item.cameras.join(", ")}</p>
                </div>
            `;

            card.addEventListener("click", () => {
                localStorage.setItem("selectedTrack", JSON.stringify(item));
                window.location.href = "/details";
            });

            // Inline styles giữ nguyên hoặc chuyển vào CSS
            card.style.cssText = "background: #1e1e1e; border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.2s;";
            const imgWrap = card.querySelector('.card-img-wrapper');
            imgWrap.style.cssText = "height: 200px; width: 100%; position: relative;";
            card.querySelector('img').style.cssText = "width: 100%; height: 100%; object-fit: cover;";
            card.querySelector('.card-info').style.cssText = "padding: 10px; color: #fff;";

            resultsGrid.appendChild(card);
        });
    }
}


/* =========================================
   2. LOGIC TRANG CHI TIẾT (DETAILS)
   ========================================= */
function initDetailPage() {
    const trackStr = localStorage.getItem("selectedTrack");
    if (!trackStr) {
        alert("No track selected. Redirecting to home.");
        window.location.href = "/";
        return;
    }

    const track = JSON.parse(trackStr);
    const allTracks = track.tracks || [];

    // 1. Fill Basic Info
    setText("headerId", track.global_id || "Unknown");
    setText("displayId", track.global_id || "Unknown");
    setText("targetConf", (track.score * 100).toFixed(1) + "%");
    setText("nodeCount", String(allTracks.length).padStart(2, "0"));

    const imgEl = document.getElementById("targetImg");
    if (imgEl) imgEl.src = track.thum_url || "";

    // 2. YÊU CẦU 3: Bảng Thống Kê & Bộ lọc
    buildStatisticsTable(allTracks);
    initFilterControls(allTracks, track);

    // Mặc định hiển thị tất cả ban đầu
    buildGrid(allTracks, track);

    // 3. Modal Close Logic
    const closeBtn = document.getElementById("closeModal");
    const modal = document.getElementById("videoModal");

    if (closeBtn) closeBtn.onclick = closeModal;
    if (modal) {
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };
    }
}

// Helper: Build Statistics Table Sidebar
function buildStatisticsTable(tracks) {
    const tableBody = document.getElementById("statsTableBody");
    if (!tableBody) return;
    tableBody.innerHTML = "";

    // Group by Seq -> List of Cams
    const stats = {};
    tracks.forEach(t => {
        const s = t.seq_id;
        const c = t.cam_id;
        if (!stats[s]) stats[s] = new Set();
        stats[s].add(c);
    });

    // Render row
    Object.keys(stats).sort().forEach(seq => {
        const cams = Array.from(stats[seq]).sort((a, b) => a - b).join(", ");
        const row = document.createElement("tr");
        row.innerHTML = `
            <td style="padding: 8px; border-bottom: 1px solid #eee;">Seq ${seq}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">Cam ${cams}</td>
        `;
        tableBody.appendChild(row);
    });
}

// Helper: Init Filter Controls (Seq & Cam Selectors)
function initFilterControls(allTracks, rootTrack) {
    const seqSelect = document.getElementById("seqSelect");
    const camSelect = document.getElementById("camSelect");

    if(!seqSelect || !camSelect) return;

    const uniqueSeqs = [...new Set(allTracks.map(t => t.seq_id))].sort();

    seqSelect.innerHTML = '<option value="all">All Sequences</option>';
    uniqueSeqs.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = `Sequence ${s}`;
        seqSelect.appendChild(opt);
    });

    function updateCamOptions() {
        const currentSeq = seqSelect.value;
        camSelect.innerHTML = '<option value="all">All Cameras</option>';
        camSelect.disabled = (currentSeq === "all");

        if (currentSeq !== "all") {
            const camsInSeq = allTracks
                .filter(t => String(t.seq_id) === String(currentSeq))
                .map(t => t.cam_id);
            const uniqueCams = [...new Set(camsInSeq)].sort((a,b)=>a-b);

            uniqueCams.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c;
                opt.textContent = `Camera ${c}`;
                camSelect.appendChild(opt);
            });
        }
    }

    seqSelect.addEventListener("change", () => {
        updateCamOptions();
        applyFilters();
    });

    camSelect.addEventListener("change", () => {
        applyFilters();
    });

    function applyFilters() {
        const sVal = seqSelect.value;
        const cVal = camSelect.value;

        let filtered = allTracks;

        if (sVal !== "all") {
            filtered = filtered.filter(t => String(t.seq_id) === String(sVal));
        }
        if (cVal !== "all") {
            filtered = filtered.filter(t => String(t.cam_id) === String(cVal));
        }

        // Đảm bảo rootTrack luôn có tracks ban đầu
        buildGrid(filtered, rootTrack);
    }
}


function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

// YÊU CẦU 2 & 3: Grid hiển thị video
function buildGrid(tracks, rootTrack) {
    const grid = document.getElementById("cctvGrid");
    if (!grid) return;
    grid.innerHTML = "";

    if (tracks.length === 0) {
        grid.innerHTML = `<p style="color:#666; grid-column: 1/-1; text-align:center;">No videos match the selected filters.</p>`;
        return;
    }

    tracks.forEach((t) => {
        const card = document.createElement("div");
        card.className = "cctv-card";

        // Fix layout: play button center (xử lý ở CSS .play-overlay)
        card.innerHTML = `
            <div class="cctv-header">
                <strong>CAM-${t.cam_id}</strong>
                <span class="rec-dot">● REC</span>
            </div>
            <div class="video-thumb-container">
                <img src="${rootTrack.thum_url}" class="thumb-bg">
                <div class="play-overlay">▶</div>
            </div>
            <div class="cctv-footer">
                SEQ-${t.seq_id} | OBJ-${t.obj_id}
            </div>
        `;

        card.addEventListener("click", () => openVideoForTrack(rootTrack, t));
        grid.appendChild(card);
    });
}

async function openVideoForTrack(rootTrack, t) {
    const modal = document.getElementById("videoModal");
    const videoPlayer = document.getElementById("modalVideoPlayer");
    const title = document.getElementById("modalNodeTitle");

    if (!modal || !videoPlayer) return;

    // Reset Player
    videoPlayer.pause();
    videoPlayer.src = "";
    if (title) title.textContent = "LOADING EVIDENCE...";
    modal.style.display = "flex";

    const payload = {
        global_id: rootTrack.global_id,
        seq_id: t.seq_id,
        cam_id: t.cam_id,
        obj_id: t.obj_id,
        tracks: rootTrack.tracks
    };

    try {
        const resp = await fetch("/api/get_video", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await resp.json();

        if (resp.ok && data.video_url) {
            if (title) title.textContent = `CAM-${t.cam_id} | SEQ-${t.seq_id}`;
            videoPlayer.src = data.video_url;
            videoPlayer.load();
            videoPlayer.play().catch(e => console.log("Auto-play blocked", e));
        } else {
            if (title) title.textContent = "VIDEO FAILED";
            alert("Could not load video: " + (data.error || "Unknown error"));
        }
    } catch (e) {
        console.error(e);
        if (title) title.textContent = "NETWORK ERROR";
    }
}

function closeModal() {
    const modal = document.getElementById("videoModal");
    const videoPlayer = document.getElementById("modalVideoPlayer");
    if (modal) modal.style.display = "none";
    if (videoPlayer) {
        videoPlayer.pause();
        videoPlayer.src = "";
    }
}