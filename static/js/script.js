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

    // Check Cache
    const cachedResults = sessionStorage.getItem("lastSearchResults");
    if (cachedResults) {
        try {
            const results = JSON.parse(cachedResults);
            if (results && results.length > 0) renderResults(results);
        } catch (e) { console.error(e); }
    }

    uploadZone.addEventListener("click", () => fileUpload.click());
    fileUpload.addEventListener("change", (e) => {
        if (e.target.files[0]) {
            selectedFile = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (ev) => {
                document.getElementById("previewImage").src = ev.target.result;
                document.getElementById("previewContainer").style.display = "block";
                uploadZone.style.display = "none";
            };
            reader.readAsDataURL(selectedFile);
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
        searchBtn.disabled = !(textQuery.value.trim().length > 0 || selectedFile);
    }

    searchBtn.addEventListener("click", async () => {
        const formData = new FormData();
        formData.append("query", textQuery.value.trim());
        if (selectedFile) formData.append("file", selectedFile);

        sessionStorage.removeItem("lastSearchResults");
        loadingBadge.style.display = "flex";
        emptyState.style.display = "none";
        resultsPanel.style.display = "none";
        searchBtn.disabled = true;
        searchBtn.textContent = "Processing...";

        try {
            const response = await fetch("/api/search", { method: "POST", body: formData });
            if (!response.ok) throw new Error("Search failed");
            const results = await response.json();
            sessionStorage.setItem("lastSearchResults", JSON.stringify(results));
            renderResults(results);
        } catch (error) {
            alert(error.message);
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
            return;
        }

        emptyState.style.display = "none";
        resultsPanel.style.display = "block";

        results.forEach((item) => {
            const card = document.createElement("div");
            card.className = "result-card";
            const imgSrc = item.thum_url || "static/assets/SmartTraceRetrieval.png";

            card.innerHTML = `
                <div class="result-thumb">
                    <img src="${imgSrc}" alt="ID ${item.global_id}">
                </div>
                <div class="result-body">
                    <div class="result-meta">
                        <div class="result-id">ID: ${item.global_id}</div>
                        <div class="badge">${(item.score * 100).toFixed(1)}%</div>
                    </div>
                    <div class="result-sub">Cams: ${item.cameras.join(", ")}</div>
                    <div class="result-actions">
                        <button class="btn-detail">VIEW DETAILS</button>
                    </div>
                </div>
            `;

            // Lưu item vào localStorage để trang detail sử dụng
            card.addEventListener("click", () => {
                localStorage.setItem("selectedTrack", JSON.stringify(item));
                window.location.href = "/details";
            });
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
        window.location.href = "/";
        return;
    }

    const track = JSON.parse(trackStr);
    const allTracks = track.tracks || [];

    // --- HIỂN THỊ THÔNG TIN SIDEBAR ---
    setText("headerId", track.global_id);
    setText("displayId", track.global_id);
    setText("targetConf", (track.score * 100).toFixed(1) + "%");
    setText("nodeCount", String(allTracks.length).padStart(2, "0"));

    // --- HIỂN THỊ ẢNH TARGET (Cải thiện) ---
    const imgEl = document.getElementById("targetImg");
    if (imgEl) {
        // Ưu tiên thum_url từ track (ảnh được chọn ở trang index)
        const targetImageSrc = track.thum_url || "static/assets/SmartTraceRetrieval.png";

        // Ẩn ảnh trước khi load
        imgEl.style.display = 'none';
        imgEl.src = targetImageSrc;

        // Xử lý khi ảnh load thành công
        imgEl.onload = function() {
            this.style.display = 'block';
            console.log("Target image loaded successfully:", this.src);
        };

        // Xử lý khi ảnh lỗi - fallback về ảnh mặc định
        imgEl.onerror = function() {
            console.warn("Failed to load target image:", targetImageSrc);
            this.src = "static/assets/SmartTraceRetrieval.png";
            this.style.display = 'block';
        };
    }

    // --- KHỞI TẠO GRID & FILTER ---
    initFilterControls(allTracks, track);
    buildGrid(allTracks, track);
    buildStatisticsTable(allTracks);

    // Modal Events
    const modal = document.getElementById("videoModal");
    const closeBtn = document.getElementById("closeModal");
    if (closeBtn) closeBtn.onclick = closeModal;
    if (modal) modal.onclick = (e) => { if (e.target === modal) closeModal(); };
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val || "N/A";
}

function buildStatisticsTable(tracks) {
    const tableBody = document.getElementById("statsTableBody");
    if (!tableBody) return;
    tableBody.innerHTML = "";

    const stats = {};
    tracks.forEach(t => {
        if (!stats[t.seq_id]) stats[t.seq_id] = new Set();
        stats[t.seq_id].add(t.cam_id);
    });

    Object.keys(stats).sort().forEach(seq => {
        const cams = Array.from(stats[seq]).sort((a,b)=>a-b).join(", ");
        const row = document.createElement("tr");
        row.innerHTML = `
            <td style="padding:8px; border-bottom:1px solid #eee;">Seq ${seq}</td>
            <td style="padding:8px; border-bottom:1px solid #eee;">Cam ${cams}</td>
        `;
        tableBody.appendChild(row);
    });
}

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

    const updateCams = () => {
        const curSeq = seqSelect.value;
        camSelect.innerHTML = '<option value="all">All Cameras</option>';
        camSelect.disabled = (curSeq === "all");
        if (curSeq !== "all") {
            const cams = [...new Set(allTracks.filter(t => String(t.seq_id) === curSeq).map(t => t.cam_id))].sort((a,b)=>a-b);
            cams.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c;
                opt.textContent = `Camera ${c}`;
                camSelect.appendChild(opt);
            });
        }
    };

    seqSelect.addEventListener("change", () => { updateCams(); applyFilters(); });
    camSelect.addEventListener("change", applyFilters);

    function applyFilters() {
        let filtered = allTracks;
        if (seqSelect.value !== "all") filtered = filtered.filter(t => String(t.seq_id) === seqSelect.value);
        if (camSelect.value !== "all") filtered = filtered.filter(t => String(t.cam_id) === camSelect.value);
        buildGrid(filtered, rootTrack);
    }
}

function buildGrid(tracks, rootTrack) {
    const grid = document.getElementById("cctvGrid");
    if (!grid) return;
    grid.innerHTML = "";

    if (tracks.length === 0) {
        grid.innerHTML = `<p style="grid-column:1/-1; text-align:center; color:#888;">No videos match filters.</p>`;
        return;
    }

    tracks.forEach((t) => {
        const card = document.createElement("div");
        card.className = "cctv-card";
        // Ưu tiên dùng ảnh thumbnail của từng track, nếu không có thì dùng ảnh root
        const thumbSrc = t.thum_url || rootTrack.thum_url;

        card.innerHTML = `
            <div class="cctv-header">
                <strong>CAM-${t.cam_id}</strong>
                <span class="rec-dot">● REC</span>
            </div>
            <div class="video-thumb-container">
                <img src="${thumbSrc}" class="thumb-bg" onerror="this.src='${rootTrack.thum_url}'">
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

    videoPlayer.pause();
    videoPlayer.src = "";
    if (title) title.textContent = "LOADING...";
    modal.style.display = "flex";

    try {
        const res = await fetch("/api/get_video", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify({
                global_id: rootTrack.global_id,
                seq_id: t.seq_id,
                cam_id: t.cam_id,
                obj_id: t.obj_id,
                tracks: rootTrack.tracks
            })
        });
        const data = await res.json();
        if (res.ok && data.video_url) {
            if (title) title.textContent = `CAM-${t.cam_id} | SEQ-${t.seq_id}`;
            videoPlayer.src = data.video_url;
            videoPlayer.load();
            videoPlayer.play().catch(e=>console.log(e));
        } else {
            alert("Video Load Failed: " + (data.error || "Unknown"));
        }
    } catch (e) {
        console.error(e);
        if (title) title.textContent = "ERROR";
    }
}

function closeModal() {
    const modal = document.getElementById("videoModal");
    const videoPlayer = document.getElementById("modalVideoPlayer");
    if (modal) modal.style.display = "none";
    if (videoPlayer) { videoPlayer.pause(); videoPlayer.src = ""; }
}