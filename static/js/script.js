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

        resultsPanel.style.display = "block";

        results.forEach((item) => {
            const card = document.createElement("div");
            card.className = "result-card"; // Cần thêm CSS cho class này nếu chưa có

            // Xử lý ảnh thumbnail
            // Giả sử server trả về đường dẫn tương đối
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

            // Click vào kết quả để sang trang chi tiết
            card.addEventListener("click", () => {
                localStorage.setItem("selectedTrack", JSON.stringify(item));
                window.location.href = "/details";
            });

            // Thêm CSS inline nhỏ cho card kết quả (hoặc thêm vào style.css)
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

    // 1. Fill Basic Info
    setText("headerId", track.global_id || "Unknown");
    setText("displayId", track.global_id || "Unknown");
    setText("targetConf", (track.score * 100).toFixed(1) + "%");
    setText("nodeCount", String(track.tracks ? track.tracks.length : 0).padStart(2, "0"));

    const imgEl = document.getElementById("targetImg");
    if (imgEl) imgEl.src = track.thum_url || "";

    // 2. Build Timeline & Grid
    // Lấy danh sách camera unique và sort
    const uniqueCams = [...new Set(track.cameras)].sort();
    buildTimeline(uniqueCams, track);
    buildGrid(track.tracks || [], track);

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

// Helpers cho Detail Page
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function buildTimeline(camIds, rootTrack) {
    const timeline = document.getElementById("interactiveTimeline");
    if (!timeline) return;
    timeline.innerHTML = "";

    camIds.forEach((camId, idx) => {
        // Vị trí dot trên timeline
        const leftPct = camIds.length === 1 ? 50 : (idx / (camIds.length - 1)) * 100;

        const dot = document.createElement("div");
        dot.className = "timeline-dot";
        dot.style.left = `${leftPct}%`;
        dot.title = `CAM ${camId}`;

        // Tìm track tốt nhất cho cam này để play
        dot.onclick = () => {
            const bestTrack = rootTrack.tracks.find(t => String(t.cam_id) === String(camId));
            if (bestTrack) openVideoForTrack(rootTrack, bestTrack);
        };

        timeline.appendChild(dot);
    });
}

function buildGrid(tracks, rootTrack) {
    const grid = document.getElementById("cctvGrid");
    if (!grid) return;
    grid.innerHTML = "";

    tracks.forEach((t, idx) => {
        const card = document.createElement("div");
        card.className = "cctv-card";

        // Tạo thẻ card động
        card.innerHTML = `
            <div style="padding:8px; display:flex; justify-content:space-between; color:white; font-size:11px;">
                <strong>CAM-${t.cam_id}</strong>
                <span style="color:#ef4444">● REC</span>
            </div>
            <div style="width:100%; aspect-ratio:16/9; background:#000; display:flex; align-items:center; justify-content:center;">
                <img src="${rootTrack.thum_url}" style="width:100%; height:100%; object-fit:cover; opacity:0.6;">
                <div style="position:absolute; color:white; font-size:24px;">▶</div>
            </div>
            <div style="padding:10px; color:#9ca3af; font-size:11px;">
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

    // Prepare Payload khớp với app.py
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