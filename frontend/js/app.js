const API_URL = "https://fashion-recommender-1r1l.onrender.com";

const imageInput = document.getElementById("image-input");
const chooseBtn = document.getElementById("choose-btn");
const recommendBtn = document.getElementById("recommend-btn");

const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const dropArea = document.getElementById("drop-area");

const loading = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");

const results = document.getElementById("results");

const imageInfo = document.getElementById("image-info");

const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const resolution = document.getElementById("image-resolution");

const modal = document.getElementById("modal");
const modalImage = document.getElementById("modal-image");
const modalTitle = document.getElementById("modal-title");
const closeModal = document.getElementById("close-modal");

let selectedFile = null;

const loadingMessages = [
    "Extracting image features...",
    "Searching similar clothing...",
    "Calculating cosine similarity...",
    "Ranking recommendations...",
    "Preparing results..."
];

let loadingInterval;


// -------------------------------
// Upload Button
// -------------------------------

chooseBtn.onclick = () => imageInput.click();
dropArea.onclick = () => imageInput.click();


// -------------------------------
// Image Selection
// -------------------------------

imageInput.addEventListener("change", () => {

    if (!imageInput.files.length) return;

    selectedFile = imageInput.files[0];

    showPreview(selectedFile);

});


// -------------------------------
// Drag & Drop
// -------------------------------

dropArea.addEventListener("dragover", e => {

    e.preventDefault();

    dropArea.style.background = "#eef6ff";

});

dropArea.addEventListener("dragleave", () => {

    dropArea.style.background = "";

});

dropArea.addEventListener("drop", e => {

    e.preventDefault();

    dropArea.style.background = "";

    if (!e.dataTransfer.files.length) return;

    imageInput.files = e.dataTransfer.files;

    selectedFile = imageInput.files[0];

    showPreview(selectedFile);

});


// -------------------------------
// Preview
// -------------------------------

function showPreview(file){

    const reader = new FileReader();

    reader.onload = e => {

        preview.src = e.target.result;

        preview.style.display = "block";

        placeholder.style.display = "none";

    };

    reader.readAsDataURL(file);

    fileName.textContent = file.name;

    fileSize.textContent =
        (file.size / 1024).toFixed(1) + " KB";

    const img = new Image();

    img.onload = () => {

        resolution.textContent =
            `${img.width} × ${img.height}`;

    };

    img.src = URL.createObjectURL(file);

    imageInfo.classList.remove("hidden");

    recommendBtn.disabled = false;

}


// -------------------------------
// Recommendation Button
// -------------------------------

recommendBtn.addEventListener("click", async () => {

    if (!selectedFile) return;

    recommendBtn.disabled = true;
    chooseBtn.disabled = true;

    results.innerHTML = "";

    document.getElementById("summary-grid").innerHTML = "";
    document.getElementById("summary-panel").classList.add("hidden");

    loading.classList.remove("hidden");

    startLoadingMessages();

    const formData = new FormData();

    formData.append("file", selectedFile);

    try{

        const response = await fetch(
            API_URL + "/recommend",
            {
                method:"POST",
                body:formData
            }
        );

        const data = await response.json();

        stopLoadingMessages();

        loading.classList.add("hidden");

        displaySummary(data.processing);

        displayResults(data.recommendations);

    }

    catch(err){

        stopLoadingMessages();

        loading.classList.add("hidden");

        alert("Unable to connect to API.");

    }

    finally{

        recommendBtn.disabled = false;
        chooseBtn.disabled = false;

    }

});


// -------------------------------
// Loading Messages
// -------------------------------

function startLoadingMessages(){

    let i = 0;

    loadingText.textContent = loadingMessages[0];

    loadingInterval = setInterval(()=>{

        i++;

        loadingText.textContent =
            loadingMessages[i % loadingMessages.length];

    },1000);

}

function stopLoadingMessages(){

    clearInterval(loadingInterval);

}

// -------------------------------
// AI Processing Summary
// -------------------------------

function displaySummary(processing){

    const panel = document.getElementById("summary-panel");

    const grid = document.getElementById("summary-grid");

    panel.classList.remove("hidden");

    grid.innerHTML = "";

    const cards = [

        {
            title:"Dataset",
            value:`${processing.dataset_size.toLocaleString()} Images`
        },

        {
            title:"AI Model",
            value:processing.feature_extractor
        },

        {
            title:"Embedding Size",
            value:`${processing.embedding_size} Dimensions`
        },

        {
            title:"Similarity",
            value:processing.similarity_metric
        },

        {
            title:"Processing Time",
            value:`${processing.processing_time} sec`
        },

        {
            title:"Recommendations",
            value:processing.recommendations_returned
        }

    ];

    cards.forEach((card,index)=>{

        const div = document.createElement("div");

        div.className = "summary-card";

        div.innerHTML = `

            <h3>${card.title}</h3>

            <p>${card.value}</p>

        `;

        grid.appendChild(div);

        setTimeout(()=>{

            div.classList.add("show");

        },index*120);

    });

}

// -------------------------------
// Rank Information
// -------------------------------

function getRankBadge(rank){

    switch(rank){

        case 1:
            return "🥇 Best Match";

        case 2:
            return "🥈 Excellent";

        case 3:
            return "🥉 Very Similar";

        default:
            return `⭐ Rank #${rank}`;

    }

}


// -------------------------------
// Display Recommendations
// -------------------------------

function displayResults(items){

    results.innerHTML = "";

    items.forEach((item,index)=>{

        const card = document.createElement("div");

        card.className = "result-card";

        const badge = getRankBadge(item.rank);

        card.innerHTML = `

            <img
                src="${API_URL}${item.image_url}"
                alt="Recommendation ${item.rank}"
            >

            <div class="result-content">

                <div class="rank-badge">
                    ${badge}
                </div>

                <h3>Recommendation #${item.rank}</h3>

                <div class="progress-container">

                    <div
                        class="progress-bar"
                        style="width:${item.similarity}%"
                    ></div>

                </div>

                <p class="similarity">

                    ${item.similarity.toFixed(2)}%

                </p>

                <div class="filename">

                    ${item.filename}

                </div>

                <button class="view-btn">

                    View Full Image

                </button>

            </div>

        `;

        card.querySelector(".view-btn").onclick = (event)=>{

            event.stopPropagation();

            modalImage.src =
                `${API_URL}${item.image_url}`;

            modalTitle.textContent =
                item.filename;

            modal.classList.remove("hidden");

        };

        card.onclick = ()=>{

            modalImage.src =
                `${API_URL}${item.image_url}`;

            modalTitle.textContent =
                item.filename;

            modal.classList.remove("hidden");

        };

        results.appendChild(card);

        setTimeout(()=>{

            card.classList.add("show");

        },index*180);

    });

}


// -------------------------------
// Modal
// -------------------------------

closeModal.onclick = ()=>{

    modal.classList.add("hidden");

};

modal.onclick = e=>{

    if(e.target===modal){

        modal.classList.add("hidden");

    }

};