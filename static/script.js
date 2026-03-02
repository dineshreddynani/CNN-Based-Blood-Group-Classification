const form = document.getElementById('upload-form');
const preview = document.getElementById('preview-image');
const uploadText = document.getElementById('upload-text');
const loader = document.getElementById('loader');
const resultCard = document.getElementById('result-card');
const predictionText = document.getElementById('prediction-text');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');

document.getElementById('file-input').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = function(event) {
        preview.src = event.target.result;
        preview.classList.remove("hidden");
        uploadText.style.display = "none";
    };
    reader.readAsDataURL(file);
});

form.onsubmit = async function(e) {
    e.preventDefault();

    loader.classList.remove("hidden");
    resultCard.classList.add("hidden");

    const formData = new FormData(this);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    loader.classList.add("hidden");

    if (result.prediction) {
        predictionText.innerText = "Predicted: " + result.prediction;
        confidenceText.innerText = "Confidence: " + result.confidence;

        const percent = parseFloat(result.confidence);
        confidenceBar.style.width = percent + "%";

        resultCard.classList.remove("hidden");
    }
};
