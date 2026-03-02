const form = document.getElementById('upload-form');
const preview = document.getElementById('preview-image');
const previewContainer = document.getElementById('preview-container');
const resultDiv = document.getElementById('result');
const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');
const barFill = document.getElementById('bar-fill');

document.getElementById('file-input').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onload = function(event) {
        preview.src = event.target.result;
        previewContainer.classList.remove("hidden");
    };

    reader.readAsDataURL(file);
});

form.onsubmit = async function(e) {
    e.preventDefault();

    predictionText.innerText = "Processing...";
    confidenceText.innerText = "";
    barFill.style.width = "0%";
    resultDiv.classList.remove("hidden");

    try {
        const formData = new FormData(this);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.prediction) {
            predictionText.innerText = "Prediction: " + result.prediction;
            confidenceText.innerText = "Confidence: " + result.confidence;

            const percent = parseFloat(result.confidence);
            barFill.style.width = percent + "%";
        } else {
            predictionText.innerText = "Error: " + result.error;
        }

    } catch (err) {
        predictionText.innerText = "Server error. Check logs.";
    }
};
