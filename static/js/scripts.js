document.addEventListener('DOMContentLoaded', () => {
    // Handle cloud cover prediction form
    document.getElementById('cloudcover-form').addEventListener('submit', async function (event) {
        event.preventDefault();
        const formData = new FormData();
        formData.append('image', document.getElementById('image-upload').files[0]);

        const response = await fetch('/predict/cloudcover', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        document.getElementById('cloudcover-result').innerText = `Predicted Cloud Cover: ${data.cloud_cover.toFixed(2)}%`;
    });

    // Handle temperature prediction form
    document.getElementById('temperature-form').addEventListener('submit', async function (event) {
        event.preventDefault();
        const cloudCover = parseFloat(document.getElementById('cloud-cover').value);
        const humidity = parseFloat(document.getElementById('humidity').value);
        const condition = document.getElementById('condition').value;

        const response = await fetch('/predict/temperature', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ cloud_cover: cloudCover, humidity: humidity, condition: condition })
        });

        const data = await response.json();
        document.getElementById('temperature-result').innerText = `Max Temp: ${data.max_temp.toFixed(2)}°C, Min Temp: ${data.min_temp.toFixed(2)}°C`;
    });
});
