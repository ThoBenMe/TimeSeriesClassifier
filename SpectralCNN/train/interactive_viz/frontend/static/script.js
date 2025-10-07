document.addEventListener("DOMContentLoaded", () => {
    const imageSelect = document.getElementById("imageSelect");
    const segmentationImage = document.getElementById("segmentationImage");
    const pixelCoordsDiv = document.getElementById("pixelCoords");
    const predictionsList = document.getElementById("predictionsList");
    const spectrumPlotDiv = document.getElementById("spectrumPlot");

    let currentImageSrc = segmentationImage.src;

    const updateImage = () => {
        const selectedMapType = imageSelect.value;
        const newImageSrc = `/image/${selectedMapType}.png?${Date.now()}`;
        segmentationImage.src = newImageSrc;
        currentImageSrc = newImageSrc; // update current image source tracker
        pixelCoordsDiv.textContent = "Hover over image...";
        predictionsList.innerHTML = "";
        Plotly.purge(spectrumPlotDiv); // clear spectrum plot
    };

    // init image load
    updateImage();

    // event listener for dropdown change
    imageSelect.addEventListener("change", updateImage);

    // event listener for mouse movement over the image
    segmentationImage.addEventListener("mousemove", async (event) => {
        const rect = segmentationImage.getBoundingClientRect();
        const imgWidth = segmentationImage.naturalWidth; // Get original image width
        const imgHeight = segmentationImage.naturalHeight; // Get original image height

        // calculate x, y relative to the image's original dimensions
        const x = Math.floor((event.clientX - rect.left) / rect.width * imgWidth);
        const y = Math.floor((event.clientY - rect.top) / rect.height * imgHeight);

        pixelCoordsDiv.textContent = `Pixel (${x}, ${y})`;

        // get the currently selected map type from the dropdown
        const selectedMapType = imageSelect.value;

        try {
            // include selectedMapType in the API call
            const response = await fetch(`/pixel_data/${selectedMapType}/${x}/${y}`);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }
            const data = await response.json();

            // update predictions list
            predictionsList.innerHTML = "";
            if (data.predictions && data.predictions.length > 0) {
                const predictionsHeader = document.querySelector('#pixelInfo h3');
                if (selectedMapType === "ground_truth_map") {
                    predictionsHeader.textContent = "Ground Truth Label:";
                    data.predictions.forEach(p => {
                        const li = document.createElement("li");
                        li.textContent = `${p.class_name} - ${(p.probability * 100).toFixed(2)}%`;
                        predictionsList.appendChild(li);
                    });
                } else if (selectedMapType === "predicted_seg_map_top1") {
                    predictionsHeader.textContent = "Top K Predictions:";
                     data.predictions.forEach(p => {
                        const li = document.createElement("li");
                        li.textContent = `${p.class_name} - ${(p.probability * 100).toFixed(2)}%`;
                        predictionsList.appendChild(li);
                    });
                } else if (selectedMapType === "uncertainty_map" || selectedMapType === "original") {
                    predictionsHeader.textContent = "Predicted Label (Top 1):";
                    data.predictions.forEach(p => {
                        const li = document.createElement("li");
                        li.textContent = `${p.class_name} - ${(p.probability * 100).toFixed(2)}%`;
                        predictionsList.appendChild(li);
                    });
                }
                else {
                    predictionsHeader.textContent = "Top K Predictions:";
                    data.predictions.forEach(p => {
                        const li = document.createElement("li");
                        li.textContent = `${p.class_name} - ${(p.probability * 100).toFixed(2)}%`;
                        predictionsList.appendChild(li);
                    });
                }

            } else {
                const predictionsHeader = document.querySelector('#pixelInfo h3');
                if (selectedMapType === "ground_truth_map") {
                    predictionsHeader.textContent = "Ground Truth Label:";
                    predictionsList.innerHTML = "<li>No ground truth data for this pixel.</li>";
                } else {
                    predictionsHeader.textContent = "Top K Predictions:";
                    predictionsList.innerHTML = "<li>No predictions for this pixel.</li>";
                }
            }

            Plotly.newPlot(spectrumPlotDiv, [{
                y: data.spectrum,
                type: 'scatter',
                mode: 'lines',
                marker: { color: 'blue' }
            }], {
                margin: { t: 30, b: 30, l: 30, r: 10 },
                xaxis: { title: 'Channel/Energy' },
                yaxis: { title: 'Intensity' },
                title: `Spectrum at (${x}, ${y})`,
                height: spectrumPlotDiv.clientHeight, 
                width: spectrumPlotDiv.clientWidth 
            }, { responsive: true });

        } catch (error) {
            console.error("Error fetching pixel data:", error);
            predictionsList.innerHTML = `<li>Error: ${error.message}</li>`;
            Plotly.purge(spectrumPlotDiv);
        }
    });

    segmentationImage.addEventListener("mouseleave", () => {
    });
});