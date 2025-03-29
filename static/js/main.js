document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const result = document.getElementById('result');
    let stream = null;
    let canvas = document.createElement('canvas');
    let context = canvas.getContext('2d');
    let isProcessing = false;

    // Set canvas size
    canvas.width = 640;
    canvas.height = 480;

    // Start camera
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 640,
                    height: 480,
                    facingMode: 'user'
                }
            });
            video.srcObject = stream;
            await video.play();
            isProcessing = true;
            processFrame();
        } catch (err) {
            console.error('Error accessing camera:', err);
            result.textContent = 'Error: Camera access denied';
        }
    }

    // Process frame and send to server
    async function processFrame() {
        if (!isProcessing || !video.videoWidth) return;

        // Draw video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to blob
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            try {
                const response = await fetch('/process_frame', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                if (data.error) {
                    console.error('Server error:', data.error);
                    return;
                }

                // Update result with confidence if available
                if (data.confidence) {
                    const confidence = Math.round(data.confidence * 100);
                    result.textContent = `${data.sign} (${confidence}%)`;
                } else {
                    result.textContent = data.sign;
                }

                // Add visual feedback based on confidence
                if (data.confidence && data.confidence > 0.5) {
                    result.style.color = '#198754'; // Green for high confidence
                } else {
                    result.style.color = '#6c757d'; // Gray for low confidence
                }
            } catch (err) {
                console.error('Error processing frame:', err);
                result.textContent = 'Error';
                result.style.color = '#dc3545'; // Red for errors
            }

            // Process next frame
            requestAnimationFrame(processFrame);
        }, 'image/jpeg', 0.8);
    }

    // Clean up when page is unloaded
    window.addEventListener('beforeunload', () => {
        isProcessing = false;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });

    // Start the application
    startCamera();
}); 