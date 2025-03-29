document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const result = document.getElementById('result');
    let stream = null;
    let canvas = document.createElement('canvas');
    let context = canvas.getContext('2d');
    let isProcessing = false;
    let processingInterval = null;

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
            console.log('Camera started successfully');
            
            // Start processing frames
            isProcessing = true;
            processingInterval = setInterval(processFrame, 500); // Process every 500ms
        } catch (err) {
            console.error('Error accessing camera:', err);
            result.textContent = 'Error: Camera access denied';
            result.style.color = '#dc3545';
        }
    }

    // Process frame and send to server
    async function processFrame() {
        if (!isProcessing || !video.videoWidth) return;

        try {
            // Draw video frame to canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            console.log('Frame captured');

            // Convert canvas to blob
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, 'image/jpeg', 0.8);
            });

            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            console.log('Sending frame to server');
            const response = await fetch('/process_frame', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Server response:', data);

            if (data.error) {
                console.error('Server error:', data.error);
                result.textContent = 'Error';
                result.style.color = '#dc3545';
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
            result.style.color = '#dc3545';
        }
    }

    // Clean up when page is unloaded
    window.addEventListener('beforeunload', () => {
        isProcessing = false;
        if (processingInterval) {
            clearInterval(processingInterval);
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });

    // Start the application
    console.log('Starting camera...');
    startCamera();
}); 