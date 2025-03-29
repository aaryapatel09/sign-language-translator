document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const result = document.getElementById('result');
    let stream = null;
    let canvas = document.createElement('canvas');
    let context = canvas.getContext('2d');

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
            video.play();
            processFrame();
        } catch (err) {
            console.error('Error accessing camera:', err);
            result.textContent = 'Error: Camera access denied';
        }
    }

    // Process frame and send to server
    async function processFrame() {
        if (!video.videoWidth) return;

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

                result.textContent = data.sign;
            } catch (err) {
                console.error('Error processing frame:', err);
            }

            // Process next frame
            requestAnimationFrame(processFrame);
        }, 'image/jpeg', 0.8);
    }

    // Clean up when page is unloaded
    window.addEventListener('beforeunload', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });

    // Start the application
    startCamera();
}); 