document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const result = document.getElementById('result');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    let stream = null;
    let isProcessing = false;

    // Start camera
    startBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640,
                    height: 480,
                    facingMode: 'user'
                } 
            });
            video.srcObject = stream;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            isProcessing = true;
            processFrame();
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Error accessing camera. Please make sure you have granted camera permissions.');
        }
    });

    // Stop camera
    stopBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            isProcessing = false;
            result.textContent = '-';
        }
    });

    // Process frames
    async function processFrame() {
        if (!isProcessing) return;

        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        try {
            // Convert canvas to blob
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
            
            // Create form data
            const formData = new FormData();
            formData.append('image', blob);

            // Send to backend
            const response = await fetch('/process_frame', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                result.textContent = data.sign || '-';
            }
        } catch (err) {
            console.error('Error processing frame:', err);
        }

        // Request next frame
        requestAnimationFrame(processFrame);
    }
}); 