let video = null;
let canvas = null;
let ctx = null;
let stream = null;
let sessionId = null;
let livenessInterval = null;

async function startLiveness() {
    const res = await fetch('/Verification/LivenessStart', {
        method: 'POST',
    });
    const data = await res.json();
    sessionId = data.session_id;
    console.log("Session started:", sessionId);

    video = document.getElementById('livenessVideo');
    canvas = document.createElement('canvas');
    ctx = canvas.getContext('2d');

    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();

    livenessInterval = setInterval(captureFrame, 500);
}

// ==================== CAPTURE AND SEND FRAME ====================
async function captureFrame() {
    if (!video || !sessionId) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        try {
            const res = await fetch(`/Verification/LivenessFrame?sessionId=${encodeURIComponent(sessionId)}`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            console.log("Frame response:", data);

            if (data.progress !== undefined) {
                const bar = document.getElementById('livenessProgress');
                if (bar) bar.value = data.progress;
            }

            if (data.stage === "done" && data.passed) {
                console.log("Liveness passed!");
                clearInterval(livenessInterval);

                const hiddenField = document.getElementById('livenessSessionId');
                if (hiddenField) {
                    hiddenField.value = sessionId;
                }

                alert("Liveness check passed! You can now submit your ID and selfie.");
            }

        } catch (err) {
            console.error("Error sending frame:", err);
        }
    }, 'image/jpeg');
}

async function cancelLiveness() {
    if (!sessionId) return;

    await fetch(`/Verification/LivenessCancel?sessionId=${encodeURIComponent(sessionId)}`, {
        method: 'POST'
    });
    clearInterval(livenessInterval);
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    console.log("Liveness session cancelled");
    sessionId = null;
}

window.startLiveness = startLiveness;
window.cancelLiveness = cancelLiveness;

async function submitIdAndSelfie() {
    if (!sessionId) {
        alert("Liveness not completed! Please complete the liveness check first.");
        return;
    }

    const idFile = document.getElementById("idInput").files[0];
    const selfieFile = document.getElementById("selfieInput").files[0];

    if (!idFile || !selfieFile) {
        alert("Please select both ID and Selfie files");
        return;
    }

    const formData = new FormData();
    formData.append('id_image', idFile);
    formData.append('selfie_image', selfieFile);
    formData.append('session_id', sessionId);

    try {
        const res = await fetch('/Verification/Verify', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        console.log("Verification result:", data);
        alert(JSON.stringify(data));
    } catch (err) {
        console.error("Error submitting ID + selfie:", err);
    }
}

window.submitIdAndSelfie = submitIdAndSelfie;
