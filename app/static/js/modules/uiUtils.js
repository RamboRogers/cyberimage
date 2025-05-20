// UI Utility Functions

export function formatDate(date) {
    return new Intl.DateTimeFormat('default', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        hour12: true
    }).format(date);
}

export function formatDateLong(date) {
    return new Intl.DateTimeFormat('default', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        second: 'numeric',
        timeZoneName: 'short'
    }).format(date);
}

export function addNeonFlash(element) {
    element.style.boxShadow = 'var(--neon-green-glow)';
    setTimeout(() => {
        element.style.boxShadow = '';
    }, 1000);
}

export function showMainFeedback(message, type = 'info') {
    let feedbackDiv = document.getElementById('main-feedback-message');
    if (!feedbackDiv) {
        feedbackDiv = document.createElement('div');
        feedbackDiv.id = 'main-feedback-message';
        feedbackDiv.style.position = 'fixed';
        feedbackDiv.style.bottom = '20px';
        feedbackDiv.style.left = '50%';
        feedbackDiv.style.transform = 'translateX(-50%)';
        feedbackDiv.style.padding = '10px 20px';
        feedbackDiv.style.borderRadius = 'var(--border-radius)';
        feedbackDiv.style.zIndex = '3000';
        feedbackDiv.style.opacity = '0';
        feedbackDiv.style.transition = 'opacity 0.3s ease';
        document.body.appendChild(feedbackDiv);
    }

    feedbackDiv.textContent = message;
    if (type === 'success') {
        feedbackDiv.style.backgroundColor = 'rgba(57, 255, 20, 0.8)'; // Neon green bg
        feedbackDiv.style.color = 'black';
        feedbackDiv.style.border = '1px solid var(--neon-green)';
    } else if (type === 'error') {
        feedbackDiv.style.backgroundColor = 'rgba(255, 68, 68, 0.8)'; // Red bg
        feedbackDiv.style.color = 'white';
        feedbackDiv.style.border = '1px solid #ff4444';
    } else {
        feedbackDiv.style.backgroundColor = 'rgba(30, 30, 30, 0.9)'; // Dark bg for info
        feedbackDiv.style.color = 'white';
        feedbackDiv.style.border = '1px solid var(--neon-green-dim)';
    }

    // Fade in
    setTimeout(() => { feedbackDiv.style.opacity = '1'; }, 10);

    // Clear previous timeout if exists
    if (feedbackDiv.dataset.timeoutId) {
        clearTimeout(parseInt(feedbackDiv.dataset.timeoutId));
    }

    // Set timeout to fade out
    const timeoutId = setTimeout(() => {
        feedbackDiv.style.opacity = '0';
        // Optional: remove element after fade out
        // setTimeout(() => { feedbackDiv.remove(); }, 300);
    }, 3000); // Display for 3 seconds
    feedbackDiv.dataset.timeoutId = timeoutId.toString();
}

// Helper function to format time in seconds to a readable format
// This was already in main.js, but it fits well with other UI utils.
export function formatTime(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)} sec`;
    } else if (seconds < 3600) {
        return `${Math.round(seconds / 60)} min`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.round((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
} 