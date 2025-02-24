// API endpoints
const API = {
    BASE: '/api',
    MODELS: '/api/models',
    GENERATE: '/api/generate',
    STATUS: (jobId) => `/api/status/${jobId}`,
    IMAGE: (imageId) => `/api/get_image/${imageId}`,
    METADATA: (imageId) => `/api/image/${imageId}/metadata`,
    QUEUE: '/api/queue'
};

document.addEventListener('DOMContentLoaded', () => {
    initializeModels();
    initializeModalHandling();
    initializeGalleryHandling();
    initializeGenerationForm();
    initializeInfiniteScroll();
    initializeMobileNav();
});

// Initialize mobile navigation
function initializeMobileNav() {
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navToggle.classList.toggle('active');
            navLinks.classList.toggle('active');
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            }
        });

        // Close menu when clicking a link
        navLinks.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            });
        });
    }
}

// Initialize available models
async function initializeModels() {
    try {
        const response = await fetch(API.MODELS);
        const data = await response.json();
        const modelSelect = document.querySelector('select[name="model"]');
        if (modelSelect && data.models) {
            modelSelect.innerHTML = '<option value="">Select Model</option>';
            Object.entries(data.models).forEach(([id, info]) => {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = `${id} - ${info.description}`;
                if (id === data.default) option.selected = true;
                modelSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Modal handling
function initializeModalHandling() {
    const modal = document.querySelector('.modal');

    // Close modal when clicking X or outside
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('close-modal') || e.target === modal) {
            modal.classList.remove('visible');
        }
    });
}

// Gallery handling with infinite scroll
function initializeGalleryHandling() {
    const galleryGrid = document.querySelector('.gallery-grid');
    const modal = document.getElementById('fullscreenModal');
    const modalImage = document.getElementById('modalImage');
    const modelInfo = document.getElementById('modelInfo');
    const promptInfo = document.getElementById('promptInfo');
    const settingsInfo = document.getElementById('settingsInfo');
    const copyPromptBtn = document.getElementById('copyPrompt');
    const downloadBtn = document.getElementById('downloadImage');
    const closeBtn = modal.querySelector('.action-close');

    let currentImageId = null;

    function showModal(imageElement) {
        const galleryItem = imageElement.closest('.gallery-item');
        currentImageId = galleryItem.dataset.imageId;
        modalImage.src = imageElement.src;
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        // Reset info sections
        modelInfo.textContent = 'Loading...';
        promptInfo.textContent = 'Loading...';
        settingsInfo.textContent = 'Loading...';

        // Fetch image metadata
        fetch(`/api/image/${currentImageId}/metadata`)
            .then(response => response.json())
            .then(data => {
                modelInfo.textContent = data.model_id || 'Not available';
                promptInfo.textContent = data.prompt || 'Not available';

                // Format settings as a readable string
                const settings = Object.entries(data.settings || {})
                    .map(([key, value]) => `${key}: ${value}`)
                    .join('\n');
                settingsInfo.textContent = settings || 'Not available';
            })
            .catch(error => {
                modelInfo.textContent = 'Error loading details';
                promptInfo.textContent = 'Error loading details';
                settingsInfo.textContent = 'Error loading details';
                console.error('Error fetching image metadata:', error);
            });
    }

    function hideModal() {
        modal.style.display = 'none';
        document.body.style.overflow = '';
        currentImageId = null;
    }

    // Copy prompt functionality
    copyPromptBtn.addEventListener('click', () => {
        const promptText = promptInfo.textContent;
        if (promptText && promptText !== 'Loading...' && promptText !== 'Error loading details') {
            navigator.clipboard.writeText(promptText).then(() => {
                const originalText = copyPromptBtn.innerHTML;
                copyPromptBtn.innerHTML = '<span class="icon">✓</span>Copied!';
                setTimeout(() => {
                    copyPromptBtn.innerHTML = originalText;
                }, 2000);
            });
        }
    });

    // Download functionality
    downloadBtn.addEventListener('click', async () => {
        if (!currentImageId) return;

        try {
            const response = await fetch(`/api/get_image/${currentImageId}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cyberimage-${currentImageId}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            // Show download feedback
            const originalText = downloadBtn.innerHTML;
            downloadBtn.innerHTML = '<span class="icon">✓</span>Downloaded!';
            setTimeout(() => {
                downloadBtn.innerHTML = originalText;
            }, 2000);
        } catch (error) {
            console.error('Error downloading image:', error);
            downloadBtn.innerHTML = '<span class="icon">❌</span>Error';
            setTimeout(() => {
                downloadBtn.innerHTML = '<span class="icon">⬇️</span>Download Image';
            }, 2000);
        }
    });

    // Close modal on button click or outside click
    closeBtn.addEventListener('click', hideModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            hideModal();
        }
    });

    // Close modal on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.style.display === 'flex') {
            hideModal();
        }
    });

    // Handle gallery item clicks
    function handleGalleryClick(e) {
        const galleryItem = e.target.closest('.gallery-item');
        if (!galleryItem) return;

        const img = galleryItem.querySelector('img');
        if (img) {
            showModal(img);
        }
    }

    galleryGrid.addEventListener('click', handleGalleryClick);

    // Handle newly loaded images
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach((node) => {
                    if (node.classList && node.classList.contains('gallery-item')) {
                        const img = node.querySelector('img');
                        if (img) {
                            img.addEventListener('load', () => {
                                // Any additional handling for newly loaded images
                            });
                        }
                    }
                });
            }
        });
    });

    observer.observe(galleryGrid, { childList: true });
}

// Timer functionality
class GenerationTimer {
    constructor(feedbackSection) {
        this.startTime = null;
        this.timerInterval = null;
        this.timerDisplay = feedbackSection.querySelector('.timer-value');
        this.timerContainer = feedbackSection.querySelector('.generation-timer');

        if (!this.timerDisplay) {
            console.error('Timer display element not found');
        }
    }

    start() {
        if (!this.timerDisplay) return;
        this.startTime = Date.now();
        this.timerDisplay.classList.add('counting');
        if (this.timerContainer) {
            this.timerContainer.style.display = 'block';
        }
        this.update();
        this.timerInterval = setInterval(() => this.update(), 1000);
    }

    stop() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        if (this.timerDisplay) {
            this.timerDisplay.classList.remove('counting');
        }
    }

    reset() {
        this.stop();
        this.startTime = null;
        if (this.timerDisplay) {
            this.timerDisplay.textContent = '00:00';
        }
        if (this.timerContainer) {
            this.timerContainer.style.display = 'block';
        }
    }

    update() {
        if (!this.startTime || !this.timerDisplay) return;

        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;

        this.timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
}

// Initialize generation form handling
function initializeGenerationForm() {
    const form = document.getElementById('generate-form');
    if (!form) return;

    const feedbackSection = form.querySelector('.generation-feedback');
    if (!feedbackSection) {
        console.error('Feedback section not found');
        return;
    }

    // Remove any inline display:none style and ensure visibility
    feedbackSection.removeAttribute('style');
    feedbackSection.style.display = 'block';

    const timer = new GenerationTimer(feedbackSection);
    const statusText = feedbackSection.querySelector('.status-text');
    const queuePosition = feedbackSection.querySelector('.queue-position');
    const estimatedTime = feedbackSection.querySelector('.estimated-time');
    const timerContainer = feedbackSection.querySelector('.generation-timer');

    // Debug check for elements
    console.log('Form elements found:', {
        feedbackSection: feedbackSection !== null,
        statusText: statusText !== null,
        queuePosition: queuePosition !== null,
        estimatedTime: estimatedTime !== null,
        timerContainer: timerContainer !== null
    });

    // Initialize sliders
    const sliders = form.querySelectorAll('.slider');
    sliders.forEach(slider => {
        const display = slider.nextElementSibling;
        slider.addEventListener('input', () => {
            display.textContent = slider.value;
        });
    });

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitButton = form.querySelector('.button-generate');
        submitButton.disabled = true;

        // Hide any previous feedback
        feedbackSection.style.display = 'none';
        feedbackSection.classList.remove('active');

        try {
            const formData = new FormData(form);
            const requestData = {
                model_id: formData.get('model'),
                prompt: formData.get('prompt'),
                negative_prompt: formData.get('negative_prompt') || undefined,
                settings: {
                    num_images: parseInt(formData.get('num_images')),
                    num_inference_steps: parseInt(formData.get('steps')),
                    guidance_scale: parseFloat(formData.get('guidance')),
                    height: parseInt(formData.get('height')),
                    width: parseInt(formData.get('width'))
                }
            };

            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();
            if (data.job_id) {
                // Show feedback section only after successful job creation
                feedbackSection.style.display = 'block';
                feedbackSection.classList.add('active');

                // Set initial status
                if (statusText) {
                    statusText.style.display = 'block';
                    statusText.textContent = 'Preparing generation...';
                }

                // Ensure timer container is visible
                if (timerContainer) {
                    timerContainer.style.display = 'flex';
                    timerContainer.style.visibility = 'visible';
                }

                // Start timer
                timer.reset();
                timer.start();

                await pollGenerationStatus(data.job_id, feedbackSection, data.num_images || 1, timer);
                window.location.reload(); // Reload to show new images
            } else {
                updateGenerationStatus(`Error: ${data.message}`, 0);
                timer.stop();
            }
        } catch (error) {
            console.error('Error generating image:', error);
            updateGenerationStatus(`Error: ${error.message}`, 0);
            timer.stop();
        } finally {
            submitButton.disabled = false;
        }
    });

    // Handle prompt enrichment
    const enrichButton = document.getElementById('enrich-prompt');
    if (enrichButton) {
        enrichButton.addEventListener('click', handlePromptEnrichment);
    }
}

// Infinite scroll implementation
function initializeInfiniteScroll() {
    let page = 1;
    let loading = false;
    const gallery = document.querySelector('.gallery-grid');

    if (!gallery) return;

    const loadMoreImages = async () => {
        if (loading) return;

        const scrollPosition = window.innerHeight + window.scrollY;
        const contentHeight = document.documentElement.scrollHeight;

        if (scrollPosition >= contentHeight - 800) { // Pre-load threshold
            loading = true;
            try {
                const response = await fetch(`/gallery?page=${page}`, {
                    headers: {
                        'Accept': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                if (data.images && data.images.length > 0) {
                    appendImagesToGallery(data.images);
                    page++;
                }
            } catch (error) {
                console.error('Error loading more images:', error);
            } finally {
                loading = false;
            }
        }
    };

    // Debounced scroll handler
    let scrollTimeout;
    window.addEventListener('scroll', () => {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(loadMoreImages, 100);
    });
}

// Update generation status
function updateGenerationStatus(message, progress) {
    const feedbackSection = document.querySelector('.generation-feedback');
    if (!feedbackSection) {
        console.error('Feedback section not found');
        return;
    }

    const statusText = feedbackSection.querySelector('.status-text');
    const progressFill = feedbackSection.querySelector('.progress-fill');
    const timerContainer = feedbackSection.querySelector('.generation-timer');

    // Show feedback section and ensure it's visible
    feedbackSection.style.display = 'block';
    feedbackSection.style.visibility = 'visible';
    feedbackSection.classList.add('active');

    // Show and update status text
    if (statusText) {
        statusText.style.display = 'block';
        statusText.style.visibility = 'visible';
        statusText.textContent = message;
        console.log('Updated status text:', statusText.textContent);
    } else {
        console.error('Status text element not found');
    }

    // Show timer
    if (timerContainer) {
        timerContainer.style.display = 'flex';
        timerContainer.style.visibility = 'visible';
    } else {
        console.error('Timer container not found');
    }

    // Update progress bar
    if (progressFill) {
        progressFill.style.width = `${progress}%`;
        progressFill.style.visibility = 'visible';

        // Only show animation when actually generating
        if (progress > 0 && progress < 100) {
            progressFill.classList.add('active');
        } else {
            progressFill.classList.remove('active');
        }
    } else {
        console.error('Progress fill element not found');
    }

    // Log status update for debugging
    console.log('Status updated:', { message, progress, elements: {
        feedbackVisible: feedbackSection.style.display,
        statusTextVisible: statusText?.style.display,
        timerVisible: timerContainer?.style.display,
        progressWidth: progressFill?.style.width
    }});
}

// Status polling with enhanced feedback
async function pollGenerationStatus(jobId, feedbackSection, totalImages = 1, timer) {
    const statusText = feedbackSection.querySelector('.status-text');
    const progressFill = feedbackSection.querySelector('.progress-fill');
    const queuePosition = feedbackSection.querySelector('.queue-position');
    const estimatedTime = feedbackSection.querySelector('.estimated-time');

    let startTime = Date.now();
    let lastQueuePosition = null;
    let estimatedTimePerJob = 30000 * totalImages; // Initial estimate: 30 seconds per image
    let currentProgress = 0;
    let targetProgress = 0;

    // Function to smoothly animate progress
    const animateProgress = () => {
        if (currentProgress < targetProgress) {
            currentProgress = Math.min(currentProgress + 1, targetProgress);
            if (progressFill) {
                progressFill.style.width = `${currentProgress}%`;
            }
            if (currentProgress < targetProgress) {
                requestAnimationFrame(animateProgress);
            }
        }
    };

    // Function to update progress smoothly
    const updateProgress = (target) => {
        targetProgress = target;
        requestAnimationFrame(animateProgress);
    };

    while (true) {
        try {
            const [statusResponse, queueResponse] = await Promise.all([
                fetch(API.STATUS(jobId)),
                fetch(API.QUEUE)
            ]);

            if (!statusResponse.ok || !queueResponse.ok) {
                throw new Error('Failed to fetch status or queue information');
            }

            const statusData = await statusResponse.json();
            const queueData = await queueResponse.json();

            const currentPosition = queueData.pending + (statusData.status === 'pending' ? 1 : 0);

            // Update queue position display
            if (queuePosition) {
                if (currentPosition > 0) {
                    queuePosition.textContent = `Queue Position: ${currentPosition}`;
                    queuePosition.style.display = 'block';
                    queuePosition.style.visibility = 'visible';
                } else {
                    queuePosition.style.display = 'none';
                }
            }

            // Update estimated time with more accuracy
            if (estimatedTime) {
                if (currentPosition > 0) {
                    const estimatedMinutes = Math.ceil((currentPosition * estimatedTimePerJob) / 60000);
                    const timeText = estimatedMinutes === 1 ? '1 minute' :
                                  estimatedMinutes < 1 ? 'less than a minute' :
                                  `${estimatedMinutes} minutes`;
                    estimatedTime.textContent = `Estimated wait: ${timeText}`;
                    estimatedTime.style.display = 'block';
                    estimatedTime.style.visibility = 'visible';
                } else {
                    estimatedTime.style.display = 'none';
                }
            }

            // Update status based on job status with detailed progress
            switch (statusData.status) {
                case 'completed':
                    timer.stop();
                    updateGenerationStatus('Generation completed!', 100);
                    updateProgress(100);
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    feedbackSection.style.display = 'none';
                    return true;

                case 'failed':
                    timer.stop();
                    const errorMsg = statusData.message || 'Unknown error';
                    updateGenerationStatus(`Generation failed: ${errorMsg}`, 0);
                    updateProgress(0);
                    return false;

                case 'processing':
                    let progress = 50; // Base progress for processing
                    let message = 'Generating your image...';

                    // If we have detailed progress info
                    if (statusData.progress) {
                        if (statusData.progress.loading_model) {
                            progress = 35;
                            message = 'Loading AI model...';
                        } else if (statusData.progress.generating) {
                            progress = 65;
                            if (statusData.message && statusData.message.includes('Generating image')) {
                                message = statusData.message; // Use the specific progress message
                            }
                        } else if (statusData.progress.saving) {
                            progress = 85;
                            message = 'Saving generated image...';
                        }
                    }

                    updateGenerationStatus(message, progress);
                    updateProgress(progress);
                    break;

                case 'pending':
                    // Calculate progress based on queue position
                    const queueProgress = Math.max(5, 25 - (currentPosition * 2));
                    updateGenerationStatus('Waiting in queue...', queueProgress);
                    updateProgress(queueProgress);
                    break;

                default:
                    updateGenerationStatus('Unexpected status: ' + statusData.status, 0);
                    break;
            }

            // Update time estimates based on queue movement
            if (lastQueuePosition !== null && lastQueuePosition > currentPosition) {
                const timeElapsed = Date.now() - startTime;
                const jobsCompleted = lastQueuePosition - currentPosition;
                estimatedTimePerJob = timeElapsed / jobsCompleted;
            }
            lastQueuePosition = currentPosition;

            // Wait before next poll
            await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (error) {
            console.error('Error polling status:', error);
            // Don't fail immediately on network errors, retry a few times
            await new Promise(resolve => setTimeout(resolve, 2000));
            updateGenerationStatus('Checking status...', currentProgress);
        }
    }
}

// Initialize queue status polling
function initializeQueueStatusPolling() {
    const updateQueueStatus = async () => {
        try {
            const response = await fetch(API.QUEUE);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            // Update any global queue status indicators
            const { pending, processing, completed, failed, queue_size } = data;

            // Update UI elements if they exist
            const queueStatus = document.querySelector('.queue-status');
            if (queueStatus) {
                if (pending > 0 || processing > 0) {
                    queueStatus.textContent = `Queue: ${pending} waiting, ${processing} processing`;
                    queueStatus.style.display = 'block';
                } else {
                    queueStatus.style.display = 'none';
                }
            }
        } catch (error) {
            console.error('Error updating queue status:', error);
            // Don't show any error to the user, just hide queue elements
            const queueElements = document.querySelectorAll('.queue-status, .queue-position, .estimated-time');
            queueElements.forEach(el => {
                if (el) el.style.display = 'none';
            });
        }
    };

    // Update every 5 seconds
    setInterval(updateQueueStatus, 5000);
    updateQueueStatus(); // Initial update
}

// Utility functions
function formatDate(date) {
    return new Intl.DateTimeFormat('default', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        hour12: true
    }).format(date);
}

function formatDateLong(date) {
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

// Image details display
async function showImageDetails(imageId) {
    try {
        const response = await fetch(API.METADATA(imageId));
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        const modal = document.querySelector('.modal');
        const content = document.querySelector('.modal-content');

        const generationTime = new Date(data.generation_time * 1000);

        content.innerHTML = `
            <span class="close-modal">&times;</span>
            <img src="${API.IMAGE(imageId)}" alt="Generated Image" style="max-width: 100%; margin-bottom: 20px;">
            <div class="image-details">
                <p><strong>Generated:</strong> <span title="${formatDateLong(generationTime)}">${formatDate(generationTime)}</span></p>
                <p><strong>Model:</strong> ${data.model_id}</p>
                <p><strong>Prompt:</strong> ${data.prompt}</p>
                <p><strong>Settings:</strong></p>
                <pre>${JSON.stringify(data.settings, null, 2)}</pre>
            </div>
        `;

        modal.classList.add('visible');

        // Preload next image if available
        const nextItem = document.querySelector(`[data-image-id="${imageId}"]`).nextElementSibling;
        if (nextItem) {
            const nextId = nextItem.dataset.imageId;
            new Image().src = API.IMAGE(nextId);
        }
    } catch (error) {
        console.error('Error fetching image details:', error);
    }
}

// Gallery image appending
function appendImagesToGallery(images) {
    const gallery = document.querySelector('.gallery-grid');
    images.forEach(image => {
        const div = document.createElement('div');
        div.className = 'gallery-item';
        div.dataset.imageId = image.id;

        const createdAt = new Date(image.created_at);

        div.innerHTML = `
            <img src="${API.IMAGE(image.id)}" alt="${image.prompt.slice(0, 50)}..." loading="lazy">
            <div class="gallery-item-info">
                <p class="prompt">${image.prompt.slice(0, 100)}...</p>
                <p class="model">${image.model_id}</p>
                <p class="date" title="${formatDateLong(createdAt)}">${formatDate(createdAt)}</p>
            </div>
        `;

        // Add click handler to new items
        div.addEventListener('click', () => {
            // Remove highlight from all items
            document.querySelectorAll('.gallery-item').forEach(i => i.classList.remove('selected'));

            // Add highlight to clicked item
            div.classList.add('selected');

            // Scroll the item into view with smooth behavior
            div.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
                inline: 'center'
            });
        });

        gallery.appendChild(div);
    });
}

// Utility functions
function addNeonFlash(element) {
    element.style.boxShadow = 'var(--neon-green-glow)';
    setTimeout(() => {
        element.style.boxShadow = '';
    }, 1000);
}

// Handle prompt enrichment
async function handlePromptEnrichment(e) {
    e.preventDefault();
    const promptInput = document.getElementById('prompt-input');
    const currentPrompt = promptInput.value.trim();

    if (!currentPrompt) {
        return;
    }

    const enrichButton = e.target;
    const originalText = enrichButton.textContent;
    enrichButton.disabled = true;
    enrichButton.textContent = 'Enriching...';

    try {
        const response = await fetch('/api/enrich', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: currentPrompt })
        });

        const data = await response.json();
        if (data.enriched_prompt) {
            promptInput.value = data.enriched_prompt;
            addNeonFlash(promptInput);
        }
    } catch (error) {
        console.error('Error enriching prompt:', error);
    } finally {
        enrichButton.disabled = false;
        enrichButton.textContent = originalText;
    }
}