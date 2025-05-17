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

// Global store for available models
let availableImageModels = {};
let availableModels = {}; // Added back: Global store for ALL models (used by modals)

// --- Revert API endpoint constants ---
const API_IMAGE_GEN = '/api/generate';
const API_I2V_GEN = '/api/generate_video';
const API_T2V_GEN = '/api/generate_t2v'; // <-- ADDED for Text-to-Video

// --- Added global hideModal function --- //
function hideModal() {
    const modal = document.getElementById('fullscreenModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }
}

// --- Added Back Missing Function Definitions --- //

// Modal handling
function initializeModalHandling() {
    const modal = document.querySelector('.modal'); // General modal selector
    const fullscreenModal = document.getElementById('fullscreenModal'); // Specific fullscreen modal

    // Close modal when clicking X or outside
    document.addEventListener('click', (e) => {
        // Close general modal
        if (modal && (e.target.classList.contains('close-modal') || e.target === modal)) {
            modal.classList.remove('visible');
        }
        // Close fullscreen modal (using its specific close button or clicking background)
        if (fullscreenModal && (e.target.classList.contains('action-close') || e.target === fullscreenModal)) {
             if (fullscreenModal.style.display === 'flex') { // Only hide if visible
                 hideModal(); // Use the specific hide function for fullscreen
             }
        }
        // Close delete confirm modal
        const deleteModal = document.getElementById('deleteConfirmModal');
        if (deleteModal && (e.target.id === 'cancelDelete' || e.target === deleteModal)) {
             deleteModal.style.display = 'none';
        }
        // Close video gen modal (using its specific close button)
        const videoGenModal = document.getElementById('videoGenModal');
        if (videoGenModal && e.target.closest('.action-close') && videoGenModal.contains(e.target.closest('.action-close'))) {
            closeVideoGenModal();
        }

    });

    // Handle Escape key for fullscreen modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && fullscreenModal && fullscreenModal.style.display === 'flex') {
            hideModal();
        }
        const deleteModal = document.getElementById('deleteConfirmModal');
        if (e.key === 'Escape' && deleteModal && deleteModal.style.display === 'block') {
            deleteModal.style.display = 'none';
        }
         const videoGenModal = document.getElementById('videoGenModal');
         if (e.key === 'Escape' && videoGenModal && videoGenModal.style.display === 'block') {
            closeVideoGenModal();
        }
    });
}

// Gallery handling with infinite scroll
function initializeGalleryHandling() {
    const galleryGrid = document.querySelector('.gallery-grid');
    const modal = document.getElementById('fullscreenModal');
    const modalImage = document.getElementById('modalImage');
    // ADD: Get reference to a video element in the modal
    const modalVideo = document.getElementById('modalVideo'); // Assumes <video id="modalVideo"> exists in index.html
    const modelInfo = document.getElementById('modelInfo');
    const promptInfo = document.getElementById('promptInfo');
    const settingsInfo = document.getElementById('settingsInfo');
    const copyPromptBtn = document.getElementById('copyPrompt');
    const downloadBtn = document.getElementById('downloadImage');
    const deleteBtn = document.getElementById('deleteImage'); // Get delete button for fullscreen modal
    const closeBtn = modal?.querySelector('.action-close'); // Safely query close button

    let currentMediaId = null; // Changed from currentImageId
    let currentMediaType = 'image'; // Added to track type

    // Define hideModal within the scope where modal is defined
    function hideModal() {
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = '';
            // ADD: Pause video when closing modal
            if (modalVideo && !modalVideo.paused) {
                modalVideo.pause();
                modalVideo.currentTime = 0; // Optional: reset time
            }
            // END ADD
            currentMediaId = null;
            currentMediaType = 'image';
        }
    }

    // Show fullscreen modal for image or video
    function showFullscreenMedia(element) { // Renamed from showModal
        if (!modal || !modalImage || !modalVideo || !modelInfo || !promptInfo || !settingsInfo || !downloadBtn) {
            console.error("Fullscreen modal elements not found.");
            return;
        }
        const galleryItem = element.closest('.gallery-item');
        if (!galleryItem) return;

        currentMediaId = galleryItem.dataset.mediaId;
        currentMediaType = galleryItem.dataset.mediaType || 'image'; // Get type

        if (!currentMediaId) {
            console.error("No media ID found on gallery item:", galleryItem);
            return;
        }

        // Reset and hide elements
        modalImage.style.display = 'none';
        modalImage.src = '';
        modalVideo.style.display = 'none';
        modalVideo.src = '';
        if (!modalVideo.paused) modalVideo.pause(); // Pause if playing

        // Set download button text and action based on type
        const downloadButtonText = currentMediaType === 'video' ? 'Download Video' : 'Download Image';
        downloadBtn.innerHTML = `<span class="icon">⬇️</span> ${downloadButtonText}`;
        downloadBtn.dataset.mediaId = currentMediaId; // Store ID for download handler
        downloadBtn.dataset.mediaType = currentMediaType; // Store type for download handler

        // Display correct element (image or video)
        if (currentMediaType === 'video') {
            modalVideo.src = `/api/get_video/${currentMediaId}`;
            modalVideo.style.display = 'block';
            modalVideo.load();
        } else { // Image
            modalImage.src = `/api/get_image/${currentMediaId}`;
            modalImage.style.display = 'block';
        }

        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        // Reset info sections
        modelInfo.textContent = 'Loading...';
        promptInfo.textContent = 'Loading...';
        settingsInfo.textContent = 'Loading...';

        // Fetch metadata (assuming universal endpoint)
        fetch(`/api/image/${currentMediaId}/metadata`) // Still using image endpoint, needs verification
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error ${response.status}`);
                return response.json();
            })
            .then(data => {
                modelInfo.textContent = data.model_id || 'Not available';
                promptInfo.textContent = data.prompt || data.settings?.prompt || 'Not available';
                const settings = Object.entries(data.settings || {})
                    .filter(([key]) => key !== 'prompt' && key !== 'negative_prompt') // Filter out prompts
                    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
                    .join('\n');
                settingsInfo.textContent = settings || 'Not available';
            })
            .catch(error => {
                console.error('Error fetching media metadata:', error);
                modelInfo.textContent = 'Metadata unavailable';
                promptInfo.textContent = 'Metadata unavailable';
                settingsInfo.textContent = 'Metadata unavailable';
            });
    }

    // Event delegation for opening the fullscreen modal
    if (galleryGrid) {
        galleryGrid.addEventListener('click', (e) => {
            const previewElement = e.target.closest('.item-preview img, .item-preview video');
            if (previewElement) {
                // Prevent triggering if a button inside quick-actions was clicked
                if (!e.target.closest('.quick-actions')) {
                   showFullscreenMedia(previewElement);
                }
            }
        });
    } else {
        console.error("Gallery grid not found for event delegation.");
    }

    // Copy prompt functionality (ensure copyPromptBtn exists) - This is for the modal
    if (copyPromptBtn) {
        copyPromptBtn.addEventListener('click', () => {
            const promptText = promptInfo.textContent;
            if (promptText && promptText !== 'Loading...' && promptText !== 'Error loading details' && promptText !== 'Metadata unavailable') {
                // Try to use clipboard API with fallback
                const copyToClipboard = (text) => {
                    // First try the modern Clipboard API
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        return navigator.clipboard.writeText(text)
                            .then(() => true)
                            .catch(err => {
                                console.error('Clipboard API failed:', err);
                                return false;
                            });
                    }
                    // Fall back to older execCommand method
                    return new Promise(resolve => {
                        const textarea = document.createElement('textarea');
                        textarea.value = text;
                        textarea.style.position = 'fixed';
                        textarea.style.opacity = '0';
                        document.body.appendChild(textarea);
                        textarea.select();

                        try {
                            const success = document.execCommand('copy');
                            resolve(success);
                        } catch (err) {
                            console.error('execCommand failed:', err);
                            resolve(false);
                        } finally {
                            document.body.removeChild(textarea);
                        }
                    });
                };

                copyToClipboard(promptText).then(success => {
                    const originalHtml = copyPromptBtn.innerHTML;
                    if (success) {
                        copyPromptBtn.innerHTML = '<span class="icon">✓</span>Copied!';
                    } else {
                        copyPromptBtn.innerHTML = '<span class="icon">❌</span>Failed';
                    }
                    setTimeout(() => { copyPromptBtn.innerHTML = originalHtml; }, 2000);
                });
            }
        });
    }

    // Download functionality for the MODAL (ensure downloadBtn exists)
    if (downloadBtn) {
        downloadBtn.addEventListener('click', async () => {
            // Retrieve media ID and type from the button's dataset
            const mediaId = downloadBtn.dataset.mediaId;
            const mediaType = downloadBtn.dataset.mediaType;

            if (!mediaId || !mediaType) {
                 console.error("Missing media ID or type for download.");
                 return;
            }

            const downloadUrl = mediaType === 'video' ? `/api/get_video/${mediaId}` : `/api/get_image/${mediaId}`;
            const filename = `cyberimage-${mediaId}.${mediaType === 'video' ? 'mp4' : 'png'}`;

            // Keep original button text
            const originalButtonHtml = downloadBtn.innerHTML;

            try {
                // Indicate download start
                downloadBtn.innerHTML = `<span class="icon">⏳</span> Downloading...`;
                downloadBtn.disabled = true;

                const response = await fetch(downloadUrl);
                if (!response.ok) {
                    throw new Error(`Failed to fetch media: ${response.statusText}`);
                }
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                // Restore button after a short delay
                 setTimeout(() => {
                    downloadBtn.innerHTML = originalButtonHtml;
                    downloadBtn.disabled = false;
                 }, 1000); // Restore after 1 second

            } catch (error) {
                console.error('Download failed:', error);
                 downloadBtn.innerHTML = `<span class="icon">❌</span> Failed`;
                 // Restore original button after a longer delay on error
                 setTimeout(() => {
                    downloadBtn.innerHTML = originalButtonHtml;
                    downloadBtn.disabled = false;
                 }, 3000);
            }
        });
    }

    // Close button for the MODAL (ensure closeBtn exists)
    if (closeBtn) {
        closeBtn.addEventListener('click', hideModal);
    }
}

// --- End Added Back Missing Function Definitions --- //

// Timer for tracking generation duration
class GenerationTimer {
    constructor(container) {
        this.container = container;
        this.timerDisplay = container?.querySelector('.timer-value');
        this.startTime = null;
        this.timerInterval = null;
        this.running = false;
    }

    start() {
        if (this.running) return;
        this.startTime = Date.now();
        this.running = true;

        if (this.timerDisplay) {
            this.timerInterval = setInterval(() => {
                if (!this.running) return;
                const elapsedSeconds = Math.floor((Date.now() - this.startTime) / 1000);
                const minutes = Math.floor(elapsedSeconds / 60).toString().padStart(2, '0');
                const seconds = (elapsedSeconds % 60).toString().padStart(2, '0');
                this.timerDisplay.textContent = `${minutes}:${seconds}`;
            }, 1000);
            // Immediate update to show 00:00
            this.timerDisplay.textContent = '00:00';
        }
    }

    stop() {
        this.running = false;
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    reset() {
        this.stop();
        if (this.timerDisplay) {
            this.timerDisplay.textContent = '00:00';
        }
    }

    getElapsedTime() {
        if (!this.startTime) return 0;
        return Math.floor((Date.now() - this.startTime) / 1000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initializeMobileNav();
    initializeModels().then(() => {
        // Only initialize form elements after models are loaded
        if (document.getElementById('generate-form')) {
            initializeGenerationForm();
            initializeModelChange(); // Ensure model change handler is set up
            initializeKeepPromptCheckbox();
            initializeEnhancedPromptUI(); // For enrich feature
        }
    }).catch(error => {
        console.error("Failed to initialize models:", error);
        // Display error to user if needed
    });

    if (document.querySelector('.gallery-grid')) {
        initializeGalleryHandling(); // Sets up modal logic
        initializeGalleryItemActions(); // ADDED: Sets up delegated actions for items
        initializeInfiniteScroll(); // Make sure this runs after gallery setup
        initializeVideoHoverPlay(); // ADDED: Enable hover-play for videos
    }

    if (document.getElementById('queue-status-text')) {
        initializeQueueStatusIndicator(); // For queue status in nav
        initializeQueueStatusPolling(); // Start polling queue status
    }

    // Initialize modal handling globally
    initializeModalHandling();

    // Initialize Video Generation Modal specific logic
    if (document.getElementById('videoGenModal')) {
        initializeVideoGenerationModal();
    }

    // DEPRECATED - Replaced by initializeGalleryItemActions
    // if (document.querySelector('.gallery-grid')) {
    //     initializeCopyPromptButtons(); // Setup copy buttons for existing items initially
    // }
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

// Initialize available models (Corrected declaration scope)
async function initializeModels() {
    try {
        const response = await fetch(API.MODELS);
        const data = await response.json();
        const modelSelect = document.querySelector('select[name="model"]');

        if (data.models) {
            availableImageModels = {}; // Clear previous image models
            availableModels = {}; // Reset the global store here
            const mainFormModels = {}; // Store models specifically for the main form dropdown (T2I & T2V)

            console.log("Models loaded from API:", data.models);

            Object.entries(data.models).forEach(([id, info]) => {
                const modelType = info.type || 'image'; // Default to image if type is missing
                // Store ALL models in the global availableModels for use elsewhere (modals)
                availableModels[id] = {
                    ...info,
                    id: id,
                    type: modelType
                };

                // Store models suitable for the main generation form (T2I and T2V)
                if (modelType === 'image' || modelType === 't2v') {
                    mainFormModels[id] = {
                       ...info,
                       id: id,
                       type: modelType
                    };
                }
            });

            console.log("Models for main form (T2I/T2V):", mainFormModels);
            console.log("All available models (global store):", availableModels);

            // Populate the main model select with T2I and T2V models
            if (modelSelect) {
                modelSelect.innerHTML = '<option value="">Select Model</option>';
                const modelsDataStore = {}; // Store full data for the selected models

                Object.entries(mainFormModels).forEach(([id, info]) => {
                    const option = document.createElement('option');
                    option.value = id;
                    // Indicate model type in the dropdown text
                    const typeLabel = info.type === 't2v' ? '[Video]' : '[Image]';
                    option.textContent = `${id} ${typeLabel} - ${info.description}`;
                    modelsDataStore[id] = info; // Store full info including type
                    if (id === data.default) option.selected = true; // Select default if applicable
                    modelSelect.appendChild(option);
                });

                const lastModelId = localStorage.getItem('lastModelId');
                if (lastModelId && modelSelect.querySelector(`option[value="${lastModelId}"]`)) {
                    modelSelect.value = lastModelId;
                }

                modelSelect.addEventListener('change', () => handleModelChange(modelsDataStore));
                handleModelChange(modelsDataStore); // Trigger initial change
            }

        } else {
             console.error('Model data structure unexpected:', data);
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Updated handleModelChange to adjust UI based on model type (T2I vs T2V)
function handleModelChange(modelsDataStore) {
    const modelSelect = document.querySelector('select[name="model"]');
    if (!modelSelect) return;

    const selectedModelId = modelSelect.value;
    const selectedModelData = modelsDataStore ? modelsDataStore[selectedModelId] : null;

    // Get references to relevant form elements
    const negativePromptGroup = document.querySelector('.input-group:has(#negative-prompt)');
    const stepsSlider = document.getElementById('steps');
    const stepsValueDisplay = document.getElementById('steps-value');
    const guidanceSlider = document.getElementById('guidance');
    const guidanceValueDisplay = document.getElementById('guidance-value');
    const generateButton = document.querySelector('#generate-form .button-generate');
    const t2vFramesGroup = document.getElementById('t2v-frames-group');
    const numFramesSlider = document.getElementById('num_frames');
    const numFramesValueDisplay = document.getElementById('num_frames-value');
    const numImagesGroup = document.querySelector('.input-group.image-only-setting');
    const widthSelect = document.getElementById('width');
    const heightSelect = document.getElementById('height');

    // Default values (can be adjusted)
    const defaults = {
        image: { steps: 30, guidance: 7.5, width: 1024, height: 1024 },
        t2v:   { steps: 50, guidance: 7.0, width: 704, height: 480, frames: 161 }
        // Add other model type defaults if needed
    };

    if (selectedModelId && selectedModelData) {
        const modelType = selectedModelData.type || 'image'; // Get model type
        const isT2V = modelType === 't2v';
        const isFluxModel = selectedModelId.toLowerCase().includes('flux'); // Keep Flux specific logic

        // 1. Update Generate Button Text
        if (generateButton) {
            generateButton.innerHTML = isT2V
                ? '<span class="button-icon">🎬</span> Generate Video'
                : '<span class="button-icon">⚡</span> Generate Image';
        }

        // 2. Show/Hide Settings Groups
        if (t2vFramesGroup) {
            t2vFramesGroup.classList.toggle('hidden', !isT2V);
        }
        if (numImagesGroup) {
            numImagesGroup.classList.toggle('hidden', isT2V);
        }

        // 3. Handle Negative Prompt (Keep Flux logic)
        if (negativePromptGroup) {
            negativePromptGroup.style.display = isFluxModel ? 'none' : 'block';
            if (isFluxModel) {
                negativePromptGroup.querySelector('textarea').value = '';
            }
        }

        // 4. Update Sliders and Selects based on type
        const currentDefaults = defaults[modelType] || defaults.image;

        // --- Steps ---
        if (stepsSlider && stepsValueDisplay) {
            const stepConfig = selectedModelData.step_config || {};
            let currentSteps = null;
            stepsSlider.disabled = false;
            stepsSlider.classList.remove('disabled');

            // Override ranges/defaults if needed based on model or type
            // Example: Set different defaults for T2V
            stepsSlider.min = stepConfig.steps?.min ?? (isT2V ? 10 : 20); // T2V might allow fewer steps
            stepsSlider.max = stepConfig.steps?.max ?? (isT2V ? 60 : 50);
            currentSteps = stepConfig.steps?.default ?? currentDefaults.steps;

            if (stepConfig.fixed_steps !== undefined) {
                 currentSteps = stepConfig.fixed_steps;
                 stepsSlider.value = currentSteps;
                 stepsSlider.min = currentSteps;
                 stepsSlider.max = currentSteps;
                 stepsSlider.disabled = true;
                 stepsSlider.classList.add('disabled');
            } else {
                // Try to load last value, ensuring it's within new bounds
                const lastStepsValue = localStorage.getItem('lastStepsValue');
                if (lastStepsValue !== null && Number(lastStepsValue) >= stepsSlider.min && Number(lastStepsValue) <= stepsSlider.max) {
                    stepsSlider.value = Number(lastStepsValue);
                } else {
                    stepsSlider.value = currentSteps;
                }
            }
            stepsValueDisplay.textContent = stepsSlider.value;
            localStorage.setItem('lastStepsValue', stepsSlider.value);
        }

        // --- Guidance ---
        if (guidanceSlider && guidanceValueDisplay) {
            const guidanceConfig = selectedModelData.step_config?.guidance || {};
            guidanceSlider.disabled = false;
            guidanceSlider.classList.remove('disabled');

            guidanceSlider.min = guidanceConfig.min ?? (isT2V ? 1 : 1);
            guidanceSlider.max = guidanceConfig.max ?? (isT2V ? 10 : 20);
            guidanceSlider.step = guidanceConfig.step ?? (isT2V ? 0.1 : 0.5);
            let currentGuidance = guidanceConfig.default ?? currentDefaults.guidance;

            const lastGuidanceValue = localStorage.getItem('lastGuidanceValue');
            if (lastGuidanceValue !== null && Number(lastGuidanceValue) >= guidanceSlider.min && Number(lastGuidanceValue) <= guidanceSlider.max) {
                guidanceSlider.value = Number(lastGuidanceValue);
            } else {
                guidanceSlider.value = currentGuidance;
            }
            guidanceValueDisplay.textContent = guidanceSlider.value;
            localStorage.setItem('lastGuidanceValue', guidanceSlider.value);
        }

        // --- Width/Height ---
        if (widthSelect && heightSelect) {
            // Set default value based on type
            if (!widthSelect.querySelector(`option[value="${currentDefaults.width}"]`)) {
                 console.warn(`Width option ${currentDefaults.width} not found for ${modelType} model.`);
            }
            if (!heightSelect.querySelector(`option[value="${currentDefaults.height}"]`)) {
                 console.warn(`Height option ${currentDefaults.height} not found for ${modelType} model.`);
            }
            widthSelect.value = currentDefaults.width;
            heightSelect.value = currentDefaults.height;
            // Note: We are not saving/loading last width/height from localStorage currently
        }

        // --- Num Frames (T2V only) ---
        if (isT2V && numFramesSlider && numFramesValueDisplay) {
            const framesConfig = selectedModelData.step_config?.frames || {}; // Check for specific frame config
            numFramesSlider.min = framesConfig.min ?? 5;
            numFramesSlider.max = framesConfig.max ?? 161; // LTX Max
            numFramesSlider.step = framesConfig.step ?? 4;
            let currentFrames = framesConfig.default ?? currentDefaults.frames;

            // Maybe load last value? For now, set default
            numFramesSlider.value = currentFrames;
            numFramesValueDisplay.textContent = numFramesSlider.value;
            // Not saving to localStorage for now
        }

    } else {
        // Reset to image defaults if no model selected
        if (generateButton) generateButton.innerHTML = '<span class="button-icon">⚡</span> Generate Image';
        if (t2vFramesGroup) t2vFramesGroup.classList.add('hidden');
        if (numImagesGroup) numImagesGroup.classList.remove('hidden');
        if (negativePromptGroup) negativePromptGroup.style.display = 'block';

        const imgDefaults = defaults.image;
        if (stepsSlider && stepsValueDisplay) { stepsSlider.min = 20; stepsSlider.max = 50; stepsSlider.value = imgDefaults.steps; stepsSlider.disabled = false; stepsValueDisplay.textContent = imgDefaults.steps; localStorage.setItem('lastStepsValue', imgDefaults.steps); }
        if (guidanceSlider && guidanceValueDisplay) { guidanceSlider.min = 1; guidanceSlider.max = 20; guidanceSlider.step = 0.5; guidanceSlider.value = imgDefaults.guidance; guidanceValueDisplay.textContent = imgDefaults.guidance; localStorage.setItem('lastGuidanceValue', imgDefaults.guidance); }
        if (widthSelect && heightSelect) { widthSelect.value = imgDefaults.width; heightSelect.value = imgDefaults.height; }
    }
    localStorage.setItem('lastModelId', selectedModelId);
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
    const generateButton = form.querySelector('.button-generate'); // Get generate button

    // Initialize general sliders
    const sliders = form.querySelectorAll(':scope > .form-layout .settings-col .slider');
    sliders.forEach(slider => {
        const display = slider.nextElementSibling;
        if (display) {
            display.textContent = slider.value;
            slider.addEventListener('input', () => { display.textContent = slider.value; });
        }
    });

    // Load/save steps slider
    const stepsSlider = document.getElementById('steps');
    const stepsValueDisplay = document.getElementById('steps-value');
    if (stepsSlider && stepsValueDisplay) {
        stepsSlider.addEventListener('input', () => { localStorage.setItem('lastStepsValue', stepsSlider.value); stepsValueDisplay.textContent = stepsSlider.value; });
        }
    // Load/save guidance slider
    const guidanceSlider = document.getElementById('guidance');
    const guidanceValueDisplay = document.getElementById('guidance-value');
     if (guidanceSlider && guidanceValueDisplay) {
        guidanceSlider.addEventListener('input', () => { localStorage.setItem('lastGuidanceValue', guidanceSlider.value); guidanceValueDisplay.textContent = guidanceSlider.value; });
    }

    // Initialize T2V frames slider listener
    const numFramesSlider = document.getElementById('num_frames');
    const numFramesValueDisplay = document.getElementById('num_frames-value');
    if (numFramesSlider && numFramesValueDisplay) {
        numFramesSlider.addEventListener('input', () => { numFramesValueDisplay.textContent = numFramesSlider.value; });
        // Note: We are NOT saving this to localStorage currently
    }

    // Set button text back to Image explicitly (will be updated by handleModelChange)
    if (generateButton) {
        generateButton.innerHTML = '<span class="button-icon">⚡</span> Generate Image';
    }

    // Handle form submission (MODIFIED to handle T2I and T2V)
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitButton = form.querySelector('.button-generate');
        submitButton.disabled = true;
        feedbackSection.style.display = 'none';
        feedbackSection.classList.remove('active');

        // Save model/steps/guidance state
        const currentModelId = document.getElementById('model')?.value;
        const currentStepsValue = document.getElementById('steps')?.value;
        const currentGuidanceValue = document.getElementById('guidance')?.value;
        if (currentModelId) localStorage.setItem('lastModelId', currentModelId);
        if (currentStepsValue) localStorage.setItem('lastStepsValue', currentStepsValue);
        if (currentGuidanceValue) localStorage.setItem('lastGuidanceValue', currentGuidanceValue);

        // Save prompt if keep prompt checked
        const keepPromptCheckbox = document.getElementById('keep_prompt');
        if (keepPromptCheckbox && keepPromptCheckbox.checked) {
            const promptInput = document.getElementById('prompt-input');
            localStorage.setItem('keptPrompt', promptInput.value);
            }

        let modelType = 'image'; // Default model type
        let originalButtonHtml = '<span class="button-icon">⚡</span> Generate Image'; // Default button text

        try {
            const formData = new FormData(form);
            const prompt = formData.get('prompt');
            const modelId = formData.get('model');

            // Determine Model Type and prepare request
            let apiUrl = '';
            let requestData = {};
            let numOutputs = 1;
            let feedbackType = 'Image'; // For user messages

            const selectedModelInfo = availableModels[modelId];
            if (!selectedModelInfo) {
                throw new Error(`Selected model (${modelId}) not found in available models.`);
            }
            modelType = selectedModelInfo.type || 'image'; // Get type, default to image

            // --- Branch based on model type --- //
            if (modelType === 't2v') {
                console.log(`Submitting T2V request for model: ${modelId}`);
                apiUrl = API_T2V_GEN;
                feedbackType = 'Video';
                originalButtonHtml = '<span class="button-icon">🎬</span> Generate Video';
                requestData = {
                    model_id: modelId,
                    prompt: prompt,
                    settings: {
                        // Add T2V specific settings if needed from UI, using defaults for now
                        fps: 16,        // Default FPS from API.md
                        duration: 8,    // Default duration from API.md - Calculated from frames/fps backend?
                        num_frames: parseInt(document.getElementById('num_frames')?.value || '17'), // Read from slider
                        type: 't2v'     // Explicitly mark type for backend
                    }
                };
                numOutputs = 1; // T2V usually produces one output
            } else { // Assume 'image'
                console.log(`Submitting T2I request for model: ${modelId}`);
                apiUrl = API_IMAGE_GEN;
                feedbackType = 'Image';
                originalButtonHtml = '<span class="button-icon">⚡</span> Generate Image';
                requestData = {
                    model_id: modelId,
                    prompt: prompt,
                negative_prompt: formData.get('negative_prompt') || undefined,
                settings: {
                        num_images: parseInt(formData.get('num_images') || '1'),
                        num_inference_steps: parseInt(formData.get('steps') || '30'),
                        guidance_scale: parseFloat(formData.get('guidance') || '7.5'),
                        height: parseInt(formData.get('height') || '1024'),
                        width: parseInt(formData.get('width') || '1024'),
                        type: 'image' // Explicitly mark type for backend
                    }
                };
                numOutputs = requestData.settings.num_images;
            }
            // --- End Branch --- //

            // Show submitting feedback
            submitButton.innerHTML = `<span class="button-icon spin">⏳</span> Submitting ${feedbackType}...`;

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();
            if (response.ok && data.job_id) {
                feedbackSection.style.display = 'block';
                feedbackSection.classList.add('active');
                if (statusText) statusText.textContent = `Preparing ${feedbackType} generation...`;
                if (timerContainer) timerContainer.style.display = 'flex';
                timer.reset();
                timer.start();

                await pollGenerationStatus(data.job_id, feedbackSection, numOutputs, timer, feedbackType);
                // Reload after generation completes
                window.location.reload();
            } else {
                showMainFeedback(`Error: ${data.message || `Failed to start ${feedbackType} job`}`, 'error');
                timer.stop();
            }
        } catch (error) {
            console.error(`Error submitting ${feedbackType} generation request:`, error);
            showMainFeedback(`Error: ${error.message}`, 'error');
            timer.stop();
        } finally {
            submitButton.disabled = false;
            // Reset button text based on the model type that was submitted
            submitButton.innerHTML = originalButtonHtml;
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

    // Determine if we're on the gallery page or index page
    const isGalleryPage = window.location.pathname.includes('/gallery');

    // Only add sentinel and infinite scroll functionality on gallery page
    if (isGalleryPage) {
        // Add a sentinel element for better infinite scroll detection
        const sentinel = document.createElement('div');
        sentinel.id = 'infinite-scroll-sentinel';
        sentinel.style.width = '100%';
        sentinel.style.height = '10px';
        sentinel.style.margin = '30px 0';
        gallery.parentNode.appendChild(sentinel);

        const loadMoreImages = async () => {
            if (loading) return;

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
        };

        // Use Intersection Observer instead of scroll event
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting && !loading) {
                loadMoreImages();
            }
        }, {
            rootMargin: '200px',
        });

        observer.observe(sentinel);
    }
    // We're not adding any scroll-blocking code for the index page
    // This ensures normal scrolling behavior is maintained
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

// Status polling with enhanced feedback (MODIFIED to accept feedbackType)
async function pollGenerationStatus(jobId, feedbackSection, totalImages = 1, timer, feedbackType = 'Image') {
    const statusText = feedbackSection.querySelector('.status-text');
    const progressFill = feedbackSection.querySelector('.progress-fill');
    const queuePosition = feedbackSection.querySelector('.queue-position');
    const estimatedTime = feedbackSection.querySelector('.estimated-time');

    let startTime = Date.now();
    let lastQueuePosition = null;
    let estimatedTimePerJob = 30000 * totalImages; // Initial estimate: 30 seconds per image
    let currentProgress = 0;
    let targetProgress = 0;
    let stageStartTime = null;
    let currentStage = 'waiting';
    let lastStatusMessage = '';
    let stuckCounter = 0; // To detect when we're stuck
    let generationSteps = 0; // Track the progression in steps
    let forcedTransition = false; // Track if we've forced a transition

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

    console.log(`Starting generation polling for job ${jobId} with ${totalImages} image(s) requested`);

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

            // Calculate queue position considering total images, not just job count
            // This will account for multi-image jobs in the queue calculation
            let estimatedTotalImages = 0;
            if (queueData.pending_jobs) {
                // If backend provides pending_jobs details with image counts
                estimatedTotalImages = queueData.pending_jobs.reduce((total, job) => total + (job.num_images || 1), 0);
            } else {
                // Fallback: estimate based on pending count and average image per job
                const avgImagesPerJob = queueData.avg_images_per_job || 1;
                estimatedTotalImages = queueData.pending * avgImagesPerJob;
            }

            // Account for own job if it's still pending
            if (statusData.status === 'pending') {
                estimatedTotalImages += totalImages;
            }

            const currentPosition = Math.max(0, estimatedTotalImages);

            // Update queue position display
            if (queuePosition) {
                if (currentPosition > 0) {
                    queuePosition.textContent = `Queue Position: ${currentPosition} ${totalImages > 1 ? `(${totalImages} images)` : 'image'}`;
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

            // Check if we're receiving the same message multiple times
            // and force progression if we appear to be stuck
            const isNewMessage = lastStatusMessage !== JSON.stringify(statusData);
            if (!isNewMessage) {
                stuckCounter++;
                console.log(`Same status received ${stuckCounter} times in a row`);
            } else {
                stuckCounter = 0;
                lastStatusMessage = JSON.stringify(statusData);

                // Check if the backend message indicates generation but we're still in loading stage
                const messageContent = statusData.message || '';
                if (currentStage === 'loading_model' &&
                    (messageContent.includes('Generating') ||
                     (statusData.progress && statusData.progress.generating))) {
                    console.log('Backend indicates generation but UI shows loading - updating stage');
                    currentStage = 'generating';
                    stageStartTime = Date.now();
                    forcedTransition = true;
                }
            }

            // Force progression if we've received the same message too many times
            const forceProgressUpdate = stuckCounter > 5;

            if (forceProgressUpdate && (currentStage === 'processing' || currentStage === 'loading_model')) {
                console.log('Forcing progress update due to stuck status');
                // Force stage advancement if we're stuck
                if (currentStage === 'loading_model') {
                    currentStage = 'generating';
                    stageStartTime = Date.now();
                    forcedTransition = true;
                    console.log('Forced transition from loading_model to generating stage');
                } else if (currentStage === 'generating' && generationSteps === 0) {
                    generationSteps = Math.floor(totalImages > 1 ? totalImages * 5 : 10);
                }
            }

            // Update status based on job status with detailed progress
            switch (statusData.status) {
                case 'completed':
                    timer.stop();
                    updateGenerationStatus(`${feedbackType} generation completed!`, 100);
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

                    // Track the current stage for better progress updates
                    if (currentStage === 'waiting' || currentStage === 'pending') {
                        currentStage = 'processing';
                        stageStartTime = Date.now();
                    }

                    // If we have detailed progress info
                    if (statusData.progress) {
                        // Get the step information if available
                        const stepInfo = statusData.progress.step;
                        const totalSteps = statusData.progress.total_steps;

                        // Check message content for status clues
                        const messageContent = statusData.message || '';
                        let isGenerating = messageContent.includes('Generating') ||
                                           messageContent.includes('Step') ||
                                           statusData.progress.generating ||
                                           statusData.progress.step !== null;
                        let isLoading = messageContent.includes('Loading model') ||
                                         statusData.progress.loading_model;
                        let isSaving = messageContent.includes('Saving') ||
                                        statusData.progress.saving;

                        // If we've forced a transition to generating, don't go back to loading
                        if (forcedTransition && currentStage === 'generating') {
                            // Override the loading status if we've forced a transition
                            if (isLoading && !isGenerating) {
                                console.log('Ignoring loading status due to forced transition');
                                isLoading = false;
                                // Simulate generating status
                                message = 'Generating your image...';
                                progress = 35 + (Math.min(100, (Date.now() - stageStartTime) / 20000) * 50);
                                updateGenerationStatus(message, progress);
                                updateProgress(progress);
                                break;
                            }
                        }

                        if (isLoading && !forcedTransition) {
                            if (currentStage !== 'loading_model') {
                                currentStage = 'loading_model';
                                stageStartTime = Date.now();
                                console.log('Entered loading_model stage');
                            }

                            // Calculate progress within the loading model stage
                            // Start at 25%, move to 35% over time
                            const loadingDuration = Date.now() - stageStartTime;
                            const maxLoadingDuration = 5000; // Assume loading takes max 5 seconds
                            const loadingProgressPercent = Math.min(100, (loadingDuration / maxLoadingDuration) * 100);
                            progress = 25 + (loadingProgressPercent / 10); // Scale to 25-35% range
                            message = 'Loading AI model...';

                            // Remove automatic transition based on time
                        } else if (isGenerating) {
                            // Also check message content as a fallback
                            if (currentStage !== 'generating') {
                                currentStage = 'generating';
                                stageStartTime = Date.now();
                                generationSteps = stepInfo;
                                console.log('Entered generating stage');
                            }

                            // If we have step information
                            if (stepInfo !== null && totalSteps !== null) {
                                generationSteps = stepInfo;
                                // Calculate progress as percentage of steps
                                const stepProgress = (stepInfo / totalSteps) * 100;

                                // Map the step progress to a range between 35-85%
                                progress = 35 + (stepProgress * 0.5); // Scale to 35-85% range
                                message = `Generating image... Step ${stepInfo}/${totalSteps}`;

                                // Update multi-image display
                                if (totalImages > 1) {
                                    // Roughly estimate which image we're generating
                                    const estimatedImageNum = Math.ceil(statusData.message?.match(/image (\d+) of \d+/)?.[1]) || 1;
                                    message = `Generating image ${estimatedImageNum} of ${totalImages}... Step ${stepInfo}/${totalSteps}`;
                                }
                            } else {
                                // No step info - estimate progress based on time
                                const generatingDuration = Date.now() - stageStartTime;
                                const estimatedFullDuration = 20000; // Assume generation takes ~20 seconds
                                const estimatedProgress = Math.min(100, (generatingDuration / estimatedFullDuration) * 100);
                                progress = 35 + (estimatedProgress * 0.5); // Scale to 35-85% range

                                // For multiple images, acknowledge that
                                if (totalImages > 1) {
                                    // Try to parse image number from status message
                                    const imgMatch = statusData.message?.match(/image (\d+) of \d+/);
                                    const estimatedImageNum = imgMatch ? parseInt(imgMatch[1]) :
                                        Math.min(totalImages, Math.floor((estimatedProgress / 100) * totalImages) + 1);
                                    message = `Generating image ${estimatedImageNum} of ${totalImages}...`;
                                } else if (feedbackType === 'Video') {
                                    message = 'Generating Video...'
                                }
                            }

                            // If the server provides a specific message, use that
                            if (statusData.message && statusData.message.includes('Generating image')) {
                                // Extract step info from message if present
                                const stepMatch = statusData.message.match(/Step (\d+)\/(\d+)/);
                                if (stepMatch) {
                                    const step = parseInt(stepMatch[1]);
                                    const total = parseInt(stepMatch[2]);
                                    if (!isNaN(step) && !isNaN(total)) {
                                        generationSteps = step;
                                        // Calculate progress percentage from steps
                                        const stepProgress = (step / total) * 100;
                                        progress = 35 + (stepProgress * 0.5); // Scale to 35-85% range
                                    }
                                }
                                message = statusData.message;
                            }
                        } else if (isSaving) {
                            if (currentStage !== 'saving') {
                                currentStage = 'saving';
                                stageStartTime = Date.now();
                                console.log('Entered saving stage');
                            }
                            progress = 85;
                            message = `Saving generated ${feedbackType}...`;
                        }
                    } else {
                        // No detailed progress info - use a time-based approach
                        const processingTime = Date.now() - startTime;
                        const estimatedTotalTime = estimatedTimePerJob;

                        // Check message content for status clues even without progress info
                        const messageContent = statusData.message || '';
                        let isGenerating = messageContent.includes('Generating') ||
                                          messageContent.includes('Step');
                        let isLoading = messageContent.includes('Loading model');
                        let isSaving = messageContent.includes('Saving');

                        // If we've forced a transition to generating, don't go back to loading
                        if (forcedTransition && currentStage === 'generating') {
                            if (isLoading && !isGenerating) {
                                console.log('Ignoring loading status due to forced transition (fallback)');
                                isLoading = false;
                                isGenerating = true; // Force generating status
                            }
                        }

                        // Estimate overall progress
                        const processingProgress = Math.min(90, (processingTime / estimatedTotalTime) * 100);
                        progress = 35 + (processingProgress * 0.5); // Scale to 35-85% range

                        // Determine message based on progress and message content
                        if (isLoading) {
                            message = 'Loading AI model...';
                            progress = Math.min(progress, 35); // Cap at 35% for loading
                        } else if (isGenerating) {
                            message = `Generating ${feedbackType}...`;
                            if (currentStage !== 'generating') {
                                currentStage = 'generating';
                                stageStartTime = Date.now();
                            }
                        } else if (isSaving) {
                            message = `Saving ${feedbackType}...`;
                            progress = Math.max(progress, 85); // At least 85% for saving
                        } else if (progress < 40) {
                            message = 'Preparing model...';
                        } else if (progress < 70) {
                            message = `Generating ${feedbackType}...`;
                        } else {
                            message = `Finalizing ${feedbackType}...`;
                        }
                    }

                    updateGenerationStatus(message, progress);
                    updateProgress(progress);

                    // Debug log to help track status changes
                    console.log(`Status update from backend: "${statusData.message}", progress info:`, statusData.progress);
                    console.log(`Current UI state: stage=${currentStage}, message="${message}", progress=${progress}, forced=${forcedTransition}`);
                    break;

                case 'pending':
                    if (currentStage !== 'pending') {
                        currentStage = 'pending';
                        stageStartTime = Date.now();
                        console.log('Entered pending stage');
                    }

                    // Calculate progress based on queue position - make it more responsive
                    const maxQueueProgress = 25; // Max progress while in queue
                    let queueProgress;

                    if (currentPosition > 10) {
                        queueProgress = 5; // Just started, long queue
                    } else if (currentPosition > 5) {
                        queueProgress = 10; // Moving up
                    } else if (currentPosition > 0) {
                        queueProgress = 15 + (5 - currentPosition) * 2; // Almost there
                    } else {
                        queueProgress = maxQueueProgress; // Next in line
                    }

                    updateGenerationStatus('Waiting in queue...', queueProgress);
                    updateProgress(queueProgress);
                    break;

                default:
                    updateGenerationStatus('Unexpected status: ' + statusData.status, currentProgress);
                    break;
            }

            // Update time estimates based on queue movement
            if (lastQueuePosition !== null && lastQueuePosition > currentPosition) {
                const timeElapsed = Date.now() - startTime;
                const jobsCompleted = lastQueuePosition - currentPosition;
                if (jobsCompleted > 0) {
                    const newEstimate = timeElapsed / jobsCompleted;
                    // Use weighted average to smooth out estimate changes
                    estimatedTimePerJob = (estimatedTimePerJob * 0.7) + (newEstimate * 0.3);
                    console.log(`Updated time estimate: ${estimatedTimePerJob/1000}s per job`);
                }
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
function appendImagesToGallery(items) {
    const gallery = document.querySelector('.gallery-grid');
    if (!gallery) return; // Ensure gallery exists

    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'gallery-item';

        // Add both attributes for compatibility and type info
        div.dataset.mediaId = item.id; // New attribute name
        div.dataset.imageId = item.id; // Legacy attribute name

        const createdAt = new Date(item.created_at);
        const prompt = item.prompt || item.settings?.prompt || 'No prompt available';
        // Ensure model_id is correctly extracted from metadata if available
        const modelId = item.model_id || item.metadata?.model_id || item.settings?.model_id || 'Unknown';

        // --- Determine Media Type based on Model ---
        let isVideoType = false;
        let mediaType = 'image'; // Default to image
        if (modelId !== 'Unknown' && availableModels[modelId]) {
            const modelType = availableModels[modelId].type;
            if (modelType === 't2v' || modelType === 'i2v') {
                isVideoType = true;
                mediaType = 'video';
            }
        } else {
            // Fallback check if model info isn't in availableModels (e.g., older items)
            // Use existing logic as a backup
            if (item.settings?.type === 'video' || isVideo(item.id)) {
                 isVideoType = true;
                 mediaType = 'video';
            }
            console.warn(`Model ID ${modelId} not found in availableModels for item ${item.id}. Falling back to settings/ID check.`);
        }
        div.dataset.mediaType = mediaType; // Set the media type dataset
        // --- End Media Type Determination ---

        const mediaUrl = isVideoType ? `/api/get_video/${item.id}` : API.IMAGE(item.id);

        let previewHtml;
        if (isVideoType) {
            previewHtml = `<video src="${mediaUrl}" controls muted loop preload="metadata" class="video-preview"></video>`;
        } else {
            previewHtml = `<img src="${mediaUrl}" alt="${prompt.slice(0, 50)}..." loading="lazy">`;
        }

        div.innerHTML = `
            <div class="item-preview">
                ${previewHtml}
                <div class="quick-actions">
                    <button class="action-copy-prompt" title="Copy Prompt" data-prompt="${prompt}">📋</button>
                    ${!isVideoType ? `<button class="action-generate-video" title="Generate Video from Image" data-image-id="${item.id}" data-image-prompt="${prompt}">🎥</button>` : ''}
                    <button class="action-download" title="Download ${isVideoType ? 'Video' : 'Image'}">⬇️</button>
                    <button class="action-delete" title="Delete ${isVideoType ? 'Video' : 'Image'}">🗑️</button>
                </div>
            </div>
            <div class="item-details">
                <p class="prompt">${prompt.slice(0, 100)}...</p>
                <p class="model">${modelId}</p>
                <p class="date" title="${formatDateLong(createdAt)}">${formatDate(createdAt)}</p>
            </div>
        `;

        // Add event listeners for quick actions (using delegation now, but individual listeners can be added here if needed)

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
    const styleSelect = document.getElementById('enrich-style');
    const currentPrompt = promptInput.value.trim();
    const selectedStyle = styleSelect.value;

    if (!currentPrompt) {
        addNeonFlash(promptInput);
        return;
    }

    const enrichButton = e.target.closest('#enrich-prompt');
    enrichButton.disabled = true;
    enrichButton.innerHTML = '<span class="button-icon spin">⏳</span> Enriching...';

    // Save original prompt for comparison
    const originalPrompt = currentPrompt;

    try {
        const response = await fetch('/api/enrich', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: currentPrompt,
                style: selectedStyle
            })
        });

        const data = await response.json();
        if (data.enriched_prompt) {
            // Update textarea with enriched prompt
            promptInput.value = data.enriched_prompt;
            addNeonFlash(promptInput);

            // Show comparison with original prompt
            showPromptComparison(originalPrompt, data.enriched_prompt);
        }
    } catch (error) {
        console.error('Error enriching prompt:', error);
    } finally {
        enrichButton.disabled = false;
        enrichButton.innerHTML = '<span class="button-icon">✨</span> Enrich';
    }
}

// Show comparison between original and enriched prompt
function showPromptComparison(original, enriched) {
    const comparisonDiv = document.getElementById('prompt-comparison');
    const originalPromptDiv = document.getElementById('original-prompt');
    const restoreButton = document.getElementById('restore-original');
    const closeButton = document.getElementById('close-comparison');
    const promptInput = document.getElementById('prompt-input');

    if (!comparisonDiv || !originalPromptDiv) return;

    // Display original prompt
    originalPromptDiv.textContent = original;
    comparisonDiv.style.display = 'block';

    // Handle restore button
    if (restoreButton) {
        restoreButton.onclick = () => {
            promptInput.value = original;
            addNeonFlash(promptInput);
        };
    }

    // Handle close button
    if (closeButton) {
        closeButton.onclick = () => {
            comparisonDiv.style.display = 'none';
        };
    }
}

// Initialize enhanced tooltip functionality
function initializeEnrichInfo() {
    const infoIcon = document.getElementById('enrich-info');
    const tooltip = document.querySelector('.enrich-tooltip');

    if (!infoIcon || !tooltip) return;

    // Show tooltip on click (more mobile-friendly)
    infoIcon.addEventListener('click', (e) => {
        e.stopPropagation();

        const isVisible = tooltip.style.display === 'block';
        tooltip.style.display = isVisible ? 'none' : 'block';
    });

    // Hide tooltip when clicking elsewhere
    document.addEventListener('click', () => {
        tooltip.style.display = 'none';
    });
}

// Initialize all enhanced prompt UI features
function initializeEnhancedPromptUI() {
    initializeEnrichInfo();
}

// Initialize queue status indicator
function initializeQueueStatusIndicator() {
    const statusIcon = document.getElementById('generation-status-icon');
    const statusText = document.getElementById('queue-status-text');
    const queueIndicator = statusIcon?.closest('.queue-indicator');

    if (!statusIcon || !statusText || !queueIndicator) return;

    // Initial update
    updateQueueStatusIndicator();

    // Update every 5 seconds
    setInterval(updateQueueStatusIndicator, 5000);

    // Add click handler to show detailed stats
    queueIndicator.addEventListener('click', showQueueDetails);

    function showQueueDetails() {
        fetch('/api/queue?detailed=true')
            .then(response => response.json())
            .then(data => {
                const { pending, processing, completed, failed, total,
                        avg_processing_time_seconds, failure_rate, models } = data;

                // Create or get the details popup
                let detailsPopup = document.getElementById('queue-details-popup');
                if (!detailsPopup) {
                    detailsPopup = document.createElement('div');
                    detailsPopup.id = 'queue-details-popup';
                    detailsPopup.className = 'queue-details-popup';
                    document.body.appendChild(detailsPopup);

                    // Add close button
                    const closeBtn = document.createElement('button');
                    closeBtn.className = 'close-popup';
                    closeBtn.innerHTML = '×';
                    closeBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        detailsPopup.classList.remove('visible');
                    });
                    detailsPopup.appendChild(closeBtn);
                }

                // Position the popup near the indicator
                const rect = queueIndicator.getBoundingClientRect();
                detailsPopup.style.top = `${rect.bottom + window.scrollY + 10}px`;
                detailsPopup.style.left = `${rect.left + window.scrollX}px`;

                // Create content for the popup
                let content = `
                    <h3>Queue Status</h3>
                    <div class="queue-stats">
                        <div class="stat-item">
                            <span class="stat-label"><i class="fas fa-hourglass-half"></i> Pending</span>
                            <span class="stat-value ${pending > 0 ? 'highlight' : ''}">${pending}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label"><i class="fas fa-cog fa-spin"></i> Processing</span>
                            <span class="stat-value ${processing > 0 ? 'highlight' : ''}">${processing}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label"><i class="fas fa-check-circle"></i> Completed</span>
                            <span class="stat-value">${completed}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label"><i class="fas fa-exclamation-triangle"></i> Failed</span>
                            <span class="stat-value ${failed > 0 ? 'error' : ''}">${failed}</span>
                        </div>
                        <div class="stat-item total">
                            <span class="stat-label"><i class="fas fa-chart-bar"></i> Total Jobs</span>
                            <span class="stat-value highlight">${total}</span>
                        </div>
                    </div>`;

                // Add performance metrics if available
                if (avg_processing_time_seconds !== undefined) {
                    const avgTimeFormatted = formatTime(avg_processing_time_seconds);
                    content += `
                        <h3>Performance</h3>
                        <div class="queue-stats">
                            <div class="stat-item">
                                <span class="stat-label"><i class="fas fa-clock"></i> Avg. Processing Time</span>
                                <span class="stat-value">${avgTimeFormatted}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label"><i class="fas fa-exclamation-circle"></i> Failure Rate</span>
                                <span class="stat-value ${failure_rate > 10 ? 'error' : ''}">${failure_rate}%</span>
                            </div>
                        </div>`;
                }

                // Add model-specific stats if available
                if (models && Object.keys(models).length > 0) {
                    content += `<h3>Models (24h)</h3><div class="queue-stats">`;

                    Object.entries(models).forEach(([modelId, stats]) => {
                        const successRate = stats.completed > 0 ?
                            Math.round((stats.completed / (stats.completed + stats.failed)) * 100) : 0;

                        content += `
                            <div class="stat-item model-stat">
                                <span class="stat-label"><i class="fas fa-robot"></i> ${modelId}</span>
                                <span class="stat-value">
                                    <span class="model-count">${stats.total} jobs</span>
                                    <span class="model-success-rate ${successRate > 90 ? 'highlight' : successRate < 70 ? 'error' : ''}">
                                        ${successRate}% success
                                    </span>
                                </span>
                            </div>`;
                    });

                    content += `</div>`;
                }

                // Set the content and show the popup
                detailsPopup.innerHTML = content;
                detailsPopup.appendChild(closeBtn);
                detailsPopup.classList.add('visible');

                // Close popup when clicking outside
                document.addEventListener('click', function closePopup(e) {
                    if (!detailsPopup.contains(e.target) && e.target !== queueIndicator) {
                        detailsPopup.classList.remove('visible');
                        document.removeEventListener('click', closePopup);
                    }
                });
            })
            .catch(error => {
                console.error('Error fetching queue details:', error);
            });
    }

    function updateQueueStatusIndicator() {
        const statusIcon = document.getElementById('generation-status-icon');
        const statusText = document.getElementById('queue-status-text');
        const queueIndicator = statusIcon?.closest('.queue-indicator');

        if (!statusIcon || !statusText || !queueIndicator) return;

        fetch('/api/queue')
            .then(response => response.json())
            .then(data => {
                const { pending, processing, completed, failed } = data;
                const totalActive = pending + processing;

                // Create tooltip with detailed stats
                const tooltipText = `
                    📊 Queue Status:
                    • ${pending} pending
                    • ${processing} processing
                    • ${completed} completed
                    • ${failed} failed

                    🖱️ Click for detailed statistics
                `;
                queueIndicator.setAttribute('title', tooltipText);

                // Update status text and icon based on queue state
                if (totalActive > 0) {
                    statusText.textContent = `Queue: ${totalActive}`;
                    statusIcon.className = 'fas fa-cog fa-spin';
                    statusIcon.style.color = '#39ff14'; // Neon green
                } else {
                    statusText.textContent = 'Queue: 0';
                    statusIcon.className = 'fas fa-check-circle';
                    statusIcon.style.color = '#39ff14'; // Neon green
                }

                // Show the indicator
                queueIndicator.style.display = 'flex';
            })
            .catch(error => {
                console.error('Error fetching queue status:', error);
                statusText.textContent = 'Queue: ?';
                statusIcon.className = 'fas fa-exclamation-triangle';
                statusIcon.style.color = '#ff4444'; // Red
                queueIndicator.setAttribute('title', 'Could not fetch queue status');
                queueIndicator.style.display = 'flex';
            });
    }
}

// Helper function to format time in seconds to a readable format
function formatTime(seconds) {
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

// Initialize Keep Prompt checkbox functionality
function initializeKeepPromptCheckbox() {
    const checkbox = document.getElementById('keep_prompt');
    const promptInput = document.getElementById('prompt-input');

    if (!checkbox || !promptInput) return;

    // Load saved state
    const savedKeep = localStorage.getItem('keepPromptChecked');
    const savedPrompt = localStorage.getItem('keptPrompt');

    if (savedKeep === 'true') {
        checkbox.checked = true;
        if (savedPrompt !== null && promptInput.value.trim() === '') {
            promptInput.value = savedPrompt;
        }
    }

    // Save checkbox state changes
    checkbox.addEventListener('change', () => {
        localStorage.setItem('keepPromptChecked', checkbox.checked);
        if (!checkbox.checked) {
            // If user unchecks, also clear saved prompt
            localStorage.removeItem('keptPrompt');
        }
    });
}

// --- ADDED: Event Delegation for Gallery Item Actions ---
function initializeGalleryItemActions() {
    const galleryGrid = document.querySelector('.gallery-grid');
    if (!galleryGrid) return;

    galleryGrid.addEventListener('click', async (e) => {
        const galleryItem = e.target.closest('.gallery-item');
        if (!galleryItem) return;

        const mediaId = galleryItem.dataset.mediaId;
        const mediaType = galleryItem.dataset.mediaType || 'image';

        // Handle Copy Prompt
        if (e.target.classList.contains('action-copy-prompt')) {
            const button = e.target;
            const promptText = button.dataset.prompt;
            if (promptText) {
                try {
                    await navigator.clipboard.writeText(promptText);
                    const originalText = button.textContent;
                    button.textContent = 'Copied!';
                    button.disabled = true;
                    setTimeout(() => {
                        button.textContent = originalText;
                        button.disabled = false;
                    }, 1500);
                } catch (err) {
                    console.error('Failed to copy prompt:', err);
                    // Optionally show feedback to user
                }
            }
        }

        // Handle Download
        else if (e.target.classList.contains('action-download')) {
            if (!mediaId) return;
            const button = e.target;
            const downloadUrl = mediaType === 'video' ? `/api/get_video/${mediaId}` : `/api/get_image/${mediaId}`;
            const filename = `cyberimage-${mediaId}.${mediaType === 'video' ? 'mp4' : 'png'}`;
            const originalText = button.textContent; // Store original icon/text if needed

             try {
                button.innerHTML = '⏳'; // Simple loading indicator
                button.disabled = true;

                const response = await fetch(downloadUrl);
                 if (!response.ok) {
                     throw new Error(`Failed to fetch media: ${response.statusText}`);
                 }
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                 setTimeout(() => {
                    button.innerHTML = originalText; // Restore icon
                    button.disabled = false;
                 }, 500);

            } catch (error) {
                console.error('Download failed:', error);
                button.innerHTML = '❌'; // Error indicator
                 setTimeout(() => {
                    button.innerHTML = originalText; // Restore icon
                    button.disabled = false;
                 }, 2000);
            }
        }

        // Handle Delete
        else if (e.target.classList.contains('action-delete')) {
            if (!mediaId) return;
            // Confirm deletion (reuse or create a confirmation modal)
            const confirmed = confirm(`Are you sure you want to delete this ${mediaType}? This cannot be undone.`);
            if (confirmed) {
                try {
                    // ALWAYS use the /api/image endpoint for deletion, regardless of mediaType
                    const deleteUrl = `/api/image/${mediaId}`;
                    const response = await fetch(deleteUrl, { method: 'DELETE' });
                    if (response.ok) {
                        galleryItem.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                        galleryItem.style.opacity = '0';
                        galleryItem.style.transform = 'scale(0.9)';
                        setTimeout(() => galleryItem.remove(), 500);
                        // Optionally update counts or show feedback
                    } else {
                        console.error('Failed to delete:', response.statusText);
                         alert(`Failed to delete ${mediaType}. Server responded: ${response.statusText}`);
                        // Show error feedback
                    }
                } catch (error) {
                    console.error('Error deleting:', error);
                     alert(`An error occurred while trying to delete the ${mediaType}.`);
                    // Show error feedback
                }
            }
        }

        // Handle Generate Video (if applicable)
        else if (e.target.classList.contains('action-generate-video') && mediaType === 'image') {
             if (!mediaId) return;
             const button = e.target;
             const sourceImageUrl = galleryItem.querySelector('img')?.src;
             const sourcePrompt = button.dataset.imagePrompt; // Get prompt from data attribute

             if (sourceImageUrl && sourcePrompt !== undefined) {
                 openVideoGenModal(mediaId, sourceImageUrl, sourcePrompt);
             } else {
                console.error("Missing image URL or prompt for video generation.", {mediaId, sourceImageUrl, sourcePrompt});
                alert("Could not retrieve necessary information to generate video from this image.");
             }
        }
    });
}
// --- END ADDED ---

// --- DEPRECATED? Remove or refactor initializeCopyPromptButtons if replaced by delegation ---
// function initializeCopyPromptButtons() { ... } // Keep if used elsewhere, otherwise remove

// --- Video Generation Modal Logic ---
function initializeVideoGenerationModal() {
    const videoGenModal = document.getElementById('videoGenModal');
    const videoGenForm = document.getElementById('video-generate-form');
    const videoModelSelect = document.getElementById('videoGenModelSelect');
    const sourceImageEl = document.getElementById('videoGenSourceImage');
    const sourcePromptEl = document.getElementById('videoGenSourcePrompt');
    const sourceImageIdInput = document.getElementById('videoGenSourceImageId');
    const videoPromptInput = document.getElementById('videoGenPromptInput');

    if (!videoGenModal || !videoGenForm || !videoModelSelect || !sourceImageEl || !sourcePromptEl || !sourceImageIdInput || !videoPromptInput) {
        console.warn('Video Generation Modal elements not found. Video generation disabled.');
        return;
    }

    // Add listener for the camera button clicks (using event delegation)
    document.body.addEventListener('click', (event) => {
        const button = event.target.closest('.action-generate-video');
        if (button) {
            const sourceImageId = button.dataset.imageId;
            const sourcePrompt = button.dataset.imagePrompt;
            const galleryItem = button.closest('.gallery-item');
            const sourceImageElement = galleryItem ? galleryItem.querySelector('img') : null;
            const sourceImageUrl = sourceImageElement ? sourceImageElement.src : '';

            if (sourceImageId && sourcePrompt) {
                openVideoGenModal(sourceImageId, sourceImageUrl, sourcePrompt);
            } else {
                console.error('Missing data on video generate button:', button);
            }
        }
    });

    // Save selected model ID
    videoModelSelect.addEventListener('change', () => {
        localStorage.setItem('lastVideoModelId', videoModelSelect.value);
    });

    // Handle I2V form submission
    videoGenForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const submitButton = videoGenForm.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="button-icon spin">⏳</span> Submitting...';

        const sourceId = document.getElementById('videoGenSourceImageId').value;
        const videoPrompt = document.getElementById('videoGenPromptInput').value.trim();
        const videoModelId = videoModelSelect.value;

        if (!videoPrompt || !videoModelId) {
            alert('Please enter a video prompt and select a model.');
            submitButton.disabled = false;
            submitButton.innerHTML = '<span class="button-icon">▶️</span> Generate Video';
            return;
        }

        try {
            showMainFeedback('Submitting I2V request...', 'info');

            const selectedModel = availableModels[videoModelId];
            if (!selectedModel) {
                throw new Error(`Model ${videoModelId} not found`);
            }

            if (selectedModel.type !== 'i2v' && !videoModelId.toLowerCase().includes('i2v')) {
                throw new Error(`Model ${videoModelId} is not an image-to-video model`);
            }

            const videoGenModalElement = document.getElementById('videoGenModal');
            const sourceWidth = videoGenModalElement.dataset.sourceWidth;
            const sourceHeight = videoGenModalElement.dataset.sourceHeight;

            let apiPayload = {
                source_image_id: sourceId,
                video_prompt: videoPrompt,
                video_model_id: videoModelId,
                guidance_scale: 3.5 // Adjusted and top-level
            };

            if (sourceWidth && sourceHeight) {
                apiPayload.width = parseInt(sourceWidth, 10);
                apiPayload.height = parseInt(sourceHeight, 10);
            } else {
                console.warn("Video Gen: Source image dimensions not available. Backend will use defaults for width/height.");
            }

            console.log("Submitting I2V job with payload:", apiPayload);

            const response = await fetch(API_I2V_GEN, { // Use I2V endpoint
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(apiPayload) // Use the modified apiPayload
            });
            const result = await response.json();
            if (response.ok && result.job_id) {
                showMainFeedback(`I2V job ${result.job_id} submitted!`, 'success');
                closeVideoGenModal();
            } else {
                throw new Error(result.message || 'Failed to submit I2V job');
            }
        } catch (error) {
            console.error('Error submitting I2V job:', error);
            showMainFeedback(`Error: ${error.message}`, 'error');
        } finally {
            submitButton.disabled = false;
            submitButton.innerHTML = '<span class="button-icon">▶️</span> Generate Video';
        }
    });
}

function openVideoGenModal(sourceImageId, sourceImageUrl, sourcePrompt) {
    const videoGenModal = document.getElementById('videoGenModal');
    const videoModelSelect = document.getElementById('videoGenModelSelect');
    const sourceImageEl = document.getElementById('videoGenSourceImage');
    const sourcePromptEl = document.getElementById('videoGenSourcePrompt');
    const sourceImageIdInput = document.getElementById('videoGenSourceImageId');
    const videoPromptInput = document.getElementById('videoGenPromptInput');

    if (!videoGenModal || !videoModelSelect || !sourceImageEl || !sourcePromptEl || !sourceImageIdInput || !videoPromptInput) return;

    // Populate fields
    sourceImageIdInput.value = sourceImageId;
    // Clear previous dimensions and set up onload for new image
    delete videoGenModal.dataset.sourceWidth;
    delete videoGenModal.dataset.sourceHeight;

    sourceImageEl.onload = function() {
        videoGenModal.dataset.sourceWidth = this.naturalWidth;
        videoGenModal.dataset.sourceHeight = this.naturalHeight;
        console.log(`Video Gen Modal: Source image loaded. Dimensions: ${this.naturalWidth}x${this.naturalHeight}`);
    };
    sourceImageEl.onerror = function() {
        console.error("Video Gen Modal: Source image failed to load.");
        // Ensure dimensions are not carried over if image fails
        delete videoGenModal.dataset.sourceWidth;
        delete videoGenModal.dataset.sourceHeight;
        sourceImageEl.alt = "Failed to load image preview"; // Update alt text
    };
    sourceImageEl.src = sourceImageUrl || ''; // Set src after onload/onerror are defined
    sourceImageEl.alt = "Source image preview"; // Reset alt text

    // If image is already loaded (e.g., from cache), onload might not fire consistently.
    // So, explicitly set dimensions if already complete and has valid dimensions.
    if (sourceImageEl.complete && sourceImageEl.naturalWidth && sourceImageEl.naturalWidth > 0) {
        // Check if onload has already run by checking dataset, to prevent issues if onload also fires
        if (!videoGenModal.dataset.sourceWidth) { 
            videoGenModal.dataset.sourceWidth = sourceImageEl.naturalWidth;
            videoGenModal.dataset.sourceHeight = sourceImageEl.naturalHeight;
            console.log(`Video Gen Modal: Source image was already complete. Dimensions: ${sourceImageEl.naturalWidth}x${sourceImageEl.naturalHeight}`);
        }
    }

    sourcePromptEl.textContent = sourcePrompt;
    videoPromptInput.value = sourcePrompt; // Pre-fill with source prompt for convenience

    // Populate video model select - FILTER FOR I2V using global availableModels
    videoModelSelect.innerHTML = '<option value="">Select I2V Model</option>';
    let modelAdded = false;

    // Log all available models to help debug
    console.log("All models available when populating video model dropdown:", availableModels);

    Object.entries(availableModels).forEach(([id, info]) => {
        // Look for models with type exactly matching "i2v" or ID containing "i2v"
        if (info.type === 'i2v' || id.toLowerCase().includes('i2v')) {
            console.log("Found I2V model:", id, info);
        const option = document.createElement('option');
        option.value = id;
            option.textContent = `${id} - ${info.description || 'Image to Video Model'}`;
        videoModelSelect.appendChild(option);
            modelAdded = true;
        }
    });

    // Set selection based on localStorage or default
    const lastVideoModelId = localStorage.getItem('lastVideoModelId');
    if (lastVideoModelId && videoModelSelect.querySelector(`option[value="${lastVideoModelId}"]`)) {
        videoModelSelect.value = lastVideoModelId;
    } else if (modelAdded) {
        videoModelSelect.selectedIndex = 1; // Select first available I2V model
    } else {
        videoModelSelect.innerHTML = '<option value="">No Image-to-Video models available</option>';
        console.error("No I2V models found in availableModels:", availableModels);
    }

    // Show modal
    videoGenModal.style.display = 'block';
    videoPromptInput.focus();
}

function closeVideoGenModal() {
    const videoGenModal = document.getElementById('videoGenModal');
    if (videoGenModal) {
        videoGenModal.style.display = 'none';
    }
}

// --- End Video Generation Modal Logic ---

// --- Simple Feedback Function for main.js ---
function showMainFeedback(message, type = 'info') {
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
// --- End Simple Feedback Function ---

// Helper function to check if a media item is a video
function isVideo(mediaId) {
    return mediaId.startsWith('vid_') || mediaId.includes('_vid_');
}

// --- ADDED: Hover-to-play for gallery videos ---
function initializeVideoHoverPlay() {
    const galleryGrid = document.querySelector('.gallery-grid');
    if (!galleryGrid) return;

    galleryGrid.addEventListener('mouseover', (e) => {
        const video = e.target.closest('.gallery-item[data-media-type="video"] video');
        if (video && !video.hasAttribute('data-playing')) {
            video.play().catch(error => {
                // Ignore errors like the user not interacting first, etc.
                // console.log("Video play interrupted or failed:", error);
            });
            video.setAttribute('data-playing', 'true'); // Mark as playing
        }
    });

    galleryGrid.addEventListener('mouseout', (e) => {
        const video = e.target.closest('.gallery-item[data-media-type="video"] video');
        if (video && video.hasAttribute('data-playing')) {
            video.pause();
            video.removeAttribute('data-playing'); // Unmark
        }
    });
}
// --- END ADDED ---