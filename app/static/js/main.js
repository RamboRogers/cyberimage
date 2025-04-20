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
    initializeQueueStatusIndicator();
    initializeEnhancedPromptUI();
    initializeKeepPromptCheckbox();
    initializeCopyPromptButtons();

    // Add direct event listener for copy button as a fallback
    const copyPromptBtn = document.getElementById('copyPrompt');
    if (copyPromptBtn) {
        console.log('Adding direct event listener to copy button');
        copyPromptBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('Copy button clicked directly');
            const promptInfo = document.getElementById('promptInfo');
            if (promptInfo) {
                const promptText = promptInfo.textContent;
                if (promptText && promptText !== 'Loading...' && promptText !== 'Error loading details') {
                    // Create a temporary textarea element to copy from
                    const textarea = document.createElement('textarea');
                    textarea.value = promptText;
                    textarea.style.position = 'fixed';
                    textarea.style.opacity = '0';
                    document.body.appendChild(textarea);
                    textarea.select();

                    try {
                        const success = document.execCommand('copy');
                        if (success) {
                            console.log('Text copied via execCommand');
                            this.innerHTML = '<span class="icon">✓</span>Copied!';
                        } else {
                            console.warn('execCommand copy failed');
                            this.innerHTML = '<span class="icon">❌</span>Failed';
                        }
                    } catch (err) {
                        console.error('Copy error:', err);
                        this.innerHTML = '<span class="icon">❌</span>Error';
                    }

                    document.body.removeChild(textarea);

                    setTimeout(() => {
                        this.innerHTML = '<span class="icon">📋</span>Copy Prompt';
                    }, 2000);
                }
            }
        });
    }
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
        const negativePromptGroup = document.querySelector('.input-group:has(#negative-prompt)');
        const promptInput = document.getElementById('prompt-input');

        if (modelSelect && data.models) {
            modelSelect.innerHTML = '<option value="">Select Model</option>';

            // Store full model data for later access
            const modelsDataStore = {};

            // Store flux models for handling negative prompt visibility
            const fluxModels = [];

            Object.entries(data.models).forEach(([id, info]) => {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = `${id} - ${info.description}`;

                // Store the full info object for this model
                modelsDataStore[id] = info;

                // Mark Flux models for special handling
                if (id.toLowerCase().includes('flux')) {
                    option.dataset.isFlux = 'true';
                    fluxModels.push(id);
                }

                if (id === data.default) option.selected = true;
                modelSelect.appendChild(option);
            });

            // --- Add localStorage logic for Model ---
            // Attempt to load last selected model ID
            const lastModelId = localStorage.getItem('lastModelId');
            if (lastModelId && modelSelect.querySelector(`option[value="${lastModelId}"]`)) {
                modelSelect.value = lastModelId;
            }
            // --- End localStorage logic ---

            // Add change handler to hide/show negative prompt based on model AND adjust steps
            modelSelect.addEventListener('change', () => {
                const selectedModelId = modelSelect.value;
                const selectedModelData = modelsDataStore[selectedModelId];
                const stepsSlider = document.getElementById('steps');
                const stepsValueDisplay = document.getElementById('steps-value');

                if (selectedModelId && selectedModelData) {
                    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
                    const isFluxModel = selectedOption.dataset.isFlux === 'true' ||
                                        fluxModels.includes(selectedOption.value);

                    // Toggle negative prompt visibility
                    if (negativePromptGroup) {
                        if (isFluxModel) {
                            negativePromptGroup.style.display = 'none';
                            negativePromptGroup.querySelector('textarea').value = ''; // Clear value when hidden
                        } else {
                            negativePromptGroup.style.display = 'block';
                        }
                    }

                    // --- Add Step Configuration Handling ---
                    if (stepsSlider && stepsValueDisplay) {
                        const stepConfig = selectedModelData.step_config || {};
                        const lastStepsValue = localStorage.getItem('lastStepsValue');
                        let currentSteps = null;

                        stepsSlider.disabled = false;
                        stepsSlider.classList.remove('disabled'); // Ensure styling reflects enabled state

                        // 1. Check for fixed steps
                        if (stepConfig.fixed_steps !== undefined) {
                            currentSteps = stepConfig.fixed_steps;
                            stepsSlider.value = currentSteps;
                            stepsSlider.min = currentSteps;
                            stepsSlider.max = currentSteps;
                            stepsSlider.disabled = true;
                            stepsSlider.classList.add('disabled'); // Ensure styling reflects disabled state
                            console.log(`Model ${selectedModelId} has fixed steps: ${currentSteps}`);
                        }
                        // 2. Check for step range
                        else if (stepConfig.steps && stepConfig.steps.min !== undefined && stepConfig.steps.max !== undefined) {
                            const min = stepConfig.steps.min;
                            const max = stepConfig.steps.max;
                            const defaultVal = stepConfig.steps.default !== undefined ? stepConfig.steps.default : Math.round((min + max) / 2);

                            stepsSlider.min = min;
                            stepsSlider.max = max;
                            // Prioritize localStorage value if valid within range, otherwise use default
                            currentSteps = (lastStepsValue !== null && Number(lastStepsValue) >= min && Number(lastStepsValue) <= max)
                                            ? Number(lastStepsValue)
                                            : defaultVal;
                            stepsSlider.value = currentSteps;
                            console.log(`Model ${selectedModelId} has range: min=${min}, max=${max}, default=${defaultVal}. Using: ${currentSteps}`);
                        }
                        // 3. Fallback to default slider settings
                        else {
                            const defaultMin = 20;
                            const defaultMax = 50;
                            const defaultVal = 30;
                            stepsSlider.min = defaultMin;
                            stepsSlider.max = defaultMax;
                            // Prioritize localStorage value if valid within range, otherwise use default
                            currentSteps = (lastStepsValue !== null && Number(lastStepsValue) >= defaultMin && Number(lastStepsValue) <= defaultMax)
                                            ? Number(lastStepsValue)
                                            : defaultVal;
                            stepsSlider.value = currentSteps;
                             console.log(`Model ${selectedModelId} using default range. Using: ${currentSteps}`);
                        }

                        // Update display and save potentially adjusted value back to localStorage
                        if (currentSteps !== null) {
                            stepsValueDisplay.textContent = stepsSlider.value; // Display the final value set on the slider
                            localStorage.setItem('lastStepsValue', stepsSlider.value);
                        }
                    }
                    // --- End Step Configuration Handling ---
                } else {
                    // Handle case where "Select Model" is chosen or data is missing
                     if (stepsSlider && stepsValueDisplay) {
                        // Reset to defaults if no model is selected
                        const defaultMin = 20;
                        const defaultMax = 50;
                        const defaultVal = 30;
                        stepsSlider.min = defaultMin;
                        stepsSlider.max = defaultMax;
                        stepsSlider.value = defaultVal;
                        stepsSlider.disabled = false;
                        stepsSlider.classList.remove('disabled');
                        stepsValueDisplay.textContent = defaultVal;
                        // Optionally clear localStorage for steps? Or leave it?
                        // Leaving it for now - if user re-selects a model, it will be reapplied if valid.
                    }
                }

                // --- Add localStorage saving for Model ---
                localStorage.setItem('lastModelId', modelSelect.value);
                // --- End localStorage saving ---
            });

            // Trigger change event AFTER setting potential localStorage value
            modelSelect.dispatchEvent(new Event('change'));

            // Enhance prompt input box
            if (promptInput) {
                // Add placeholder text enhancement
                promptInput.placeholder = "Describe your vision in detail for better results...";

                // Auto resize the textarea based on content
                promptInput.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = (this.scrollHeight) + 'px';

                    // Simplified feedback - just show positive feedback when there's content
                    if (this.value.length > 0) {
                        this.classList.add('prompt-ideal');
                        this.classList.remove('prompt-short', 'prompt-good');
                    } else {
                        this.classList.remove('prompt-ideal', 'prompt-good', 'prompt-short');
                    }
                });

                // Add length indicator
                const lengthIndicator = document.createElement('div');
                lengthIndicator.className = 'prompt-length-indicator';
                lengthIndicator.innerHTML = '<span class="current">0</span> chars (no limit)';
                promptInput.parentNode.appendChild(lengthIndicator);

                promptInput.addEventListener('input', function() {
                    const currentLength = this.value.length;
                    const currentSpan = lengthIndicator.querySelector('.current');
                    if (currentSpan) {
                        currentSpan.textContent = currentLength;

                        // Always show positive feedback regardless of length
                        if (currentLength > 0) {
                            currentSpan.className = 'current ideal';
                        } else {
                            currentSpan.className = 'current';
                        }
                    }
                });
            }
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

    // Debug log to check if elements are found
    console.log('Gallery elements initialized:', {
        modal: modal !== null,
        copyPromptBtn: copyPromptBtn !== null,
        promptInfo: promptInfo !== null
    });

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
            // Try to use the Clipboard API with fallback
            const copyToClipboard = async (text) => {
                try {
                    // Modern method - Clipboard API
                    await navigator.clipboard.writeText(text);
                    return true;
                } catch (err) {
                    console.warn('Clipboard API failed:', err);

                    // Fallback method - create temporary textarea
                    try {
                        const textarea = document.createElement('textarea');
                        textarea.value = text;
                        textarea.style.position = 'fixed';  // Prevent scrolling to bottom
                        textarea.style.opacity = '0';
                        document.body.appendChild(textarea);
                        textarea.select();
                        const success = document.execCommand('copy');
                        document.body.removeChild(textarea);
                        return success;
                    } catch (fallbackErr) {
                        console.error('Fallback clipboard method failed:', fallbackErr);
                        return false;
                    }
                }
            };

            // Execute copy and show feedback
            const originalText = copyPromptBtn.innerHTML;
            copyToClipboard(promptText).then(success => {
                if (success) {
                    copyPromptBtn.innerHTML = '<span class="icon">✓</span>Copied!';
                } else {
                    copyPromptBtn.innerHTML = '<span class="icon">❌</span>Failed to copy';
                }

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

    // --- Add localStorage loading for Steps ---
    const stepsSlider = document.getElementById('steps');
    const stepsValueDisplay = document.getElementById('steps-value');
    if (stepsSlider && stepsValueDisplay) {
        const lastStepsValue = localStorage.getItem('lastStepsValue');
        if (lastStepsValue) {
            stepsSlider.value = lastStepsValue;
            stepsValueDisplay.textContent = lastStepsValue;
        }
        // Add listener to save steps value on change
        stepsSlider.addEventListener('input', () => {
            localStorage.setItem('lastStepsValue', stepsSlider.value);
            stepsValueDisplay.textContent = stepsSlider.value; // Ensure display updates here too
        });
    }
    // --- End localStorage loading ---

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitButton = form.querySelector('.button-generate');
        submitButton.disabled = true;

        // Hide any previous feedback
        feedbackSection.style.display = 'none';
        feedbackSection.classList.remove('active');

        // --- Add localStorage saving on Submit ---
        const currentModelId = document.getElementById('model')?.value;
        const currentStepsValue = document.getElementById('steps')?.value;
        if (currentModelId) {
            localStorage.setItem('lastModelId', currentModelId);
        }
        if (currentStepsValue) {
            localStorage.setItem('lastStepsValue', currentStepsValue);
        }

        // Save the prompt when generating an image if "Keep Prompt" is checked
        const keepPromptCheckbox = document.getElementById('keep_prompt');
        if (keepPromptCheckbox && keepPromptCheckbox.checked) {
            const promptInput = document.getElementById('prompt-input');
            const promptToKeep = promptInput.value;
            if (promptToKeep) {
                localStorage.setItem('keptPrompt', promptToKeep);
                console.log('Prompt saved at generation time:', promptToKeep.substring(0, 50) + (promptToKeep.length > 50 ? '...' : ''));
            }
        }
        // --- End localStorage saving ---

        try {
            const formData = new FormData(form);
            const promptInput = document.getElementById('prompt-input');
            const prompt = formData.get('prompt');

            // No validation on prompt length - accept any length
            // The backend/LLM will handle truncation if needed

            const requestData = {
                model_id: formData.get('model'),
                prompt: prompt, // Use prompt without any length validation
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
                            message = totalImages > 1 ?
                                `Saving ${totalImages} generated images...` :
                                'Saving generated image...';
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
                            message = `Generating ${totalImages > 1 ? 'images' : 'image'}...`;
                            if (currentStage !== 'generating') {
                                currentStage = 'generating';
                                stageStartTime = Date.now();
                            }
                        } else if (isSaving) {
                            message = `Saving ${totalImages > 1 ? 'images' : 'image'}...`;
                            progress = Math.max(progress, 85); // At least 85% for saving
                        } else if (progress < 40) {
                            message = 'Preparing model...';
                        } else if (progress < 70) {
                            message = `Generating ${totalImages > 1 ? 'images' : 'image'}...`;
                        } else {
                            message = `Finalizing ${totalImages > 1 ? 'images' : 'image'}...`;
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

// Initialize copy prompt buttons in the gallery
function initializeCopyPromptButtons() {
    const copyButtons = document.querySelectorAll('.action-copy-prompt');

    copyButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent opening the modal

            const promptText = button.dataset.prompt;
            if (!promptText) return;

            // Copy to clipboard
            const copyToClipboard = async (text) => {
                try {
                    // Modern method - Clipboard API
                    await navigator.clipboard.writeText(text);
                    return true;
                } catch (err) {
                    console.warn('Clipboard API failed:', err);

                    // Fallback method - create temporary textarea
                    try {
                        const textarea = document.createElement('textarea');
                        textarea.value = text;
                        textarea.style.position = 'fixed';
                        textarea.style.opacity = '0';
                        document.body.appendChild(textarea);
                        textarea.select();
                        const success = document.execCommand('copy');
                        document.body.removeChild(textarea);
                        return success;
                    } catch (fallbackErr) {
                        console.error('Fallback clipboard method failed:', fallbackErr);
                        return false;
                    }
                }
            };

            // Show copy feedback
            const originalText = button.innerHTML;
            copyToClipboard(promptText).then(success => {
                if (success) {
                    button.innerHTML = '✓';
                    button.classList.add('copied');
                } else {
                    button.innerHTML = '❌';
                }

                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.classList.remove('copied');
                }, 2000);
            });
        });
    });
}