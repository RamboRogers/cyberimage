import { ModalManager } from './modules/modalManager.js';
import { showMainFeedback } from './modules/uiUtils.js'; // Assuming showMainFeedback is exported from uiUtils

// API endpoints
const API = {
    BASE: '/api',
    MODELS: '/api/models',
    GENERATE: '/api/generate',
    STATUS: (jobId) => `/api/status/${jobId}`,
    IMAGE: (imageId) => `/api/get_image/${imageId}`,
    VIDEO: (videoId) => `/api/get_video/${videoId}`,
    METADATA: (imageId) => `/api/image/${imageId}/metadata`,
    QUEUE: '/api/queue'
};

// Global store for available models
// availableModels is used by ModalManager.openVideoGenerator, ensure it's accessible or passed.
// For now, keeping it global as ModalManager expects.
let availableImageModels = {};
let availableModels = {};
let submittedJobIdsFromIndexPage = [];
let indexPageJobPollingInterval = null;
let pendingIndexPageReload = false;
let indexReloadNotificationTimer = null;
// globalCurrentMediaId and globalCurrentMediaType are now managed within ModalManager.js

const API_IMAGE_GEN = '/api/generate';
const API_I2V_GEN = '/api/generate_video';
const API_T2V_GEN = '/api/generate_t2v';

// --- All Modal functions (hideModal, showFullscreenMedia, initializeModalHandling, initializeGalleryHandling for modal parts, confirmDeleteMedia, video gen modal functions) are moved to modalManager.js ---

// Timer for tracking generation duration (KEEPING in main.js for now, or move to generationForm.js later)
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
    initializeMobileNav(); // Can be called early

    initializeModels().then(() => {
        // Initialize ModalManager AFTER models are loaded and availableModels is populated
        ModalManager.initialize({
            callbacks: {
                refreshRecentGenerationsSection: refreshRecentGenerationsSection
            },
            API: API,
            availableModels: availableModels, // Now this will have data
            API_I2V_GEN: API_I2V_GEN
        });

        // Other initializations that might depend on models or ModalManager
        if (document.getElementById('generate-form')) {
            initializeGenerationForm();
            initializeKeepPromptCheckbox();
            initializeEnhancedPromptUI();
        }
        // initializeGalleryHandling() removed - Its modal parts are in ModalManager.
        // Gallery grid click to open fullscreen modal needs to be re-established if not covered by ModalManager's general click handling
        // For now, relying on ModalManager to handle clicks that should open fullscreen, if HTML is structured correctly.
        // OR, add a specific listener here if needed:
        const mainGalleryGrid = document.querySelector('.gallery-grid'); // General selector, might be too broad
        if (mainGalleryGrid) {
            mainGalleryGrid.addEventListener('click', (e) => {
                // Trigger fullscreen for main gallery images/videos when clicking on preview (but not on quick actions)
                const previewElement = e.target.closest('.item-preview img, .item-preview video');
                if (previewElement && !e.target.closest('.quick-actions') && !e.target.closest('.action-button')) { // ensure not a button
                    // Check if the gallery item is from recent generations or a main gallery page
                    const galleryItem = previewElement.closest('.gallery-item');
                    if (galleryItem) {
                        ModalManager.showFullscreen(galleryItem);
                    }
                }
            });
    }

    if (document.getElementById('queue-status-text')) {
            initializeQueueStatusIndicator();
            initializeQueueStatusPolling();
        }

        if (document.querySelector('.recent-images .gallery-grid')) {
            // initializeRecentGenerationsViewer(); // This was mostly a placeholder
            refreshRecentGenerationsSection(); // Load initial recent items
            // Event listeners for recent generations actions are set up within appendItemsToRecentGenerations
        }

        initializeGalleryItemActions();

        if (submittedJobIdsFromIndexPage.length > 0 && !indexPageJobPollingInterval) {
            indexPageJobPollingInterval = setInterval(pollSubmittedIndexPageJobs, 5000);
        }

        const promptInput = document.querySelector('#generation-form #prompt-input');
        if(promptInput) {
            promptInput.addEventListener('focus', () => checkQueueAndReloadIfIndexPending());
            promptInput.addEventListener('blur', () => setTimeout(checkQueueAndReloadIfIndexPending, 100));
        }

    }).catch(error => {
        console.error("Failed to initialize models (and subsequently ModalManager):", error);
        if(typeof showMainFeedback === 'function') showMainFeedback("Error loading critical components. Page functionality may be limited.", "error");
    });
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
        document.addEventListener('click', (e) => {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            }
        });
        navLinks.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            });
        });
    }
}

async function initializeModels() {
    try {
        const response = await fetch(API.MODELS);
        const data = await response.json();
        const modelSelect = document.querySelector('select[name="model"]');

        if (data.models) {
            availableModels = {}; // Reset the global store
            const mainFormModels = {};
            Object.entries(data.models).forEach(([id, info]) => {
                const modelType = info.type || 'image';
                availableModels[id] = { ...info, id: id, type: modelType };
                if (modelType === 'image' || modelType === 't2v') {
                    mainFormModels[id] = { ...info, id: id, type: modelType };
                }
            });

            if (modelSelect) {
                modelSelect.innerHTML = '<option value="">Select Model</option>';
                const modelsDataStore = {};
                const localModels = [];
                const apiModels = [];
                Object.entries(mainFormModels).forEach(([id, info]) => {
                    modelsDataStore[id] = info;
                    if (info.source === 'huggingface_api' || info.source === 'fal_api') {
                        apiModels.push({ id, ...info });
                    } else {
                        localModels.push({ id, ...info });
                    }
                });
                localModels.forEach(info => {
                    const option = document.createElement('option');
                    option.value = info.id;
                    const typeLabel = info.type === 't2v' ? '[Video]' : '[Image]';
                    option.textContent = `${info.id} ${typeLabel} - ${info.description}`;
                    if (info.id === data.default) option.selected = true;
                    modelSelect.appendChild(option);
                });
                apiModels.forEach(info => {
                    const option = document.createElement('option');
                    option.value = info.id;
                    const typeLabel = info.type === 't2v' ? '[Video]' : '[Image]';
                    const apiProvider = info.provider || (info.source === 'fal_api' ? 'Fal.ai' : 'Default');
                    option.textContent = `${info.id} ${typeLabel} - ${info.description} (API: ${apiProvider})`;
                    if (info.id === data.default && !modelSelect.querySelector('option[selected]')) option.selected = true;
                    modelSelect.appendChild(option);
                });
                const lastModelId = localStorage.getItem('lastModelId');
                if (lastModelId && modelSelect.querySelector(`option[value="${lastModelId}"]`)) {
                    modelSelect.value = lastModelId;
                }
                modelSelect.addEventListener('change', () => handleModelChange(modelsDataStore));
                handleModelChange(modelsDataStore);
            }
        } else {
             console.error('Model data structure unexpected:', data);
        }
    } catch (error) {
        console.error('Error loading models:', error);
        throw error;
    }
}

function handleModelChange(modelsDataStore) {
    const modelSelect = document.querySelector('select[name="model"]');
    if (!modelSelect) return;
    const selectedModelId = modelSelect.value;
    const selectedModelData = modelsDataStore ? modelsDataStore[selectedModelId] : null;
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
    const defaults = {
        image: { steps: 30, guidance: 7.5, width: 1024, height: 1024 },
        t2v:   { steps: 50, guidance: 7.0, width: 704, height: 480, frames: 161 }
    };

    if (selectedModelId && selectedModelData) {
        const modelType = selectedModelData.type || 'image';
        const isT2V = modelType === 't2v';
        const isFluxModel = selectedModelId.toLowerCase().includes('flux');
        if (generateButton) {
            generateButton.innerHTML = isT2V
                ? '<span class="button-icon">🎬</span> Generate Video'
                : '<span class="button-icon">⚡</span> Generate Image';
        }
        if (t2vFramesGroup) t2vFramesGroup.classList.toggle('hidden', !isT2V);
        if (numImagesGroup) numImagesGroup.classList.toggle('hidden', isT2V);
        if (negativePromptGroup) {
            negativePromptGroup.style.display = isFluxModel ? 'none' : 'block';
            if (isFluxModel) negativePromptGroup.querySelector('textarea').value = '';
            }
        const currentDefaults = defaults[modelType] || defaults.image;
        if (stepsSlider && stepsValueDisplay) {
            const stepConfig = selectedModelData.step_config || {};
            let currentStepsVal = null;
            stepsSlider.disabled = false;
            stepsSlider.classList.remove('disabled');
            stepsSlider.min = stepConfig.steps?.min ?? (isT2V ? 10 : 20);
            stepsSlider.max = stepConfig.steps?.max ?? (isT2V ? 60 : 50);
            currentStepsVal = stepConfig.steps?.default ?? currentDefaults.steps;
            if (stepConfig.fixed_steps !== undefined) {
                 currentStepsVal = stepConfig.fixed_steps;
                 stepsSlider.value = currentStepsVal;
                 stepsSlider.min = currentStepsVal;
                 stepsSlider.max = currentStepsVal;
                 stepsSlider.disabled = true;
                 stepsSlider.classList.add('disabled');
            } else {
                const lastStepsValue = localStorage.getItem('lastStepsValue');
                if (lastStepsValue !== null && Number(lastStepsValue) >= parseFloat(stepsSlider.min) && Number(lastStepsValue) <= parseFloat(stepsSlider.max)) {
                    stepsSlider.value = Number(lastStepsValue);
                } else {
                    stepsSlider.value = currentStepsVal;
                }
            }
            stepsValueDisplay.textContent = stepsSlider.value;
            localStorage.setItem('lastStepsValue', stepsSlider.value);
        }
        if (guidanceSlider && guidanceValueDisplay) {
            const guidanceConfig = selectedModelData.step_config?.guidance || {};
            guidanceSlider.disabled = false;
            guidanceSlider.classList.remove('disabled');
            guidanceSlider.min = guidanceConfig.min ?? (isT2V ? 1 : 1);
            guidanceSlider.max = guidanceConfig.max ?? (isT2V ? 10 : 20);
            guidanceSlider.step = guidanceConfig.step ?? (isT2V ? 0.1 : 0.5);
            let currentGuidance = guidanceConfig.default ?? currentDefaults.guidance;
            const lastGuidanceValue = localStorage.getItem('lastGuidanceValue');
            if (lastGuidanceValue !== null && Number(lastGuidanceValue) >= parseFloat(guidanceSlider.min) && Number(lastGuidanceValue) <= parseFloat(guidanceSlider.max)) {
                guidanceSlider.value = Number(lastGuidanceValue);
            } else {
                guidanceSlider.value = currentGuidance;
            }
            guidanceValueDisplay.textContent = guidanceSlider.value;
            localStorage.setItem('lastGuidanceValue', guidanceSlider.value);
        }
        if (widthSelect && heightSelect) {
            if (!widthSelect.querySelector(`option[value="${currentDefaults.width}"]`)) console.warn(`Width option ${currentDefaults.width} not found for ${modelType} model.`);
            if (!heightSelect.querySelector(`option[value="${currentDefaults.height}"]`)) console.warn(`Height option ${currentDefaults.height} not found for ${modelType} model.`);
            widthSelect.value = currentDefaults.width;
            heightSelect.value = currentDefaults.height;
        }
        if (isT2V && numFramesSlider && numFramesValueDisplay) {
            const framesConfig = selectedModelData.step_config?.frames || {};
            numFramesSlider.min = framesConfig.min ?? 5;
            numFramesSlider.max = framesConfig.max ?? 161;
            numFramesSlider.step = framesConfig.step ?? 4;
            numFramesSlider.value = framesConfig.default ?? currentDefaults.frames;
            numFramesValueDisplay.textContent = numFramesSlider.value;
        }
    } else {
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

function initializeGenerationForm() {
    const form = document.getElementById('generate-form');
    if (!form) return;
    const feedbackSection = form.querySelector('.generation-feedback');
    if (!feedbackSection) {
        console.error('Feedback section not found');
        return;
    }
    feedbackSection.removeAttribute('style');
    feedbackSection.style.display = 'block'; // Should be controlled by active class
    const timer = new GenerationTimer(feedbackSection);
    const statusText = feedbackSection.querySelector('.status-text');
    const timerContainer = feedbackSection.querySelector('.generation-timer');
    const generateButton = form.querySelector('.button-generate');

    const sliders = form.querySelectorAll(':scope > .form-layout .settings-col .slider');
    sliders.forEach(slider => {
        const display = slider.nextElementSibling;
        if (display && display.classList.contains('value-display')) {
            display.textContent = slider.value;
            slider.addEventListener('input', () => { display.textContent = slider.value; });
        }
    });
    const stepsSlider = document.getElementById('steps');
    if (stepsSlider) stepsSlider.addEventListener('input', () => { localStorage.setItem('lastStepsValue', stepsSlider.value); });
    const guidanceSlider = document.getElementById('guidance');
    if (guidanceSlider) guidanceSlider.addEventListener('input', () => { localStorage.setItem('lastGuidanceValue', guidanceSlider.value); });

    ['prompt-input', 'negative-prompt'].forEach(id => {
        const inputElement = document.getElementById(id);
        if (inputElement) {
            inputElement.addEventListener('blur', () => {
                if (pendingIndexPageReload) {
                    setTimeout(() => { checkQueueAndReloadIfIndexPending(); }, 100);
                }
            });
        }
    });
    if (generateButton) generateButton.innerHTML = '<span class="button-icon">⚡</span> Generate Image';

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitButton = form.querySelector('.button-generate');
        submitButton.disabled = true;
        feedbackSection.style.display = 'none';
        feedbackSection.classList.remove('active');

        const currentModelId = document.getElementById('model')?.value;
        if (currentModelId) localStorage.setItem('lastModelId', currentModelId);
        const keepPromptCheckbox = document.getElementById('keep_prompt');
        if (keepPromptCheckbox && keepPromptCheckbox.checked) {
            localStorage.setItem('keptPrompt', document.getElementById('prompt-input').value);
            }

        let modelType = 'image';
        let feedbackTypeText = 'Image';

        try {
            const formData = new FormData(form);
            const prompt = formData.get('prompt');
            const modelId = formData.get('model');
            const selectedModelInfo = availableModels[modelId];
            if (!selectedModelInfo) throw new Error(`Selected model (${modelId}) not found.`);
            modelType = selectedModelInfo.type || 'image';

            let apiUrl, requestData, numOutputs;
            if (modelType === 't2v') {
                apiUrl = API_T2V_GEN;
                feedbackTypeText = 'Video';
                requestData = {
                    model_id: modelId, prompt: prompt,
                    settings: { fps: 16, num_frames: parseInt(document.getElementById('num_frames')?.value || '17'), type: 't2v' }
                };
                numOutputs = 1;
            } else {
                apiUrl = API_IMAGE_GEN;
                feedbackTypeText = 'Image';
                requestData = {
                    model_id: modelId, prompt: prompt, negative_prompt: formData.get('negative_prompt') || undefined,
                settings: {
                        num_images: parseInt(formData.get('num_images') || '1'),
                        num_inference_steps: parseInt(formData.get('steps') || '30'),
                        guidance_scale: parseFloat(formData.get('guidance') || '7.5'),
                        height: parseInt(formData.get('height') || '1024'),
                        width: parseInt(formData.get('width') || '1024'),
                        type: 'image'
                    }
                };
                numOutputs = requestData.settings.num_images;
            }
            submitButton.innerHTML = `<span class="button-icon spin">⏳</span> Submitting ${feedbackTypeText}...`;

            const response = await fetch(apiUrl, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestData)
            });
            const data = await response.json();

            if (response.ok && data.job_id) {
                feedbackSection.style.display = 'block';
                feedbackSection.classList.add('active');
                if (statusText) statusText.textContent = `Preparing ${feedbackTypeText} generation...`;
                if (timerContainer) timerContainer.style.display = 'flex';
                timer.reset(); timer.start();
                await pollGenerationStatus(data.job_id, feedbackSection, numOutputs, timer, feedbackTypeText);
                if (data.job_id) {
                    submittedJobIdsFromIndexPage.push(data.job_id);
                    if (!indexPageJobPollingInterval) {
                        indexPageJobPollingInterval = setInterval(pollSubmittedIndexPageJobs, 10000);
                    }
                }
            } else {
                if(typeof showMainFeedback === 'function') showMainFeedback(`Error: ${data.message || `Failed to start ${feedbackTypeText} job`}`, 'error');
                timer.stop();
            }
        } catch (error) {
            console.error(`Error submitting ${feedbackTypeText} generation request:`, error);
            if(typeof showMainFeedback === 'function') showMainFeedback(`Error: ${error.message}`, 'error');
            timer.stop();
        } finally {
             // Restore button text based on model type, needs handleModelChange to be callable or similar logic here
            const finalModelId = document.getElementById('model')?.value;
            const finalSelectedModel = availableModels[finalModelId];
            const finalIsT2V = finalSelectedModel && finalSelectedModel.type === 't2v';
            submitButton.innerHTML = finalIsT2V ? '<span class="button-icon">🎬</span> Generate Video' : '<span class="button-icon">⚡</span> Generate Image';
            submitButton.disabled = false;
        }
    });
}

function initializeQueueStatusPolling() {
    const updateQueueStatus = async () => {
        try {
            const response = await fetch(API.QUEUE);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            const { pending, processing } = data;
            const queueStatusEl = document.querySelector('.queue-status'); // In nav, from HTML structure
            if (queueStatusEl) {
                if (pending > 0 || processing > 0) {
                    queueStatusEl.textContent = `Queue: ${pending} waiting, ${processing} processing`;
                    queueStatusEl.style.display = 'block'; // Or 'flex' depending on CSS
                } else {
                    queueStatusEl.style.display = 'none';
                }
            }
        } catch (error) {
            console.error('Error updating queue status:', error);
            const queueStatusEl = document.querySelector('.queue-status');
            if (queueStatusEl) queueStatusEl.style.display = 'none';
        }
    };
    setInterval(updateQueueStatus, 5000);
    updateQueueStatus();
}

async function showImageDetails(imageId) { // This seems like an older modal, might be deprecated or for admin use.
    try {
        const response = await fetch(API.METADATA(imageId));
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        const modal = document.querySelector('.modal'); // Generic .modal
        const content = modal?.querySelector('.modal-content');
        if (!modal || !content) return;

        const generationTime = new Date(data.generation_time * 1000);
        // Assuming formatDate and formatDateLong are available (e.g. from uiUtils.js)
        const formattedDate = typeof formatDate === 'function' ? formatDate(generationTime) : generationTime.toLocaleTimeString();
        const formattedDateLong = typeof formatDateLong === 'function' ? formatDateLong(generationTime) : generationTime.toLocaleString();

        content.innerHTML = `
            <span class="close-modal">&times;</span>
            <img src="${API.IMAGE(imageId)}" alt="Generated Image" style="max-width: 100%; margin-bottom: 20px;">
            <div class="image-details">
                <p><strong>Generated:</strong> <span title="${formattedDateLong}">${formattedDate}</span></p>
                <p><strong>Model:</strong> ${data.model_id}</p>
                <p><strong>Prompt:</strong> ${data.prompt}</p>
                <p><strong>Settings:</strong></p>
                <pre>${JSON.stringify(data.settings, null, 2)}</pre>
            </div>
        `;
        modal.classList.add('visible');
    } catch (error) {
        console.error('Error fetching image details:', error);
    }
}

async function handlePromptEnrichment(e) {
    e.preventDefault();
    const promptInput = document.getElementById('prompt-input');
    const styleSelect = document.getElementById('enrich-style');
    const currentPrompt = promptInput.value.trim();
    const selectedStyle = styleSelect.value;
    if (!currentPrompt) {
        if(typeof addNeonFlash === 'function') addNeonFlash(promptInput); // from uiUtils
        return;
    }
    const enrichButton = e.target.closest('#enrich-prompt');
    enrichButton.disabled = true;
    enrichButton.innerHTML = '<span class="button-icon spin">⏳</span> Enriching...';
    const originalPrompt = currentPrompt;
    try {
        const response = await fetch('/api/enrich', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: currentPrompt, style: selectedStyle })
        });
        const data = await response.json();
        if (data.enriched_prompt) {
            promptInput.value = data.enriched_prompt;
            if(typeof addNeonFlash === 'function') addNeonFlash(promptInput);
            showPromptComparison(originalPrompt, data.enriched_prompt);
        }
    } catch (error) {
        console.error('Error enriching prompt:', error);
    } finally {
        enrichButton.disabled = false;
        enrichButton.innerHTML = '<span class="button-icon">✨</span> Enrich';
    }
}

function showPromptComparison(original, enriched) {
    const comparisonDiv = document.getElementById('prompt-comparison');
    const originalPromptDiv = document.getElementById('original-prompt');
    const restoreButton = document.getElementById('restore-original');
    const closeButton = document.getElementById('close-comparison');
    const promptInput = document.getElementById('prompt-input');
    if (!comparisonDiv || !originalPromptDiv) return;
    originalPromptDiv.textContent = original;
    comparisonDiv.style.display = 'block';
    if (restoreButton) {
        restoreButton.onclick = () => {
            promptInput.value = original;
            if(typeof addNeonFlash === 'function') addNeonFlash(promptInput);
        };
    }
    if (closeButton) closeButton.onclick = () => { comparisonDiv.style.display = 'none'; };
}

function initializeEnrichInfo() {
    const infoIcon = document.getElementById('enrich-info');
    const tooltip = document.querySelector('.enrich-tooltip');
    if (!infoIcon || !tooltip) return;
    infoIcon.addEventListener('click', (e) => {
        e.stopPropagation();
        tooltip.style.display = tooltip.style.display === 'block' ? 'none' : 'block';
    });
    document.addEventListener('click', () => { tooltip.style.display = 'none'; });
}

function initializeEnhancedPromptUI() {
    initializeEnrichInfo();
    const enrichButton = document.getElementById('enrich-prompt');
    if (enrichButton) enrichButton.addEventListener('click', handlePromptEnrichment);
}

function initializeQueueStatusIndicator() {
    const statusIcon = document.getElementById('generation-status-icon');
    const statusText = document.getElementById('queue-status-text');
    const queueIndicator = statusIcon?.closest('.queue-indicator');
    if (!statusIcon || !statusText || !queueIndicator) return;

    function formatTime(seconds) { // Inner helper, or move to uiUtils
        if (seconds < 60) return `${Math.round(seconds)} sec`;
        if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.round((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }

    function showQueueDetailsPopup() {
        fetch('/api/queue?detailed=true')
            .then(response => response.json())
            .then(data => {
                let detailsPopup = document.getElementById('queue-details-popup');
                if (!detailsPopup) {
                    detailsPopup = document.createElement('div');
                    detailsPopup.id = 'queue-details-popup';
                    detailsPopup.className = 'queue-details-popup'; // Add CSS for this
                    document.body.appendChild(detailsPopup);
                    const closeBtn = document.createElement('button');
                    closeBtn.className = 'close-popup'; closeBtn.innerHTML = '×';
                    closeBtn.addEventListener('click', (e) => { e.stopPropagation(); detailsPopup.classList.remove('visible'); });
                    detailsPopup.appendChild(closeBtn);
                }
                const rect = queueIndicator.getBoundingClientRect();
                detailsPopup.style.top = `${rect.bottom + window.scrollY + 10}px`;
                detailsPopup.style.left = `${rect.left + window.scrollX}px`;

                let content = `<h3>Queue Status</h3><div class="queue-stats">
                    <div class="stat-item"><span class="stat-label"><i class="fas fa-hourglass-half"></i> Pending</span><span class="stat-value ${data.pending > 0 ? 'highlight' : ''}">${data.pending}</span></div>
                    <div class="stat-item"><span class="stat-label"><i class="fas fa-cog fa-spin"></i> Processing</span><span class="stat-value ${data.processing > 0 ? 'highlight' : ''}">${data.processing}</span></div>
                    <div class="stat-item"><span class="stat-label"><i class="fas fa-check-circle"></i> Completed</span><span class="stat-value">${data.completed}</span></div>
                    <div class="stat-item"><span class="stat-label"><i class="fas fa-exclamation-triangle"></i> Failed</span><span class="stat-value ${data.failed > 0 ? 'error' : ''}">${data.failed}</span></div>
                    <div class="stat-item total"><span class="stat-label"><i class="fas fa-chart-bar"></i> Total Jobs</span><span class="stat-value highlight">${data.total}</span></div></div>`;

                if (data.avg_processing_time_seconds !== undefined) {
                    content += `<h3>Performance</h3><div class="queue-stats">
                        <div class="stat-item"><span class="stat-label"><i class="fas fa-clock"></i> Avg. Time</span><span class="stat-value">${formatTime(data.avg_processing_time_seconds)}</span></div>
                        <div class="stat-item"><span class="stat-label"><i class="fas fa-exclamation-circle"></i> Failure Rate</span><span class="stat-value ${data.failure_rate > 10 ? 'error' : ''}">${data.failure_rate}%</span></div></div>`;
                }
                if (data.models && Object.keys(data.models).length > 0) {
                    content += `<h3>Models (24h)</h3><div class="queue-stats">`;
                    Object.entries(data.models).forEach(([modelId, stats]) => {
                        const successRate = stats.completed > 0 ? Math.round((stats.completed / (stats.completed + stats.failed)) * 100) : 0;
                        content += `<div class="stat-item model-stat"><span class="stat-label"><i class="fas fa-robot"></i> ${modelId}</span><span class="stat-value"><span class="model-count">${stats.total} jobs</span> <span class="model-success-rate ${successRate > 90 ? 'highlight' : successRate < 70 ? 'error' : ''}">${successRate}% success</span></span></div>`;
                    });
                    content += `</div>`;
                }
                detailsPopup.innerHTML = content;
                const closeBtn = detailsPopup.querySelector('.close-popup') || document.createElement('button'); // Re-add if innerHTML overwrote
                if (!detailsPopup.querySelector('.close-popup')) {
                     closeBtn.className = 'close-popup'; closeBtn.innerHTML = '×';
                     closeBtn.addEventListener('click', (e) => { e.stopPropagation(); detailsPopup.classList.remove('visible'); });
                     detailsPopup.prepend(closeBtn); // Prepend to keep at top
                }
                detailsPopup.classList.add('visible');
                document.addEventListener('click', function closePopupOnClickOutside(e) {
                    if (!detailsPopup.contains(e.target) && e.target !== queueIndicator) {
                        detailsPopup.classList.remove('visible');
                        document.removeEventListener('click', closePopupOnClickOutside);
                    }
                });
            }).catch(error => console.error('Error fetching queue details:', error));
    }

    function updateIndicatorUI() {
        fetch('/api/queue')
            .then(response => response.json())
            .then(data => {
                const { pending, processing, completed, failed } = data;
                const totalActive = pending + processing;
                const tooltipText = `📊 Queue: ${pending} pending, ${processing} processing, ${completed} completed, ${failed} failed. Click for details.`;
                queueIndicator.setAttribute('title', tooltipText);
                if (totalActive > 0) {
                    statusText.textContent = `Queue: ${totalActive}`;
                    statusIcon.className = 'fas fa-cog fa-spin'; statusIcon.style.color = 'var(--neon-green)';
                } else {
                    statusText.textContent = 'Queue: 0';
                    statusIcon.className = 'fas fa-check-circle'; statusIcon.style.color = 'var(--neon-green)';
                }
                queueIndicator.style.display = 'flex';
            }).catch(error => {
                console.error('Error fetching queue status for indicator:', error);
                statusText.textContent = 'Queue: ?';
                statusIcon.className = 'fas fa-exclamation-triangle'; statusIcon.style.color = '#ff4444';
                queueIndicator.setAttribute('title', 'Could not fetch queue status');
                queueIndicator.style.display = 'flex';
            });
    }
    updateIndicatorUI();
    setInterval(updateIndicatorUI, 5000);
    queueIndicator.addEventListener('click', showQueueDetailsPopup);
}

function initializeKeepPromptCheckbox() {
    const checkbox = document.getElementById('keep_prompt');
    const promptInput = document.getElementById('prompt-input');
    if (!checkbox || !promptInput) return;
    const savedKeep = localStorage.getItem('keepPromptChecked');
    const savedPrompt = localStorage.getItem('keptPrompt');
    if (savedKeep === 'true') {
        checkbox.checked = true;
        if (savedPrompt !== null && promptInput.value.trim() === '') promptInput.value = savedPrompt;
        }
    checkbox.addEventListener('change', () => {
        localStorage.setItem('keepPromptChecked', checkbox.checked);
        if (!checkbox.checked) localStorage.removeItem('keptPrompt');
    });
}

function initializeGalleryItemActions() {
    const galleryGrid = document.querySelector('.gallery-grid'); // This could be any gallery grid on the page
    if (!galleryGrid) return;

    galleryGrid.addEventListener('click', async (e) => {
        const galleryItem = e.target.closest('.gallery-item');
        if (!galleryItem) return;
        const mediaId = galleryItem.dataset.mediaId;
        const mediaType = galleryItem.dataset.mediaType || 'image';

        if (e.target.closest('.action-copy-prompt')) { //delegated from initializeRecentGenerationsActionListeners
            const button = e.target.closest('.action-copy-prompt');
            const promptText = button.dataset.prompt;
            let copied = false;

            if (promptText) {
                // Try modern clipboard API first
                if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                try {
                    await navigator.clipboard.writeText(promptText);
                        copied = true;
                } catch (err) {
                        console.warn('navigator.clipboard.writeText failed (galleryItemActions):', err);
                    }
                }

                // Fallback to document.execCommand('copy')
                if (!copied) {
                    const textArea = document.createElement("textarea");
                    textArea.value = promptText;
                    textArea.style.position = "fixed"; textArea.style.top = "0"; textArea.style.left = "0";
                    textArea.style.width = "2em"; textArea.style.height = "2em";
                    textArea.style.padding = "0"; textArea.style.border = "none";
                    textArea.style.outline = "none"; textArea.style.boxShadow = "none";
                    textArea.style.background = "transparent";
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    try {
                        document.execCommand('copy');
                        copied = true;
                    } catch (err) {
                        console.warn('document.execCommand(\'copy\') failed (galleryItemActions):', err);
                    } finally {
                        document.body.removeChild(textArea);
                    }
                }

                const originalText = button.textContent; // Assuming button only contains text or simple icon
                if (copied) {
                    button.textContent = 'Copied!'; button.disabled = true;
                    setTimeout(() => { button.textContent = originalText; button.disabled = false; }, 1500);
                    if(typeof showMainFeedback === 'function') showMainFeedback('Prompt copied!', 'success', 1500);
                } else {
                    console.error('All copy methods failed for gallery item.');
                    if(typeof showMainFeedback === 'function') showMainFeedback('Could not copy prompt. Please try manually.', 'error');
                    // No direct text selection fallback here as it's a small button, not a text area.
                    // The error message should guide the user.
                    button.textContent = 'Fail'; // Keep it short
                    button.disabled = true;
                    setTimeout(() => { button.textContent = originalText; button.disabled = false; }, 2000);
                }
            }
        }
        else if (e.target.closest('.action-download')) {
            if (!mediaId) return;
            const button = e.target.closest('.action-download');
            const downloadUrl = mediaType === 'video' ? API.IMAGE(mediaId).replace("/get_image/", "/get_video/") : API.IMAGE(mediaId);
            const filename = `cyberimage-${mediaId}.${mediaType === 'video' ? 'mp4' : 'png'}`;
            const originalText = button.innerHTML;
             try {
                button.innerHTML = '⏳'; button.disabled = true;
                const response = await fetch(downloadUrl);
                if (!response.ok) throw new Error(`Failed to fetch media: ${response.statusText}`);
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a'); a.href = url; a.download = filename;
                document.body.appendChild(a); a.click(); document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                setTimeout(() => { button.innerHTML = originalText; button.disabled = false; }, 500);
            } catch (error) {
                console.error('Download failed (galleryItemActions):', error);
                button.innerHTML = '❌';
                setTimeout(() => { button.innerHTML = originalText; button.disabled = false; }, 2000);
            }
        }
        else if (e.target.closest('.action-delete')) {
            if (!mediaId) return;
             // Use ModalManager.confirmDelete if available and appropriate
             // This is more for a generic gallery page; recent-generations uses its own listeners
             // which call ModalManager.confirmDelete.
             if (typeof ModalManager !== 'undefined' && ModalManager.confirmDelete) {
                ModalManager.confirmDelete(mediaId, mediaType, galleryItem);
             } else {
                // Fallback to simple confirm if ModalManager isn't fully set up or for a different context
            const confirmed = confirm(`Are you sure you want to delete this ${mediaType}? This cannot be undone.`);
            if (confirmed) {
                try {
                        const response = await fetch(API.IMAGE(mediaId), { method: 'DELETE' });
                    if (response.ok) {
                            galleryItem.remove();
                            if(typeof showMainFeedback === 'function') showMainFeedback('Deleted successfully.', 'success');
                    } else {
                             if(typeof showMainFeedback === 'function') showMainFeedback(`Failed to delete ${mediaType}.`, 'error');
                        }
                    } catch (err) {
                        if(typeof showMainFeedback === 'function') showMainFeedback(`Error deleting ${mediaType}.`, 'error');
                    }
                }
             }
        }
        else if (e.target.closest('.action-generate-video') && mediaType === 'image') {
             if (!mediaId) return;
            const button = e.target.closest('.action-generate-video');
             const sourceImageUrl = galleryItem.querySelector('img')?.src;
            const sourcePrompt = button.dataset.imagePrompt;
            if (sourceImageUrl && sourcePrompt !== undefined && typeof ModalManager !== 'undefined' && ModalManager.openVideoGenerator) {
                ModalManager.openVideoGenerator(mediaId, sourceImageUrl, sourcePrompt);
             } else {
                console.error("Missing data or ModalManager.openVideoGenerator for video generation.");
                if(typeof showMainFeedback === 'function') showMainFeedback("Cannot generate video. Info missing.", "error");
             }
        }
    });
}

async function pollGenerationStatus(jobId, feedbackSection, numOutputs, timer, feedbackType) {
    const progressBar = feedbackSection?.querySelector('.progress-fill');
    const statusTextEl = feedbackSection?.querySelector('.status-text');
    const queuePositionEl = feedbackSection?.querySelector('.queue-position');
    const estimatedTimeEl = feedbackSection?.querySelector('.estimated-time');
    const POLLING_INTERVAL = 2000;
    let pollCount = 0;
    const MAX_POLLS_WITHOUT_PROGRESS = 30;
    let pollsWithoutProgress = 0;

    return new Promise((resolve, reject) => {
        const intervalId = setInterval(async () => {
            pollCount++;
            try {
                const response = await fetch(API.STATUS(jobId));
                if (!response.ok) {
                    if (response.status === 404 && pollCount > 5) throw new Error(`${feedbackType} job ${jobId} not found.`);
                    console.warn(`Error fetching status for ${jobId} (Attempt ${pollCount}): ${response.status}`);
                    pollsWithoutProgress++;
                    if (pollsWithoutProgress > MAX_POLLS_WITHOUT_PROGRESS) throw new Error(`No progress for ${jobId} after ${MAX_POLLS_WITHOUT_PROGRESS} attempts.`);
        return;
    }
                const data = await response.json();
                pollsWithoutProgress = 0;
                if (statusTextEl) statusTextEl.textContent = data.status_message || data.status || 'Processing...';
                if (progressBar) progressBar.style.width = `${data.progress || 0}%`;
                if (queuePositionEl) queuePositionEl.textContent = data.queue_position ? `Queue Pos: ${data.queue_position}` : '';
                if (estimatedTimeEl) estimatedTimeEl.textContent = data.estimated_processing_time_seconds ? `Est: ${Math.round(data.estimated_processing_time_seconds)}s` : '';

                if (data.status === 'completed') {
                    clearInterval(intervalId); timer.stop();
                    if (statusTextEl) statusTextEl.textContent = `${feedbackType} generation complete!`;
                    if (progressBar) progressBar.style.width = '100%';
                    resolve(data);
                } else if (data.status === 'failed') {
                    clearInterval(intervalId); timer.stop();
                    const errorMessage = data.error_message || `${feedbackType} generation failed.`;
                    if (statusTextEl) statusTextEl.textContent = errorMessage;
                    reject(new Error(errorMessage));
                } else if (!['processing', 'pending', 'queued'].includes(data.status)){
                    console.warn(`Unknown status for job ${jobId}: ${data.status}`);
                }
            } catch (error) {
                console.error(`Polling error for job ${jobId}:`, error);
                pollsWithoutProgress++;
                if (pollsWithoutProgress > MAX_POLLS_WITHOUT_PROGRESS || pollCount > MAX_POLLS_WITHOUT_PROGRESS * 2) {
                    clearInterval(intervalId); timer.stop();
                    if (statusTextEl) statusTextEl.textContent = `Error checking status.`;
                    reject(error);
                }
            }
        }, POLLING_INTERVAL);
    });
}

function showIndexPendingReloadNotification(message) {
    if (indexReloadNotificationTimer) clearTimeout(indexReloadNotificationTimer);
    if(typeof showMainFeedback === 'function') showMainFeedback(message, 'info', 15000);
}

async function checkQueueAndReloadIfIndexPending() {
    const mainForm = document.getElementById('generation-form');
    const promptInput = mainForm ? mainForm.querySelector('#prompt-input') : null;
    const videoGenModal = document.getElementById('videoGenModal');
    const isPromptActive = promptInput && (document.activeElement === promptInput || promptInput.value.trim() !== '');
    const isModalActive = videoGenModal && videoGenModal.style.display === 'block';
    console.log(`PollIndex: checkQueueAndReloadIfIndexPending - isPromptActive: ${isPromptActive}, isModalActive: ${isModalActive}, pendingIndexPageReload: ${pendingIndexPageReload}`);

    if (!isPromptActive && !isModalActive && pendingIndexPageReload) {
        console.log('PollIndex: checkQueueAndReloadIfIndexPending - Conditions met, refreshing!');
        await refreshRecentGenerationsSection();
        pendingIndexPageReload = false;
        if (indexReloadNotificationTimer) clearTimeout(indexReloadNotificationTimer);
    } else if (pendingIndexPageReload) {
        console.log("PollIndex: checkQueueAndReloadIfIndexPending - Reload pending, but inputs still active or other condition not met.");
    }
}

async function pollSubmittedIndexPageJobs() {
    if (submittedJobIdsFromIndexPage.length === 0) {
        // console.log('PollIndex: No jobs to poll, clearing interval.');
        clearInterval(indexPageJobPollingInterval); indexPageJobPollingInterval = null; return;
    }
    let newCompletions = false;
    // console.log('PollIndex: Polling for jobs:', submittedJobIdsFromIndexPage);
    const jobIdsToPoll = [...submittedJobIdsFromIndexPage]; // Create a copy to iterate over

    for (const jobId of jobIdsToPoll) {
        try {
            const response = await fetch(API.STATUS(jobId));
            if (!response.ok) {
                if (response.status === 404) {
                    // console.log(`PollIndex: Job ${jobId} not found (404), removing from tracking.`);
                    submittedJobIdsFromIndexPage = submittedJobIdsFromIndexPage.filter(id => id !== jobId);
                }
                continue;
            }
            const data = await response.json();
            // console.log(`PollIndex: Status for job ${jobId}:`, data.status);
            if (data.status === 'completed' || data.status === 'failed') {
                // console.log(`PollIndex: Job ${jobId} finished with status: ${data.status}. Removing from tracking.`);
                submittedJobIdsFromIndexPage = submittedJobIdsFromIndexPage.filter(id => id !== jobId);
                newCompletions = true;
            }
        } catch (error) { console.error('PollIndex: Error polling job status for recent generations:', error); }
    }

    if (newCompletions) {
        console.log('PollIndex: New completions detected.');
        const mainForm = document.getElementById('generation-form');
        const promptInput = mainForm ? mainForm.querySelector('#prompt-input') : null;
        const videoGenModal = document.getElementById('videoGenModal');
        
        const isPromptActive = promptInput && (document.activeElement === promptInput || promptInput.value.trim() !== '');
        const isModalActive = videoGenModal && videoGenModal.style.display === 'block';
        console.log(`PollIndex: Input states - isPromptActive: ${isPromptActive} (ActiveElem: ${document.activeElement === promptInput}, PromptText: '${promptInput ? promptInput.value.trim() : 'N/A'}'), isModalActive: ${isModalActive}`);

        if (isPromptActive || isModalActive) {
            if (!pendingIndexPageReload) {
                console.log('PollIndex: Inputs active, setting pendingIndexPageReload = true.');
                pendingIndexPageReload = true;
                showIndexPendingReloadNotification("New content ready! Finish typing or close modals to see updates.");
            }
            } else {
            console.log('PollIndex: Inputs NOT active. Calling refreshRecentGenerationsSection().');
            await refreshRecentGenerationsSection();
            pendingIndexPageReload = false;
            if (indexReloadNotificationTimer) clearTimeout(indexReloadNotificationTimer);
        }
    }

    if (submittedJobIdsFromIndexPage.length === 0) {
        // console.log('PollIndex: All tracked jobs are done. Clearing interval.');
        clearInterval(indexPageJobPollingInterval); indexPageJobPollingInterval = null;
        if (pendingIndexPageReload) {
            console.log('PollIndex: All jobs done, and a reload was pending. Checking inputs again.');
            checkQueueAndReloadIfIndexPending();
        }
    }
}

async function refreshRecentGenerationsSection() {
    console.log('refreshRecentGenerationsSection: Called.'); // Log entry
    const galleryGrid = document.querySelector('.recent-images .gallery-grid');
    const loadingIndicator = document.querySelector('.recent-images .loading-indicator');
    if (!galleryGrid) {
        console.error("refreshRecentGenerationsSection: Could not find '.recent-images .gallery-grid'.");
        return;
    }
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    try {
        const itemsToFetch = parseInt(galleryGrid.dataset.itemsToShow) || 10;
        console.log(`refreshRecentGenerationsSection: Fetching ${itemsToFetch} items.`); // Log itemsToFetch
        const response = await fetch(`${API.BASE}/gallery?page=1&per_page=${itemsToFetch}`);
        console.log(`refreshRecentGenerationsSection: Fetch response OK: ${response.ok}, Status: ${response.status}`); // Log response status
        if (!response.ok) throw new Error(`Failed to fetch recent items: ${response.status}`);
        const data = await response.json();
        // Log only a summary of data to avoid overly verbose logs in case of many items.
        console.log('refreshRecentGenerationsSection: Data received. Items length:', data.images ? data.images.length : 'undefined', 'Full data keys:', Object.keys(data));

        if (data.images && data.images.length > 0) {
            console.log(`refreshRecentGenerationsSection: Found ${data.images.length} items. Calling appendItemsToRecentGenerations.`);
            appendItemsToRecentGenerations(data.images);
            } else {
            console.log('refreshRecentGenerationsSection: No items found or data.images is empty/undefined. Displaying empty message.'); // Log empty case, reflect change in log
            galleryGrid.innerHTML = '<p class="empty-gallery-message">No recent generations yet.</p>';
            }
        } catch (error) {
        console.error('Error refreshing recent generations:', error);
        galleryGrid.innerHTML = '<p class="error-message">Could not load recent items.</p>';
        } finally {
        if (loadingIndicator) loadingIndicator.style.display = 'none';
    }
}

function appendItemsToRecentGenerations(items) {
    const galleryGrid = document.querySelector('.recent-images .gallery-grid');
    if (!galleryGrid) {
        console.error("appendItemsToRecentGenerations: Could not find '.recent-images .gallery-grid'.");
        return;
    }
    galleryGrid.innerHTML = ''; // Clear previous items

    // Create a document fragment to batch DOM manipulations
    const fragment = document.createDocumentFragment();

    items.forEach(item => {
        // console.log('appendItemsToRecentGenerations - Processing item:', item); // Kept for debugging if needed
        const itemElement = document.createElement('div');
        itemElement.classList.add('gallery-item');
        itemElement.dataset.mediaId = item.id;

        let determinedMediaType = 'image'; // Default to image

        // Prioritize .mp4 extension check
        if (item.file_path && typeof item.file_path === 'string' && item.file_path.toLowerCase().endsWith('.mp4')) {
            determinedMediaType = 'video';
            // console.log(`appendItemsToRecentGenerations: Media type set to video for ${item.id} based on .mp4 file_path.`);
        } else if (item.metadata && item.metadata.type === 'video') {
            determinedMediaType = 'video';
        } else if (item.type === 'video') {
            determinedMediaType = 'video';
        }
        // If none of the above, it remains 'image' by default
        
        itemElement.dataset.mediaType = determinedMediaType;

        itemElement.dataset.fullPrompt = item.full_prompt || item.prompt || '';
        itemElement.dataset.modelId = item.model_id || '';
        if (item.settings) itemElement.dataset.settings = JSON.stringify(item.settings);

        const itemThumbnailUrl = item.thumbnail_url || (determinedMediaType === 'video' ? `/api/get_video/${item.id}/thumbnail` : `/api/get_image/${item.id}?size=small`);
        const itemPreviewUrl = item.preview_url || (determinedMediaType === 'video' ? `/api/get_video/${item.id}` : `/api/get_image/${item.id}?size=medium`);
        
        let previewHtml;
        if (determinedMediaType === 'video') {
            previewHtml = `
                <video preload="metadata" muted loop poster="${itemThumbnailUrl}">
                    <source src="${itemPreviewUrl}#t=0.1" type="video/mp4">
                </video>
                <div class="video-indicator"><span class="icon">▶️</span></div>`;
    } else {
            previewHtml = `<img src="${itemThumbnailUrl}" alt="${(item.prompt || 'Generated image').substring(0,50)}" loading="lazy">`;
        }

        const canGenerateVideo = availableModels && Object.values(availableModels).some(m => m.type === 'i2v');
        let quickActionsHtml = `
            <div class="quick-actions">
                <button class="quick-action-btn action-view" title="View Fullscreen"><span class="icon">👁️</span></button>`;
        if (determinedMediaType === 'image' && canGenerateVideo) {
             quickActionsHtml += `<button class="quick-action-btn action-generate-video-from-recent" title="Generate Video"><span class="icon">🎬</span></button>`;
        }
        quickActionsHtml += `<button class="quick-action-btn action-delete-from-recent" title="Delete"><span class="icon">🗑️</span></button></div>`;

        itemElement.innerHTML = `
            <div class="item-preview">${previewHtml}</div>
            <div class="item-info"><p class="item-prompt" title="${item.full_prompt || item.prompt || ''}">${(item.prompt || 'No prompt').substring(0,100)}</p></div>
            ${quickActionsHtml}`;
        fragment.appendChild(itemElement); // Append to fragment
    });

    galleryGrid.appendChild(fragment); // Append fragment to the DOM once

    initializeVideoHoverPlayForGrid('.recent-images .gallery-grid');
    initializeRecentGenerationsActionListeners();
}

function initializeRecentGenerationsActionListeners() {
    const recentGalleryGrid = document.querySelector('.recent-images .gallery-grid');
    if (!recentGalleryGrid) {
        console.error("initializeRecentGenerationsActionListeners: Could not find '.recent-images .gallery-grid'.");
        return;
    }
    recentGalleryGrid.addEventListener('click', function(event) {
        const target = event.target;
        const galleryItem = target.closest('.gallery-item');
        if (!galleryItem) return;
        const mediaId = galleryItem.dataset.mediaId;
        const mediaType = galleryItem.dataset.mediaType;

        if (target.closest('.action-view')) {
            ModalManager.showFullscreen(galleryItem);
        }
        else if (target.closest('.action-generate-video-from-recent')) {
            if (mediaType === 'image') {
                const imageUrl = API.IMAGE(mediaId); // Use API const
                const prompt = galleryItem.dataset.fullPrompt || '';
                ModalManager.openVideoGenerator(mediaId, imageUrl, prompt);
            }
        }
        else if (target.closest('.action-delete-from-recent')) {
            ModalManager.confirmDelete(mediaId, mediaType, galleryItem);
        }
        // Click on preview itself (not buttons) to open fullscreen
        else if (target.closest('.item-preview') && !target.closest('.quick-actions')) {
             ModalManager.showFullscreen(galleryItem);
        }
    });
}

function initializeVideoHoverPlayForGrid(gridSelector) {
    const galleryGrid = document.querySelector(gridSelector);
    if (!galleryGrid) return;
    galleryGrid.addEventListener('mouseover', function(event) {
        const galleryItem = event.target.closest('.gallery-item');
        if (galleryItem && galleryItem.dataset.mediaType === 'video') {
            const video = galleryItem.querySelector('video');
            if (video && video.paused) video.play().catch(e => {});
        }
    });
    galleryGrid.addEventListener('mouseout', function(event) {
        const galleryItem = event.target.closest('.gallery-item');
        if (galleryItem && galleryItem.dataset.mediaType === 'video') {
            const video = galleryItem.querySelector('video');
            if (video && !video.paused) video.pause();
        }
    });
}

// --- Removed confirmDeleteMedia (moved to ModalManager) ---
// --- Removed initializeVideoGenerationModal, openVideoGenModal, closeVideoGenModal (moved to ModalManager) ---
// --- Removed initializeModalHandling, initializeGalleryHandling, showFullscreenMedia, hideModal (moved/integrated into ModalManager) ---
