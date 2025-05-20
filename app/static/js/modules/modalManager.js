import { showMainFeedback } from './uiUtils.js';
// AppModule: modalManager.js

// Ensure API, availableModels are accessible (e.g., global or imported)
// For now, assumes they are global as per current main.js structure.
// API_I2V_GEN is also assumed global from main.js

let _callbacks = {}; // Store callbacks from main.js or other modules
let _API = null;
let _availableModels = null;
let _API_I2V_GEN = null; // Will also need to be passed or derived

// --- Start Fullscreen Modal ---
let globalCurrentMediaId = null;
let globalCurrentMediaType = 'image';

function _hideFullscreenModal() {
    const modal = document.getElementById('fullscreenModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
        const modalVideo = document.getElementById('modalVideo');
        if (modalVideo && !modalVideo.paused) {
            modalVideo.pause();
            modalVideo.currentTime = 0;
        }
    }
    globalCurrentMediaId = null;
    globalCurrentMediaType = 'image'; // Reset global type
}

function _showFullscreenMedia(element) {
    const modal = document.getElementById('fullscreenModal');
    const modalImage = document.getElementById('modalImage');
    const modalVideo = document.getElementById('modalVideo');
    const modelInfo = document.getElementById('modelInfo');
    const promptInfo = document.getElementById('promptInfo');
    const settingsInfo = document.getElementById('settingsInfo');
    const downloadBtn = document.getElementById('downloadImage'); // For fullscreen modal

    if (!modal || !modalImage || !modalVideo || !modelInfo || !promptInfo || !settingsInfo || !downloadBtn) {
        console.error("Fullscreen modal elements not found for _showFullscreenMedia.");
        return;
    }

    const galleryItem = element.closest('.gallery-item');
    let mediaIdToLoad, mediaTypeToLoad, itemFullPrompt, itemModelId;

    if (galleryItem) {
        mediaIdToLoad = galleryItem.dataset.mediaId;
        mediaTypeToLoad = galleryItem.dataset.mediaType || 'image';
        itemFullPrompt = galleryItem.dataset.fullPrompt; // Ensure these are consistently set on gallery items
        itemModelId = galleryItem.dataset.modelId;       // Ensure these are consistently set on gallery items
    } else if (element.classList && element.classList.contains('gallery-item')) { // If element itself is the gallery item
        mediaIdToLoad = element.dataset.mediaId;
        mediaTypeToLoad = element.dataset.mediaType || 'image';
        itemFullPrompt = element.dataset.fullPrompt;
        itemModelId = element.dataset.modelId;
    }
     else { // Fallback for elements not directly .gallery-item or child of it (e.g., direct click on image in some contexts)
        mediaIdToLoad = element.dataset.mediaId;
        mediaTypeToLoad = element.dataset.mediaType || 'image';
        if (!mediaIdToLoad) {
            console.error("No media ID found for fullscreen view. Element:", element);
            return;
        }
    }

    globalCurrentMediaId = mediaIdToLoad;
    globalCurrentMediaType = mediaTypeToLoad;

    if (!globalCurrentMediaId) {
        console.error("No media ID could be determined for fullscreen view from element:", element);
        return;
    }

    modalImage.style.display = 'none';
    modalImage.src = '';
    modalVideo.style.display = 'none';
    modalVideo.src = '';
    if (modalVideo && !modalVideo.paused) modalVideo.pause();

    const downloadButtonText = globalCurrentMediaType === 'video' ? 'Download Video' : 'Download Image';
    downloadBtn.innerHTML = `<span class="icon">⬇️</span> ${downloadButtonText}`;
    downloadBtn.dataset.mediaId = globalCurrentMediaId;
    downloadBtn.dataset.mediaType = globalCurrentMediaType;

    const deleteBtnFullscreen = modal.querySelector('#deleteImage'); 
    if (deleteBtnFullscreen) {
        deleteBtnFullscreen.style.display = 'inline-block'; 
        deleteBtnFullscreen.onclick = () => {
            _confirmDeleteMedia(globalCurrentMediaId, globalCurrentMediaType, null, modal);
        };
    }

    if (globalCurrentMediaType === 'video') {
        if (_API && typeof _API.VIDEO === 'function') {
            modalVideo.src = _API.VIDEO(globalCurrentMediaId);
        } else {
            // Fallback or error if _API.VIDEO is not available - though it should be after main.js update
            console.warn('_API.VIDEO function not available, attempting fallback URL construction for video.');
            modalVideo.src = (_API && _API.IMAGE ? _API.IMAGE(globalCurrentMediaId) : `/api/get_image/${globalCurrentMediaId}`).replace('/get_image/', '/get_video/');
        }
        modalVideo.style.display = 'block';
        modalVideo.load();
    } else {
        if (_API && typeof _API.IMAGE === 'function') {
            modalImage.src = _API.IMAGE(globalCurrentMediaId);
        } else {
            console.warn('_API.IMAGE function not available, attempting fallback URL construction for image.');
            modalImage.src = `/api/get_image/${globalCurrentMediaId}`;
        }
        modalImage.style.display = 'block';
    }

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    modelInfo.textContent = 'Loading...';
    promptInfo.textContent = 'Loading...';
    settingsInfo.textContent = 'Loading...';

    fetch(_API.METADATA(globalCurrentMediaId))
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error ${response.status} fetching metadata for ${globalCurrentMediaId}`);
            return response.json();
        })
        .then(data => {
            const displayPrompt = itemFullPrompt || data.prompt || data.settings?.prompt || 'Not available';
            promptInfo.textContent = displayPrompt;
            modelInfo.textContent = data.model_id || itemModelId || 'Not available';

            const settings = Object.entries(data.settings || {})
                .filter(([key]) => !['prompt', 'negative_prompt', 'source_image_id', 'type', 'model_id', 'num_images'].includes(key))
                .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
                .join('\n');
            settingsInfo.textContent = settings || 'Not available';
        })
        .catch(error => {
            console.error('Error fetching media metadata for fullscreen view:', error);
            modelInfo.textContent = 'Metadata unavailable';
            promptInfo.textContent = 'Metadata unavailable';
            settingsInfo.textContent = 'Metadata unavailable';
        });
}

function _initializeFullscreenModalActions() {
    const modal = document.getElementById('fullscreenModal');
    if (!modal) return;

    const copyPromptBtn = modal.querySelector('#copyPrompt');
    const downloadBtn = modal.querySelector('#downloadImage'); 
    const closeBtn = modal.querySelector('.action-close');    

    if (copyPromptBtn) {
        copyPromptBtn.addEventListener('click', async () => {
            const promptInfoEl = modal.querySelector('#promptInfo');
            const promptText = promptInfoEl ? promptInfoEl.textContent : '';
            let copied = false;

            if (promptText && promptText !== 'Loading...' && promptText !== 'Metadata unavailable') {
                // Try modern clipboard API first
                if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                    try {
                        await navigator.clipboard.writeText(promptText);
                        copied = true;
                    } catch (err) {
                        console.warn('navigator.clipboard.writeText failed:', err);
                        // Proceed to fallback if this fails
                    }
                }

                // Fallback to document.execCommand('copy') if modern API failed or not available
                if (!copied) {
                    const textArea = document.createElement("textarea");
                    textArea.value = promptText;
                    textArea.style.position = "fixed"; // Prevent scrolling to bottom of page in MS Edge.
                    textArea.style.top = "0";
                    textArea.style.left = "0";
                    textArea.style.width = "2em";
                    textArea.style.height = "2em";
                    textArea.style.padding = "0";
                    textArea.style.border = "none";
                    textArea.style.outline = "none";
                    textArea.style.boxShadow = "none";
                    textArea.style.background = "transparent";
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    try {
                        document.execCommand('copy');
                        copied = true;
                    } catch (err) {
                        console.warn('document.execCommand(\'copy\') failed:', err);
                        // Proceed to further fallback if this also fails
                    } finally {
                        document.body.removeChild(textArea);
                    }
                }

                const originalHtml = copyPromptBtn.innerHTML;
                if (copied) {
                    copyPromptBtn.innerHTML = '<span class="icon">✓</span>Copied!';
                    setTimeout(() => { copyPromptBtn.innerHTML = originalHtml; }, 2000);
                    if(typeof showMainFeedback === 'function') showMainFeedback('Prompt copied to clipboard!', 'success', 2000);
                } else {
                    // Last resort: select text for manual copy
                    console.error('All copy methods failed. Fallback to text selection.');
                    if(typeof showMainFeedback === 'function') showMainFeedback('Auto-copy failed. Prompt selected for manual copy.', 'error');
                    const promptPre = modal.querySelector('#promptInfo');
                    if (promptPre) {
                        try {
                            const range = document.createRange();
                            range.selectNodeContents(promptPre);
                            const selection = window.getSelection();
                            if (selection) {
                                selection.removeAllRanges();
                                selection.addRange(range);
                            }
                        } catch (selectErr) {
                            console.error('Fallback prompt selection failed:', selectErr);
                            if(typeof showMainFeedback === 'function') showMainFeedback('Could not select prompt text.', 'error');
                        }
                    }
                    copyPromptBtn.innerHTML = '<span class="icon">❌</span>Copy Failed'; // Indicate failure on button
                    setTimeout(() => { copyPromptBtn.innerHTML = originalHtml; }, 3000);
                }
            }
        });
    }

    if (downloadBtn) {
        downloadBtn.addEventListener('click', async () => {
            const mediaId = downloadBtn.dataset.mediaId;
            const mediaType = downloadBtn.dataset.mediaType;

            if (!mediaId || !mediaType) {
                console.error("Fullscreen Modal: Missing media ID or type for download.");
                if (typeof showMainFeedback === 'function') showMainFeedback("Error: Cannot download, media details missing.", "error");
                return;
            }

            const downloadUrl = mediaType === 'video' ? _API.IMAGE(mediaId).replace('/get_image/', '/get_video/') : _API.IMAGE(mediaId);
            const filename = `cyberimage-${mediaId}.${mediaType === 'video' ? 'mp4' : 'png'}`;
            const originalButtonHtml = downloadBtn.innerHTML;

            try {
                downloadBtn.innerHTML = '<span class="icon">⏳</span> Downloading...';
                downloadBtn.disabled = true;

                const response = await fetch(downloadUrl);
                if (!response.ok) throw new Error(`Failed to fetch media: ${response.statusText}`);
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
                    downloadBtn.innerHTML = originalButtonHtml;
                    downloadBtn.disabled = false;
                }, 1000);
            } catch (error) {
                console.error('Fullscreen Modal download failed:', error);
                if (typeof showMainFeedback === 'function') showMainFeedback("Download failed. See console for details.", "error");
                downloadBtn.innerHTML = '<span class="icon">❌</span> Failed';
                setTimeout(() => {
                    downloadBtn.innerHTML = originalButtonHtml;
                    downloadBtn.disabled = false;
                }, 3000);
            }
        });
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', _hideFullscreenModal);
    }
}
// --- End Fullscreen Modal ---


// --- Start Video Generation Modal ---
let _videoGenModalElement = null;
let _videoModelSelectElement = null;
let _videoGenSourceImageElement = null;
let _videoGenSourcePromptElement = null;
let _videoGenSourceImageIdInputElement = null;
let _videoGenPromptInputElement = null;
let _videoGenFormElement = null;

function _closeVideoGenModal() {
    if (_videoGenModalElement) {
        _videoGenModalElement.style.display = 'none';
    }
}

function _openVideoGenModal(sourceImageId, sourceImageUrl, sourcePrompt) {
    if (!_videoGenModalElement || !_videoModelSelectElement || !_videoGenSourceImageElement || !_videoGenSourcePromptElement || !_videoGenSourceImageIdInputElement || !_videoGenPromptInputElement) {
        console.error("Video Generation Modal elements not fully initialized for _openVideoGenModal.");
        return;
    }

    _videoGenSourceImageIdInputElement.value = sourceImageId;
    delete _videoGenModalElement.dataset.sourceWidth;
    delete _videoGenModalElement.dataset.sourceHeight;

    _videoGenSourceImageElement.onload = function() {
        _videoGenModalElement.dataset.sourceWidth = this.naturalWidth;
        _videoGenModalElement.dataset.sourceHeight = this.naturalHeight;
    };
    _videoGenSourceImageElement.onerror = function() {
        console.error("Video Gen Modal: Source image failed to load for _openVideoGenModal.");
        delete _videoGenModalElement.dataset.sourceWidth;
        delete _videoGenModalElement.dataset.sourceHeight;
        _videoGenSourceImageElement.alt = "Failed to load image preview";
    };
    _videoGenSourceImageElement.src = sourceImageUrl || '';
    _videoGenSourceImageElement.alt = "Source image preview";

    if (_videoGenSourceImageElement.complete && _videoGenSourceImageElement.naturalWidth && _videoGenSourceImageElement.naturalWidth > 0) {
        if (!_videoGenModalElement.dataset.sourceWidth) {
            _videoGenModalElement.dataset.sourceWidth = _videoGenSourceImageElement.naturalWidth;
            _videoGenModalElement.dataset.sourceHeight = _videoGenSourceImageElement.naturalHeight;
        }
    }

    _videoGenSourcePromptElement.textContent = sourcePrompt;
    _videoGenPromptInputElement.value = '';

    _videoModelSelectElement.innerHTML = `<option value="">Select I2V Model</option>`;
    let modelAdded = false;

    // --- Enhanced Logging Start ---
    if (typeof _availableModels !== 'undefined' && _availableModels && Object.keys(_availableModels).length > 0) {
        console.log("ModalManager: _availableModels in _openVideoGenModal:", JSON.parse(JSON.stringify(_availableModels)));
        Object.entries(_availableModels).forEach(([id, info]) => {
            const isI2VByType = info.type === 'i2v';
            const isI2VByName = id.toLowerCase().includes('i2v');
            console.log(`ModalManager: Checking model ${id}: type='${info.type}', isI2VByType=${isI2VByType}, isI2VByName=${isI2VByName}`);
            if (isI2VByType || isI2VByName) {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = `${id} - ${info.description || 'Image to Video Model'}`;
                _videoModelSelectElement.appendChild(option);
                modelAdded = true;
                console.log(`ModalManager: Added I2V model ${id} to select list.`);
            }
        });
    } else {
        console.error("ModalManager: `_availableModels` is undefined, empty, or not an object in _openVideoGenModal. Content:", _availableModels);
    }
    // --- Enhanced Logging End ---

    const lastVideoModelId = localStorage.getItem('lastVideoModelId');
    if (lastVideoModelId && _videoModelSelectElement.querySelector(`option[value="${lastVideoModelId}"]`)) {
        _videoModelSelectElement.value = lastVideoModelId;
    } else if (modelAdded) {
        _videoModelSelectElement.selectedIndex = 1;
    } else {
        _videoModelSelectElement.innerHTML = `<option value="">No I2V models available</option>`;
    }

    _videoGenModalElement.style.display = 'block';
    _videoGenPromptInputElement.focus();
}

function _initializeVideoGenerationModal() {
    _videoGenModalElement = document.getElementById('videoGenModal');
    _videoGenFormElement = document.getElementById('video-generate-form');
    _videoModelSelectElement = document.getElementById('videoGenModelSelect');
    _videoGenSourceImageElement = document.getElementById('videoGenSourceImage');
    _videoGenSourcePromptElement = document.getElementById('videoGenSourcePrompt');
    _videoGenSourceImageIdInputElement = document.getElementById('videoGenSourceImageId');
    _videoGenPromptInputElement = document.getElementById('videoGenPromptInput');

    if (!_videoGenModalElement || !_videoGenFormElement || !_videoModelSelectElement || !_videoGenSourceImageElement || !_videoGenSourcePromptElement || !_videoGenSourceImageIdInputElement || !_videoGenPromptInputElement) {
        console.warn('Video Generation Modal elements not found during _initializeVideoGenerationModal. Video generation features might be limited.');
        return;
    }

    if(_videoModelSelectElement) {
        _videoModelSelectElement.addEventListener('change', () => {
            localStorage.setItem('lastVideoModelId', _videoModelSelectElement.value);
        });
    }

    // Add event listeners for closing the video generation modal
    const closeButtonHeader = _videoGenModalElement?.querySelector('.modal-header .action-close');
    if (closeButtonHeader) {
        closeButtonHeader.addEventListener('click', _closeVideoGenModal);
    }

    const cancelButton = _videoGenModalElement?.querySelector('.modal-actions .button-cancel');
    if (cancelButton) {
        cancelButton.addEventListener('click', _closeVideoGenModal);
    }

    if(_videoGenFormElement){
        _videoGenFormElement.addEventListener('submit', async (event) => {
            event.preventDefault();
            const submitButton = _videoGenFormElement.querySelector('button[type="submit"]');
            if(!submitButton) return;

            submitButton.disabled = true;
            submitButton.innerHTML = '<span class="button-icon spin">⏳</span> Submitting...';

            const sourceId = _videoGenSourceImageIdInputElement.value;
            const videoPrompt = _videoGenPromptInputElement.value.trim();
            const videoModelId = _videoModelSelectElement.value;

            if (!videoPrompt || !videoModelId) {
                alert('Please enter a video prompt and select a model.');
                submitButton.disabled = false;
                submitButton.innerHTML = '<span class="button-icon">▶️</span> Generate Video';
                return;
            }

            try {
                if (typeof showMainFeedback !== 'function') {
                    console.error("showMainFeedback function is not available for I2V submission.");
                    alert("Feedback system error.");
                } else {
                    showMainFeedback('Submitting I2V request...', 'info');
                }

                const selectedModel = (typeof _availableModels !== 'undefined' && _availableModels) ? _availableModels[videoModelId] : null;
                if (!selectedModel) throw new Error(`Model ${videoModelId} not found`);
                if (selectedModel.type !== 'i2v' && !videoModelId.toLowerCase().includes('i2v')) {
                    throw new Error(`Model ${videoModelId} is not an image-to-video model`);
                }

                const sourceWidth = _videoGenModalElement.dataset.sourceWidth;
                const sourceHeight = _videoGenModalElement.dataset.sourceHeight;
                let apiPayload = {
                    source_image_id: sourceId,
                    video_prompt: videoPrompt,
                    video_model_id: videoModelId,
                    guidance_scale: 3.5
                };
                if (sourceWidth && sourceHeight) {
                    apiPayload.width = parseInt(sourceWidth, 10);
                    apiPayload.height = parseInt(sourceHeight, 10);
                }

                const response = await fetch(_API_I2V_GEN, { 
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(apiPayload)
                });
                const result = await response.json();

                if (response.ok && result.job_id) {
                    if (typeof showMainFeedback === 'function') showMainFeedback(`I2V job ${result.job_id} submitted!`, 'success');
                    _closeVideoGenModal();
                    if (typeof submittedJobIdsFromIndexPage !== 'undefined' && Array.isArray(submittedJobIdsFromIndexPage)) {
                        submittedJobIdsFromIndexPage.push(result.job_id);
                         if (typeof indexPageJobPollingInterval === 'undefined' || indexPageJobPollingInterval === null) {
                            if (typeof pollSubmittedIndexPageJobs === 'function') {
                                console.log('modalManager.js (I2V): Polling for index page jobs would start here if fully managed.');
                            }
                        }
                    }
                } else {
                    throw new Error(result.message || 'Failed to submit I2V job');
                }
            } catch (error) {
                console.error('Error submitting I2V job:', error);
                 if (typeof showMainFeedback === 'function') showMainFeedback(`Error: ${error.message}`, 'error');
            } finally {
                submitButton.disabled = false;
                submitButton.innerHTML = '<span class="button-icon">▶️</span> Generate Video';
            }
        });
    }
}
// --- End Video Generation Modal ---


// --- Start Delete Confirmation Modal ---
async function _confirmDeleteMedia(mediaId, mediaType, galleryItemElement, callingModalElement = null) {
    const deleteModal = document.getElementById('deleteConfirmModal');
    const confirmBtn = document.getElementById('confirmDelete');
    const cancelBtn = document.getElementById('cancelDelete');
    const deleteMessage = document.getElementById('deleteMessage');

    if (!deleteModal || !confirmBtn || !cancelBtn || !deleteMessage) {
        console.error('Delete confirmation modal elements not found for _confirmDeleteMedia.');
        if (typeof showMainFeedback === 'function') showMainFeedback('Error: Could not initiate delete.', 'error');
        else alert('Error: Could not initiate delete. Feedback system unavailable.');
        return;
    }

    deleteMessage.textContent = `Are you sure you want to delete this ${mediaType} (ID: ${mediaId})? This action cannot be undone.`;
    deleteModal.style.display = 'block';

    const newConfirmBtn = confirmBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
    const newCancelBtn = cancelBtn.cloneNode(true);
    cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);

    newConfirmBtn.onclick = async () => {
        try {
            const correctDeleteUrl = `/api/image/${mediaId}`;
            const response = await fetch(correctDeleteUrl, { method: 'DELETE' });
            if (!response.ok) {
                let errorText = await response.text(); // Get response text for better error info
                let errorMessage = `Failed to delete ${mediaType}. Status: ${response.status}`;
                try {
                    const errorData = JSON.parse(errorText);
                    if (errorData && errorData.message) {
                        errorMessage = errorData.message;
                    }
                } catch (e) {
                    // Not a JSON response, use the raw text if it's not too long, or a generic message
                    if (errorText && errorText.length < 200) { // Avoid overly long error messages
                        errorMessage = errorText;
                    } else if (errorText) {
                        errorMessage = `Server error (${response.status}). Check console for full response.`;
                        console.error("Full error response from server during delete:", errorText);
                    }
                }
                throw new Error(errorMessage);
            }

            if (galleryItemElement) { 
                galleryItemElement.remove();
            } else if (callingModalElement && callingModalElement.id === 'fullscreenModal') {
                _hideFullscreenModal();
            }
            
            if (typeof showMainFeedback === 'function') {
                showMainFeedback(`${mediaType.charAt(0).toUpperCase() + mediaType.slice(1)} deleted successfully.`, 'success');
            }

            const recentGenerationsGrid = document.querySelector('#recent-generations .gallery-grid');
            if (recentGenerationsGrid && typeof _callbacks.refreshRecentGenerationsSection === 'function') { 
                if (galleryItemElement && galleryItemElement.closest('#recent-generations')) {
                    _callbacks.refreshRecentGenerationsSection();
                } else if (callingModalElement && callingModalElement.id === 'fullscreenModal') {
                    _callbacks.refreshRecentGenerationsSection();
                }
            } else if (window.location.pathname.includes('gallery.html')) {
                // window.location.reload(); 
            }
        } catch (error) {
            console.error('Error deleting media in _confirmDeleteMedia:', error);
            if (typeof showMainFeedback === 'function') showMainFeedback(error.message || `Error deleting ${mediaType}.`, 'error');
        } finally {
            deleteModal.style.display = 'none';
        }
    };

    newCancelBtn.onclick = () => {
        deleteModal.style.display = 'none';
    };
}
// --- End Delete Confirmation Modal ---


// --- General Modal Handling (click outside, escape key) ---
function _initializeGeneralModalBehavior() {
    const generalModal = document.querySelector('.modal'); 
    const fullscreenModal = document.getElementById('fullscreenModal');
    const videoGenModal = document.getElementById('videoGenModal');
    const deleteConfirmModal = document.getElementById('deleteConfirmModal');

    document.addEventListener('click', (e) => {
        if (generalModal && (e.target.classList.contains('close-modal') || e.target === generalModal)) {
            if(generalModal.classList.contains('visible')) generalModal.classList.remove('visible');
        }
        if (fullscreenModal && (e.target.classList.contains('action-close') || e.target === fullscreenModal)) {
             if (fullscreenModal.style.display === 'flex') _hideFullscreenModal();
        }
        if (deleteConfirmModal && (e.target.id === 'cancelDelete' || e.target === deleteConfirmModal || e.target.closest('.close-delete-modal-btn') )) { 
             if(deleteConfirmModal.style.display === 'block') deleteConfirmModal.style.display = 'none';
        }
        if (videoGenModal && e.target.closest('.action-close') && videoGenModal.contains(e.target.closest('.action-close'))) {
             if(videoGenModal.style.display === 'block') _closeVideoGenModal();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (fullscreenModal && fullscreenModal.style.display === 'flex') {
                _hideFullscreenModal();
            }
            if (deleteConfirmModal && deleteConfirmModal.style.display === 'block') {
                deleteConfirmModal.style.display = 'none';
            }
            if (videoGenModal && videoGenModal.style.display === 'block') {
                _closeVideoGenModal();
            }
             if (generalModal && generalModal.classList.contains('visible')) {
                generalModal.classList.remove('visible');
            }
        }
    });
}

// --- Public API for ModalManager ---
export const ModalManager = {
    initialize: (config = {}) => {
        _callbacks = config.callbacks || {}; 
        _API = config.API;
        _availableModels = config.availableModels;
        _API_I2V_GEN = config.API_I2V_GEN; // Add API_I2V_GEN to config

        if (!_API || !_availableModels || !_API_I2V_GEN) {
            console.error("ModalManager FATAL: API, availableModels, or API_I2V_GEN not provided during initialization!");
            // Optionally throw an error or disable functionality
            return;
        }

        _initializeGeneralModalBehavior();
        _initializeFullscreenModalActions(); 
        _initializeVideoGenerationModal();   
        console.log("ModalManager initialized with config:", config);
    },
    showFullscreen: _showFullscreenMedia,
    hideFullscreen: _hideFullscreenModal,
    openVideoGenerator: _openVideoGenModal,
    closeVideoGenerator: _closeVideoGenModal,
    confirmDelete: _confirmDeleteMedia 
};

// Expose necessary functions globally if needed by non-module scripts (legacy)
// This is generally not recommended for ES modules. Prefer importing ModalManager.
// window.showFullscreenMedia = _showFullscreenMedia;
// window.hideModal = _hideFullscreenModal; // Keep a generic hideModal if something relies on that name
// window.confirmDeleteMedia = _confirmDeleteMedia;
// window.openVideoGenModal = _openVideoGenModal;
// window.closeVideoGenModal = _closeVideoGenModal;

// The `initializeGalleryHandling` from main.js primarily dealt with:
// 1. Getting fullscreen modal elements (done in _initializeFullscreenModalActions)
// 2. Defining local hideModal/showFullscreenMedia (now part of this module)
// 3. Event delegation on galleryGrid to *open* the fullscreen modal. This part will
//    remain in main.js or a gallery-specific module and will call ModalManager.showFullscreen().
// 4. Copy/Download/Close actions for fullscreen modal (done in _initializeFullscreenModalActions)
