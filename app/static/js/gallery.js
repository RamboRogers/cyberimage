// API and utility functions - reusing those defined in main.js if already available
if (typeof window.API === 'undefined') {
    window.API = {
        BASE: '/api',
        MODELS: '/api/models',
        GENERATE: '/api/generate',
        STATUS: (jobId) => `/api/status/${jobId}`,
        IMAGE: (imageId) => `/api/get_image/${imageId}`,
        METADATA: (imageId) => `/api/image/${imageId}/metadata`,
        QUEUE: '/api/queue',
        GALLERY: (page = 1) => `/api/gallery?page=${page}&limit=20`,
        DELETE_IMAGE: (imageId) => `/api/image/${imageId}`,
        FAVORITE: (imageId) => `/api/favorite/${imageId}`,
        TAGS: '/api/tags',
        ADD_TAG: (imageId, tag) => `/api/image/${imageId}/tag/${encodeURIComponent(tag)}`,
        REMOVE_TAG: (imageId, tag) => `/api/image/${imageId}/tag/${encodeURIComponent(tag)}`,
        VIDEO_GEN: '/api/generate_video',
        GET_VIDEO: (videoId) => `/api/get_video/${videoId}`,
    };
}

// Add CSS for focus highlighting
(function() {
    const style = document.createElement('style');
    style.textContent = `
        .gallery-item.focused {
            outline: 3px solid #39ff14 !important; /* Neon green outline */
            box-shadow: 0 0 10px rgba(57, 255, 20, 0.7) !important;
            position: relative;
            z-index: 1;
            transform: translateZ(0); /* Force GPU acceleration for smoother animations */
        }

        .gallery-item.selected {
            outline: 2px solid #1e90ff !important; /* Blue outline for selected items */
        }

        .gallery-item.focused.selected {
            outline: 3px solid #f5f242 !important; /* Yellow outline for focused+selected */
            box-shadow: 0 0 12px rgba(245, 242, 66, 0.7) !important;
        }

        /* Better hover state for gallery items */
        .gallery-item:hover {
            transform: translateY(-2px);
            transition: transform 0.2s ease;
        }

        /* Visual feedback when using space to select */
        .gallery-item.selection-flash {
            animation: selection-flash 0.5s ease;
        }

        @keyframes selection-flash {
            0% { background-color: rgba(57, 255, 20, 0); }
            50% { background-color: rgba(57, 255, 20, 0.3); }
            100% { background-color: rgba(57, 255, 20, 0); }
        }

        /* Make the scroll sentinel more visible when debugging */
        .scroll-sentinel {
            background: linear-gradient(transparent, rgba(57, 255, 20, 0.05), transparent);
        }
    `;
    document.head.appendChild(style);
})();

// Utility functions
const Utilities = {
    /**
     * Format date for display
     * @param {Date} date - Date to format
     * @returns {string} Formatted date string
     */
    formatDate(date) {
        return new Intl.DateTimeFormat('default', {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: 'numeric',
            hour12: true
        }).format(date);
    },

    /**
     * Format date with more details
     * @param {Date} date - Date to format
     * @returns {string} Detailed formatted date string
     */
    formatDateLong(date) {
        return new Intl.DateTimeFormat('default', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: 'numeric',
            second: 'numeric',
            timeZoneName: 'short'
        }).format(date);
    },

    /**
     * Escape HTML special characters
     * @param {string} unsafe - String to escape
     * @returns {string} Escaped string
     */
    escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    },

    /**
     * Debounce function to limit execution frequency
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
     */
    debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    },

    /**
     * Throttle function to limit execution frequency
     * @param {Function} func - Function to throttle
     * @param {number} limit - Limit time in milliseconds
     * @returns {Function} Throttled function
     */
    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    /**
     * Copy text to clipboard with proper error handling
     * @param {string} text - Text to copy
     * @returns {Promise<boolean>} Success status
     */
    async copyToClipboard(text) {
        if (!text) return false;

        try {
            // Method 1: Clipboard API (modern browsers)
            if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                try {
                    await navigator.clipboard.writeText(text);
                    return true;
                } catch (clipboardErr) {
                    console.error('Clipboard API failed:', clipboardErr);
                    // Fall through to fallback
                }
            }

            // Method 2: execCommand fallback (older browsers)
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.left = '0';
            textarea.style.top = '0';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();

            try {
                const success = document.execCommand('copy');
                if (success) {
                    return true;
                }
            } catch (execErr) {
                console.error('execCommand fallback failed:', execErr);
            } finally {
                document.body.removeChild(textarea);
            }

            // Method 3: Manual copy (last resort)
            console.log('All clipboard methods failed, showing manual copy dialog');
            alert(`Please copy this text manually: ${text}`);
            return false;
        } catch (error) {
            console.error('Copy operation completely failed:', error);
            return false;
        }
    },

    /**
     * Create and show a feedback toast message
     * @param {string} message - Message to display
     * @param {string} type - Message type (success, error, info)
     * @param {number} duration - Duration in milliseconds
     */
    showFeedback(message, type = 'info', duration = 3000) {
        // Remove any existing feedback with the same ID
        const existingFeedback = document.getElementById('gallery-feedback');
        if (existingFeedback) {
            existingFeedback.remove();
        }

        const feedback = document.createElement('div');
        feedback.id = 'gallery-feedback';
        feedback.className = `gallery-feedback ${type}`;
        feedback.setAttribute('role', 'alert');
        feedback.textContent = message;

        document.body.appendChild(feedback);

        // Trigger animation
        requestAnimationFrame(() => {
            feedback.style.opacity = '1';
            feedback.style.transform = 'translateY(0)';
        });

        // Remove after animation
        setTimeout(() => {
            feedback.style.opacity = '0';
            feedback.style.transform = 'translateY(-20px)';
            setTimeout(() => feedback.remove(), 300);
        }, duration);
    }
};

/**
 * ImageLoader - Handles efficient image loading with advanced intersection observer
 */
class ImageLoader {
    constructor(options = {}) {
        this.options = {
            rootMargin: '200px',
            threshold: 0.1,
            loadingClass: 'loading',
            loadedClass: 'loaded',
            errorClass: 'error',
            ...options
        };

        this.observer = this.createObserver();
        this.observedElements = new Set();
        this.retryQueue = new Map(); // For tracking failed loads and retries
        this.maxRetries = 3;
    }

    /**
     * Create the intersection observer
     * @returns {IntersectionObserver} Configured observer
     */
    createObserver() {
        return new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.loadImage(entry.target, observer);
                }
            });
        }, {
            root: this.options.root || null,
            rootMargin: this.options.rootMargin,
            threshold: this.options.threshold
        });
    }

    /**
     * Load an image when it intersects the viewport
     * @param {HTMLImageElement} img - Image element to load
     * @param {IntersectionObserver} observer - Observer instance
     */
    loadImage(img, observer) {
        // Skip if already loaded or loading
        if (!img.dataset.src || img.src === img.dataset.src) {
            observer.unobserve(img);
            return;
        }

        // Track loading state
        img.classList.add(this.options.loadingClass);

        // Set up load and error handlers
        img.onload = () => {
            img.classList.remove(this.options.loadingClass);
            img.classList.add(this.options.loadedClass);
            observer.unobserve(img);
            this.observedElements.delete(img);
            this.retryQueue.delete(img);

            // Animate the image in
            img.style.opacity = '0';
            requestAnimationFrame(() => {
                img.style.transition = 'opacity 0.3s ease';
                img.style.opacity = '1';
            });
        };

        img.onerror = () => {
            // Track failed loads for retry
            const retries = this.retryQueue.get(img) || 0;

            if (retries < this.maxRetries) {
                // Exponential backoff for retries
                const delay = Math.pow(2, retries) * 1000;

                setTimeout(() => {
                    // Use a cache-busting parameter for CDN/cache issues
                    const cacheBuster = `?retry=${Date.now()}`;
                    img.src = `${img.dataset.src}${cacheBuster}`;
                    this.retryQueue.set(img, retries + 1);
                }, delay);
            } else {
                // Give up after max retries
                img.classList.remove(this.options.loadingClass);
                img.classList.add(this.options.errorClass);

                // Add error indicator
                this.showImageError(img);

                observer.unobserve(img);
                this.observedElements.delete(img);
                this.retryQueue.delete(img);
            }
        };

        // Actually set the src to begin loading
        img.src = img.dataset.src;
    }

    /**
     * Show error state for failed images
     * @param {HTMLImageElement} img - Failed image element
     */
    showImageError(img) {
        const container = img.closest('.item-preview');
        if (!container) return;

        // Check if error overlay already exists
        if (container.querySelector('.error-overlay')) return;

        const errorOverlay = document.createElement('div');
        errorOverlay.className = 'error-overlay';
        errorOverlay.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>Failed to load image</span>
            <button class="retry-button">Retry</button>
        `;

        container.appendChild(errorOverlay);

        // Add retry button functionality
        const retryButton = errorOverlay.querySelector('.retry-button');
        retryButton.addEventListener('click', (e) => {
            e.stopPropagation();
            errorOverlay.remove();
            img.classList.remove(this.options.errorClass);
            img.classList.add(this.options.loadingClass);

            // Reset retry count and try again
            this.retryQueue.delete(img);
            img.src = img.dataset.src + `?retry=${Date.now()}`;

            // Re-observe the image
            this.observe(img);
        });
    }

    /**
     * Observe an image element for lazy loading
     * @param {HTMLImageElement} img - Image element to observe
     */
    observe(img) {
        if (!this.observedElements.has(img)) {
            this.observer.observe(img);
            this.observedElements.add(img);
        }
    }

    /**
     * Observe multiple image elements
     * @param {NodeList|Array} images - Collection of image elements
     */
    observeAll(images) {
        images.forEach(img => this.observe(img));
    }

    /**
     * Disconnect observer and clean up
     */
    disconnect() {
        this.observer.disconnect();
        this.observedElements.clear();
        this.retryQueue.clear();
    }

    /**
     * Reset and reconnect observer
     */
    reset() {
        this.disconnect();
        this.observer = this.createObserver();
    }
}

/**
 * ViewManager - Handles gallery view modes and layout
 */
class ViewManager {
    constructor(galleryGrid) {
        this.galleryGrid = galleryGrid;
        this.availableViews = ['grid', 'list', 'compact'];
        this.currentView = localStorage.getItem('preferred-view') || 'grid';
        this.toggles = document.querySelectorAll('.view-toggle');

        this.initialize();
    }

    initialize() {
        // Set initial view
        this.setView(this.currentView);

        // Update active toggle button
        this.updateActiveToggle();

        // Set up event listeners
        this.toggles.forEach(toggle => {
            toggle.addEventListener('click', () => {
                const view = toggle.dataset.view;
                this.setView(view);
                this.savePreference(view);
            });
        });
    }

    /**
     * Set the current view mode
     * @param {string} view - View mode name
     */
    setView(view) {
        if (!this.availableViews.includes(view)) {
            view = 'grid'; // Default fallback
        }

        this.currentView = view;
        this.galleryGrid.dataset.view = view;
        this.updateActiveToggle();

        // Notify about view change
        window.dispatchEvent(new CustomEvent('gallery:viewChanged', {
            detail: { view: this.currentView }
        }));
    }

    /**
     * Update the active toggle button
     */
    updateActiveToggle() {
        this.toggles.forEach(toggle => {
            toggle.classList.toggle('active', toggle.dataset.view === this.currentView);
        });
    }

    /**
     * Save view preference to localStorage
     * @param {string} view - View to save
     */
    savePreference(view) {
        localStorage.setItem('preferred-view', view);
    }

    /**
     * Get the current view mode
     * @returns {string} Current view mode
     */
    getCurrentView() {
        return this.currentView;
    }
}

/**
 * SelectionManager - Handles image selection and batch operations
 */
class SelectionManager {
    constructor(galleryGrid) {
        this.galleryGrid = galleryGrid;
        this.selectedItems = new Set();
        this.batchOperationsContainer = document.querySelector('.batch-operations');
        this.selectedCountElement = document.querySelector('.selected-count');

        this.initialize();
    }

    initialize() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Handle selection toggle by clicking on image items
        this.galleryGrid.addEventListener('click', (e) => {
            // Skip if clicking on action buttons
            if (e.target.closest('.quick-actions') || e.target.closest('.fullscreen-modal')) {
                return;
            }

            // Toggle selection with Shift or Ctrl key
            if (e.shiftKey || e.ctrlKey || e.metaKey) {
                e.preventDefault();
                const item = e.target.closest('.gallery-item');
                if (item) {
                    this.toggleSelection(item);

                    // Add a visual flash effect for feedback
                    item.classList.add('selection-flash');
                    setTimeout(() => item.classList.remove('selection-flash'), 500);
                }
            }
        });

        // Batch operation buttons
        document.querySelector('.batch-download')?.addEventListener('click', () => this.downloadSelected());
        document.querySelector('.batch-delete')?.addEventListener('click', () => this.deleteSelected());
        document.querySelector('.batch-tag')?.addEventListener('click', () => this.tagSelected());
    }

    /**
     * Toggle selection state of an item
     * @param {HTMLElement} item - Gallery item to toggle
     */
    toggleSelection(item) {
        if (this.isSelected(item)) {
            this.deselectItem(item);
        } else {
            this.selectItem(item);
        }

        // Update any favorite/select buttons
        const selectButton = item.querySelector('.action-favorite');
        if (selectButton) {
            selectButton.textContent = this.isSelected(item) ? '✓' : '⭐';
            selectButton.title = this.isSelected(item) ? 'Deselect Image' : 'Select Image';
        }

        return this.isSelected(item); // Return current selection state
    }

    /**
     * Select an item
     * @param {HTMLElement} item - Gallery item to select
     */
    selectItem(item) {
        // Support both data attributes for backwards compatibility
        const imageId = item.dataset.imageId || item.dataset.mediaId;
        if (!imageId) return;

        item.classList.add('selected');
        this.selectedItems.add(imageId);
        this.updateBatchOperationsVisibility();
    }

    /**
     * Deselect an item
     * @param {HTMLElement} item - Gallery item to deselect
     */
    deselectItem(item) {
        // Support both data attributes for backwards compatibility
        const imageId = item.dataset.imageId || item.dataset.mediaId;
        if (!imageId) return;

        item.classList.remove('selected');
        this.selectedItems.delete(imageId);
        this.updateBatchOperationsVisibility();
    }

    /**
     * Check if an item is selected
     * @param {HTMLElement} item - Gallery item to check
     * @returns {boolean} Whether the item is selected
     */
    isSelected(item) {
        // Support both data attributes for backwards compatibility
        const imageId = item.dataset.imageId || item.dataset.mediaId;
        return imageId ? this.selectedItems.has(imageId) : false;
    }

    /**
     * Select all visible items
     */
    selectAll() {
        document.querySelectorAll('.gallery-item:not(.hidden)').forEach(item => {
            this.selectItem(item);

            // Update select button
            const selectButton = item.querySelector('.action-favorite');
            if (selectButton) {
                selectButton.textContent = '✓';
                selectButton.title = 'Deselect Image';
            }
        });
    }

    /**
     * Deselect all items
     */
    deselectAll() {
        const selectedElements = document.querySelectorAll('.gallery-item.selected');

        selectedElements.forEach(item => {
            item.classList.remove('selected');

            // Update select button
            const selectButton = item.querySelector('.action-favorite');
            if (selectButton) {
                selectButton.textContent = '⭐';
                selectButton.title = 'Select Image';
            }
        });

        this.selectedItems.clear();
        this.updateBatchOperationsVisibility();
    }

    /**
     * Update the visibility of batch operations toolbar
     */
    updateBatchOperationsVisibility() {
        if (this.selectedItems.size > 0) {
            this.batchOperationsContainer.style.display = 'flex';
            this.selectedCountElement.textContent = `${this.selectedItems.size} selected`;
        } else {
            this.batchOperationsContainer.style.display = 'none';
        }
    }

    /**
     * Get all selected image IDs
     * @returns {Array} Array of selected image IDs
     */
    getSelectedIds() {
        return Array.from(this.selectedItems);
    }

    /**
     * Download all selected images
     */
    async downloadSelected() {
        const selectedIds = this.getSelectedIds();
        if (selectedIds.length === 0) return;

        try {
            // For small batches, download directly
            if (selectedIds.length <= 5) {
                Utilities.showFeedback(`Downloading ${selectedIds.length} images...`, 'info');

                for (const id of selectedIds) {
                    await this.downloadImage(id);
                }

                Utilities.showFeedback(`Downloaded ${selectedIds.length} images successfully!`, 'success');
            } else {
                // For larger batches, request a zip file
                Utilities.showFeedback(`Preparing ${selectedIds.length} images for download...`, 'info');

                // This would be a server-side API that packages multiple images
                const response = await fetch('/api/batch_download', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_ids: selectedIds })
                });

                if (!response.ok) throw new Error('Batch download failed');

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `cyberimage-batch-${Date.now()}.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                Utilities.showFeedback('Batch download complete!', 'success');
            }
        } catch (error) {
            console.error('Download error:', error);
            Utilities.showFeedback(`Download failed: ${error.message}`, 'error');
        }
    }

    /**
     * Download a single image
     * @param {string} imageId - ID of image to download
     */
    async downloadImage(imageId) {
        try {
            const response = await fetch(window.API.IMAGE(imageId));
            if (!response.ok) throw new Error('Failed to download image');

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cyberimage-${imageId}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            Utilities.showFeedback('Image downloaded successfully!', 'success');
        } catch (error) {
            console.error('Error downloading image:', error);
            Utilities.showFeedback('Failed to download image', 'error');
        }
    }

    /**
     * Delete all selected images
     */
    async deleteSelected() {
        const selectedIds = this.getSelectedIds();
        if (selectedIds.length === 0) return;

        // Confirm deletion
        if (!confirm(`Are you sure you want to delete ${selectedIds.length} images? This cannot be undone.`)) {
            return;
        }

        Utilities.showFeedback(`Deleting ${selectedIds.length} images...`, 'info');

        let successCount = 0;
        let failCount = 0;

        try {
            await Promise.all(selectedIds.map(async (id) => {
                try {
                    const response = await fetch(window.API.DELETE_IMAGE(id), {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        // Remove item from DOM
                        const item = document.querySelector(`[data-image-id="${id}"]`);
                        if (item) item.remove();
                        successCount++;
                    } else {
                        failCount++;
                    }
                } catch (error) {
                    console.error(`Error deleting image ${id}:`, error);
                    failCount++;
                }
            }));

            this.selectedItems.clear();
            this.updateBatchOperationsVisibility();

            if (failCount === 0) {
                Utilities.showFeedback(`Successfully deleted ${successCount} images!`, 'success');
            } else {
                Utilities.showFeedback(`Deleted ${successCount} images, ${failCount} failed.`, 'warning');
            }
        } catch (error) {
            console.error('Batch delete error:', error);
            Utilities.showFeedback(`Error during batch delete: ${error.message}`, 'error');
        }
    }

    /**
     * Tag all selected images
     */
    async tagSelected() {
        const selectedIds = this.getSelectedIds();
        if (selectedIds.length === 0) return;

        const tag = prompt('Enter tag to add to selected images:');
        if (!tag) return;

        Utilities.showFeedback(`Adding tag "${tag}" to ${selectedIds.length} images...`, 'info');

        let successCount = 0;
        let failCount = 0;

        try {
            await Promise.all(selectedIds.map(async (id) => {
                try {
                    const response = await fetch(window.API.ADD_TAG(id, tag), {
                        method: 'POST'
                    });

                    if (response.ok) {
                        successCount++;
                    } else {
                        failCount++;
                    }
                } catch (error) {
                    console.error(`Error tagging image ${id}:`, error);
                    failCount++;
                }
            }));

            if (failCount === 0) {
                Utilities.showFeedback(`Successfully tagged ${successCount} images!`, 'success');
            } else {
                Utilities.showFeedback(`Tagged ${successCount} images, ${failCount} failed.`, 'warning');
            }
        } catch (error) {
            console.error('Batch tag error:', error);
            Utilities.showFeedback(`Error during batch tagging: ${error.message}`, 'error');
        }
    }
}

/**
 * ModalManager - Handles the fullscreen image modal
 */
class ModalManager {
    constructor() {
        this.modal = document.getElementById('fullscreenModal');
        this.modalContent = this.modal.querySelector('.fullscreen-content');
        this.currentImageId = null;
        this.isOpen = false;
        this.modalImage = document.getElementById('modalImage');
        this.modalVideoContainer = this.modal.querySelector('.fullscreen-image');
        this.modalVideo = null;
        this.modelInfo = document.getElementById('modelInfo');
        this.promptInfo = document.getElementById('promptInfo');
        this.settingsInfo = document.getElementById('settingsInfo');
        this.copyPromptBtn = document.getElementById('copyPrompt');
        this.downloadBtn = document.getElementById('downloadImage');
        this.deleteBtn = document.getElementById('deleteImage');
        this.closeBtn = this.modal.querySelector('.action-close');

        this.initialize();
    }

    initialize() {
        // Close modal handlers
        this.closeBtn.addEventListener('click', () => this.hide());
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) this.hide();
        });

        // Keyboard handlers
        document.addEventListener('keydown', (e) => {
            if (!this.isOpen) return;

            switch (e.key) {
                case 'Escape':
                    this.hide();
                    break;
                case 'ArrowRight':
                    this.nextImage();
                    break;
                case 'ArrowLeft':
                    this.prevImage();
                    break;
            }
        });

        // Button handlers
        if (this.copyPromptBtn) {
            this.copyPromptBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.copyPrompt();
            });
        }

        if (this.downloadBtn) {
            this.downloadBtn.addEventListener('click', () => this.downloadCurrentImage());
        }

        if (this.deleteBtn) {
            this.deleteBtn.addEventListener('click', () => this.deleteCurrentImage());
        }
    }

    /**
     * Hide the modal
     */
    hide() {
        if (this.modal) {
            this.modal.style.display = 'none';
            document.body.style.overflow = '';
            this.isOpen = false;
            this.currentImageId = null;
        }
    }

    /**
     * Show the modal for an image
     * @param {HTMLImageElement} imageElement - Image element to show
     */
    show(imageElement) {
        const galleryItem = imageElement.closest('.gallery-item');
        if (!galleryItem) return;

        // Support both data-media-id (newer) and data-image-id (older gallery items)
        this.currentImageId = galleryItem.dataset.mediaId || galleryItem.dataset.imageId;
        if (!this.currentImageId) {
            console.error("No image ID found on gallery item:", galleryItem);
            return;
        }

        const isVideo = galleryItem.classList.contains('gallery-item-video');

        // Store reference for navigation
        this.modalContent.dataset.currentImageId = this.currentImageId;
        this.modalContent.dataset.isVideo = isVideo;

        // Hide/show image/video elements
        this.modalImage.style.display = isVideo ? 'none' : 'block';
        if (this.modalVideo) this.modalVideo.style.display = isVideo ? 'block' : 'none';

        // Set image source
        if (isVideo) {
            // Create video element if it doesn't exist
            if (!this.modalVideo) {
                this.modalVideo = document.createElement('video');
                this.modalVideo.id = 'modalVideo';
                this.modalVideo.controls = true;
                this.modalVideo.style.maxWidth = '100%';
                this.modalVideo.style.maxHeight = '70vh';
                this.modalVideo.style.display = 'none'; // Initially hidden
                this.modalVideoContainer.appendChild(this.modalVideo);
            }
            this.modalVideo.src = window.API.GET_VIDEO(this.currentImageId);
            this.modalVideo.style.display = 'block';
            this.modalImage.style.display = 'none';
        } else {
            this.modalImage.src = imageElement.src || imageElement.dataset.src;
            this.modalImage.style.display = 'block';
        }

        // Make sure modal is visible
        this.modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        this.isOpen = true;

        // Fetch and display metadata
        this.fetchImageMetadata(this.currentImageId);
    }

    /**
     * Download the current image
     */
    async downloadCurrentImage() {
        if (!this.currentImageId) return;

        try {
            const response = await fetch(window.API.IMAGE(this.currentImageId));
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cyberimage-${this.currentImageId}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            // Show feedback
            const originalText = this.downloadBtn.innerHTML;
            this.downloadBtn.innerHTML = '<span class="icon">✓</span>Downloaded!';
            setTimeout(() => {
                this.downloadBtn.innerHTML = originalText;
            }, 2000);

            Utilities.showFeedback('Image downloaded successfully!', 'success');
        } catch (error) {
            console.error('Error downloading image:', error);
            this.downloadBtn.innerHTML = '<span class="icon">❌</span>Error';
            setTimeout(() => {
                this.downloadBtn.innerHTML = '<span class="icon">⬇️</span>Download Image';
            }, 2000);

            Utilities.showFeedback('Failed to download image', 'error');
        }
    }

    /**
     * Delete the current image
     */
    async deleteCurrentImage() {
        if (!this.currentImageId) return;

        const deleteConfirmModal = document.getElementById('deleteConfirmModal');
        if (deleteConfirmModal) {
            deleteConfirmModal.style.display = 'flex';

            // Set up the confirmation to delete this specific image
            const confirmBtn = document.getElementById('confirmDelete');
            const cancelBtn = document.getElementById('cancelDelete');

            // Remove existing listeners to prevent duplicates
            const newConfirmBtn = confirmBtn.cloneNode(true);
            const newCancelBtn = cancelBtn.cloneNode(true);
            confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
            cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);

            newConfirmBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch(window.API.DELETE_IMAGE(this.currentImageId), {
                        method: 'DELETE'
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();

                    if (data.status === 'success') {
                        // Remove the image from gallery - support both attribute types
                        const galleryItem = document.querySelector(`[data-image-id="${this.currentImageId}"]`) ||
                                          document.querySelector(`[data-media-id="${this.currentImageId}"]`);
                        if (galleryItem) {
                            galleryItem.remove();
                        }

                        // Close both modals
                        deleteConfirmModal.style.display = 'none';
                        this.hide();

                        // Show success message
                        Utilities.showFeedback('Image deleted successfully!', 'success');
                    } else {
                        throw new Error(data.message || 'Failed to delete image');
                    }
                } catch (error) {
                    console.error('Error deleting image:', error);
                    deleteConfirmModal.style.display = 'none';
                    Utilities.showFeedback(`Failed to delete image: ${error.message}`, 'error');
                }
            });

            newCancelBtn.addEventListener('click', () => {
                deleteConfirmModal.style.display = 'none';
            });

            // Close on outside click
            deleteConfirmModal.addEventListener('click', (e) => {
                if (e.target === deleteConfirmModal) {
                    deleteConfirmModal.style.display = 'none';
                }
            });
        }
    }

    /**
     * Copy the current prompt to clipboard
     */
    async copyPrompt() {
        if (!this.currentImageId || !this.promptInfo) return;

        // Get prompt text from the element
        const promptText = this.promptInfo.textContent;
        if (!promptText || promptText === 'Loading...' || promptText === 'Metadata unavailable') {
            Utilities.showFeedback('No prompt available to copy', 'error');
            return;
        }

        try {
            // Visual feedback
            const originalButtonHtml = this.copyPromptBtn.innerHTML;
            this.copyPromptBtn.innerHTML = '<span class="icon">⏳</span> Copying...';

            // Try to copy
            const success = await Utilities.copyToClipboard(promptText);

            if (success) {
                this.copyPromptBtn.innerHTML = '<span class="icon">✓</span> Copied!';
                Utilities.showFeedback('Prompt copied to clipboard!', 'success');
            } else {
                this.copyPromptBtn.innerHTML = '<span class="icon">❌</span> Failed';
                Utilities.showFeedback('Failed to copy prompt automatically. Try selecting and copying manually.', 'warning');

                // Make the text selectable and bring attention to it
                this.promptInfo.style.userSelect = 'text';
                this.promptInfo.style.webkitUserSelect = 'text';
                this.promptInfo.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                this.promptInfo.focus();
            }

            // Reset button after delay
            setTimeout(() => {
                this.copyPromptBtn.innerHTML = originalButtonHtml;
                // Reset highlighting after a delay
                setTimeout(() => {
                    this.promptInfo.style.backgroundColor = '';
                }, 3000);
            }, 2000);

        } catch (error) {
            console.error('Error in modal copyPrompt:', error);
            this.copyPromptBtn.innerHTML = '<span class="icon">❌</span> Error';

            // Reset button after delay
            setTimeout(() => {
                this.copyPromptBtn.innerHTML = '<span class="icon">📋</span> Copy Prompt';
            }, 2000);

            Utilities.showFeedback(`Failed to copy: ${error.message}`, 'error');
        }
    }

    /**
     * Fetch and display image metadata
     * @param {string} imageId - ID of the image to fetch metadata for
     */
    async fetchImageMetadata(imageId) {
        // Reset metadata fields
        if (this.modelInfo) this.modelInfo.textContent = 'Loading...';
        if (this.promptInfo) this.promptInfo.textContent = 'Loading...';
        if (this.settingsInfo) this.settingsInfo.textContent = 'Loading...';

        try {
            const response = await fetch(window.API.METADATA(imageId));

            if (!response.ok) {
                throw new Error(`Failed to fetch metadata (${response.status})`);
            }

            const data = await response.json();

            // Update modal with metadata
            if (this.modelInfo) {
                this.modelInfo.textContent = data.model_id || 'Not available';
            }

            if (this.promptInfo) {
                this.promptInfo.textContent = data.prompt ||
                                             data.settings?.prompt ||
                                             'Not available';
            }

            if (this.settingsInfo && data.settings) {
                try {
                    // Format settings nicely
                    const settings = Object.entries(data.settings || {})
                        .filter(([key]) => key !== 'prompt') // Skip prompt as it's shown separately
                        .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
                        .join('\n');

                    this.settingsInfo.textContent = settings || 'No additional settings';
                } catch (formatError) {
                    console.error('Error formatting settings:', formatError);
                    this.settingsInfo.textContent = 'Error formatting settings';
                }
            }
        } catch (error) {
            console.error('Error fetching image metadata:', error);

            // Set error messages in modal
            if (this.modelInfo) this.modelInfo.textContent = 'Metadata unavailable';
            if (this.promptInfo) this.promptInfo.textContent = 'Metadata unavailable';
            if (this.settingsInfo) this.settingsInfo.textContent = 'Metadata unavailable';

            Utilities.showFeedback(`Failed to load image details: ${error.message}`, 'error');
        }
    }
}

/**
 * Main Gallery Manager class
 */
class GalleryManager {
    constructor() {
        // DOM Elements
        this.galleryGrid = document.querySelector('.gallery-grid');
        this.loadingIndicator = document.querySelector('.loading-indicator');
        this.quickFilter = document.querySelector('.quick-filter');
        this.viewToggle = document.getElementById('view-toggle');
        this.gridViewBtn = document.getElementById('grid-view');
        this.listViewBtn = document.getElementById('list-view');
        this.selectAllBtn = document.getElementById('select-all');
        this.deselectAllBtn = document.getElementById('deselect-all');
        this.batchActionsBar = document.getElementById('batch-actions');
        this.selectedCountDisplay = document.getElementById('selected-count');
        this.sentinel = document.getElementById('sentinel');

        // State
        this.currentImage = null;
        this.loading = false;
        this.page = 0;
        this.allImagesLoaded = false;
        this.filter = '';
        this.modelFilter = null; // Track model filter for API calls
        this.debugMode = true; // Enable debug mode by default

        // Check for URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('search')) {
            this.filter = urlParams.get('search');
        }
        if (urlParams.has('model')) {
            this.modelFilter = urlParams.get('model');
        }

        // Sub-managers
        this.imageLoader = new ImageLoader();
        this.viewManager = new ViewManager(this.galleryGrid);
        this.selectionManager = new SelectionManager(this.galleryGrid);
        this.modalManager = new ModalManager();

        // Initialize
        this.initialize();
    }

    initialize() {
        this.setupInfiniteScroll();
        this.setupImageActions();
        this.setupFilterHandling();
        this.setupShortcuts();
        this.setupTouchGestures();

        // Load initial images if none are present
        if (this.galleryGrid.querySelectorAll('.gallery-item').length === 0) {
            this.loadMoreImages();
        } else {
            // Observe existing images for lazy loading
            this.galleryGrid.querySelectorAll('.gallery-item img[data-src]').forEach(img => {
                this.imageLoader.observe(img);
            });
        }

        // Show loading state
        this.updateLoadingState(false);

        // Log initialization
        if (this.debugMode) {
            console.log('Gallery initialized', {
                items: this.galleryGrid.querySelectorAll('.gallery-item').length,
                page: this.page
            });
        }
    }

    /**
     * Set up infinite scroll with improved visibility and debugging
     */
    setupInfiniteScroll() {
        // Use a clearly visible sentinel for infinite scroll
        const sentinel = document.createElement('div');
        sentinel.className = 'scroll-sentinel';
        sentinel.style.width = '100%';
        sentinel.style.height = '10px';
        sentinel.style.margin = '30px 0';
        sentinel.id = 'infinite-scroll-sentinel';

        // Add the sentinel AFTER the gallery grid but BEFORE the loading indicator
        if (this.loadingIndicator && this.loadingIndicator.parentNode) {
            this.loadingIndicator.parentNode.insertBefore(sentinel, this.loadingIndicator);
        } else {
            // Fallback - append after gallery
            this.galleryGrid.parentNode.appendChild(sentinel);
        }

        // Log sentinel creation
        if (this.debugMode) {
            console.log('Infinite scroll sentinel created and positioned', {
                sentinel: sentinel,
                position: 'before loading indicator'
            });
        }

        // Create a more aggressive intersection observer
        const observer = new IntersectionObserver((entries) => {
            if (this.debugMode) {
                console.log('Sentinel intersection', {
                    isIntersecting: entries[0].isIntersecting,
                    loading: this.loading,
                    allLoaded: this.allImagesLoaded
                });
            }

            if (entries[0].isIntersecting && !this.loading && !this.allImagesLoaded) {
                this.loadMoreImages();
            }
        }, {
            rootMargin: '300px', // More aggressive rootMargin
            threshold: 0.1 // Trigger when just a small part is visible
        });

        observer.observe(sentinel);

        // Enhanced scroll handler that's more aggressive
        window.addEventListener('scroll', Utilities.throttle(() => {
            // Check if we're near the bottom of the page
            const scrollPosition = window.scrollY + window.innerHeight;
            const contentHeight = document.documentElement.scrollHeight;
            const scrollPercentage = scrollPosition / contentHeight;

            // If we're within 30% of the bottom and not loading/finished
            if (scrollPercentage > 0.7 && !this.loading && !this.allImagesLoaded) {
                this.loadMoreImages();

                if (this.debugMode) {
                    console.log('Scroll triggered load', {
                        scrollPercentage: scrollPercentage,
                        position: scrollPosition,
                        contentHeight: contentHeight
                    });
                }
            }
        }, 200)); // Throttle to 200ms for better performance
    }

    /**
     * Load more images via API with enhanced error handling
     */
    async loadMoreImages() {
        if (this.loading) return;

        // Reset the allImagesLoaded flag when starting a new search
        if (this.page === 0) {
            this.allImagesLoaded = false;
        }

        // Skip if all images are already loaded (only for browsing, not search)
        if (this.allImagesLoaded && !this.filter) return;

        this.loading = true;
        this.updateLoadingState(true);

        // Scroll to top when starting a new search
        if (this.page === 0 && this.filter) {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        if (this.debugMode) {
            console.log('Loading more images', {
                page: this.page,
                filter: this.filter
            });
        }

        try {
            // Build the URL with search parameters
            let url = `${window.API.GALLERY(this.page + 1)}`;

            // Add search parameter if we have a filter
            if (this.filter && this.filter.trim() !== '') {
                url += `&search=${encodeURIComponent(this.filter.trim())}`;

                // Show searching feedback
                this.loadingIndicator.innerHTML = `
                    <div class="spinner"></div>
                    <p>Searching across all images for "${this.filter}"...</p>
                `;
            } else {
                // When not searching, show normal loading message
                this.loadingIndicator.innerHTML = `
                    <div class="spinner"></div>
                    <p>Loading more images...</p>
                `;
            }

            // Add any model filter if applicable
            if (this.modelFilter) {
                url += `&model=${encodeURIComponent(this.modelFilter)}`;
            }

            const response = await fetch(url, {
                headers: {
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('API Error Response:', {
                    status: response.status,
                    statusText: response.statusText,
                    responseText: errorText
                });
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText || response.statusText}`);
            }

            const data = await response.json();
            this.page++; // Only increment page after successful load

            if (this.debugMode) {
                console.log('Images loaded', {
                    count: data.images?.length || 0,
                    page: this.page,
                    total: data.total || 'unknown',
                    searchActive: Boolean(this.filter && this.filter.trim() !== '')
                });
            }

            if (data.images && data.images.length > 0) {
                // Clear existing content if this is a new search
                if (this.page === 1) {
                    this.galleryGrid.innerHTML = '';

                    // Remove any existing search banner
                    const existingBanner = document.querySelector('.search-results-banner');
                    if (existingBanner) {
                        existingBanner.remove();
                    }
                }

                this.renderImages(data.images);

                // Show appropriate feedback based on search vs regular loading
                if (this.filter && this.filter.trim() !== '') {
                    if (this.page === 1) {
                        // First page of search results
                        const resultMessage = `Found ${data.total} image${data.total !== 1 ? 's' : ''} matching "${this.filter}"`;

                        // Show a banner at the top with search results count
                        const searchBanner = document.createElement('div');
                        searchBanner.className = 'search-results-banner';
                        searchBanner.innerHTML = `
                            <div class="search-results-count">
                                <i class="fas fa-search"></i>
                                ${resultMessage}
                            </div>
                            <button class="clear-search-button">Clear Search</button>
                        `;

                        // Remove any existing banner first
                        const existingBanner = document.querySelector('.search-results-banner');
                        if (existingBanner) {
                            existingBanner.remove();
                        }

                        // Insert banner before gallery grid
                        this.galleryGrid.parentNode.insertBefore(searchBanner, this.galleryGrid);

                        // Add clear button functionality
                        searchBanner.querySelector('.clear-search-button').addEventListener('click', () => {
                            if (this.quickFilter) {
                                this.quickFilter.value = '';
                                this.filter = '';
                                this.page = 0;
                                this.allImagesLoaded = false;
                                this.galleryGrid.innerHTML = '';
                                searchBanner.remove();

                                // Update URL
                                const url = new URL(window.location);
                                url.searchParams.delete('search');
                                window.history.replaceState({}, '', url);

                                // Update clear button visibility
                                const clearBtn = document.querySelector('.clear-filter');
                                if (clearBtn) clearBtn.style.display = 'none';

                                this.loadMoreImages();
                            }
                        });

                        Utilities.showFeedback(resultMessage, 'success', 3000);

                        // Also scroll to top to show results
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                    } else if (data.images.length >= 10) {
                        // Loading more search results
                        Utilities.showFeedback(`Loaded ${data.images.length} more matching images`, 'info', 1500);
                    }
                } else if (data.images.length >= 10) {
                    // Regular loading more
                    Utilities.showFeedback(`Loaded ${data.images.length} more images`, 'info', 1500);
                }

                // Check if there are more images to load
                if (!data.has_more) {
                    this.allImagesLoaded = true;
                    this.loadingIndicator.innerHTML = 'All images loaded';
                }
            } else {
                this.allImagesLoaded = true;

                // Different messages for search vs regular browsing
                if (this.filter && this.filter.trim() !== '') {
                    if (this.page === 1) {
                        // No results at all for search
                        this.loadingIndicator.innerHTML = `
                            <div class="no-results">
                                <p>No images found matching "${this.filter}"</p>
                                <button id="clear-search-btn" class="button">Clear Search</button>
                            </div>
                        `;

                        document.getElementById('clear-search-btn')?.addEventListener('click', () => {
                            if (this.quickFilter) {
                                this.quickFilter.value = '';
                                this.filter = '';
                                this.page = 0;
                                this.allImagesLoaded = false;
                                this.galleryGrid.innerHTML = '';

                                // Update URL
                                const url = new URL(window.location);
                                url.searchParams.delete('search');
                                window.history.replaceState({}, '', url);

                                // Update clear button visibility
                                const clearBtn = document.querySelector('.clear-filter');
                                if (clearBtn) clearBtn.style.display = 'none';

                                this.loadMoreImages();
                            }
                        });

                        Utilities.showFeedback(`No images found matching "${this.filter}"`, 'warning');
                    } else {
                        // No more results for search
                        this.loadingIndicator.innerHTML = `No more images matching "${this.filter}"`;
                        Utilities.showFeedback(`No more images matching "${this.filter}"`, 'info');
                    }
                } else {
                    // No more results for regular browsing
                    this.loadingIndicator.innerHTML = 'All images loaded';
                    Utilities.showFeedback('All images loaded', 'info');
                }
            }
        } catch (error) {
            console.error('Error loading more images:', error);
            this.page--; // Revert page increment on error

            this.loadingIndicator.innerHTML = 'Error loading images. <button id="retry-load">Retry</button>';
            document.getElementById('retry-load')?.addEventListener('click', () => {
                this.loading = false;
                this.loadMoreImages();
            });

            Utilities.showFeedback(`Failed to load images: ${error.message}`, 'error');
        } finally {
            this.loading = false;
            this.updateLoadingState(false);
        }
    }

    /**
     * Render images to the gallery
     * @param {Array} images - Array of image data objects
     */
    renderImages(images) {
        if (!images || !Array.isArray(images) || images.length === 0) {
            return;
        }

        // Apply client-side filtering if a filter is set
        let filteredImages = images;
        if (this.filter && this.filter.trim() !== '') {
            const queryTerms = this.filter.toLowerCase().split(/\s+/).filter(term => term.length > 0);
            filteredImages = images.filter(image => {
                // Get all searchable content from the image
                const prompt = (image.prompt || '').toLowerCase();
                const modelId = (image.model_id || '').toLowerCase();
                const tags = Array.isArray(image.tags) ? image.tags.join(' ').toLowerCase() : '';
                const searchableContent = `${prompt} ${modelId} ${tags}`;

                // Check if ALL search terms appear in the searchable content
                return queryTerms.every(term => searchableContent.includes(term));
            });

            if (this.debugMode) {
                console.log(`Filtered ${images.length} images to ${filteredImages.length} results using ${queryTerms.length} search terms: "${this.filter}"`);
            }
        }

        // If no images match the filter, update state accordingly
        if (filteredImages.length === 0) {
            if (images.length > 0 && this.filter) {
                // Images were filtered out
                this.loadingIndicator.innerHTML = `No matches for "${this.filter}". <button id="clear-filter-btn">Clear filter</button>`;
                document.getElementById('clear-filter-btn')?.addEventListener('click', () => {
                    if (this.quickFilter) this.quickFilter.value = '';
                    this.filter = '';
                    this.galleryGrid.innerHTML = '';
                    this.page = 0;
                    this.allImagesLoaded = false;
                    this.loadMoreImages();
                });
            }
            return;
        }

        // Create and append elements for filtered images
        const fragment = document.createDocumentFragment();
        filteredImages.forEach(image => {
            const imgEl = this.createImageElement(image);
            fragment.appendChild(imgEl);
        });

        this.galleryGrid.appendChild(fragment);

        // Observe images for lazy loading
        this.galleryGrid.querySelectorAll('.gallery-item:not(.observed) img[data-src]').forEach(img => {
            this.imageLoader.observe(img);
            img.closest('.gallery-item').classList.add('observed');
        });
    }

    /**
     * Create a gallery item element with updated select button
     * @param {Object} image - Image data object
     * @returns {HTMLElement} Created gallery item
     */
    createImageElement(image) {
        const div = document.createElement('div');
        div.className = 'gallery-item';

        // Add both attributes for compatibility
        div.dataset.imageId = image.id; // Original attribute
        div.dataset.mediaId = image.id; // New attribute for compatibility

        // Handle potential missing data gracefully
        const prompt = image.prompt || 'No prompt available';
        const modelId = image.model_id || 'Unknown model';

        // Format dates
        const createdAt = new Date(image.created_at);
        const displayDate = Utilities.formatDate(createdAt);
        const fullDate = Utilities.formatDateLong(createdAt);

        // Highlight search terms in prompt if we have a filter
        let highlightedPrompt = Utilities.escapeHtml(prompt);
        let highlightedModelId = Utilities.escapeHtml(modelId);

        if (this.filter && this.filter.trim() !== '') {
            const keywords = this.filter.toLowerCase().split(' ');

            // Highlight each keyword in the prompt
            keywords.forEach(keyword => {
                if (keyword && keyword.trim()) {
                    const regex = new RegExp(`(${keyword})`, 'gi');
                    highlightedPrompt = highlightedPrompt.replace(
                        regex,
                        '<span class="search-highlight">$1</span>'
                    );
                    highlightedModelId = highlightedModelId.replace(
                        regex,
                        '<span class="search-highlight">$1</span>'
                    );
                }
            });
        }

        div.innerHTML = `
            <div class="item-preview">
                <img data-src="${window.API.IMAGE(image.id)}"
                     alt="${Utilities.escapeHtml(prompt.substring(0, 50))}"
                     loading="lazy">
                <div class="quick-actions">
                    <button class="action-copy-prompt" title="Copy Prompt">📋</button>
                    <button class="action-favorite" title="Select Image">⭐</button>
                    <button class="action-download" title="Download">⬇️</button>
                    <button class="action-delete" title="Delete Image">🗑️</button>
                </div>
            </div>
            <div class="item-details">
                <div class="prompt-preview">${highlightedPrompt}</div>
                <div class="metadata">
                    <span class="model-info">${highlightedModelId}</span>
                    <span class="generation-time" title="${fullDate}">
                        ${displayDate}
                    </span>
                </div>
                ${image.tags && image.tags.length > 0 ? `
                <div class="tags">
                    ${image.tags.map(tag => `<span class="tag">${Utilities.escapeHtml(tag)}</span>`).join('')}
                </div>` : ''}
            </div>
        `;

        return div;
    }

    /**
     * Update loading indicator state
     * @param {boolean} isLoading - Whether images are being loaded
     */
    updateLoadingState(isLoading) {
        if (isLoading) {
            this.loadingIndicator.style.display = 'block';
            this.loadingIndicator.innerHTML = `
                <div class="spinner"></div>
                <p>Loading more images...</p>
            `;
        } else if (this.allImagesLoaded) {
            this.loadingIndicator.style.display = 'block';
            this.loadingIndicator.innerHTML = 'All images loaded';
        } else {
            this.loadingIndicator.style.display = 'none';
        }
    }

    /**
     * Set up image action buttons with favorite button repurposed as select
     */
    setupImageActions() {
        // Use event delegation for better performance
        this.galleryGrid.addEventListener('click', (e) => {
            // Get closest gallery item
            const galleryItem = e.target.closest('.gallery-item');
            if (!galleryItem) return;

            // Handle action buttons
            if (e.target.classList.contains('action-copy-prompt')) {
                e.stopPropagation();
                this.copyPrompt(galleryItem.dataset.imageId, e.target);
            } else if (e.target.classList.contains('action-favorite')) {
                e.stopPropagation();
                // CHANGED: Repurposed favorite button as select button
                this.selectionManager.toggleSelection(galleryItem);
                e.target.textContent = this.selectionManager.isSelected(galleryItem) ? '✓' : '⭐';
            } else if (e.target.classList.contains('action-download')) {
                e.stopPropagation();
                this.downloadImage(galleryItem.dataset.imageId);
            } else if (e.target.classList.contains('action-delete')) {
                e.stopPropagation();
                this.deleteImage(galleryItem.dataset.imageId);
            } else if (!e.target.closest('.quick-actions')) {
                // Show image in modal if not clicking on action buttons
                const img = galleryItem.querySelector('img');
                if (img) {
                    this.modalManager.show(img);
                }
            }
        });
    }

    /**
     * Set up filter handling
     */
    setupFilterHandling() {
        if (!this.quickFilter) return;

        // Add clear button if it doesn't exist
        if (!document.querySelector('.clear-filter')) {
            const clearButton = document.createElement('button');
            clearButton.className = 'clear-filter';
            clearButton.setAttribute('type', 'button');
            clearButton.innerHTML = '×';
            clearButton.title = 'Clear search';
            this.quickFilter.parentNode.appendChild(clearButton);
        }

        // Update search status on page load
        if (this.filter && this.filter.trim() !== '') {
            this.quickFilter.parentNode.classList.add('search-active');
        }

        // Debounce filter input
        this.quickFilter.addEventListener('input', Utilities.debounce((e) => {
            const query = e.target.value.toLowerCase();
            const previousFilter = this.filter;
            this.filter = query;

            // Skip API call if the filter hasn't actually changed
            if (previousFilter === query) {
                return;
            }

            // Reset pagination to start fresh
            this.page = 0;
            this.allImagesLoaded = false;
            this.galleryGrid.innerHTML = '';

            if (this.debugMode) {
                console.log('Applied filter:', query);
            }

            // Update search visual state
            if (query && query.trim() !== '') {
                this.quickFilter.parentNode.classList.add('search-active');
            } else {
                this.quickFilter.parentNode.classList.remove('search-active');

                // Remove any search results banner if clearing search
                const searchBanner = document.querySelector('.search-results-banner');
                if (searchBanner) {
                    searchBanner.remove();
                }
            }

            // Update URL with search parameter for bookmarking/sharing
            const url = new URL(window.location);
            if (query) {
                url.searchParams.set('search', query);
            } else {
                url.searchParams.delete('search');
            }
            window.history.replaceState({}, '', url);

            // Show loading indicator with appropriate message
            this.loadingIndicator.style.display = 'block';
            this.loadingIndicator.innerHTML = query ?
                `<div class="spinner"></div><p>Searching for "${query}"...</p>` :
                `<div class="spinner"></div><p>Loading gallery...</p>`;

            // Fetch filtered results from server
            this.loadMoreImages();

            // Update clear button visibility
            const clearBtn = document.querySelector('.clear-filter');
            if (clearBtn) {
                clearBtn.style.display = query ? 'block' : 'none';
            }
        }, 500));

        // Add clear button functionality
        const clearBtn = document.querySelector('.clear-filter');
        if (clearBtn) {
            // Hide initially if no search
            clearBtn.style.display = this.quickFilter.value ? 'block' : 'none';

            clearBtn.addEventListener('click', () => {
                // Skip if already empty
                if (this.quickFilter.value === '') return;

                this.quickFilter.value = '';
                this.filter = '';
                this.quickFilter.parentNode.classList.remove('search-active');

                // Reset and reload
                this.page = 0;
                this.allImagesLoaded = false;
                this.galleryGrid.innerHTML = '';

                // Update URL by removing search parameter
                const url = new URL(window.location);
                url.searchParams.delete('search');
                window.history.replaceState({}, '', url);

                // Hide the clear button
                clearBtn.style.display = 'none';

                // Remove any search results banner
                const searchBanner = document.querySelector('.search-results-banner');
                if (searchBanner) {
                    searchBanner.remove();
                }

                this.loadMoreImages();
            });
        }

        // Check for URL search parameter on page load
        const urlParams = new URLSearchParams(window.location.search);
        const searchParam = urlParams.get('search');
        if (searchParam) {
            this.quickFilter.value = searchParam;
            this.filter = searchParam.toLowerCase();

            // Make clear button visible
            const clearBtn = document.querySelector('.clear-filter');
            if (clearBtn) {
                clearBtn.style.display = 'block';
            }

            // Add active search class
            this.quickFilter.parentNode.classList.add('search-active');
        }
    }

    /**
     * Set up keyboard shortcuts with improved arrow key navigation
     */
    setupShortcuts() {
        // Clear any existing keyboard listeners to prevent duplicates
        document.removeEventListener('keydown', this._keydownHandler);

        // Create a handler function we can reference later
        this._keydownHandler = (e) => {
            // Skip if in input field or modal is open
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            // Only handle modal-specific keys if modal is open
            if (this.modalManager.isOpen) {
                switch (e.key) {
                    case 'Escape':
                        this.modalManager.hide();
                        break;
                    case 'ArrowRight':
                        this.modalManager.nextImage();
                        break;
                    case 'ArrowLeft':
                        this.modalManager.prevImage();
                        break;
                }
                return;
            }

            // Handle gallery navigation keys
            switch (e.key) {
                case 'ArrowDown':
                case 'j': // Keep j as alternative
                    e.preventDefault();
                    this.selectionManager.moveFocus('down');
                    break;

                case 'ArrowUp':
                case 'k': // Keep k as alternative
                    e.preventDefault();
                    this.selectionManager.moveFocus('up');
                    break;

                case 'ArrowRight':
                    e.preventDefault();
                    this.selectionManager.moveFocus('right');
                    break;

                case 'ArrowLeft':
                    e.preventDefault();
                    this.selectionManager.moveFocus('left');
                    break;

                case '/': // Focus search
                    e.preventDefault();
                    this.quickFilter?.focus();
                    break;

                case ' ': // Toggle selection of focused item
                    // Prevent space from scrolling the page
                    e.preventDefault();
                    this.selectionManager.toggleFocusedSelection();
                    break;
            }
        };

        // Add the event listener
        document.addEventListener('keydown', this._keydownHandler);

        if (this.debugMode) {
            console.log('Keyboard shortcuts initialized');
        }
    }

    /**
     * Set up touch gestures for mobile
     */
    setupTouchGestures() {
        let touchStartX = 0;
        let touchStartY = 0;

        // Set up swipe detection on the modal
        this.modalManager.modal.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
            touchStartY = e.changedTouches[0].screenY;
        }, { passive: true });

        this.modalManager.modal.addEventListener('touchend', (e) => {
            if (!this.modalManager.isOpen) return;

            const touchEndX = e.changedTouches[0].screenX;
            const touchEndY = e.changedTouches[0].screenY;

            const diffX = touchEndX - touchStartX;
            const diffY = touchEndY - touchStartY;

            // Only trigger if horizontal swipe is dominant
            if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 50) {
                if (diffX > 0) {
                    // Swipe right - previous image
                    this.modalManager.prevImage();
                } else {
                    // Swipe left - next image
                    this.modalManager.nextImage();
                }
            }
        }, { passive: true });

        if (this.debugMode) {
            console.log('Touch gestures initialized');
        }
    }

    /**
     * Get the currently focused item
     * @returns {HTMLElement|null} Focused gallery item or null
     */
    getFocusedItem() {
        return document.querySelector('.gallery-item.focused');
    }

    /**
     * Clear focus from all items
     */
    clearFocus() {
        document.querySelectorAll('.gallery-item.focused').forEach(item => {
            item.classList.remove('focused');
        });
    }

    /**
     * Navigate through images with keyboard
     * @param {string} direction - Direction to navigate (next/prev/left/right)
     */
    navigateImages(direction) {
        const items = Array.from(document.querySelectorAll('.gallery-item:not(.hidden)'));
        if (items.length === 0) return;

        const focused = this.getFocusedItem();
        let index = focused ? items.indexOf(focused) : -1;
        let newIndex;

        // Get current view to understand the layout
        const currentView = this.viewManager.getCurrentView();
        const isGridView = currentView === 'grid' || currentView === 'compact';

        // Calculate items per row for grid views
        let itemsPerRow = 1;
        if (isGridView && items.length > 0) {
            const firstItem = items[0];
            const itemRect = firstItem.getBoundingClientRect();
            const containerRect = this.galleryGrid.getBoundingClientRect();
            itemsPerRow = Math.floor(containerRect.width / itemRect.width);
        }

        if (this.debugMode) {
            console.log('Navigation calculation', {
                direction,
                currentIndex: index,
                totalItems: items.length,
                currentView,
                itemsPerRow
            });
        }

        switch (direction) {
            case 'next':
                newIndex = index < items.length - 1 ? index + 1 : 0;
                break;
            case 'prev':
                newIndex = index > 0 ? index - 1 : items.length - 1;
                break;
            case 'right':
                // If we're in a grid view, move right
                if (isGridView) {
                    newIndex = index < items.length - 1 ? index + 1 : 0;
                } else {
                    // In list view, right doesn't move
                    newIndex = index;
                }
                break;
            case 'left':
                // If we're in a grid view, move left
                if (isGridView) {
                    newIndex = index > 0 ? index - 1 : items.length - 1;
                } else {
                    // In list view, left doesn't move
                    newIndex = index;
                }
                break;
            default:
                newIndex = index;
                break;
        }

        const target = items[newIndex];
        if (target) {
            this.clearFocus();
            target.classList.add('focused');
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });

            if (this.debugMode) {
                console.log('Navigated to new item', {
                    newIndex,
                    imageId: target.dataset.imageId
                });
            }
        }
    }

    /**
     * Copy prompt for an image
     * @param {string} imageId - ID of image to copy prompt for
     * @param {HTMLElement} button - Button element that was clicked
     */
    async copyPrompt(imageId, button) {
        try {
            // Visual feedback - start
            if (button) {
                const originalText = button.textContent;
                button.textContent = '⏳';
                button.classList.add('copying');
            }

            // Fetch the metadata with proper error handling
            const response = await fetch(window.API.METADATA(imageId));

            if (!response.ok) {
                throw new Error(`Failed to fetch prompt (${response.status})`);
            }

            const data = await response.json();
            const promptText = data.prompt;

            if (!promptText) {
                throw new Error('No prompt available in metadata');
            }

            // Try to copy the text
            const success = await Utilities.copyToClipboard(promptText);

            if (success) {
                // Success feedback
                if (button) {
                    button.textContent = '✓';
                    button.classList.add('active');
                    button.classList.remove('copying');
                    setTimeout(() => {
                        button.textContent = '📋';
                        button.classList.remove('active');
                    }, 1500);
                }
                Utilities.showFeedback('Prompt copied to clipboard!', 'success');
            } else {
                throw new Error('Copy operation failed');
            }
        } catch (error) {
            console.error('Error copying prompt:', error);

            // Error feedback
            if (button) {
                button.textContent = '❌';
                button.classList.remove('copying');
                button.classList.add('error');
                setTimeout(() => {
                    button.textContent = '📋';
                    button.classList.remove('error');
                }, 1500);
            }

            Utilities.showFeedback(`Failed to copy prompt: ${error.message}`, 'error');
        }
    }

    /**
     * Download an image
     * @param {string} imageId - ID of image to download
     */
    async downloadImage(imageId) {
        try {
            const response = await fetch(window.API.IMAGE(imageId));
            if (!response.ok) throw new Error('Failed to download image');

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cyberimage-${imageId}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            Utilities.showFeedback('Image downloaded successfully!', 'success');
        } catch (error) {
            console.error('Error downloading image:', error);
            Utilities.showFeedback('Failed to download image', 'error');
        }
    }

    /**
     * Delete an image with confirmation dialog
     * @param {string} imageId - ID of image to delete
     */
    deleteImage(imageId) {
        // Get element for both potential attribute names
        const galleryItem = document.querySelector(`[data-image-id="${imageId}"]`) ||
                           document.querySelector(`[data-media-id="${imageId}"]`);

        // Reuse the deleteCurrentImage method from ModalManager
        this.modalManager.currentImageId = imageId;
        this.modalManager.deleteCurrentImage();
    }

    /**
     * Toggle display of keyboard shortcuts help
     */
    toggleShortcutsHelp() {
        const shortcutsHelp = document.querySelector('.shortcuts-help');
        if (shortcutsHelp) {
            const isVisible = shortcutsHelp.style.display === 'block';
            shortcutsHelp.style.display = isVisible ? 'none' : 'block';
        }
    }
}

// Initialize gallery when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.galleryManager = new GalleryManager();

    // Initialize queue status indicator
    updateQueueStatusIndicator();
    // Set interval for updating queue status
    setInterval(updateQueueStatusIndicator, 5000);

    // Initialize mobile navigation
    initializeMobileNav();

    // Register service worker for offline support if available
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('ServiceWorker registration successful');
            })
            .catch(error => {
                console.error('ServiceWorker registration failed:', error);
            });
    }
});

// Queue status functionality - direct implementation (no recursion)
function updateQueueStatusIndicator() {
    const statusIcon = document.getElementById('generation-status-icon');
    const statusText = document.getElementById('queue-status-text');
    const queueIndicator = statusIcon?.closest('.queue-indicator');

    if (!statusIcon || !statusText || !queueIndicator) return;

    fetch('/api/queue')
        .then(response => response.json())
        .then(data => {
            const { pending, processing } = data;
            const totalActive = pending + processing;

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
            queueIndicator.style.display = 'flex';
        });
}

// Mobile navigation functionality - direct implementation
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