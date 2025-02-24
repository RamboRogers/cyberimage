import { shortcutManager } from './modules/shortcuts.js';
import { viewManager } from './modules/viewManager.js';
import { selectionManager } from './modules/selectionManager.js';
import { imageLoader } from './modules/imageLoader.js';

class GalleryManager {
    constructor() {
        this.currentImageId = null;
        this.isModalOpen = false;  // Track modal state

        this.initializeModules();
        this.setupShortcuts();
        this.setupInfiniteScroll();
        this.setupQuickSearch();
        this.setupImageActions();
        this.setupFullscreenModal();
    }

    initializeModules() {
        // Initialize view manager
        viewManager.initialize();

        // Initialize image loading
        imageLoader.observeAll();

        // Setup event listeners for dynamic content
        this.setupDynamicContentHandlers();
    }

    setupShortcuts() {
        // Navigation shortcuts
        shortcutManager.register('j', () => this.navigateImages('next'));
        shortcutManager.register('k', () => this.navigateImages('prev'));

        // Selection shortcuts
        shortcutManager.register('space', (e) => {
            e.preventDefault();
            const focusedItem = this.getFocusedItem();
            if (focusedItem) selectionManager.toggleSelection(focusedItem);
        });

        shortcutManager.register('a', () => selectionManager.selectAll());
        shortcutManager.register('shift+a', () => selectionManager.deselectAll());

        // Search shortcut
        shortcutManager.register('/', () => {
            const searchInput = document.querySelector('.quick-filter');
            searchInput?.focus();
        });
    }

    setupInfiniteScroll() {
        let loading = false;
        let page = 1;

        const loadMore = async () => {
            if (loading) return;

            const scrollPosition = window.scrollY + window.innerHeight;
            const scrollThreshold = document.documentElement.scrollHeight - 800;

            if (scrollPosition >= scrollThreshold) {
                loading = true;
                try {
                    const response = await fetch(`/api/gallery?page=${++page}`);
                    const data = await response.json();

                    if (data.images?.length) {
                        this.appendImages(data.images);
                    }
                } catch (error) {
                    console.error('Error loading more images:', error);
                } finally {
                    loading = false;
                }
            }
        };

        window.addEventListener('scroll', () => {
            requestAnimationFrame(loadMore);
        });
    }

    setupQuickSearch() {
        const searchInput = document.querySelector('.quick-filter');
        let searchTimeout;

        searchInput?.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                const query = e.target.value.toLowerCase();
                this.filterImages(query);
            }, 300);
        });
    }

    setupDynamicContentHandlers() {
        // Handle new content from infinite scroll
        const galleryGrid = document.querySelector('.gallery-grid');
        const observer = new MutationObserver((mutations) => {
            mutations.forEach(mutation => {
                if (mutation.addedNodes.length) {
                    imageLoader.handleNewImages(galleryGrid);
                }
            });
        });

        observer.observe(galleryGrid, {
            childList: true,
            subtree: true
        });
    }

    appendImages(images) {
        const container = document.querySelector('.gallery-grid');

        images.forEach(image => {
            const element = this.createImageElement(image);
            container.appendChild(element);
        });
    }

    createImageElement(image) {
        const div = document.createElement('div');
        div.className = 'gallery-item';
        div.dataset.imageId = image.id;

        div.innerHTML = `
            <div class="item-preview">
                <img data-src="/api/get_image/${image.id}"
                     alt="${this.escapeHtml(image.prompt)}"
                     loading="lazy">
                <div class="quick-actions">
                    <button class="action-copy-prompt" title="Copy Prompt">📋</button>
                    <button class="action-favorite" title="Add to Favorites">⭐</button>
                    <button class="action-download" title="Download">⬇️</button>
                    <button class="action-delete" title="Delete Image">🗑️</button>
                </div>
            </div>
            <div class="item-details">
                <div class="prompt-preview">${this.escapeHtml(image.prompt)}</div>
                <div class="metadata">
                    <span class="model-info">${image.model_id}</span>
                    <span class="generation-time">${this.formatDate(image.created_at)}</span>
                </div>
            </div>
        `;

        return div;
    }

    filterImages(query) {
        document.querySelectorAll('.gallery-item').forEach(item => {
            const prompt = item.querySelector('.prompt-preview').textContent.toLowerCase();
            const model = item.querySelector('.model-info').textContent.toLowerCase();

            if (prompt.includes(query) || model.includes(query)) {
                item.classList.remove('hidden');
            } else {
                item.classList.add('hidden');
            }
        });
    }

    navigateImages(direction) {
        const items = Array.from(document.querySelectorAll('.gallery-item:not([style*="display: none"])'));
        const focused = this.getFocusedItem();
        let index = focused ? items.indexOf(focused) : -1;

        if (direction === 'next') {
            index = index < items.length - 1 ? index + 1 : 0;
        } else {
            index = index > 0 ? index - 1 : items.length - 1;
        }

        const target = items[index];
        if (target) {
            this.focusItem(target);
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    getFocusedItem() {
        return document.querySelector('.gallery-item.focused');
    }

    focusItem(item) {
        document.querySelectorAll('.gallery-item.focused').forEach(i => i.classList.remove('focused'));
        item.classList.add('focused');
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return new Intl.RelativeTimeFormat('en', { numeric: 'auto' }).format(
            Math.floor((date - Date.now()) / (1000 * 60 * 60 * 24)),
            'day'
        );
    }

    setupImageActions() {
        const self = this;
        const deleteConfirmModal = document.getElementById('deleteConfirmModal');
        const confirmDeleteBtn = document.getElementById('confirmDelete');
        const cancelDeleteBtn = document.getElementById('cancelDelete');

        console.log('Delete modal elements:', {
            modal: deleteConfirmModal,
            confirmBtn: confirmDeleteBtn,
            cancelBtn: cancelDeleteBtn
        });

        // Handle all delete button clicks (both gallery and fullscreen view)
        document.addEventListener('click', async (e) => {
            const target = e.target;

            // Handle delete button clicks
            if (target.classList.contains('action-delete') || target.id === 'deleteImage') {
                e.preventDefault();
                e.stopPropagation();

                // Get the image ID either from gallery item or fullscreen modal
                const galleryItem = target.closest('.gallery-item');
                self.currentImageId = galleryItem
                    ? galleryItem.dataset.imageId
                    : document.querySelector('#modalImage')?.closest('.fullscreen-content')
                        ?.querySelector('.gallery-item')?.dataset.imageId;

                if (self.currentImageId && deleteConfirmModal) {
                    deleteConfirmModal.style.display = 'flex';
                    self.manageModalState(true);
                }
                return;
            }

            // Other action handlers (download, favorite, copy) remain unchanged
            const galleryItem = target.closest('.gallery-item');
            if (!galleryItem) return;

            const imageId = galleryItem.dataset.imageId;

            if (target.classList.contains('action-download')) {
                e.stopPropagation();
                await self.downloadImage(imageId);
            } else if (target.classList.contains('action-favorite')) {
                e.stopPropagation();
                self.toggleFavorite(target, imageId);
            } else if (target.classList.contains('action-copy-prompt')) {
                e.stopPropagation();
                await self.copyPrompt(imageId);
            }
        });

        // Confirm delete action
        if (confirmDeleteBtn) {
            confirmDeleteBtn.addEventListener('click', async () => {
                if (!self.currentImageId) return;

                try {
                    const response = await fetch(`/api/image/${self.currentImageId}`, {
                        method: 'DELETE'
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();

                    if (data.status === 'success') {
                        // Remove the image from gallery
                        const galleryItem = document.querySelector(`[data-image-id="${self.currentImageId}"]`);
                        if (galleryItem) {
                            galleryItem.remove();
                        }

                        // Close both modals
                        deleteConfirmModal.style.display = 'none';
                        const fullscreenModal = document.getElementById('fullscreenModal');
                        if (fullscreenModal) {
                            fullscreenModal.style.display = 'none';
                        }

                        // Reset modal state and scrolling
                        self.manageModalState(false);

                        // Show success message
                        self.showFeedback('Image deleted successfully', 'success');

                        // Reset the current image ID
                        self.currentImageId = null;
                    } else {
                        throw new Error(data.message || 'Failed to delete image');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    self.showFeedback(error.message || 'Failed to delete image', 'error');
                    self.manageModalState(false);
                }
            });
        }

        // Cancel delete
        if (cancelDeleteBtn) {
            cancelDeleteBtn.addEventListener('click', () => {
                deleteConfirmModal.style.display = 'none';
                self.manageModalState(false);
                self.currentImageId = null;
            });
        }

        // Close modal when clicking outside
        window.addEventListener('click', (event) => {
            if (event.target === deleteConfirmModal) {
                deleteConfirmModal.style.display = 'none';
                self.manageModalState(false);
                self.currentImageId = null;
            }
        });
    }

    async downloadImage(imageId) {
        try {
            const response = await fetch(`/api/get_image/${imageId}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cyberimage-${imageId}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error downloading image:', error);
        }
    }

    async toggleFavorite(button, imageId) {
        try {
            const response = await fetch(`/api/favorite/${imageId}`, {
                method: 'POST'
            });
            const data = await response.json();

            if (data.success) {
                button.classList.toggle('active');
            }
        } catch (error) {
            console.error('Error toggling favorite:', error);
        }
    }

    async copyPrompt(imageId) {
        try {
            const response = await fetch(`/api/image/${imageId}/metadata`);
            const data = await response.json();
            await navigator.clipboard.writeText(data.prompt);

            // Show feedback
            const button = document.querySelector(`[data-image-id="${imageId}"] .action-copy-prompt`);
            const originalText = button.textContent;
            button.textContent = '✓';
            button.classList.add('active');
            setTimeout(() => {
                button.textContent = originalText;
                button.classList.remove('active');
            }, 1000);
        } catch (error) {
            console.error('Error copying prompt:', error);
        }
    }

    setupFullscreenModal() {
        const self = this;
        const modal = document.getElementById('fullscreenModal');
        const modalImage = document.getElementById('modalImage');
        const modelInfo = document.getElementById('modelInfo');
        const promptInfo = document.getElementById('promptInfo');
        const settingsInfo = document.getElementById('settingsInfo');
        const copyPromptBtn = document.getElementById('copyPrompt');
        const downloadBtn = document.getElementById('downloadImage');
        const closeBtn = modal.querySelector('.action-close');

        const showModal = (imageElement) => {
            const galleryItem = imageElement.closest('.gallery-item');
            if (!galleryItem) return;

            // Clone the gallery item and attach it to the modal for reference
            const clonedItem = galleryItem.cloneNode(true);
            clonedItem.style.display = 'none';
            modal.querySelector('.fullscreen-content').appendChild(clonedItem);

            modalImage.src = imageElement.src;
            modal.style.display = 'flex';
            self.manageModalState(true);

            // Reset info sections
            modelInfo.textContent = 'Loading...';
            promptInfo.textContent = 'Loading...';
            settingsInfo.textContent = 'Loading...';

            // Fetch image metadata
            const imageId = galleryItem.dataset.imageId;
            fetch(`/api/image/${imageId}/metadata`)
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
        };

        const hideModal = () => {
            modal.style.display = 'none';
            self.manageModalState(false);

            // Remove the cloned gallery item
            const clonedItem = modal.querySelector('.gallery-item');
            if (clonedItem) {
                clonedItem.remove();
            }
        };

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
            const galleryItem = modal.querySelector('.gallery-item');
            if (!galleryItem) return;

            const imageId = galleryItem.dataset.imageId;
            await this.downloadImage(imageId);
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
        document.querySelector('.gallery-grid').addEventListener('click', (e) => {
            const galleryItem = e.target.closest('.gallery-item');
            if (!galleryItem) return;

            const img = galleryItem.querySelector('img');
            if (img) {
                showModal(img);
            }
        });
    }

    showFeedback(message, type = 'success') {
        const feedback = document.createElement('div');
        feedback.className = `gallery-feedback ${type}`;
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
        }, 3000);
    }

    // Add a new method to manage modal state and scrolling
    manageModalState(isOpen) {
        this.isModalOpen = isOpen;
        document.body.style.overflow = isOpen ? 'hidden' : '';
    }
}

// Initialize gallery when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new GalleryManager();
});