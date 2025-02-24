export class SelectionManager {
    constructor() {
        this.selectedItems = new Set();
        this.batchOperationsEl = document.querySelector('.batch-operations');
        this.selectedCountEl = document.querySelector('.selected-count');
        this.initializeListeners();
    }

    initializeListeners() {
        // Handle item selection
        document.addEventListener('click', (e) => {
            const item = e.target.closest('.gallery-item');
            if (!item) return;

            if (e.shiftKey) {
                this.toggleSelection(item);
            }
        });

        // Handle batch operations
        const batchButtons = document.querySelectorAll('.batch-actions button');
        batchButtons.forEach(button => {
            button.addEventListener('click', () => {
                const action = button.classList[0].replace('batch-', '');
                this.executeBatchAction(action);
            });
        });
    }

    toggleSelection(item) {
        const itemId = item.dataset.imageId;

        if (this.selectedItems.has(itemId)) {
            this.selectedItems.delete(itemId);
            item.classList.remove('selected');
        } else {
            this.selectedItems.add(itemId);
            item.classList.add('selected');
        }

        this.updateUI();
    }

    selectAll() {
        document.querySelectorAll('.gallery-item').forEach(item => {
            const itemId = item.dataset.imageId;
            this.selectedItems.add(itemId);
            item.classList.add('selected');
        });
        this.updateUI();
    }

    deselectAll() {
        this.selectedItems.clear();
        document.querySelectorAll('.gallery-item').forEach(item => {
            item.classList.remove('selected');
        });
        this.updateUI();
    }

    updateUI() {
        const count = this.selectedItems.size;

        if (count > 0) {
            this.batchOperationsEl.classList.add('visible');
            this.selectedCountEl.textContent = `${count} selected`;
        } else {
            this.batchOperationsEl.classList.remove('visible');
        }
    }

    async executeBatchAction(action) {
        const selectedIds = Array.from(this.selectedItems);

        switch (action) {
            case 'download':
                this.downloadSelected(selectedIds);
                break;
            case 'delete':
                await this.deleteSelected(selectedIds);
                break;
            case 'tag':
                this.showTagDialog(selectedIds);
                break;
        }
    }

    async downloadSelected(ids) {
        for (const id of ids) {
            const response = await fetch(`/api/get_image/${id}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `image-${id}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    }

    async deleteSelected(ids) {
        if (!confirm(`Are you sure you want to delete ${ids.length} images?`)) {
            return;
        }

        try {
            await Promise.all(ids.map(id =>
                fetch(`/api/delete_image/${id}`, { method: 'DELETE' })
            ));

            // Remove deleted items from DOM
            ids.forEach(id => {
                const item = document.querySelector(`[data-image-id="${id}"]`);
                if (item) item.remove();
            });

            this.selectedItems.clear();
            this.updateUI();
        } catch (error) {
            console.error('Error deleting images:', error);
            alert('Failed to delete some images. Please try again.');
        }
    }

    showTagDialog(ids) {
        // Implementation for tag dialog
        console.log('Show tag dialog for:', ids);
    }
}

export const selectionManager = new SelectionManager();