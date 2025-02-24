export class ViewManager {
    constructor() {
        this.currentView = 'grid';
        this.container = document.querySelector('.gallery-grid');
        this.initializeViewToggles();
    }

    initializeViewToggles() {
        const toggles = document.querySelectorAll('.view-toggle');
        toggles.forEach(toggle => {
            toggle.addEventListener('click', () => {
                const view = toggle.dataset.view;
                this.setView(view);

                // Update active state
                toggles.forEach(t => t.classList.remove('active'));
                toggle.classList.add('active');
            });
        });
    }

    setView(view) {
        if (this.currentView === view) return;

        // Remove old view
        this.container.classList.remove(`view-${this.currentView}`);

        // Add new view
        this.container.classList.add(`view-${view}`);
        this.container.dataset.view = view;

        // Store current view
        this.currentView = view;

        // Save preference
        localStorage.setItem('preferred-view', view);

        // Dispatch event for other modules
        window.dispatchEvent(new CustomEvent('viewchange', { detail: { view } }));
    }

    getStoredView() {
        return localStorage.getItem('preferred-view') || 'grid';
    }

    initialize() {
        const storedView = this.getStoredView();
        this.setView(storedView);

        // Set active state on the correct toggle
        const activeToggle = document.querySelector(`[data-view="${storedView}"]`);
        if (activeToggle) {
            activeToggle.classList.add('active');
        }
    }
}

export const viewManager = new ViewManager();