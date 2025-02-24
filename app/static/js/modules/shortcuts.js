export const SHORTCUTS = {
    'j': 'Next image',
    'k': 'Previous image',
    'f': 'Toggle favorite',
    'c': 'Copy prompt',
    'd': 'Download image',
    '/': 'Focus search',
    'esc': 'Close modal/clear selection',
    'space': 'Toggle selection',
    'a': 'Select all',
    'shift+a': 'Deselect all'
};

class ShortcutManager {
    constructor() {
        this.handlers = new Map();
        this.isEnabled = true;
        this.initializeShortcuts();
    }

    initializeShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (!this.isEnabled) return;

            // Don't trigger shortcuts when typing in input fields
            if (e.target.matches('input, textarea')) return;

            const key = this.getKeyCombo(e);
            const handler = this.handlers.get(key);

            if (handler) {
                e.preventDefault();
                handler();
            }
        });
    }

    getKeyCombo(e) {
        const parts = [];
        if (e.shiftKey) parts.push('shift');
        if (e.ctrlKey) parts.push('ctrl');
        if (e.altKey) parts.push('alt');
        parts.push(e.key.toLowerCase());
        return parts.join('+');
    }

    register(key, handler) {
        this.handlers.set(key.toLowerCase(), handler);
    }

    unregister(key) {
        this.handlers.delete(key.toLowerCase());
    }

    disable() {
        this.isEnabled = false;
    }

    enable() {
        this.isEnabled = true;
    }
}

export const shortcutManager = new ShortcutManager();