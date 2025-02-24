export class ImageLoader {
    constructor() {
        this.observer = null;
        this.initializeObserver();
    }

    initializeObserver() {
        this.observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        this.loadImage(img);
                        this.observer.unobserve(img);
                    }
                });
            },
            {
                rootMargin: '50px 0px',
                threshold: 0.1
            }
        );
    }

    loadImage(img) {
        if (!img.dataset.src) return;

        // Start loading animation
        img.classList.add('loading');

        // Create a temporary image to load in background
        const tempImg = new Image();

        tempImg.onload = () => {
            img.src = tempImg.src;
            img.classList.remove('loading');
            img.classList.add('loaded');
        };

        tempImg.onerror = () => {
            img.classList.remove('loading');
            img.classList.add('error');
            // Add error placeholder
            img.src = '/static/images/error-placeholder.png';
        };

        tempImg.src = img.dataset.src;
    }

    observe(img) {
        if (!img.dataset.src) {
            img.dataset.src = img.src;
            img.src = '/static/images/placeholder.png';
        }
        this.observer.observe(img);
    }

    observeAll() {
        document.querySelectorAll('.gallery-item img').forEach(img => {
            this.observe(img);
        });
    }

    // Handle newly added images (e.g., from infinite scroll)
    handleNewImages(container) {
        container.querySelectorAll('img:not(.loaded)').forEach(img => {
            this.observe(img);
        });
    }
}

export const imageLoader = new ImageLoader();