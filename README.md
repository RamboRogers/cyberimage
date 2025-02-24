# CyberImage

**AI Image Generation Platform**

🎨 Stable Diffusion | 🌍 Web Interface | ⚡ Fast | 🎯 Precise | 🖼️ Gallery | 🔄 Queue System

## 🌟 Features

### Image Generation
- Multiple state-of-the-art AI models
- Customizable generation parameters
- Negative prompt support
- Batch image generation
- Real-time generation progress
- Smart queue management

### Web Interface
- Beautiful cyberpunk-themed UI
- Real-time status updates
- Interactive image gallery
- Mobile-responsive design
- Keyboard shortcuts
- Image metadata viewing

### Gallery Features
- Grid, list, and compact views
- Image details modal
- Quick actions (download, delete, copy prompt)
- Infinite scroll loading
- Search and filtering
- Batch operations

### Performance
- Smart model caching
- Memory-optimized pipeline
- Efficient queue system
- Background job processing
- Real-time progress tracking

### Security & Privacy
- No data collection
- Local image storage
- Secure file handling
- Privacy-focused design

## ⚡ Installation

### Requirements
- Python 3.12+ (for local installation)
- CUDA-capable GPU
- 16GB+ RAM recommended
- 24GB+ VRAM recommended
- 250GB+ disk space for models
- Docker & Docker Compose (for containerized installation)
- Huggingface API Key (free! for downloading models)

### Environment Setup
1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure the environment variables in `.env`:

- For a local install, you can use the MODEL_FOLDER and IMAGES_FOLDER to store the models and images locally in different directories.

- For a docker install, you can use the EXTERNAL_MODEL_FOLDER and EXTERNAL_IMAGES_FOLDER to store the models and images externally in different directories.

- For a docker install, you will need a Huggingface API Key to download the models.

- The openai endpoints works fine with Ollama with 127.0.0.1:11434/v1 as the endpoint and the openai api key as the key, or host.docker.internal:11434/v1 as the endpoint and the openai api key as the key if ollama is running on the host machine. The model needs to be something in 127.0.0.1:11434/v1/models on your system.

> ***If you don't configure the openai endpoint, the enrich prompt will not work.***

- The civitai api key is optional, it is used to download models from civitai (not configured currently).

```env
MODEL_FOLDER=./models
IMAGES_FOLDER=./images
EXTERNAL_MODEL_FOLDER=
EXTERNAL_IMAGES_FOLDER=
HF_TOKEN=
OPENAI_ENDPOINT=
OPENAI_API_KEY=
OPENAI_MODEL=
CIVITAI_API_KEY=
```

### Docker Installation (Recommended)

1. Using docker-compose (easiest):
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

2. Using Docker directly:
```bash
# Build the image
docker build -t cyberimage .

# Run the container
docker run -d \
  --name cyberimage \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/images:/app/images \
  --env-file .env \
  cyberimage
```

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cyberimage.git
cd cyberimage
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download models:
```bash
python download_models.py
```

### Quick Start Script
You can use the provided `run.sh` script for common operations:
```bash
# Make the script executable
chmod +x run.sh

# Start the application
./run.sh start

# Stop the application
./run.sh stop

# View logs
./run.sh logs

# Update dependencies
./run.sh update

# Clean temporary files
./run.sh clean
```

## 🚀 Usage

1. Start the server:
```bash
# Using run.sh
./run.sh start

# Or manually
python run.py
```

2. Access the web interface:
```
http://localhost:5000
```

### Generation Parameters
- **Model**: Choose from multiple AI models
- **Prompt**: Describe your desired image
- **Negative Prompt**: Specify elements to exclude
- **Size**: Select output dimensions
- **Steps**: Control generation quality
- **Guidance**: Adjust prompt adherence
- **Batch Size**: Generate multiple images

## 💡 Use Cases
- **Digital Art**: Create unique artwork
- **Concept Design**: Generate design ideas
- **Visual Inspiration**: Explore creative concepts
- **Content Creation**: Generate visual content

## 🎯 Features in Detail

### Queue Management
- Efficient job scheduling
- Real-time queue status
- Smart resource allocation
- Progress tracking
- Job prioritization

### Gallery Management
- Image organization
- Metadata storage
- Quick filtering
- Batch operations
- Export functionality

### User Interface
- Intuitive controls
- Real-time feedback
- Responsive design
- Keyboard shortcuts
- Dark theme

## 🔧 Configuration

Key settings in `config.py`:
```python
# Model Settings
MODEL_CACHE_SIZE = 1
CUDA_DEVICE = "cuda"
PRECISION = "float16"

# Queue Settings
MAX_QUEUE_SIZE = 100
JOB_TIMEOUT = 300

# Storage Settings
IMAGE_STORAGE = "images/"
MODEL_STORAGE = "models/"
```

## ⌨️ Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Focus Search | `/` |
| Next Image | `j` |
| Previous Image | `k` |
| Copy Prompt | `c` |
| Download | `d` |
| Toggle Selection | `Space` |
| Select All | `a` |
| Deselect All | `Shift + a` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## ⚖️ License

CyberImage is licensed under the MIT License.

## 🔗 Connect

- GitHub: @ramborogers
- Twitter: @rogerscissp
- Website: https://matthewrogers.org

---

Made with 💚 by [Matthew Rogers]