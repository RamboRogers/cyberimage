# CyberImage

**AI Image Generation Platform**

<table>
<tr>
<td><img src="media/generate.png"></td>
<td><img src="media/gallery.png"></td>
</tr>
<tr>
<td><img src="media/queue.png"></td>
<td><img src="media/single.png"></td>
</tr>
</table>

<div align="center">

🎨 Stable Diffusion | 🌍 Web Interface | ⚡ Fast | 🎯 Precise | 🖼️ Gallery | 🔄 Queue System | 📡 API | 🤖 MCP Support

</div>

## 🌟 Features

<div align="center">

| 🎨 **Image Generation** | 🖥️ **Web Interface** | 🖼️ **Gallery Features** | ⚡ **Performance** |
|------------------------|---------------------|------------------------|-------------------|
| 🤖 State-of-the-art AI models | 🌃 Cyberpunk-themed UI | 📊 Multiple view options | 💾 Smart model caching |
| 🎛️ Customizable parameters | ⏱️ Real-time status updates | 🔍 Detailed image modal | 🧠 Memory optimization |
| 🚫 Negative prompt support | 🖱️ Interactive gallery | ⬇️ Quick download actions | 🔄 Efficient queue system |
| 📦 Batch image generation | 📱 Mobile-responsive design | 📋 Copy prompt feature | 🏃‍♂️ Background processing |
| 📈 Real-time progress | 🌈 Beautiful UI | 🔍 Search and filtering | 🔒 No data collection |
| 🎯 Precise control | 🎮 Intuitive controls | 🏷️ Tagging system | 🏠 Local image storage |
| 🧩 Model compatibility | 🌙 Dark mode support | | |
| 🤖 **MCP Integration** | 🔌 **AI Accessibility** | | |
| 🔗 AI assistant support | 🔄 JSON-RPC interface | | |

</div>

## 🤖 Model Context Protocol (MCP) Support

CyberImage now implements the [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/specification/2024-11-05/), enabling AI assistants and other tools to seamlessly generate images through a standardized interface.

### What is MCP?

MCP is an open protocol that enables AI systems to interact with external tools and capabilities in a standardized way. With MCP support, AI assistants can generate images directly through CyberImage using JSON-RPC calls.

### Key MCP Features

- **JSON-RPC 2.0 Interface**: Simple, standardized format for all requests
- **Dynamic Default Model**: Uses the system's default model (same as the web UI), prioritizing "flux-1" if available
- **Seamless Queue Integration**: Jobs from AI assistants are integrated into the same queue as web UI requests
- **Progress Tracking**: AI systems can track generation progress in real-time
- **Standard Format**: Follows the MCP specification for interoperability with any MCP-compatible AI system

### Supported MCP Methods

| Method | Description |
|--------|-------------|
| `context.image_generation.models` | List all available models |
| `context.image_generation.generate` | Generate images based on a prompt |
| `context.image_generation.status` | Check the status of a generation job |

### Using the MCP Endpoint

AI assistants can connect to the MCP endpoint at:

```
http://localhost:5050/api/mcp
```

For implementation examples, see the `examples/` directory:
- `mcp_client_example.py`: General MCP client implementation
- `ai_assistant_mcp_example.py`: Specialized client for AI assistants

## The Enhance/Enrich Button

The Enrich button uses the openai api to enhance the image. It uses the openai api key and the openai model to enhance the image. The openai model is the same as the model used for generation.

This is my favorite feature, it allows you to take a basic image prompt and enhance it to make it better using a number of different techniques shared with myself by an AI expert.

<table>
<tr>
<td><img src="media/basic.png"></td>
<td><img src="media/enrich.png"></td>
</tr>
</table>


## ⚡ Installation

### Requirements

| Requirement | Specification |
|-------------|---------------|
| Python | 3.12 (for local installation) |
| GPU | CUDA-capable |
| RAM | 16GB+ recommended |
| VRAM | 24GB+ recommended |
| Disk Space | 250GB+ for models |
| Container | Docker & Docker Compose (for containerized installation) |
| API | Huggingface API Key (free! for downloading models) |

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

  1. Clone the repository:
```bash
git clone https://github.com/ramborogers/cyberimage.git
cd cyberimage
```
2. Use the run.sh script to start the application (easiest):
```bash
# This will start the application in a container
./run.sh start
```

3. Using docker-compose :
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

3. Using Docker directly:
```bash
# Build the image
docker build -t cyberimage .

# Run the container
docker run -d \
  --name cyberimage \
  --gpus all \
  -p 7860:5050 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/images:/app/images \
  --env-file .env \
  cyberimage
```

4. Open in browser:
```
http://localhost:7860
```

![CyberImage](media/generate.png)

![CyberImage](media/gallery.png)

![CyberImage](media/queue.png)

![CyberImage](media/single.png)


### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/ramborogers/cyberimage.git
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

### Generation Parameters
- **Model**: Choose from multiple AI models
- **Prompt**: Describe your desired image
- **Negative Prompt**: Specify elements to exclude
- **Size**: Select output dimensions
- **Steps**: Control generation quality
- **Guidance**: Adjust prompt adherence
- **Batch Size**: Generate multiple images

## 💡 Use Cases
- **AI Assistant Integration**: Allow AI assistants to generate images based on user conversations
- **Family Images**: My children love to use this
- **Digital Art**: Create unique artwork
- **Concept Design**: Generate design ideas
- **Visual Inspiration**: Explore creative concepts
- **Content Creation**: Generate visual content


## 🔧 Configuration

Key settings in `.env`:
```python
# Model Settings

# Don't change these unless you are NOT using docker
MODEL_FOLDER=./models
IMAGES_FOLDER=./images

# For usage with docker (models will grow > 250GB)
EXTERNAL_MODEL_FOLDER=
EXTERNAL_IMAGES_FOLDER=

# Huggingface API Key (free! for downloading models)
HF_TOKEN=

# OpenAI API Key (for enrich prompt), ollama works fine with 127.0.0.1:11434/v1 as the endpoint and the openai api key as the key, or host.docker.internal:11434/v1 as the endpoint and the openai api key as the key if ollama is running on the host machine. The model needs to be something in 127.0.0.1:11434/v1/models on your system.
OPENAI_ENDPOINT=
OPENAI_API_KEY=

# OpenAI Model (for enrich prompt button to work)
OPENAI_MODEL=

# Civitai API Key (optional, for downloading models) (not configured currently)
CIVITAI_API_KEY=

# Format: MODEL_NAME=<name>;<repo>;<description>;<source>;<requires_auth>
# Note: When sourcing this file in a shell, quotes are required around values with semicolons
# When used as .env file directly, quotes are optional but must be consistent
MODEL_1="flux-1;black-forest-labs/FLUX.1-dev;FLUX Dev;huggingface;true"
MODEL_2="sd-3.5;stabilityai/stable-diffusion-3.5-large;Stable Diffusion 3.5;huggingface;true"
MODEL_3="flux-schnell;black-forest-labs/FLUX.1-schnell;FLUX Schnell;huggingface;true"
# MODEL_4="my-custom-model;civitai:12345;My Custom Model;civitai;true"

# Enable/disable downloading specific models (values: true/false)
DOWNLOAD_MODEL_1=true
DOWNLOAD_MODEL_2=true
DOWNLOAD_MODEL_3=tue
# DOWNLOAD_MODEL_4=false
```

## 🖼️ Managing Models

CyberImage uses environment variables to configure models. You can easily add, remove, or modify models by editing the `.env` file.

### Model Configuration Format

Models are defined using the following format:

```
MODEL_<N>=<name>;<repo>;<description>;<source>;<requires_auth>
```

Where:
- `<N>`: Numerical index (1, 2, 3, etc.)
- `<name>`: Unique identifier for the model (used as directory name)
- `<repo>`: HuggingFace repository path or model identifier
- `<description>`: Human-readable description
- `<source>`: Source platform (huggingface, civitai, etc.)
- `<requires_auth>`: Whether authentication is required (true/false)

### Adding Models

To add a new model, simply add a new line to your `.env` file with an unused index:

```
MODEL_1=flux-1;black-forest-labs/FLUX.1-dev;FLUX base model;huggingface;true
MODEL_2=sd-3.5;stabilityai/stable-diffusion-3.5-large;Stable Diffusion 3.5;huggingface;true
MODEL_3=animagine-xl;cagliostrolab/animagine-xl-4.0;Animagine XL;huggingface;true
```

### Disabling Models

You can disable a model's download without removing it from the configuration:

```
MODEL_3=animagine-xl;cagliostrolab/animagine-xl-4.0;Animagine XL;huggingface;true
DOWNLOAD_MODEL_3=false
```

This keeps the model in the UI but prevents it from being downloaded automatically.

### Removing Models

To completely remove a model, simply delete or comment out its configuration line in the `.env` file:

```
MODEL_1=flux-1;black-forest-labs/FLUX.1-dev;FLUX base model;huggingface;true
MODEL_2=sd-3.5;stabilityai/stable-diffusion-3.5-large;Stable Diffusion 3.5;huggingface;true
# MODEL_3=animagine-xl;cagliostrolab/animagine-xl-4.0;Animagine XL;huggingface;true
```

### Model Type Detection

CyberImage automatically detects the model type based on the model name:
- Names containing "flux" are treated as FLUX models
- Names containing "sd-3" are treated as SD3 models
- Names containing "xl" or "sdxl" are treated as SDXL models
- Names containing "animagine" are treated as SDXL architecture models

### Example Configuration

Here's a complete example with multiple models:

```
# Format: MODEL_NAME=<name>;<repo>;<description>;<source>;<requires_auth>
# Note: When sourcing this file in a shell, quotes are required around values with semicolons
# When used as .env file directly, quotes are optional but must be consistent
MODEL_1="flux-1;black-forest-labs/FLUX.1-dev;FLUX Dev;huggingface;true"
MODEL_2="sd-3.5;stabilityai/stable-diffusion-3.5-large;Stable Diffusion 3.5;huggingface;true"
MODEL_3="flux-schnell;black-forest-labs/FLUX.1-schnell;FLUX Schnell;huggingface;true"
# MODEL_4="my-custom-model;civitai:12345;My Custom Model;civitai;true"

# Enable/disable downloading specific models (values: true/false)
DOWNLOAD_MODEL_1=true
DOWNLOAD_MODEL_2=true
DOWNLOAD_MODEL_3=tue
# DOWNLOAD_MODEL_4=false
```

After changing model configurations, restart the application to apply the changes.


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

<div align="center">

## ⚖️ License

<p>
NetVentory is licensed under the GNU General Public License v3.0 (GPLv3).<br>
<em>Free Software</em>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)

### Connect With Me 🤝

[![GitHub](https://img.shields.io/badge/GitHub-RamboRogers-181717?style=for-the-badge&logo=github)](https://github.com/RamboRogers)
[![Twitter](https://img.shields.io/badge/Twitter-@rogerscissp-1DA1F2?style=for-the-badge&logo=twitter)](https://x.com/rogerscissp)
[![Website](https://img.shields.io/badge/Web-matthewrogers.org-00ADD8?style=for-the-badge&logo=google-chrome)](https://matthewrogers.org)

![RamboRogers](https://github.com/RamboRogers/netventory/raw/master/media/ramborogers.png)

</div>

---

Made with 💚 by [Matthew Rogers]