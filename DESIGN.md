- **Project Name:** CyberImage
- **Overview:** Design a dynamic cyberpunk flux image generator that allows users to create, view, and explore images with advanced customization and user experience.
- **Core Features:**
  - **Model Selection:** Users can choose from various generation models for different styles and effects.
  - **Queue System:** Implement a queuing system to manage multiple image generation requests efficiently.
  - **Dynamic Gallery View:** A gallery displaying all generated images in PNG or JPG format, with a grid view for effective browsing.
  - **Dynamic Generation View:** Real-time display of image generation progress and result preview.
  - **Enrich Prompt Button:** A feature that sends the current prompt to an OpenAI-compatible endpoint to expand the prompt into a more detailed and epic version.
  - **Image Display and Format:** Generated images should be displayed in high-quality PNG or JPG format.
  - **Prompt and Audit Logging:** Save all prompts along with generation data (model, time, and settings) in a SQLite table for auditing and reference.
  - **Image Details View:** Clicking on an image shows:
    - Image preview
    - Date and time of generation
    - Model used
    - Generation prompt
    - All step settings and seed values
  - **Unique Image Naming:** Each image is named using a unique ID for easy reference and retrieval.
- **Technical Requirements:
  - **Technologies to be Used:**
    - **Python 3.12** for backend logic and system integrations.
    - **Flask** for building the web application with dynamic routing and responsive views.
    - **Hugging Face** for model integration and prompt enhancement.
    - **Docker** for containerization, ensuring consistent deployment across environments.
  - **Storage Design Consideration:**
    - External volumes or folders should be used for image storage to separate application data from container storage.
    - Image storage paths should be configurable for flexible deployment and scalability.
  - **Event Logging and Auditing:**
    - All user actions and system events are logged in SQLite for complete auditing.
    - An audit view is available to review SQLite audit data, including prompts, model selections, generation settings, and image views.
    - All system events are logged to the console using Python logging with detailed datetime stamps and event context for debugging and monitoring.**
  - **Frontend:** Dynamic and responsive UI for gallery and generation views.
  - **Backend:** Efficient handling of image generation requests and prompt enrichment via OpenAI-compatible API.
  - **Database:** SQLite database to store prompts, audit data, and image metadata.
- **User Experience Flow:**
  1. User enters a prompt and selects a generation model.
  2. User can enhance the prompt using the Enrich Prompt Button.
     - cinematic
     - asymmetrical
     - unique
     - rule of thirds
     - strong negative space
     - production photography
  3. Generation request is added to the queue and processed.
  4. Image generation progress is shown in real-time.
  5. Result is displayed in the dynamic generation view.
  6. Image is saved in the gallery with the unique ID.
  7. User can click on an image to view details and audit information.
- **- **UI Design:**
  - **Color Scheme:**
    - Primary Color: Neon Green
    - Background: Black
    - Data Text: White
  - **Theme and Style:**
    - Cyberpunk dynamic, vivid, and beautiful aesthetics.
    - High contrast with vibrant neon elements for a futuristic feel.
    - Sleek animations and fluid transitions to enhance user experience.
    - Dynamic layouts for a visually engaging experience.

- **Additional Notes:****
  - Ensure the UI/UX is sleek and intuitive for seamless navigation.
  - Optimize image loading and caching for faster gallery performance.
  - Secure API endpoints and data storage for user-generated content.

