# water-treatment-cv-llm
An example of a computer vision monitoring app with a LLM frontend built in Python.

The backend monitors a synthetic water treatment plant. Can easily be modified to incorporate connection to real APIs.

An LLM using Antrophic is built into the backend so it can query a synthetic segmentation model which monitors the water treatment plant. In production this would be replaced by e.g. a mask RCNN monitoring model.

The frontend provides a monitoring interface (built with tailwind css for a nice look) and with an LLM interface.

## System Architecture

React Frontend (localhost:3000) <br>
       ↕ HTTP API calls <br>
FastAPI Backend (localhost:8000)  <br>
       ↕ Processes <br>
LLM Parser + Mask R-CNN Simulator <br>

# HOW TO USE:

Create virtual environment for backend and frontend
  
  ## Frontend
  
  1) conda create -n front-end python=3.9
  2) pip install -r water-treatment-cv-llm/frontend-requirements.txt

- 1. Create React app
  1) cd water-treatment-cv-llm
  npx create-react-app water-treatment-frontend
  cd water-treatment-frontend

- 2. Install dependencies
  npm install lucide-react
  npm install -D tailwindcss postcss autoprefixer
  npx tailwindcss init -p
  
## Backend
  1) conda create -n back-end python=3.9
  2) pip install -r water-treatment-cv-llm/water-treatment-cv-backend/requirements.txt
  
  
- put Anthropic API key in .env file


- Start frontend:
  1) cd water-treatment-cv-llm/water-treatment-cv-frontend/src
  2( conda activate frontend	  
  3) npm run start

- Start backend:
  1) cd water-treatment-cv-llm/water-treatment-cv-backend
  2) uvicorn main:app --reload --host 0.0.0.0 --port 8000


##
Now you can access the app at localhost:3000!

