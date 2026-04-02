# Naive Bayes Worked Example

This folder contains a short teaching notebook for introducing Gaussian Naive Bayes with a materials-science-flavored dataset from the Materials Project API.

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create a local environment file:

   ```bash
   copy .env.example .env
   ```

3. Add your Materials Project API key to `.env`:

   ```env
   MP_API_KEY=your_api_key_here
   ```

4. Start Jupyter:

   ```bash
   uv run jupyter lab
   ```

The `.env` file is ignored by the repository root `.gitignore`, so the key will not be committed if you push this folder to GitHub.
