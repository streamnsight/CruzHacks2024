# Create AI apps with LLMs

## Requirements

This has been tested with Python 3.11

## Getting started

- Create a virtual environment, and activate it

    ```bash
    python3 -m venv .venv
    . .venv/bin/activate
    ```

- Upgrade pip (just a good thing to do)

    ```bash
    pip install --upgrade pip
    ```

- Install requirements

    ```bash
    pip install -r requirements.txt
    ```

    if you are on Mac M1/M2, you might want to use 

    ```bash
    CT_METAL=1 pip install ctransformers --no-binary ctransformers
    ```

    To use a llama2 model locally on M1/M2 use:

    ```
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
    ```

    - To use OpenAI ChatGPT, you need to set your API key in the environment by setting:

    ```
    export OPENAI_API_KEY="..."
    ```

## Use the Notebook

- Open Jupyter Lab
```bash
cd notebooks
jupyter lab
```

- Then open the notebook


## Use the app

```bash
cd app
gradio app.py
```


If you want to share the app over the internet, use the `demo.launch(share=True)` option in the code.

Note this creates a proxy to your local computer, and is NOT a solution for actual deployment. Consider some cloud provider instance with public IP.

