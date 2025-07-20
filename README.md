# retico-language-practice-network

An end-to-end incremental Retico system for live spoken language practice with Gemini.

## Contents

- `gemini.py`: A Python module that defines the logic for dialoguing with the Gemini API and streaming the response tokens.
- `gemini_llm_module.py`: A Retico module that defines incremental processing logic for the LLM interaction part of the network.
- `topics.json`: The list of registered topics broken down by level for the curated language learning track.
- `run_network.py`: A simple script for running the network.

## Installation

  ```bash
  pip install --upgrade pip setuptools wheel
  git clone https://github.com/mi-1000/retico-language-practice-network
  ```
Follow the [setup](#Setup) steps, then run:

  ```bash
  python run_network.py
  ```

### Troubleshooting

The following errors can arise when using the recommended versions for Python and libraries:

- ```plain

  File ".venv/lib/python3.9/site-packages/bangla/__init__.py", line 139, in <module>
      def get_date(passed_date=None, passed_month=None, passed_year=None, ordinal: bool | None = False):
  TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
  ```

  - **Solution:** This error occurs because the `|` operator is not supported in Python 3.9. You can fix this by removing `bool | None` in the source code. Alternatively, you can reinstall an older version of this package, or upgrade to Python 3.10 or later, although this is not recommended at the moment due to potential compatibility issues with other Retico modules.

- ```plain
  File ".venv/lib/python3.9/site-packages/TTS/utils/io.py", line 54, in load_fsspec
      return torch.load(f, map_location=map_location, **kwargs)
  File ".venv/lib/python3.9/site-packages/torch/serialization.py", line 1470, in load
      raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
  _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.

  (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
  ```

  - **Solution:** You can fix this by adding `weights_only=False` in the source code of the module, or by downgrading to PyTorch<=2.5, although the latter solution is also not recommended at the moment due to potential compatibility issues with other Retico modules.

## Setup

> [!WARNING]
> It is recommended to use **Python 3.9.22** with this network in order to avoid any compatibility issues.

- If you haven't yet, create a virtual environment and activate it:

  - **Linux/MacOS**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
  - **Windows**:

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

    If you encounter an error when activating the virtual environment, retry the above command after running the following line:

    ```powershell
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
- Install the dependencies:

  ```bash
  python3 -m pip install -r requirements.txt
  ```
- Set the environment variable `GOOGLE_API_KEY`. The simplest way is to create an `.env` file in the root folder of this repository, and write the following contents:

  ```dotenv
   GOOGLE_API_KEY=your-api-key
  ```

  The API key will be automatically loaded by the network.

  You can get an API key for free on the [Google Cloud website](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys).
