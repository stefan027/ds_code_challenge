<img src="img/city_emblem.png" alt="City Logo"/>

# City of Cape Town - Data Science Unit Code Challenge

Solution by Stefan Strydom (stefan.strydom87@gmail.com)

## Shortlisted positions
- Head: Data Science
- Senior Professional Officer: Data Science

## Setup

### Install `uv`
`uv` is the recommended package manager. To install `uv` on Linux or MacOS, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or to use `wget` to download the script, use:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
To install `uv` on Windows, execute the following in `PowerShell`:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For detailed installation instructions, see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/).

### Clone this repository
```bash
git clone https://github.com/stefan027/ds_code_challenge.git
```

### Install dependencies
Use `uv` to create a virtual environment and to install all dependencies in the virtual environment. The commands below creates the environment `.venv` in the current directory.

```bash
cd ds_code_challenge
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Data requirements
All data are assumed to be in the `./data` directory. A script to create the `./data` directory and to download the data is provided. To run the script, AWS credentials must already be configured. Alternatively, the dummy credentials file can be downloaded with `wget`:
```bash
wget -O .ds_code_challenge_creds.json https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/ds_code_challenge_creds.json
```

To download all the data from S3 using the dummy credentials file, run the following:
```bash
python scripts/download_data.py -c .ds_code_challenge_creds.json
```

If AWS credentials were already configured, simply run:
```bash
python scripts/download_data.py
```
