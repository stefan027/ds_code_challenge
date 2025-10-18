<img src="img/city_emblem.png" alt="City Logo"/>

# City of Cape Town - Data Science Unit Code Challenge

Solution by Stefan Strydom (stefan.strydom87@gmail.com)

## Shortlisted positions
- Head: Data Science
- Senior Professional Officer: Data Science

## Setup
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Data is assumed to be in the `./data` directory. To download all the data from S3, run the following:
```bash
python scripts/download_data.py -c .ds_code_challenge_creds.json
```