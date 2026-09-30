#!/bin/bash
set -e

docker compose build app

# Always synchronize filesystem -> database first
docker compose run --rm app python -m musicrec scan

# Process new/changed/failed songs
docker compose run --rm app python -m musicrec extract --retry

# Rebuild embeddings / language / mood
docker compose run --rm app python -m musicrec build

# Show database/pipeline status
docker compose run --rm app python -m musicrec report

# Start API
docker compose up -d app