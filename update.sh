#!/bin/bash
docker compose build app                                   # only the last layer rebuilds, seconds
docker compose run app python -m musicrec extract --retry   # the 3 failed files, ~1 min
docker compose run app python -m musicrec build
docker compose run app python -m musicrec report
docker compose up -d app                                   # http://localhost:8000