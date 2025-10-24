#!/bin/bash
sudo apt update -y
sudo apt install -y python3-pip python3-venv
sudo pip install -U pip setuptools wheel
sudo pip install -r requirements.txt