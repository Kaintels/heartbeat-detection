#!/bin/bash

curl.exe --output ".\data\mit-bih-arrhythmia-database-1.0.0.zip" https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip
sleep 1
echo -e "\n"
echo "Download successfully. Auto script shutdown after 3 seconds..."
sleep 3