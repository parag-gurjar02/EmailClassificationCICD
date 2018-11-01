#!/bin/bash

#This file is to download pretrained model file. It is assumed that these two files are already in a blob container

# Downloading Azure CLI on the VSTS build agent machine
apt-get update -y && apt-get install -y python libssl-dev libffi-dev python-dev build-essential
curl -L https://azurecliprod.blob.core.windows.net/install.py -o install.py
printf "/usr/azure-cli\n/usr/bin" | python install.py
az

#Setting environment variables to access the blob container
export AZURE_STORAGE_ACCOUNT=$1
export AZURE_STORAGE_KEY=$2
export container_name=$3
export blob_name1=$4


echo "Azure Storage Account" $AZURE_STORAGE_ACCOUNT
echo "Azure Storage key" $AZURE_STORAGE_KEY
echo "Azure BLOB Container name" $container_name
echo "Azure File Name 1" $blob_name1

echo "Starting download of model file : " $blob_name1

az storage blob download --container-name $container_name --name $blob_name1 --file /$blob_name1 --output table
az storage blob list --container-name $container_name --output table

echo "Download of model file complete : " $blob_name1
