#!/bin/sh

mkdir -p /home/workspace/.vscode/user-data/
mkdir -p /home/workspace/.vscode/extensions

code --user-data-dir /home/workspace/.vscode/user-data/ --extensions-dir /home/workspace/.vscode/extensions/

