#!/bin/sh

echo "Verifying virtual_envs"
ls -lh /data/virtual_envs/ 2> /dev/null

echo "Setting up vscode extensions & user-data dir"
mkdir -p /home/workspace/.vscode/user-data/
mkdir -p /home/workspace/.vscode/extensions
echo "code --user-data-dir /home/workspace/.vscode/user-data/ --extensions-dir /home/workspace/.vscode/extensions/" > ./code.sh
chmod u+x ./code.sh

echo "vs code is set to run using custom user-data-dir and extensions-dir"
echo "\n"
echo "installing vs code launch.json & settings.json"
# download launch.json (Vscode debug settings)
curl -s https://raw.githubusercontent.com/brijs/udacity-self-driving/main/scripts/launch.json --output /home/workspace/.vscode/launch.json

# download User settings.json
curl -s https://raw.githubusercontent.com/brijs/udacity-self-driving/main/scripts/user_data_user_settings.json --output /home/workspace/.vscode/user-data/User/settings.json

echo "Done"
echo "\n"
echo " Run the following to launch vs code"
echo " ./code.sh /home/workspace/"

