# Setup udacity workspace

- copy [code_udacity.sh](code_udacity.sh) in `/home/workspace`
```bash
code --user-data-dir /home/workspace/.vscode/user-data/ --extensions-dir /home/workspace/.vscode/extensions/
```
- save debugger launch config [launch.json](launch.json) in `/home/workspace/.vscode/`
- copy [settings.json](user_data_user_settings.json) to `/home/workspace/.vscode/user-data/user/Settings.json`
- copy [workspaces.json](user_data_workspaces.json) to `/home/workspace/.vscode/user-data/Workspaces/xxxx/workspace.json`

### Note
- virtualenvs are located in `/data/virtual_envs/`. See [.student_bashrc](.student_bashrc) in `/home/workspace`