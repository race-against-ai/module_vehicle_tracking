# RAAI Module Template
A GitHub Template for creating modular RAAI Components

## Stuff you need to change
- Change the ```README.md``` according to your project
- rename the ```your_project_folder``` to the name of your project
- edit ```.gitattributes``` to the new folder
- put your GitHub or Volkswagen email in the setup.py
- import and execute your main file from the project folder into the root ```main.py```
- adjust the ```pyinstaller.spec``` according to your project (mainly the name)

## Code Syntax
To test your code type syntax run

```
tox -e types
```

or to check the style syntax run
```
tox -e styles
```