Hello World
 
# Project Structure

```
.
├── main.py
├── pyproject.toml
├── .python-version
├── README.md
├── uv.lock
└── .venv
    ├── bin
    │   ├── activate
    │   ├── activate.bat
    │   ├── activate.csh
    │   ├── activate.fish
    │   ├── activate.nu
    │   ├── activate.ps1
    │   ├── activate_this.py
    │   ├── deactivate.bat
    │   ├── pydoc.bat
    │   ├── python -> /usr/bin/python3.9
    │   ├── python3 -> python
    │   └── python3.9 -> python
    ├── CACHEDIR.TAG
    ├── .gitignore
    ├── lib
    │   └── python3.9
    │       └── site-packages
    │           ├── __pycache__
    │           │   └── _virtualenv.cpython-39.pyc
    │           ├── _virtualenv.pth
    │           └── _virtualenv.py
    ├── lib64 -> lib
    ├── .lock
    └── pyvenv.cfg

7 directories, 24 files
```
 
## Managing version

`uv version`  
hello-world 0.1.0  

`uv version --short`  
0.1.0  

`uv version --output-format json`  

{
  "package_name": "hello-world",
  "version": "0.1.0",
  "commit_info": null
}
 
## Building distributions from project  
 
`uv build`  
 
```

.
├── dist
│   ├── hello_world-0.1.0-py3-none-any.whl
│   └── hello_world-0.1.0.tar.gz
├── example.py
├── hello_world.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── main.py
├── __pycache__
│   └── app.cpython-39.pyc
├── pyproject.toml
├── README.md
└── uv.lock

3 directories, 13 files
```
