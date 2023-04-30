# mv-klotsid

My machine vision project is to track and count custom wooden bricks.

Project documentation on my [https://taunoerik.art/2023/04/30/machine-vision-project/](blog).

## Set Up a Virtual Environment

```Bash
pip install virtualenv
python3 --version
python3.10 -m venv env
source env/bin/activate
pip list

pip freeze > requirements.txt
~ pip install -r requirements.txt

~ deactivate
```

## CUDA error out of memory

```Bash
nvidia-smi
kill -KILL PID_number
```

