# mv-klotsid

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

