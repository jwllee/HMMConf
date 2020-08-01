#!/usr/bin/env bash

cd /opt/hmmconf/workspace

# while developing we generally want to update all dependencies on container
# in Production they get baked into the image on build and this is not required
if [[ "${UPDATE_PYTHON_REQUIREMENTS_ON_CONTAINERSTART}" == "true" ]]; then
	printf "updating Python requirements:\n"
	dev pipi
fi

# in local requirements, additional requirements might exist which we always want to install
if [[ -f requirements-local.txt ]]; then
	dev pipi -r requirements-local.txt
fi

# image can run in multiple modes
if [[ "${RUNTYPE}" == "bash" ]]; then
	printf "started Docker container as runtype \e[1;93mbash\e[0m\n"

	if [ $# -eq 0 ]; then
		exec /bin/bash
	else
		"$@" && exec /bin/bash
	fi

elif [[ "${RUNTYPE}" == "jupyterlab" ]]; then
	printf "started Docker container as runtype \e[1;93mjupyterlab\e[0m\n"

	dev pipi jupyterlab==1.2.6 nbresuse==0.3.3

	mkdir ~/.jupyter 2>/dev/null
	jupyter serverextension enable --py nbresuse
	echo "c.NotebookApp.terminals_enabled = False" >> ~/.jupyter/jupyter_notebook_config.py
	echo "c.MappingKernelManager.cull_idle_timeout = 900" >> ~/.jupyter/jupyter_notebook_config.py
	exec jupyter lab --ip=0.0.0.0 --port=8000 --no-browser --NotebookApp.token='' --NotebookApp.password=''
fi
