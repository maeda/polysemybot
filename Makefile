FILE?=starwars.cpu.latest.tar.gz
.ONESHELL:

default: setup deploy-model

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || python -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate

setup: venv

delete-model:
	rm -Rf ./target

deploy-model:
	tar -zxvf $(FILE)

console: venv
	. venv/bin/activate ; \
	python app.py -te starwars

