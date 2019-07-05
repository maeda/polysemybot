FILE?=starwars.cpu.latest.tar.gz
DATASET?=./data/starwars.txt
.ONESHELL:

default: setup deploy-model

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || python -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate

setup: venv

clean:
	rm -Rf ./target

package:
	tar -zcvf $(FILE) ./target/starwars

deploy:
ifeq "$(wildcard ./target)" ""
	mkdir ./target
endif
	tar -zxvf $(FILE)

console: venv
	. venv/bin/activate ; \
	python app.py -te starwars

train: venv
	. venv/bin/activate ; \
	python app.py -tr ${DATASET} -it 10000 -hi 300 -s 1000 -lr 0.01 -la 1 -d 0.0
