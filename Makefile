run-original:
	python3 main.py --algo original --network row

run-dynamic:
	python3 main.py --algo dynamic --network radius --radius 250

save-original:
	python3 main.py --algo original --network row --save --file data.info

save-dynamic:
	python3 main.py --algo dynamic --network radius --radius 250 --save --file data.info

load-original:
	python3 main.py --algo original --network row --load --file data/data.info

load-dynamic:
	python3 main.py --algo dynamic --network radius --radius 250 --load --file data/data.info

.PHONY: save-original save-dynamic load-original load-dynamic
