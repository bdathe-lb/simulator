save-original:
	python3 main.py --algo original --network row --save --file data.info

save-v1:
	python3 main.py --algo v1 --network radius --radius 350 --save --file data.info

save-v2:
	python3 main.py --algo v2 --network radius --radius 350 --save --file data.info

save-v3:
	python3 main.py --algo v3 --network radius --radius 350 --save --file data.info

load-original:
	python3 main.py --algo original --network row --load --file data/data.info

load-v1:
	python3 main.py --algo v1 --network radius --radius 350 --load --file data/data.info

load-v2:
	python3 main.py --algo v2 --network radius --radius 350 --load --file data/data.info

load-v3:
	python3 main.py --algo v3 --network radius --radius 350 --load --file data/data.info

.PHONY: save-original save-v1 save-v2 save-v3  load-original load-v1 load-v2 load-v3
