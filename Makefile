

default:
	@echo "Please specify a make command."

main:
	@python drl-chess/main.py main

gen:
	@python drl-chess/main.py gen

load:
	@python drl-chess/main.py load
