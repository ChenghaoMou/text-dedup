DOCKER = docker
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

build:
	${DOCKER} compose build

up:
	${DOCKER} compose up --detach

down:
	${DOCKER} compose down

build-doc: up
	${DOCKER} compose run $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

serve: up build-doc
	cd "$(BUILDDIR)" && python3 -m http.server

test: up
	${DOCKER} compose exec local poetry run coverage run -m pytest -vvv -s --doctest-modules . --ignore deduplicate-text-datasets --ignore docs --ignore text_dedup/minhash_spark.py --ignore tests/test_benchmark.py
	${DOCKER} compose exec local poetry run coverage xml -o cobertura.xml
	${DOCKER} compose exec local poetry run coverage report -m
	${DOCKER} compose cp local:/app/cobertura.xml cobertura.xml

benchmark: up
	${DOCKER} compose exec local poetry run python tests/test_benchmark_core.py
	${DOCKER} compose exec local poetry run python tests/test_benchmark_news.py

spark_test: up
	${DOCKER} compose exec local poetry run pytest -vvv -s --doctest-modules tests/test_minhash_spark.py

clean:
	${DOCKER} system prune -a
