SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build
ENV           = regular

build:
	docker compose build $(ENV)

up:
	docker compose up $(ENV) --detach

down:
	docker compose down $(ENV)

build-doc: up
	docker compose run $(ENV) $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

serve: up build-doc
	cd "$(BUILDDIR)" && python3 -m http.server

test: up
	docker compose exec regular poetry run coverage run -m pytest -vvv -s --doctest-modules . --ignore deduplicate-text-datasets --ignore docs --ignore text_dedup/minhash_spark.py --ignore reference --ignore tests/test_minhash_spark.py --ignore tests/test_benchmark.py
	docker compose exec regular poetry run coverage xml -o cobertura.xml
	docker compose exec regular poetry run coverage report -m
	docker compose cp regular:/app/cobertura.xml cobertura.xml

spark_test: up
	docker compose exec spark poetry run pytest -vvv -s --doctest-modules tests/test_minhash_spark.py

clean:
	docker system prune -a
