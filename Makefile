SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build
ENV           = regular

build:
	docker compose build $(ENV)

run:
	docker compose up $(ENV) --detach

stop:
	docker compose down $(ENV)

build-doc:
	docker compose run $(ENV) --rm local $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

serve: run
	docker compose exec $(ENV) @$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	docker compose exec $(ENV) -p 8000:8000 -v $(PWD)/docs/build/html:/app/docs/build "cd docs/build && python3 -m http.server"

test: run
	docker compose exec regular poetry run coverage run -m pytest -vvv -s --doctest-modules . --ignore deduplicate-text-datasets --ignore docs --ignore text_dedup/minhash_spark.py --ignore reference
	docker compose exec regular poetry run coverage xml -o cobertura.xml
	docker compose exec regular poetry run coverage report -m
	docker compose cp regular:/app/cobertura.xml cobertura.xml

spark_test: run
	docker compose exec spark poetry run pytest -vvv -s --doctest-modules tests/test_minhash_spark.py

clean:
	docker system prune -a
