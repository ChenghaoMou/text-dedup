SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

build:
	docker compose build

run:
	docker compose up --detach

stop:
	docker compose down

build-doc:
	docker compose run --rm local $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

serve: run
	docker compose exec local @$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	docker compose exec local -p 8000:8000 -v $(PWD)/docs/build/html:/app/docs/build "cd docs/build && python3 -m http.server"

test: run
	docker compose exec local poetry run coverage run -m pytest -vvv -s --doctest-modules . --ignore deduplicate-text-datasets --ignore docs --ignore text_dedup/minhash_spark.py --ignore reference
	docker compose exec local poetry run coverage xml -o cobertura.xml
	docker compose exec local poetry run coverage report -m
	docker compose cp local:/app/cobertura.xml cobertura.xml
